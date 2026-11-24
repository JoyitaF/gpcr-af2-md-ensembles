"""Post-processing of the Cartesian results: cluster occupancy, nearest-MD distances,
model-index effects and the paired seed/MD-block bootstrap.

Everything here reads `results/<mode>/cartesian/<receptor>/cache.pkl` (written by the
Cartesian stage) and the AF2 file list; nothing recomputes the core analyses.

The number of MD replicates is a parameter, so the same code runs on the public
single-replicate data and on the full three-replicate data.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from .config import Layout, Settings

# --- loading -----------------------------------------------------------------

_SAFE = {
    "numpy": {"ndarray", "dtype"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.numeric": {"_frombuffer"},
    "sklearn.preprocessing._data": {"StandardScaler"},
    "sklearn.decomposition._pca": {"PCA"},
    "sklearn.cluster._kmeans": {"KMeans"},
}


class _CacheReader(pickle.Unpickler):
    """Load only the array and scikit-learn classes the Cartesian cache contains."""

    def find_class(self, module, name):
        module = module.replace("numpy.core.", "numpy._core.")
        if name in _SAFE.get(module, ()):
            return super().find_class(module, name)
        raise ValueError(f"Unexpected class in cache: {module}.{name}")


def load_cache(cart_dir: Path) -> tuple[dict, dict]:
    with (cart_dir / "cache.pkl").open("rb") as fh:
        cache = _CacheReader(fh).load()
    prov = json.loads((cart_dir / "PROVENANCE.json").read_text())
    n_rep = len(prov["md_inputs"])
    if len(cache["md"]["pooled"]) % n_rep:
        raise ValueError(f"{cart_dir}: pooled MD frames do not split into {n_rep} equal replicates")
    return cache, prov


def af2_paths(layout: Layout, receptor: str, label: str) -> list[Path]:
    """Same order the core reads them in (sorted glob), so labels line up with files."""
    root = layout.af2() / receptor / label.replace(":", "x")
    return sorted(root.glob("seed_*/*_unrelaxed_rank_*.pdb"))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- clusters and model-index effects ------------------------------------------

MODEL_RE = re.compile(r"model_(\d+)")


def cluster_tables(cache: dict, paths_by_label: dict[str, list[Path]], receptor: str) -> tuple[list, list]:
    s = cache["states"]
    pop_md = np.asarray(s["pop_md"])
    occupancy, agreement = [], []
    for label, labels in s["labels_af2"].items():
        labels = np.asarray(labels)
        used, counts = np.unique(labels, return_counts=True)
        occupancy.append({"receptor": receptor, "msa_depth": label, "n_models": len(labels),
                          "clusters_occupied": len(used),
                          "af2_counts": ";".join(f"C{c}:{n}" for c, n in zip(used, counts)),
                          "md_percent_of_occupied": ";".join(f"C{c}:{100 * pop_md[c]:.1f}" for c in used)})
        paths = paths_by_label[label]
        if len(paths) != len(labels):
            raise ValueError(f"{receptor} {label}: {len(paths)} files but {len(labels)} labels")
        seeds = [p.parent.name for p in paths]
        models = [int(MODEL_RE.search(p.name).group(1)) if MODEL_RE.search(p.name) else None for p in paths]
        same_model, same_seed = [], []
        for i, j in combinations(range(len(labels)), 2):
            if models[i] is not None and models[i] == models[j] and seeds[i] != seeds[j]:
                same_model.append(labels[i] == labels[j])
            elif seeds[i] == seeds[j] and models[i] != models[j]:
                same_seed.append(labels[i] == labels[j])
        agreement.append({"receptor": receptor, "msa_depth": label,
                          "same_model_index_across_seeds_pct": 100 * np.mean(same_model) if same_model else np.nan,
                          "same_seed_across_model_indices_pct": 100 * np.mean(same_seed) if same_seed else np.nan})
    return occupancy, agreement


# --- nearest MD frame ---------------------------------------------------------

def temporally_excluded_reference(x, frames_per_replica, radius, dimension, n_rep=3, chunk=128):
    """Exact nearest different MD frame, excluding ±radius in same replica."""
    count = len(x)
    if count != n_rep * frames_per_replica:
        raise ValueError("replica size mismatch")
    out = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        d = cdist(x[start:stop, :dimension], x[:, :dimension], "euclidean")
        query = np.arange(start, stop, dtype=int)
        for row, i in enumerate(query):
            a = (i // frames_per_replica) * frames_per_replica
            low, high = max(a, i - radius), min(a + frames_per_replica, i + radius + 1)
            d[row, low:high] = np.inf
        out[start:stop] = np.min(d, axis=1)
    if not np.all(np.isfinite(out)):
        raise ValueError("No temporally independent MD neighbor found")
    return out


def nearest_md(cache: dict, receptor: str, n_rep: int, exclusion_frames: int, pcs=(3, 20)) -> list[dict]:
    xs = cache["states"]["X_md"]
    per_rep = len(xs) // n_rep
    rows = []
    for n_pc in pcs:
        if xs.shape[1] < n_pc:
            raise ValueError(f"{receptor}: fewer than {n_pc} MD distance PCs")
        ref = temporally_excluded_reference(xs, per_rep, radius=exclusion_frames, dimension=n_pc, n_rep=n_rep)
        threshold = float(np.quantile(ref, .95))
        for label, x_af2 in cache["states"]["X_af2"].items():
            nearest = np.min(cdist(x_af2[:, :n_pc], xs[:, :n_pc]), axis=1)
            above = int(np.count_nonzero(nearest > threshold))
            rows.append({"receptor": receptor, "msa_depth": label, "n_pc": n_pc, "n_models": len(nearest),
                         "reference_md_frames": len(ref), "md_distance_p95": threshold,
                         "af2_above_p95_n": above, "af2_above_p95_pct": 100 * above / len(nearest)})
    return rows


# --- paired bootstrap ---------------------------------------------------------

def modes_from_gram(aligned):
    """Exact top-10 right singular vectors using the n<=100 sample Gram matrix."""
    n = len(aligned)
    if n < 11:
        raise ValueError("Need >=11 structures for ten PCA modes")
    x = aligned.reshape(n, -1).astype(np.float64, copy=False)
    y = x - x.mean(axis=0)
    gram = y @ y.T  # (n,n), avoiding eigen-decomposition of 3*N Cartesian matrix
    eigen, U = eigh(gram, subset_by_index=(n-10, n-1), driver="evr", check_finite=False)
    if eigen[0] <= 1e-10:
        raise ValueError("Rank-deficient AF2 sample, cannot compute ten PCA modes")
    v = y.T @ U
    v /= np.sqrt(eigen)[None, :]
    rmsf = np.sqrt(np.mean(np.sum(y.reshape(n, -1, 3)**2, axis=2), axis=0))
    return v, rmsf


def md_stats_from_moments(sumx, xx, n):
    """MD PCA covariance and per-residue RMSF from sufficient statistics."""
    mean = sumx / n
    cross_centered = xx - np.outer(sumx, mean)
    cov = (cross_centered + cross_centered.T) * (0.5 / (n - 1))
    d = len(mean)
    eig, v = eigh(cov, subset_by_index=(d - 10, d - 1), driver="evr", check_finite=False)
    if eig[0] <= 1e-10:
        raise ValueError("MD covariance has fewer than 10 nonzero modes")
    var = np.maximum(np.diag(xx) / n - mean ** 2, 0)
    rmsf = np.sqrt(var.reshape(-1, 3).sum(axis=1))
    return v, rmsf


def moments(x):
    return x.sum(axis=0, dtype=np.float64), x.T @ x


def md_blocks(aligned, per_replica, block_size, n_rep=3):
    x = aligned.reshape(len(aligned), -1).astype(np.float64, copy=False)
    nblocks = per_replica // block_size
    if nblocks < 2:
        raise ValueError(f"Not enough MD blocks for resampling: {per_replica}, {block_size}")
    block_sums, block_cross = [], []
    for replica in range(n_rep):
        start = replica * per_replica
        for j in range(nblocks):
            group = x[start+j*block_size:start+(j+1)*block_size]
            s, q = moments(group)
            block_sums.append(s)
            block_cross.append(q)
    return np.asarray(block_sums), np.asarray(block_cross), nblocks


def sample_md(block_sums, block_cross, nblocks, block_frames, rng, n_rep=3):
    weights = np.concatenate([
        np.bincount(rng.integers(0, nblocks, size=nblocks), minlength=nblocks)
        for _ in range(n_rep)
    ]).astype(np.float64)
    summed = weights @ block_sums
    cross = np.einsum("i,ijk->jk", weights, block_cross, optimize=True)
    n = int(n_rep * nblocks * block_frames)
    return md_stats_from_moments(summed, cross, n)


def score(af2_stats, md_stats):
    av, ar = af2_stats
    mv, mr = md_stats
    overlap = float(np.sqrt(((av.T @ mv)**2).sum() / 10))
    corr = float(np.corrcoef(ar, mr)[0, 1])
    if not np.isfinite(overlap + corr):
        raise ValueError("Nonfinite bootstrap metric")
    return (overlap, corr)


def seed_groups(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    found: dict[str, list[int]] = {}
    for i, p in enumerate(paths):
        found.setdefault(p.parent.name, []).append(i)
    sizes = {len(v) for v in found.values()}
    if len(sizes) != 1:
        raise ValueError(f"Unequal seed groups: {[(k, len(v)) for k, v in found.items()]}")
    keys = sorted(found)
    return keys, np.asarray([found[k] for k in keys], dtype=int)


def receptor_bootstrap(core, cache, prov, paths_by_label, receptor, cfg, receptor_index):
    """Paired draws: the same MD block weights and seed groups are used for every cap in a draw."""
    md = cache["md"]
    reference = md["ref"]
    coords = md["pooled"]
    n_rep = len(prov["md_inputs"])
    per_replica = len(coords) // n_rep
    X_md = core.superpose_to(coords, reference)
    md_rmsf = np.sqrt(np.mean(np.sum((X_md-X_md.mean(axis=0))**2, axis=2), axis=0))
    fixed_md = (np.asarray(md["V"][:, :10], float), md_rmsf)
    ref_label, comparators = cfg["reference"], list(cfg["comparators"])
    labels = [ref_label] + comparators
    baseline_rows = {r["depth"]: r for r in cache["af2_rows"]}
    af2, fixed_af2s, baseline, seed_keys = {}, {}, {}, None
    for label in labels:
        aligned = core.superpose_to(cache["coords_bd"][label], reference)
        keys, groups = seed_groups(paths_by_label[label])
        if groups.size != len(aligned):
            raise ValueError(f"{receptor} {label}: file list does not match cached coordinates")
        if seed_keys is not None and seed_keys != keys:
            raise ValueError(f"{receptor}: seed groups differ between MSA caps")
        seed_keys = keys
        af2[label] = (aligned, groups)
        fixed_af2s[label] = modes_from_gram(aligned)
        baseline[label] = score(fixed_af2s[label], fixed_md)
        expected = baseline_rows[label]
        if (abs(baseline[label][0] - expected["RMSIP_pca"]) > .002 or
                abs(baseline[label][1] - expected["rmsf_corr"]) > .002):
            raise ValueError(f"{receptor} {label}: reconstructed values {baseline[label]} disagree with the "
                             f"numerical core ({expected['RMSIP_pca']}, {expected['rmsf_corr']}); refusing bootstrap")
    n_groups = len(seed_keys)
    rows = []
    rseed = int(cfg["seed_root"]) + receptor_index * 100000
    first_block = min(cfg["block_ns"])
    for block_ns in cfg["block_ns"]:
        block_frames = int(round(block_ns * 1000 / prov["frame_dt_ps"]))
        sums, crosses, nblocks = md_blocks(X_md, per_replica, block_frames, n_rep)
        rng = np.random.default_rng(rseed + block_ns)
        for draw in range(int(cfg["draws"])):
            picked = rng.integers(0, n_groups, size=n_groups)
            sampled_md = sample_md(sums, crosses, nblocks, block_frames, rng, n_rep)
            estimates = {"md_only": {}, "joint": {}}
            if block_ns == first_block:
                estimates["seed_only"] = {}
            for label in labels:
                aligned, groups = af2[label]
                sampled_af2 = modes_from_gram(aligned[groups[picked].ravel()])
                estimates["md_only"][label] = score(fixed_af2s[label], sampled_md)
                estimates["joint"][label] = score(sampled_af2, sampled_md)
                if block_ns == first_block:
                    estimates["seed_only"][label] = score(sampled_af2, fixed_md)
            for scheme, vals in estimates.items():
                for comp in comparators:
                    a, b = vals[ref_label], vals[comp]
                    rows.append({"receptor": receptor, "block_ns": block_ns, "scheme": scheme, "draw": draw,
                                 "comparator": comp, "delta_rmsip": a[0] - b[0], "delta_rmsf_corr": a[1] - b[1],
                                 "n_blocks_per_replica": nblocks,
                                 "discarded_tail_frames_per_replica": per_replica - nblocks * block_frames})
    observed = [{"receptor": receptor, "comparator": comp,
                 "delta_rmsip": baseline[ref_label][0] - baseline[comp][0],
                 "delta_rmsf_corr": baseline[ref_label][1] - baseline[comp][1]} for comp in comparators]
    return rows, observed


def aggregate_bootstrap(draws: dict[str, list[dict]], observed: dict[str, list[dict]], cfg) -> list[dict]:
    """Panel = unweighted mean over receptors of each paired draw; percentile intervals."""
    receptors = sorted(draws)
    by_key = defaultdict(dict)
    for rec in receptors:
        for row in draws[rec]:
            by_key[(int(row["block_ns"]), row["scheme"], int(row["draw"]), row["comparator"])][rec] = row
    drawn = defaultdict(list)
    for (block_ns, scheme, draw, comp), d in by_key.items():
        for metric in ("delta_rmsip", "delta_rmsf_corr"):
            drawn[(block_ns, scheme, comp, metric)].append((draw, float(np.mean([float(x[metric]) for x in d.values()]))))
    rows = []
    for (block_ns, scheme, comp, metric), points in sorted(drawn.items()):
        values = np.array([v for _, v in sorted(points)])
        est = float(np.mean([float(next(o for o in observed[r] if o["comparator"] == comp)[metric]) for r in receptors]))
        rows.append({"reference": cfg["reference"], "comparator": comp, "scheme": scheme,
                     "block_ns": block_ns if scheme != "seed_only" else "", "metric": metric,
                     "observed_difference": est, "bootstrap_median": float(np.median(values)),
                     "ci_2p5": float(np.quantile(values, .025)), "ci_97p5": float(np.quantile(values, .975)),
                     "draws": len(values), "receptors": ";".join(receptors)})
    return rows


# --- stage entry points -----------------------------------------------------------

def run_states(settings: Settings, layout: Layout, receptor: str, mode: str) -> Path:
    cart = layout.cartesian(receptor, mode)
    cache, prov = load_cache(cart)
    n_rep = len(prov["md_inputs"])
    paths = {label: af2_paths(layout, receptor, label) for label in cache["states"]["labels_af2"]}
    occ, agr = cluster_tables(cache, paths, receptor)
    exclusion = int(round(settings.analysis["states"]["nearest_md_exclusion_ns"] * 1000 / prov["frame_dt_ps"]))
    near = nearest_md(cache, receptor, n_rep, exclusion)
    out = layout.results / mode / "states" / receptor
    _write(out / "cluster_occupancy.csv", occ)
    _write(out / "model_index_agreement.csv", agr)
    _write(out / "nearest_md.csv", near)
    print(f"{receptor}: states, model-index agreement and nearest-MD tables -> {out}", flush=True)
    return out


def run_bootstrap(settings: Settings, layout: Layout, receptors: list[str], mode: str) -> Path:
    from .cartesian import load_core
    core = load_core()
    cfg = settings.analysis["bootstrap"]
    order = list(settings.receptors)
    out = layout.results / mode / "bootstrap"
    draws, observed = {}, {}
    for r in receptors:
        cache, prov = load_cache(layout.cartesian(r, mode))
        labels = [cfg["reference"], *cfg["comparators"]]
        paths = {label: af2_paths(layout, r, label) for label in labels}
        draws[r], observed[r] = receptor_bootstrap(core, cache, prov, paths, r, cfg, order.index(r))
        _write(out / f"{r}_draws.csv", draws[r])
        _write(out / f"{r}_observed.csv", observed[r])
        print(f"{r}: {cfg['draws']} paired bootstrap draws", flush=True)
    rows = aggregate_bootstrap(draws, observed, cfg)
    _write(out / "bootstrap_all_schemes.csv", rows)
    _write(out / "bootstrap_joint.csv", [x for x in rows if x["scheme"] == "joint"])
    (out / "resampling_specification.json").write_text(json.dumps({
        **cfg, "paired_across_caps": True, "resampling_within_each_md_replica": True,
        "trailing_incomplete_md_blocks": "not drawn; retained in the observed estimate",
        "interval": "pointwise percentile [2.5, 97.5]", "panel": "unweighted mean across receptors",
        "cartesian_skip_ns": settings.analysis["cartesian"]["skip_ns"]}, indent=2) + "\n")
    return out
