"""Backbone dihedral PCA (dPCA) of MD, with AF2 models projected onto the MD basis.

For each receptor, fit a PCA on (sin, cos) of backbone phi/psi from the pooled MD
frames, project every AF2 model onto dPC1/dPC2, and compare distributions with a
Jensen-Shannon divergence on an 8 x 8 grid of MD quantile bins. The PCA is never
refitted to AF2 and receptors are never pooled.

"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import jensenshannon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import MDAnalysis as mda  # noqa: E402

AA3 = dict(zip(
    "ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCEQGHILKMFPSTWYV",
))
AA3.update({"HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H", "MSE": "M"})
BACKBONE = ("N", "CA", "C")

# Constants hard-coded in the numerical functions below. config/analysis.yaml must
# agree with them; `check_settings` stops the run if someone edits only the YAML.
HARD_CODED = {"n_components": 5, "histogram_bins": 8, "pseudocount": 0.1,
              "peptide_bond_min_A": 0.9, "peptide_bond_max_A": 2.2, "min_torsions": 60}


# ---------------------------------------------------------------------------
# Numerical functions
# ---------------------------------------------------------------------------
def pdb_backbone(path: Path) -> tuple[dict, dict, dict]:
    """Return coordinates, sequence letters, (chain, segment) per canonical position.

    Ignore alternate-location B+ and HETATM. Refuse ambiguous backbone atoms.
    Coordinates are read directly so each AF2 PDB is processed only once.
    """
    atoms, identity, segment = {}, {}, {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            name = line[12:16].strip()
            if name not in BACKBONE or line[16] not in (" ", "A"):
                continue
            if line[26].strip():
                continue  # Never turn insertion codes into canonical residue numbers.
            try:
                resid = int(line[22:26])
                coords = np.array([float(line[k:k + 8]) for k in (30, 38, 46)], dtype=float)
            except ValueError as exc:
                raise ValueError(f"Malformed backbone PDB record in {path}: {line.strip()}") from exc
            key = (resid, name)
            if key in atoms:
                raise ValueError(f"Duplicate backbone atom {key} in {path}")
            atoms[key] = coords
            letter = AA3.get(line[17:20].strip(), "?")
            if resid in identity and identity[resid] != letter:
                raise ValueError(f"Conflicting residue names at {resid} in {path}")
            identity[resid] = letter
            loc = (line[21].strip(), line[72:76].strip())
            if resid in segment and segment[resid] != loc:
                raise ValueError(f"Conflicting segment labels at {resid} in {path}")
            segment[resid] = loc
    return atoms, identity, segment



def candidate_reason(feature, atoms, segments):
    """Check each directed torsion independently, not requiring both neighbours."""
    quad = atom_quads([feature])[0]
    r, kind = feature
    adjacent = r-1 if kind == 'phi' else r+1
    if adjacent not in segments:
        return 'neighbor_not_modeled'
    if segments[adjacent] != segments[r]:
        return 'different_segment_or_chain'
    if not all(k in atoms for k in quad):
        return 'missing_MD_backbone_atoms'
    if not valid_peptide_bonds(atoms, feature):
        return 'MD_peptide_geometry_first_frame'
    return ''


def make_candidates(atoms: dict, segments: dict, cartesian: list[int]):
    candidates, exclusions = [], {}
    for r in cartesian:
        for feature in ((r, 'phi'), (r, 'psi')):
            reason = candidate_reason(feature, atoms, segments)
            if reason:
                exclusions[feature] = reason
            else:
                candidates.append(feature)
    return candidates, exclusions


def valid_peptide_bonds(atoms: dict, feature: tuple[int, str]) -> bool:
    r, kind = feature
    if kind == "phi":
        a, b = (r - 1, "C"), (r, "N")
    else:
        a, b = (r, "C"), (r + 1, "N")
    if a not in atoms or b not in atoms:
        return False
    dist = float(np.linalg.norm(atoms[a] - atoms[b]))
    return 0.9 <= dist <= 2.2


def atom_quads(features: list[tuple[int, str]]) -> list[tuple[tuple[int, str], ...]]:
    quads = []
    for r, kind in features:
        if kind == "phi":
            quads.append(((r - 1, "C"), (r, "N"), (r, "CA"), (r, "C")))
        else:
            quads.append(((r, "N"), (r, "CA"), (r, "C"), (r + 1, "N")))
    return quads


def dihedral_angles(quad_coords: np.ndarray) -> np.ndarray:
    """Standard signed dihedral angles in radians; shape (torsions,4,3)."""
    p0, p1, p2, p3 = np.moveaxis(quad_coords, 1, 0)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norms = np.linalg.norm(b1, axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError("Degenerate dihedral: central bond has zero length")
    b1 = b1 / norms[:, None]
    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    if np.any((np.linalg.norm(v, axis=-1) < 1e-8) |
              (np.linalg.norm(w, axis=-1) < 1e-8)):
        raise ValueError("Degenerate dihedral: adjacent bond vectors are collinear")
    return np.arctan2(y, x)


def circular_features(angles: np.ndarray) -> np.ndarray:
    """Interleave sine and cosine for every torsion; avoid angular wraparound."""
    return np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(-1)


def md_feature_matrix(top: Path, trajectories: list[Path], quads: list, seq: str) -> tuple[np.ndarray, list[int]]:
    matrices = []
    frames = []
    required = set(key for q in quads for key in q)
    for traj in trajectories:
        u = mda.Universe(str(top), str(traj))
        indexes = {}
        for atom in u.atoms:
            key = (int(atom.resid), atom.name)
            if key in required:
                if key in indexes:
                    raise ValueError(f"{top}: ambiguous MD atom {key}")
                indexes[key] = atom.index
        missing = required - indexes.keys()
        if missing:
            raise ValueError(f"{top}: missing MD backbone atoms {list(missing)[:8]}")
        ind = np.asarray([[indexes[key] for key in q] for q in quads], dtype=np.int32)
        series = np.empty((len(u.trajectory), 2 * len(quads)), dtype=np.float64)
        for i, _ in enumerate(u.trajectory):
            a = dihedral_angles(u.atoms.positions[ind])
            series[i] = circular_features(a)
        frames.append(series.shape[0])
        matrices.append(series)
        print(f"  MD {traj.name}: {series.shape[0]} frames", flush=True)
    return np.vstack(matrices), frames


def af2_feature_matrix(paths: list[Path], quads: list, seq: str, selected: set[int]) -> np.ndarray:
    required = set(key for q in quads for key in q)
    data = np.empty((len(paths), 2 * len(quads)), dtype=np.float64)
    for i, pdb in enumerate(paths):
        atoms, identities, _ = pdb_backbone(pdb)
        for r in selected:
            if identities.get(r) != seq[r - 1]:
                raise ValueError(f"{pdb}: AF2/canonical sequence mismatch at residue {r}")
        missing = required - atoms.keys()
        if missing:
            raise ValueError(f"{pdb}: missing backbone atoms {list(missing)[:8]}")
        for feature in quads_to_features(quads):
            if not valid_peptide_bonds(atoms, feature):
                raise ValueError(f"{pdb}: a selected torsion crosses a broken peptide bond: {feature}")
        xyz = np.asarray([[atoms[key] for key in q] for q in quads], dtype=np.float64)
        data[i] = circular_features(dihedral_angles(xyz))
    return data


def quads_to_features(quads: list) -> list:
    # One feature/quad; avoids global mutable state. Called for every model;
    # build once in run_receptor and pass it instead when optimizing later.
    result = []
    for q in quads:
        if q[0][1] == "C":
            result.append((q[1][0], "phi"))
        else:
            result.append((q[0][0], "psi"))
    return result


def fit_md_pca(data: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data.shape[0] < 5 or data.shape[1] < 4:
        raise ValueError("Not enough observations or torsions for dPCA")
    mu = data.mean(axis=0)
    centered = data - mu
    covariance = (centered.T @ centered) / (len(centered) - 1)
    k = min(k, len(mu))
    eigvals, eigvecs = eigh(covariance, subset_by_index=(len(mu) - k, len(mu) - 1), check_finite=False)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0)
    eigvecs = eigvecs[:, order]
    total_variance = float(np.trace(covariance))
    if not np.isfinite(total_variance) or total_variance <= 1e-12:
        raise ValueError("MD torsional variation is insufficient for PCA")
    explained = eigvals / total_variance
    # Deterministic sign convention by largest magnitude loading.
    for i in range(eigvecs.shape[1]):
        strongest = np.argmax(np.abs(eigvecs[:, i]))
        if eigvecs[strongest, i] < 0:
            eigvecs[:, i] *= -1
    return mu, eigvecs, eigvals, explained


def quantile_hist(scores: np.ndarray, boundaries: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    first = np.searchsorted(boundaries[0], scores[:, 0], side="right")
    second = np.searchsorted(boundaries[1], scores[:, 1], side="right")
    hist = np.zeros((8, 8), dtype=np.float64)
    np.add.at(hist, (first, second), 1)
    hist += 0.1  # Fixed pseudocount for both MD and AF2 distributions.
    return (hist / hist.sum()).ravel()


def score_metrics(md_proj: np.ndarray, af_proj: np.ndarray, boundaries: tuple[np.ndarray, np.ndarray], threshold: float) -> dict:
    p = quantile_hist(md_proj, boundaries)
    q = quantile_hist(af_proj, boundaries)
    js = float(jensenshannon(p, q, base=2) ** 2)
    cov_diag = np.maximum(md_proj.var(axis=0, ddof=1), 1e-12)
    af_radius2 = ((af_proj - md_proj.mean(axis=0)) ** 2 / cov_diag).sum(axis=1)
    return {
        "js_divergence_8x8_bits": js,
        "af2_within_md_95pct_pc_region": float(np.mean(af_radius2 <= threshold)),
        "af2_md_centroid_distance_in_pc_sd": float(np.linalg.norm((af_proj.mean(axis=0) - md_proj.mean(axis=0)) / np.sqrt(cov_diag))),
    }


def reference_distribution(md_proj: np.ndarray, frame_counts: list[int], bins, window_frames=100, draws=1000, seed=2026):
    """20-ns contiguous-window versus random-frame JSD, same 8x8 bin/pseudocount calibration."""
    if min(frame_counts) < window_frames:
        raise ValueError(f"MD replica shorter than 20-ns window: {frame_counts}")
    ref = quantile_hist(md_proj, bins)
    js = lambda points: float(jensenshannon(ref, quantile_hist(points, bins), base=2)**2)
    windows=[]; start=0
    for n in frame_counts:
        replica=md_proj[start:start+n]
        windows.extend(js(replica[i:i+window_frames]) for i in range(n-window_frames+1))
        start += n
    rng=np.random.default_rng(seed)
    randoms=[js(md_proj[rng.choice(len(md_proj),window_frames,replace=False)]) for _ in range(draws)]
    return {'window_frames':window_frames,'window_ns':20.0,'window_jsd_p95':float(np.quantile(windows,.95)),
            'random_jsd_p95':float(np.quantile(randoms,.95)), 'window_count':len(windows),
            'random_draws':draws, 'seed':seed, 'calibration':'full MD 8x8 quantile bins; pseudocount 0.1 per bin; JSD bits'}


def save_overlay(md_proj: np.ndarray, projections: dict, explained: np.ndarray, out: Path, receptor: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True, constrained_layout=True)
    all_pts = np.vstack([md_proj] + list(projections.values()))
    xlow, ylow = all_pts.min(axis=0)
    xhigh, yhigh = all_pts.max(axis=0)
    # Fixed axes across depths, including every MD and AF2 point.
    xrange = max(xhigh - xlow, 0.05)
    yrange = max(yhigh - ylow, 0.05)
    for ax, (label, points) in zip(axes.flat, projections.items()):
        ax.scatter(md_proj[:, 0], md_proj[:, 1], s=2, c="#999999", alpha=0.14, rasterized=True, label="MD")
        ax.scatter(points[:, 0], points[:, 1], s=13, alpha=0.75, color="#cc462e", rasterized=True, label="AF2")
        ax.set_title(f"MSA {label} (n={len(points)})", fontsize=10)
        ax.set_xlim(xlow - 0.08*xrange, xhigh + 0.08*xrange)
        ax.set_ylim(ylow - 0.08*yrange, yhigh + 0.08*yrange)
        ax.grid(alpha=0.15)
    fig.supxlabel(f"dPC1 ({explained[0] * 100:.1f}% MD variance)")
    fig.supylabel(f"dPC2 ({explained[1] * 100:.1f}% MD variance)")
    fig.suptitle(f"{receptor}: MD-fitted circular backbone dPCA; MD gray, AF2 orange", fontsize=12)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_loadings(features: list, vectors: np.ndarray, outdir: Path) -> None:
    # Eigenvector weights for sin/cos of each torsion, aggregated per residue.
    loadings = defaultdict(lambda: [0.0, 0.0])
    with (outdir / "torsion_loadings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_residue", "torsion", "dPC1_squared_loading", "dPC2_squared_loading"])
        for i, (r, kind) in enumerate(features):
            writer.writerow([r, kind, float(np.sum(vectors[2*i:2*i+2, 0]**2)),
                                      float(np.sum(vectors[2*i:2*i+2, 1]**2))])
    for i, (r, kind) in enumerate(features):
        for pc in (0, 1):
            loadings[r][pc] += float(np.sum(vectors[2*i:2*i+2, pc] ** 2))
    with (outdir / "residue_loadings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_residue", "dPC1_squared_loading", "dPC2_squared_loading"])
        for r in sorted(loadings):
            writer.writerow([r, *loadings[r]])
    top = sorted(loadings, key=lambda r: sum(loadings[r]), reverse=True)[:20]
    top.sort()
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    x = np.arange(len(top))
    ax.bar(x-0.18, [loadings[r][0] for r in top], width=0.35, label="dPC1")
    ax.bar(x+0.18, [loadings[r][1] for r in top], width=0.35, label="dPC2")
    ax.set_xticks(x, [str(r) for r in top], rotation=50)
    ax.set_ylabel("Sum of squared sin/cos loadings")
    ax.set_xlabel("Canonical residue")
    ax.set_title("Top 20 residue contributions to dPC1/dPC2")
    ax.legend()
    fig.savefig(outdir / "residue_loadings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)




# ---------------------------------------------------------------------------
# Torsion selection, fitting and output
# ---------------------------------------------------------------------------
def check_settings(dpca_cfg: dict) -> None:
    for key, value in HARD_CODED.items():
        if dpca_cfg.get(key) != value:
            raise ValueError(f"config/analysis.yaml dpca.{key}={dpca_cfg.get(key)!r}, but the "
                             f"numerical code uses {value!r}. Edit both together, or neither.")


def cartesian_mask(cartesian_dir: Path) -> list[int]:
    """Use the identical C-alpha mask written by the Cartesian stage."""
    path = cartesian_dir / "cache.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Cartesian stage must finish first: {path}")
    with path.open("rb") as fh:
        payload = pickle.load(fh)  # cache written by this package's own Cartesian stage
    common = [int(x) for x in payload["common"]]
    if not common or len(common) != len(set(common)):
        raise ValueError(f"Invalid residue mask in {path}")
    return sorted(common)


def run(*, receptor: str, top: Path, trajectories: list[Path], seq: str,
        by_depth: dict[str, list[Path]], cartesian_dir: Path, final: Path,
        dpca_cfg: dict, provenance_extra: dict | None = None) -> Path:
    check_settings(dpca_cfg)
    if (final / "metrics_by_depth.csv").is_file():
        print(f"{receptor}: dPCA results already exist at {final}", flush=True)
        return final
    staged = final.parent / f".incomplete_{receptor}"
    if final.exists() or staged.exists():
        raise FileExistsError(f"Refusing to overwrite {final} or {staged}; delete it to rerun")

    md_atoms, md_identity, segments = pdb_backbone(top)
    common = cartesian_mask(cartesian_dir)
    candidates, excluded = make_candidates(md_atoms, segments, common)
    if len(candidates) < 60:
        raise ValueError(f"{receptor}: too few MD-eligible backbone torsions ({len(candidates)})")
    for r in {x for x, _ in candidates}:
        if md_identity.get(r) != seq[r - 1]:
            raise ValueError(f"{receptor}: MD/canonical sequence mismatch at {r}")
    # Audit every candidate peptide bond in every frame of every replicate first.
    # This MD-eligible feature set is the broader sensitivity selection.
    bonds = [(f, (f[0] - 1, 'C'), (f[0], 'N')) if f[1] == 'phi'
             else (f, (f[0], 'C'), (f[0] + 1, 'N')) for f in candidates]
    broken_md = set()
    for traj in trajectories:
        u = mda.Universe(str(top), str(traj))
        lookup = {(int(a.resid), str(a.name)): a.index for a in u.atoms if a.name in ('N', 'C')}
        ia = np.array([lookup[a] for _, a, b in bonds])
        ib = np.array([lookup[b] for _, a, b in bonds])
        count_bad = 0
        for ts in u.trajectory:
            d = np.linalg.norm(u.atoms.positions[ia] - u.atoms.positions[ib], axis=1)
            bad = np.where((d < 0.9) | (d > 2.2))[0]
            broken_md.update(bonds[i][0] for i in bad)
            count_bad += len(bad)
        print(f"[{receptor}] {traj.name}: {count_bad} MD peptide-bond geometry failures", flush=True)
    for f in broken_md:
        excluded[f] = 'MD_peptide_geometry_any_frame'
    candidates = [f for f in candidates if f not in broken_md]
    # The AF2-matched primary set keeps a torsion only if its backbone atoms and
    # peptide bond are intact in every AF2 model at every MSA setting.
    features = candidates[:]
    exclusion_reasons = dict(excluded)
    first_failure: dict[tuple[int, str], str] = {}  # diagnostic only: first MSA setting that broke a torsion
    for label, paths in by_depth.items():
        print(f"[{receptor}] validating {label} AF2 torsion geometry ({len(paths)} structures)", flush=True)
        for pdb in paths:
            af_atoms, identities, _ = pdb_backbone(pdb)
            survivors = []
            for f in features:
                quad = atom_quads([f])[0]
                if not all(key in af_atoms for key in quad):
                    exclusion_reasons[f] = 'AF2_missing_backbone_atoms'
                elif any(identities.get(key[0]) != seq[key[0] - 1] for key in quad):
                    exclusion_reasons[f] = 'AF2_canonical_identity_mismatch'
                elif not valid_peptide_bonds(af_atoms, f):
                    exclusion_reasons[f] = 'AF2_peptide_geometry'
                    first_failure.setdefault(f, label)
                else:
                    survivors.append(f)
            features = survivors
    removed_peptide_bond = [f for f, r in exclusion_reasons.items() if 'geometry' in r]
    removed_af2_availability = [f for f, r in exclusion_reasons.items() if r.startswith('AF2')]
    if len(features) < 60:
        raise ValueError(f"{receptor}: only {len(features)} valid shared torsions")
    quads = atom_quads(features)
    required_residues = {r for q in quads for r, _ in q}
    print(f"[{receptor}] {len(features)} common torsions, {len(trajectories)} MD replica(s)", flush=True)
    staged.mkdir(parents=True)
    with (staged / 'torsion_coverage.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['canonical_residue', 'torsion', 'status', 'first_failing_msa_setting'])
        kept = set(features)
        w.writerows([r, kind, 'retained' if (r, kind) in kept else exclusion_reasons.get((r, kind), 'excluded'),
                     first_failure.get((r, kind), '')]
                    for r in common for kind in ('phi', 'psi'))
    md_data, frame_counts = md_feature_matrix(top, trajectories, quads, seq)
    mu, vectors, eigvals, explained = fit_md_pca(md_data)
    md_proj = (md_data - mu) @ vectors[:, :2]
    thresholds = tuple(np.quantile(md_proj[:, axis], np.arange(1, 8) / 8) for axis in range(2))
    calibration = reference_distribution(md_proj, frame_counts, thresholds,
                                         window_frames=int(dpca_cfg["window_frames"]),
                                         draws=int(dpca_cfg["random_draws"]), seed=int(dpca_cfg["seed"]))
    (staged / "window_reference.json").write_text(json.dumps(calibration, indent=2) + "\n")
    broad = None
    if set(candidates) != set(features):
        broad_data, _ = md_feature_matrix(top, trajectories, atom_quads(candidates), seq)
        mu_b, vec_b, _, var_b = fit_md_pca(broad_data)
        proj_b = (broad_data - mu_b) @ vec_b[:, :2]
        bins_b = tuple(np.quantile(proj_b[:, axis], np.arange(1, 8) / 8) for axis in range(2))
        broad_calibration = reference_distribution(proj_b, frame_counts, bins_b,
                                                   window_frames=int(dpca_cfg["window_frames"]),
                                                   draws=int(dpca_cfg["random_draws"]), seed=int(dpca_cfg["seed"]))
        removed = set(candidates) - set(features)
        fraction = float(sum(np.sum(vec_b[2 * i:2 * i + 2, 0] ** 2) for i, f in enumerate(candidates) if f in removed))
        broad = {'feature_set': 'MD-eligible broader sensitivity', 'torsions': len(candidates),
                 'removed_to_obtain_matched': sorted([list(f) for f in removed]),
                 'removed_dPC1_squared_loading_fraction': fraction,
                 **broad_calibration}
    if broad is None:
        broad = {'feature_set': 'MD-eligible broader sensitivity', 'torsions': len(candidates),
                 'removed_to_obtain_matched': [], 'removed_dPC1_squared_loading_fraction': 0.0,
                 **calibration}
    comparison = {'primary': {'feature_set': 'AF2-matched', 'torsions': len(features), **calibration},
                  'sensitivity': broad, 'same_calibration_procedure': True}
    (staged / 'window_reference_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    cov_diag = np.maximum(md_proj.var(axis=0, ddof=1), 1e-12)
    md_rad2 = ((md_proj - md_proj.mean(axis=0)) ** 2 / cov_diag).sum(axis=1)
    threshold = float(np.quantile(md_rad2, 0.95))
    all_proj, rows = {}, []
    for label, paths in by_depth.items():
        print(f"  AF2 {label}: {len(paths)} PDBs", flush=True)
        af2_data = af2_feature_matrix(paths, quads, seq, required_residues)
        af_proj = (af2_data - mu) @ vectors[:, :2]
        all_proj[label] = af_proj
        metrics = score_metrics(md_proj, af_proj, thresholds, threshold)
        rows.append({"receptor": receptor, "msa_depth": label, "md_replicas": len(trajectories),
                     "md_frames": len(md_data), "af2_models": len(paths), "torsions": len(features),
                     "dpca_pc1_md_variance": float(explained[0]), "dpca_pc2_md_variance": float(explained[1]),
                     **metrics})
    with (staged / "metrics_by_depth.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (staged / "torsion_features.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_residue", "torsion"])
        writer.writerows(features)
    save_loadings(features, vectors, staged)
    if len(all_proj) == 8:
        save_overlay(md_proj, all_proj, explained, staged / "md_vs_af2_dpca.png", receptor)
    else:
        save_overlay_any(md_proj, all_proj, explained, staged / "md_vs_af2_dpca.png", receptor)
    replica_index = np.concatenate([np.full(n, i) for i, n in enumerate(frame_counts)])
    np.savez_compressed(staged / "dpca_scores.npz", md=md_proj.astype(np.float32), md_replica_index=replica_index,
                        **{f"af2_{label.replace(':', '_')}": v.astype(np.float32) for label, v in all_proj.items()})
    between = [float(np.var([md_proj[replica_index == k, a].mean() for k in np.unique(replica_index)])
                     / md_proj[:, a].var()) for a in (0, 1)]
    metadata = {
        "receptor": receptor, "method": "separate per-receptor MD-fitted circular backbone phi/psi PCA",
        "dpca_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "md_topology": str(top), "md_files": [str(p) for p in trajectories],
        "md_frames_by_replica": frame_counts, "md_replicas": len(trajectories),
        "af2_counts": {label: len(paths) for label, paths in by_depth.items()},
        "msa_depths": list(by_depth),
        "cartesian_residues": len(common), "possible_directed_torsions": 2 * len(common),
        "md_eligible_torsions": len(candidates), "retained_torsions": len(features),
        "exclusion_counts": dict(Counter(exclusion_reasons.values())),
        "af2_geometry_exclusions_by_first_failing_setting": dict(Counter(first_failure.values())),
        "excluded_due_bond_geometry": removed_peptide_bond,
        "excluded_due_af2_absence": removed_af2_availability,
        "explained_md_variance_first_5pcs": explained.tolist(),
        "between_replicate_variance_fraction_dPC1_dPC2": between if len(trajectories) > 1 else None,
        "temporal_handling": "all stored MD frames; no skip or stride",
        "comparison": "8x8 MD-quantile-bin Jensen-Shannon divergence in bits; pseudocount 0.1 per bin",
        "window_reference": calibration,
        **(provenance_extra or {}),
    }
    (staged / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.rename(staged, final)
    print(f"[{receptor}] COMPLETE -> {final}", flush=True)
    return final


def save_overlay_any(md_proj, projections, explained, out, receptor):
    """Same overlay as `save_overlay` for a smaller set of MSA settings (smoke tests)."""
    n = len(projections)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, (label, points) in zip(axes.flat, projections.items()):
        ax.scatter(md_proj[:, 0], md_proj[:, 1], s=2, c="#999999", alpha=0.14, rasterized=True)
        ax.scatter(points[:, 0], points[:, 1], s=13, alpha=0.75, color="#cc462e", rasterized=True)
        ax.set_title(f"MSA {label} (n={len(points)})", fontsize=10)
        ax.grid(alpha=0.15)
    fig.supxlabel(f"dPC1 ({explained[0] * 100:.1f}% MD variance)")
    fig.supylabel(f"dPC2 ({explained[1] * 100:.1f}% MD variance)")
    fig.suptitle(f"{receptor}: MD-fitted backbone dPCA; MD grey, AF2 red", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def audit_exclusions(coverage_csv: Path, by_depth: dict[str, list[Path]], seq: str) -> list[dict]:
    """For every torsion excluded on AF2 grounds, count failing models at each MSA setting.

    Unlike the selection loop in `run`, every model at every setting is checked, so a
    torsion that first breaks at 64:640 also shows whether it breaks at 1:2.
    """
    with Path(coverage_csv).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    excluded = [(int(r["canonical_residue"]), r["torsion"]) for r in rows if r["status"].startswith("AF2_")]
    quads = dict(zip(excluded, atom_quads(excluded)))
    counts = {f: Counter() for f in excluded}
    for label, paths in by_depth.items():
        for pdb in paths:
            atoms, identities, _ = pdb_backbone(pdb)
            for f, quad in quads.items():
                if (any(k not in atoms for k in quad)
                        or any(identities.get(r) != seq[r - 1] for r, _ in quad)
                        or not valid_peptide_bonds(atoms, f)):
                    counts[f][label] += 1
    never = [f for f, c in counts.items() if not c]
    if never:
        raise ValueError(f"{len(never)} excluded torsions do not fail in these AF2 files: {never[:5]}")
    labels = list(by_depth)
    return [{"canonical_residue": r, "torsion": k,
             "first_failing_setting": next(l for l in labels if counts[(r, k)][l]),
             "fails_only_at_1:2": set(counts[(r, k)]) == {"1:2"},
             **{l: counts[(r, k)][l] for l in labels}} for (r, k) in excluded]
