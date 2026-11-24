"""Panel-level tables and figures from per-receptor results.

Writes CSV tables (one per manuscript table) plus `summary.md`, a plain-text digest
with the numbers quoted in the paper. Nothing here recomputes a metric: it only
reads `results/<mode>/cartesian/*` and `results/<mode>/dpca/*`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import Layout, Settings


def _read(layout: Layout, receptors: list[str], mode: str):
    cart, dp, summ, ref, prov = [], [], {}, {}, {}
    for r in receptors:
        cdir, ddir = layout.cartesian(r, mode), layout.dpca(r, mode)
        if not (cdir / "metrics_by_depth.csv").is_file() or not (ddir / "metrics_by_depth.csv").is_file():
            raise FileNotFoundError(f"{r}: run `gpcr-af2md cartesian` and `gpcr-af2md dpca` first")
        c = pd.read_csv(cdir / "metrics_by_depth.csv")
        c.insert(0, "receptor", r)
        cart.append(c)
        dp.append(pd.read_csv(ddir / "metrics_by_depth.csv"))
        summ[r] = json.loads((cdir / "summary.json").read_text())
        ref[r] = json.loads((ddir / "window_reference_comparison.json").read_text())
        prov[r] = json.loads((ddir / "provenance.json").read_text())
    return pd.concat(cart, ignore_index=True), pd.concat(dp, ignore_index=True), summ, ref, prov


def _heatmap(matrix: np.ndarray, receptors, labels, title, cmap, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 1.5, 0.75 * len(receptors) + 1.2), layout="constrained")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(receptors)), receptors)
    ax.set_xlabel("MSA caps (max_seq:max_extra_seq)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(settings: Settings, layout: Layout, receptors: list[str], mode: str) -> Path:
    out = layout.summary(mode)
    out.mkdir(parents=True, exist_ok=True)
    cart, dp, summ, ref, prov = _read(layout, receptors, mode)
    labels = [l for l in settings.depth_labels if l in set(dp["msa_depth"])]

    dp.to_csv(out / "dpca_metrics.csv", index=False)
    cart.to_csv(out / "cartesian_metrics.csv", index=False)

    # JSD calibration: AF2 JSD against the 20-ns window and random-frame P95 references.
    rows = []
    for _, row in dp.iterrows():
        primary = ref[row.receptor]["primary"]
        rows.append({"receptor": row.receptor, "msa_depth": row.msa_depth,
                     "af2_jsd_bits": row.js_divergence_8x8_bits,
                     "window_20ns_p95_bits": primary["window_jsd_p95"],
                     "random_frames_p95_bits": primary["random_jsd_p95"],
                     "at_or_below_window_p95": row.js_divergence_8x8_bits <= primary["window_jsd_p95"],
                     "above_random_p95": row.js_divergence_8x8_bits > primary["random_jsd_p95"]})
    calib = pd.DataFrame(rows)
    calib.to_csv(out / "jsd_reference_table.csv", index=False)

    calib_rec = []
    for r in receptors:
        sub = dp[dp.receptor == r]
        p, s = ref[r]["primary"], ref[r]["sensitivity"]
        calib_rec.append({"receptor": r, "torsions_primary": p["torsions"], "torsions_sensitivity": s["torsions"],
                          "af2_jsd_min": sub.js_divergence_8x8_bits.min(), "af2_jsd_max": sub.js_divergence_8x8_bits.max(),
                          "random_p95": p["random_jsd_p95"], "window_p95": p["window_jsd_p95"],
                          "sensitivity_random_p95": s["random_jsd_p95"], "sensitivity_window_p95": s["window_jsd_p95"],
                          "removed_dPC1_squared_loading_fraction": s["removed_dPC1_squared_loading_fraction"],
                          "n_at_or_below_window_p95": int(calib[calib.receptor == r].at_or_below_window_p95.sum())})
    pd.DataFrame(calib_rec).to_csv(out / "jsd_calibration_by_receptor.csv", index=False)

    merged = cart.rename(columns={"depth": "msa_depth"}).merge(
        dp[["receptor", "msa_depth", "js_divergence_8x8_bits"]], on=["receptor", "msa_depth"])
    means = merged.groupby("msa_depth")[["RMSIP_pca", "rmsf_corr", "mean_pLDDT", "js_divergence_8x8_bits"]].mean()
    means = means.loc[labels]
    means.to_csv(out / "depth_means.csv")

    rec_rows, end_rows, anm_rows, tors_rows = [], [], [], []
    for r in receptors:
        c = cart[cart.receptor == r].set_index("depth")
        d0 = dp[dp.receptor == r].iloc[0]
        i, j = c.RMSIP_pca.idxmax(), c.rmsf_corr.idxmax()
        rec_rows.append({"receptor": r, "peak_rmsip": c.RMSIP_pca[i], "peak_rmsip_at": i,
                         "peak_rmsf_corr": c.rmsf_corr[j], "peak_rmsf_corr_at": j,
                         "md_anm_rmsip": summ[r]["md_anm_baseline"],
                         "md_md_replica_rmsip": summ[r]["md_inter_replica_rmsip"],
                         "md_pc1_cosine_mean": summ[r]["md_pc1_cosine_mean"],
                         "dpc12_variance_percent": 100 * (d0.dpca_pc1_md_variance + d0.dpca_pc2_md_variance),
                         "between_replicate_fraction_dPC1": (prov[r].get("between_replicate_variance_fraction_dPC1_dPC2") or [None])[0]})
        a, b = c.min_rmsd_inactive.idxmin(), c.min_rmsd_active.idxmin()
        end_rows.append({"receptor": r, "best_inactive_A": c.min_rmsd_inactive[a], "best_inactive_at": a,
                         "best_active_A": c.min_rmsd_active[b], "best_active_at": b,
                         "inactive_at_1:2": c.min_rmsd_inactive.get("1:2", np.nan),
                         "active_at_1:2": c.min_rmsd_active.get("1:2", np.nan)})
        deep = c.drop(index="1:2", errors="ignore")
        anm_rows.append({"receptor": r, "af2_anm_rmsip_min_excl_1:2": deep.RMSIP_anm.min(),
                         "af2_anm_rmsip_max_excl_1:2": deep.RMSIP_anm.max(),
                         "af2_anm_rmsip_1:2": c.RMSIP_anm.get("1:2", np.nan)})
        tors_rows.append({"receptor": r, "cartesian_residues": prov[r]["cartesian_residues"],
                          "possible_directed_torsions": prov[r]["possible_directed_torsions"],
                          "md_eligible": prov[r]["md_eligible_torsions"], "retained": prov[r]["retained_torsions"],
                          **{f"excluded_{k}": v for k, v in prov[r]["exclusion_counts"].items()},
                          **{f"af2_break_first_at_{k}": v for k, v in
                             prov[r].get("af2_geometry_exclusions_by_first_failing_setting", {}).items()}})
    pd.DataFrame(rec_rows).to_csv(out / "receptor_table.csv", index=False)
    pd.DataFrame(end_rows).to_csv(out / "endpoint_table.csv", index=False)
    pd.DataFrame(anm_rows).to_csv(out / "anm_table.csv", index=False)
    pd.DataFrame(tors_rows).to_csv(out / "torsion_coverage.csv", index=False)

    for metric, title, name, cmap in (
            ("js_divergence_8x8_bits", "MD-AF2 dPCA JSD (bits; lower = closer)", "jsd_heatmap.png", "viridis_r"),
            ("af2_within_md_95pct_pc_region", "AF2 models inside MD 95% dPCA region", "coverage_heatmap.png", "viridis")):
        m = np.array([[float(dp[(dp.receptor == r) & (dp.msa_depth == l)][metric].iloc[0]) for l in labels]
                      for r in receptors])
        _heatmap(m, receptors, labels, title, cmap, out / name)

    extra = _states_and_bootstrap(settings, layout, receptors, mode, out)

    n = len(calib)
    lines = [f"# Summary ({mode} MD replicates)", "",
             f"- Receptor-setting comparisons: {n}",
             f"- AF2 JSD range: {calib.af2_jsd_bits.min():.3f}-{calib.af2_jsd_bits.max():.3f} bits",
             f"- Above random-frame P95: {int(calib.above_random_p95.sum())} of {n}",
             f"- At or below 20-ns window P95: {int(calib.at_or_below_window_p95.sum())} of {n} "
             f"({', '.join(f'{r} {int(calib[calib.receptor == r].at_or_below_window_p95.sum())}' for r in receptors)})",
             *extra, "", "See the CSV files in this folder for every table."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return out


def _states_and_bootstrap(settings: Settings, layout: Layout, receptors: list[str], mode: str, out: Path) -> list[str]:
    """Tables from the `states` and `bootstrap` stages, when those stages have been run."""
    lines: list[str] = []
    ref = settings.analysis["bootstrap"]["reference"]
    sdirs = {r: layout.results / mode / "states" / r for r in receptors}
    if all((d / "nearest_md.csv").is_file() for d in sdirs.values()):
        near = pd.concat([pd.read_csv(d / "nearest_md.csv") for d in sdirs.values()], ignore_index=True)
        occ = pd.concat([pd.read_csv(d / "cluster_occupancy.csv") for d in sdirs.values()], ignore_index=True)
        agr = pd.concat([pd.read_csv(d / "model_index_agreement.csv") for d in sdirs.values()], ignore_index=True)
        near.to_csv(out / "nearest_md_all_settings.csv", index=False)
        occ.to_csv(out / "cluster_occupancy.csv", index=False)
        agr.to_csv(out / "model_index_agreement.csv", index=False)
        rows = []
        for r in receptors:
            sub = near[near.receptor == r]
            at = sub[sub.msa_depth == ref].set_index("n_pc").af2_above_p95_pct
            p20 = sub[sub.n_pc == 20].af2_above_p95_pct
            rows.append({"receptor": r, f"3PC_at_{ref}_pct": at.get(3), f"20PC_at_{ref}_pct": at.get(20),
                         "20PC_min_over_settings_pct": p20.min(), "20PC_max_over_settings_pct": p20.max()})
        pd.DataFrame(rows).to_csv(out / "nearest_md_table.csv", index=False)
        single = occ[occ.clusters_occupied == 1]
        lines += [f"- All AF2 models in one MD cluster: {len(single)} of {len(occ)} receptor-setting combinations",
                  f"- At {ref}: " + "; ".join(f"{r} {occ[(occ.receptor == r) & (occ.msa_depth == ref)].af2_counts.item()}"
                                            for r in receptors),
                  f"- Nearest-MD, 20 PCs, at {ref}, % above MD P95: " +
                  ", ".join(f"{x['receptor']} {x[f'20PC_at_{ref}_pct']:.0f}" for x in rows)]
    boot = layout.results / mode / "bootstrap" / "bootstrap_joint.csv"
    if boot.is_file():
        b = pd.read_csv(boot)
        b.to_csv(out / "bootstrap_joint.csv", index=False)
        for _, x in b[b.block_ns == min(b.block_ns)].iterrows():
            lines.append(f"- Joint bootstrap, {x.block_ns:g}-ns blocks, {x.reference} minus {x.comparator}, "
                         f"{x.metric}: {x.observed_difference:+.3f} [{x.ci_2p5:+.3f}, {x.ci_97p5:+.3f}]")
    return lines
