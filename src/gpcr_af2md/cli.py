"""Command-line interface: `gpcr-af2md <command> [options]` (or `python -m gpcr_af2md`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import REPLICA_MODES, Layout, Settings


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--receptors", nargs="+", metavar="NAME",
                   help="receptors to process (default: all in config/receptors.yaml)")
    p.add_argument("--md-replicas", choices=REPLICA_MODES, default="public",
                   help="'public': GPCRmd replicate 1 only, no account needed (default). "
                        "'all': all three replicates, as in the paper; replicates 2-3 need a GPCRmd account.")
    p.add_argument("--data", type=Path, help="data folder (default: ./data)")
    p.add_argument("--results", type=Path, help="results folder (default: ./results)")
    p.add_argument("--config", type=Path, help="folder with receptors.yaml and analysis.yaml (default: ./config)")


def _subset(p: argparse.ArgumentParser) -> None:
    p.add_argument("--depths", nargs="+", metavar="CAPS",
                   help="MSA settings to analyse, e.g. 64:640 1:2 (default: all eight)")
    p.add_argument("--af2-models-per-depth", type=int,
                   help="required models per MSA setting (default: 100 when all eight settings are used)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gpcr-af2md", description=__doc__)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, help_text in (
            ("fetch", "download UniProt sequences and GPCRmd files (public ones only)"),
            ("prepare", "extract receptor-only trajectories from GPCRmd full-system files"),
            ("checksums", "print SHA-256 of local GPCRmd files, for config/receptors.yaml"),
            ("af2-msa", "build one MSA per receptor with the ColabFold MMseqs2 server"),
            ("dpca", "backbone dihedral PCA and JSD calibration"),
            ("states", "cluster occupancy, model-index agreement, nearest-MD distances"),
            ("bootstrap", "paired seed/MD-block bootstrap of the depth contrasts (needs all compared caps)"),
            ("torsion-audit", "count AF2 failures per MSA setting for every excluded torsion"),
            ("summarize", "panel tables, heatmaps and summary.md")):
        _common(sub.add_parser(name, help=help_text))

    p = sub.add_parser("af2-tasks", help="write the cluster task list (receptor, caps, seed)")
    _common(p)
    p.add_argument("--out", type=Path, default=Path("workflow/af2/tasks.tsv"))

    p = sub.add_parser("af2-predict", help="run AF2 tasks locally with colabfold_batch (GPU)")
    _common(p)
    p.add_argument("--depths", nargs="+", required=True, metavar="CAPS")
    p.add_argument("--seeds", nargs="+", type=int, required=True)

    p = sub.add_parser("cartesian", help="Cartesian PCA/RMSIP, RMSF, ANM, states, descriptors")
    _common(p)
    _subset(p)

    p = sub.add_parser("run", help="fetch, prepare, cartesian, dpca, states, bootstrap, summarize (AF2 models must exist)")
    _common(p)
    _subset(p)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load(args.config)
    layout = Layout.from_args(args.data, args.results)
    receptors = args.receptors or list(settings.receptors)
    for r in receptors:
        settings.receptor(r)
    mode = args.md_replicas
    cmd = args.command

    if cmd == "checksums":
        from .download import record_checksums
        print(json.dumps(record_checksums(settings, layout), indent=2))
        return 0
    if cmd == "af2-tasks":
        from .af2 import write_tasks
        write_tasks(settings, receptors, args.out)
        return 0
    if cmd == "af2-msa":
        from .af2 import make_msa
        from .download import fetch_fasta
        for r in receptors:
            fetch_fasta(settings, layout, r)
            print(make_msa(settings, layout, r))
        return 0
    if cmd == "af2-predict":
        from .af2 import predict
        for r in receptors:
            predict(settings, layout, r, args.depths, args.seeds)
        return 0

    if cmd == "torsion-audit":
        run_torsion_audit(settings, layout, receptors, mode)
        return 0
    stages = {"fetch": ["fetch"], "prepare": ["prepare"], "cartesian": ["cartesian"], "dpca": ["dpca"],
              "states": ["states"], "bootstrap": ["bootstrap"], "summarize": ["summarize"],
              "run": ["fetch", "prepare", "cartesian", "dpca", "states", "bootstrap", "summarize"]}[cmd]
    labels = getattr(args, "depths", None) or settings.depth_labels
    per_depth = getattr(args, "af2_models_per_depth", None)

    for r in receptors:
        if "fetch" in stages:
            from .download import fetch_fasta, fetch_gpcrmd
            fetch_fasta(settings, layout, r)
            fetch_gpcrmd(settings, layout, r, mode)
        if "prepare" in stages:
            from .prepare import prepare
            prepare(settings, layout, r, mode)
        if "cartesian" in stages:
            from . import cartesian
            cartesian.run(settings, layout, r, mode, labels=labels, models_per_depth=per_depth)
        if "dpca" in stages:
            run_dpca(settings, layout, r, mode, getattr(args, "depths", None), per_depth)
        if "states" in stages:
            from .states import run_states
            run_states(settings, layout, r, mode)
    if "bootstrap" in stages:
        from .states import run_bootstrap
        needed = {settings.analysis["bootstrap"]["reference"], *settings.analysis["bootstrap"]["comparators"]}
        if cmd == "run" and not needed <= set(labels):
            print(f"Skipping bootstrap: it compares {sorted(needed)}, not all analysed here", flush=True)
        else:
            run_bootstrap(settings, layout, receptors, mode)
    if "summarize" in stages:
        from .summarize import summarize
        summarize(settings, layout, receptors, mode)
    return 0


def run_dpca(settings: Settings, layout: Layout, receptor: str, mode: str,
             labels: list[str] | None = None, per_depth: int | None = None) -> Path:
    from . import dpca
    from .cartesian import af2_ensemble
    from .download import read_fasta
    from .prepare import prepared_inputs
    from .provenance import environment_record

    cart = layout.cartesian(receptor, mode)
    if (cart / "PROVENANCE.json").is_file():
        used = json.loads((cart / "PROVENANCE.json").read_text())["af2_models_per_setting"]
        labels = labels or list(used)
        if set(labels) != set(used):
            raise ValueError(f"{receptor}: dPCA settings {labels} differ from the Cartesian run {list(used)}")
    labels = labels or settings.depth_labels
    if per_depth is None and labels == settings.depth_labels:
        per_depth = int(settings.analysis["af2"]["models_per_depth"])
    top, trajectories, meta = prepared_inputs(layout, receptor, mode)
    return dpca.run(receptor=receptor, top=top, trajectories=trajectories,
                    seq=read_fasta(layout.fasta(receptor)),
                    by_depth=af2_ensemble(layout, receptor, labels, per_depth),
                    cartesian_dir=cart, final=layout.dpca(receptor, mode),
                    dpca_cfg=settings.analysis["dpca"],
                    provenance_extra={"md_replicas_mode": mode, "gpcrmd_record": meta["gpcrmd_record"],
                                      "config_sha256": {"receptors.yaml": settings.receptors_sha256,
                                                        "analysis.yaml": settings.analysis_sha256},
                                      "environment": environment_record()})


def run_torsion_audit(settings: Settings, layout: Layout, receptors: list[str], mode: str) -> None:
    import csv
    from . import dpca
    from .cartesian import af2_ensemble
    from .download import read_fasta
    for r in receptors:
        d = layout.dpca(r, mode)
        used = json.loads((d / "provenance.json").read_text())["af2_counts"]
        by_depth = af2_ensemble(layout, r, list(used), None)
        rows = dpca.audit_exclusions(d / "torsion_coverage.csv", by_depth, read_fasta(layout.fasta(r)))
        out = d / "af2_exclusions_by_setting.csv"
        if rows:
            with out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        only = sum(bool(x["fails_only_at_1:2"]) for x in rows)
        print(f"{r}: {len(rows)} AF2-excluded torsions; {only} fail only at 1:2 -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
