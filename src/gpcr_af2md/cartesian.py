"""Cartesian analyses: PCA/RMSIP, RMSF, ANM baseline, distance-PCA states, descriptors.

The numerical work is done by `_core/analysis_core.py`, whose SHA-256 is checked
before every run. This module gathers inputs, sets the receptor-specific
configuration and residue masks, and writes provenance.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Layout, Settings, depth_dirname, file_sha256
from .download import read_fasta
from .prepare import AA3, prepared_inputs
from .provenance import environment_record

CORE_DIR = Path(__file__).resolve().parent / "_core"
CORE_SHA256 = {"analysis_core.py": "40338325e227fda2be77cd7177c204b1f4ea220a0415110c826c09fcaefa333b"}
BACKBONE = frozenset(("N", "CA", "C", "O"))


def load_core():
    """Import the analysis core after confirming it is the checked version."""
    for name, expected in CORE_SHA256.items():
        digest = file_sha256(CORE_DIR / name)
        if digest != expected:
            raise RuntimeError(f"{CORE_DIR / name} has changed (expected {expected}, found {digest}); "
                               "refusing to run. Update CORE_SHA256 only after reviewing the change.")
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))
    import analysis_core as core  # noqa: E402  (path added above)
    return core


def excluded_residues(cfg: dict) -> list[int]:
    """Everything outside the analysis range, plus ICL3, is left out of the main comparisons."""
    lo, hi = map(int, cfg["analysis_keep_range"])
    il, ih = map(int, cfg["icl3_range"])
    keep = set(range(lo, hi + 1))
    return sorted(r for r in range(1, int(cfg["sequence_length"]) + 1) if (r not in keep) or (il <= r <= ih))


def configure_core(core, cfg: dict, patterns: dict[str, str], skip_ns: float, stride: int) -> None:
    core.CONFIG["md_skip_ns"] = float(skip_ns)
    core.CONFIG["md_stride"] = int(stride)
    core.CONFIG["md_resid_offset"] = int(cfg["md_resid_offset"])
    core.CONFIG["af2_resid_offset"] = int(cfg["af2_resid_offset"])
    core.CONFIG["xtal_resid_offset"] = int(cfg["xtal_resid_offset"])
    core.CONFIG["exclude_resids"] = excluded_residues(cfg)
    core.CONFIG["ionic_lock"] = dict(cfg["ionic_lock"])
    core.CONFIG["npxxy"] = dict(cfg["npxxy"])
    core.CONFIG["contact_label"] = cfg.get("contact_label", "TM3-TM6 ionic lock (A)")
    core.CONFIG["af2_ensembles"] = dict(patterns)
    for key in ("cg", "cg_backmapped"):  # no coarse-grained inputs in this study
        core.CONFIG[key]["topology"] = None
        core.CONFIG[key]["trajectory"] = None
    # The dedicated ICL3 PCA uses this receptor's loop range.
    lo, hi = map(int, cfg["icl3_range"])
    if not getattr(core, "_icl_wrapped", False):
        core._plot_icl_pca_unwrapped = core.plot_icl_pca
        core._icl_wrapped = True
    core.plot_icl_pca = lambda out, args, **_: core._plot_icl_pca_unwrapped(out, args, icl_lo=lo, icl_hi=hi)


def check_frame_spacing(top: Path, trajectories: list[Path], dt_ps: float) -> None:
    """The 25-ns skip is converted to frames from the trajectory time step, so it must be right."""
    import MDAnalysis as mda
    for path in trajectories:
        u = mda.Universe(str(top), str(path))
        if len(u.trajectory) > 1 and abs(u.trajectory.dt - dt_ps) > 0.1:
            raise ValueError(f"{path}: frame spacing {u.trajectory.dt} ps, expected {dt_ps}; rerun `prepare`")


def af2_ensemble(layout: Layout, receptor: str, labels: list[str], expected: int | None) -> dict[str, list[Path]]:
    """Ranked, unrelaxed ColabFold models per MSA setting: data/af2/<R>/<depth>/seed_*/..."""
    root = layout.af2() / receptor
    found = {label: sorted((root / depth_dirname(label)).glob("seed_*/*_unrelaxed_rank_*.pdb"))
             for label in labels}
    counts = {k: len(v) for k, v in found.items()}
    if any(n == 0 for n in counts.values()):
        raise FileNotFoundError(f"{receptor}: AF2 models missing under {root}: {counts}")
    if expected is not None and any(n != expected for n in counts.values()):
        raise ValueError(f"{receptor}: expected {expected} AF2 models per setting, found {counts}. "
                         "Pass --af2-models-per-depth to analyse a smaller ensemble on purpose.")
    return found


def backbone_positions(path: Path) -> set[int]:
    """Residues with all of N/CA/C/O; only these can enter backbone PCA."""
    import MDAnalysis as mda
    names: dict[int, set[str]] = defaultdict(set)
    for atom in mda.Universe(str(path)).select_atoms("name N CA C O"):
        names[int(atom.resid)].add(str(atom.name))
    return {r for r, seen in names.items() if BACKBONE <= seen}


def inspect_af2_pdb(path: Path, required: set[int], seq: str) -> None:
    """Every selected residue must have exactly one N/CA/C/O and match the canonical residue."""
    atoms: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    aa: dict[int, str] = {}
    with Path(path).open() as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            resid = int(line[22:26])
            name = line[12:16].strip()
            if resid not in required or name not in BACKBONE:
                continue
            atoms[resid][name] += 1
            if name == "CA":
                aa[resid] = AA3.get(line[17:20].strip(), "?")
    for r in sorted(required):
        if not 1 <= r <= len(seq):
            raise ValueError(f"{path}: residue {r} outside canonical length")
        for name in BACKBONE:
            if atoms[r][name] != 1:
                raise ValueError(f"{path}: residue {r} {name}: {atoms[r][name]} copies (expected 1)")
        if aa.get(r) != seq[r - 1]:
            raise ValueError(f"{path}: residue {r} is {aa.get(r)}, canonical {seq[r - 1]}")


def run(settings: Settings, layout: Layout, receptor: str, mode: str, *,
        labels: list[str] | None = None, models_per_depth: int | None = None,
        backbone_states: bool = True) -> Path:
    core = load_core()
    labels = labels or settings.depth_labels
    if models_per_depth is None and labels == settings.depth_labels:
        models_per_depth = int(settings.analysis["af2"]["models_per_depth"])
    cfg = settings.core_config(receptor)
    final = layout.cartesian(receptor, mode)
    if (final / "PROVENANCE.json").is_file():
        print(f"{receptor}: Cartesian results already exist at {final}", flush=True)
        return final
    stage = final.parent / f".incomplete_{receptor}"
    if final.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite {final} or {stage}; delete it to rerun")

    top, trajectories, md_meta = prepared_inputs(layout, receptor, mode)
    inactive = layout.endpoints(receptor) / "inactive.pdb"
    active = layout.endpoints(receptor) / "active.pdb"
    for p in (inactive, active):
        if not p.is_file():
            raise FileNotFoundError(p)
    seq = read_fasta(layout.fasta(receptor))
    if len(seq) != cfg["sequence_length"]:
        raise ValueError(f"{receptor}: FASTA length {len(seq)} != {cfg['sequence_length']}")
    af2 = af2_ensemble(layout, receptor, labels, models_per_depth)
    skip_ns = float(settings.analysis["cartesian"]["skip_ns"])

    check_frame_spacing(top, trajectories, cfg["frame_dt_ps"])
    patterns = {label: str(layout.af2() / receptor / depth_dirname(label) / "seed_*" / "*_unrelaxed_rank_*.pdb")
                for label in labels}
    configure_core(core, cfg, patterns, skip_ns, int(settings.analysis["md"]["stride"]))

    # Same complete-backbone mask for every receptor (C-alpha overlap alone is not
    # enough for backbone-state PCA).
    ca_common = set(core.build_common_set(str(inactive), str(active), str(top), list(patterns.values())))
    backbone_sets = [backbone_positions(p) for p in (top, inactive, active)]
    backbone_sets += [backbone_positions(paths[0]) for paths in af2.values()]
    common = ca_common & set.intersection(*backbone_sets)
    minimum = int(settings.analysis["cartesian"]["min_common_residues"])
    if len(common) < minimum:
        raise ValueError(f"{receptor}: only {len(common)} common complete-backbone residues (< {minimum})")
    core.CONFIG["exclude_resids"] = sorted(set(core.CONFIG["exclude_resids"]) | (ca_common - common))
    for paths in af2.values():
        for pdb in paths:
            inspect_af2_pdb(pdb, common, seq)

    stage.mkdir(parents=True)
    argv = [str(CORE_DIR / "analysis_core.py")]
    for xtc in trajectories:
        argv += ["--md", str(top), str(xtc)]
    argv += ["--inactive", str(inactive), "--active", str(active), "--out", str(stage)]
    print(f"{receptor}: {len(trajectories)} replicate(s) x {md_meta['replicas'][0]['retained_frames']} frames; "
          f"skip {skip_ns} ns; AF2 {', '.join(f'{k}={len(v)}' for k, v in af2.items())}", flush=True)
    saved = sys.argv, core.analyze_backbone_states, core.plot_backbone_states
    if not backbone_states:
        # The N/CA/C/O distance-state analysis is not used in the paper and needs far more
        # memory than the C-alpha analyses; replace it with an empty result.
        print(f"{receptor}: skipping backbone (N/CA/C/O) distance states", flush=True)
        core.analyze_backbone_states = lambda *a, **k: {"pop_md": np.array([]), "pop_cg_bb": None}
        core.plot_backbone_states = lambda *a, **k: None
    try:
        sys.argv = argv
        core.main()
    finally:
        sys.argv, core.analyze_backbone_states, core.plot_backbone_states = saved

    (stage / "PROVENANCE.json").write_text(json.dumps({
        "receptor": receptor,
        "md_replicas_mode": mode,
        "gpcrmd_record": md_meta["gpcrmd_record"],
        "md_inputs": md_meta["replicas"],
        "prepared_sha256": {p.name: file_sha256(p) for p in (top, *trajectories)},
        "frame_dt_ps": cfg["frame_dt_ps"], "skip_ns": skip_ns, "md_stride": settings.analysis["md"]["stride"],
        "cartesian_common_residues": sorted(common),
        "af2_models_per_setting": {k: len(v) for k, v in af2.items()},
        "backbone_states": backbone_states,
        "endpoints_sha256": {p.name: file_sha256(p) for p in (inactive, active)},
        "analysis_core_sha256": CORE_SHA256,
        "config_sha256": {"receptors.yaml": settings.receptors_sha256, "analysis.yaml": settings.analysis_sha256},
        "environment": environment_record(),
    }, indent=2) + "\n")
    os.rename(stage, final)
    print(f"COMPLETE: {final}", flush=True)
    return final


__all__ = ["run", "load_core", "af2_ensemble"]
