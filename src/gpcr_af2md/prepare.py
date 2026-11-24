"""Extract receptor-only trajectories from GPCRmd full-system files.

Steps and checks:

* the receptor is selected by audited segment IDs (CHRM2, OPRD1) or, otherwise, by
  matching canonical residue identities segment by segment;
* every selected C-alpha must match the canonical FASTA at its residue number;
* PSF and PDB atom order must agree for the selected atoms;
* unequal replicates are trimmed to the shortest, so replicates carry equal weight;
* output frames are re-stamped at the physical 200-ps spacing (raw GPCRmd XTC
  headers store 1000x the deposited interval). Coordinates are not modified.

Output: `<data>/prepared/<mode>/<receptor>/{minimal.pdb, replica_NN.xtc, provenance.json}`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import Layout, Settings, file_sha256
from .download import check_file_kind, read_fasta

AA3 = dict(zip("ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
               "ARNDCEQGHILKMFPSTWYV"))
AA3.update({"HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H", "MSE": "M"})
REPLICA_POLICY = "retain first min(source_frames) frames from each replica; equal frame weights"
MIN_RECEPTOR_CA = 180


def receptor_selection(universe, receptor: str, gpcrmd_cfg: dict, sequence: str):
    segments = gpcrmd_cfg.get("receptor_segments", "auto")
    if segments != "auto":
        selected = universe.select_atoms("protein and segid " + " ".join(segments))
        ranges = gpcrmd_cfg.get("expected_residue_ranges")
        if ranges:
            expected = {i for lo, hi in ranges for i in range(lo, hi + 1)}
            found = {int(a.resid) for a in selected.select_atoms("name CA")}
            if found != expected:
                raise ValueError(f"{receptor}: audited GPCRmd residue coverage changed; "
                                 f"missing={sorted(expected - found)[:20]}, extra={sorted(found - expected)[:20]}")
    else:
        # A segment is the receptor if >=95% of its C-alphas match the canonical
        # residue at the same number. Other proteins cannot match a whole GPCR.
        by_segment: dict[str, list[int]] = {}
        for atom in universe.select_atoms("protein and name CA"):
            r = int(atom.resid)
            if not 1 <= r <= len(sequence):
                continue
            counts = by_segment.setdefault(str(atom.segid), [0, 0])
            counts[0] += 1
            counts[1] += AA3.get(str(atom.resname), "?") == sequence[r - 1]
        accepted = [seg for seg, (n, matched) in by_segment.items()
                    if n >= 20 and matched >= 20 and matched / n >= 0.95]
        if not accepted:
            raise ValueError(f"{receptor}: no canonical GPCRmd receptor segment; stats={by_segment}")
        selected = universe.select_atoms("protein and segid " + " ".join(accepted))
    ca = selected.select_atoms("name CA")
    ids = [int(a.resid) for a in ca]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{receptor}: duplicate canonical C-alpha residue numbers")
    mismatches = [(int(a.resid), str(a.resname)) for a in ca
                  if not (1 <= int(a.resid) <= len(sequence)
                          and AA3.get(str(a.resname), "?") == sequence[int(a.resid) - 1])]
    if mismatches:
        raise ValueError(f"{receptor}: receptor differs from the canonical sequence: {mismatches[:20]}")
    if len(ca) < MIN_RECEPTOR_CA:
        raise ValueError(f"{receptor}: suspiciously short receptor mapping ({len(ca)} C-alphas)")
    return selected


def raw_inputs(settings: Settings, layout: Layout, receptor: str, mode: str):
    g = settings.receptor(receptor)["gpcrmd"]
    raw = layout.raw(int(g["record"]))
    topology, model = raw / g["topology"]["filename"], raw / g["model"]["filename"]
    trajectories = [raw / e["filename"] for e in settings.replicates(receptor, mode)]
    for path in (topology, model, *trajectories):
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run `gpcr-af2md fetch` first")
        check_file_kind(path)
    return topology, model, trajectories


def prepare(settings: Settings, layout: Layout, receptor: str, mode: str, *, check_only: bool = False) -> Path:
    import MDAnalysis as mda

    dest = layout.prepared(receptor, mode)
    if (dest / "provenance.json").is_file():
        print(f"{receptor}: already prepared at {dest}", flush=True)
        return dest
    cfg = settings.receptor(receptor)
    dt = float(settings.analysis["md"]["frame_dt_ps"])
    topology, model, trajectories = raw_inputs(settings, layout, receptor, mode)
    sequence = read_fasta(layout.fasta(receptor))

    universe = mda.Universe(str(topology), str(trajectories[0]))
    reference = mda.Universe(str(model))
    if len(reference.atoms) != len(universe.atoms):
        raise ValueError(f"{receptor}: topology and model PDB atom counts differ")
    group = receptor_selection(universe, receptor, cfg["gpcrmd"], sequence)
    idx = group.indices
    if not (np.array_equal(universe.atoms.names[idx], reference.atoms.names[idx])
            and np.array_equal(universe.atoms.resids[idx], reference.atoms.resids[idx])):
        raise ValueError(f"{receptor}: receptor atom order differs between topology and model PDB")
    counts = []
    for path in trajectories:
        u = mda.Universe(str(topology), str(path))
        if len(u.atoms) != len(universe.atoms):
            raise ValueError(f"{path}: atom count differs from topology")
        counts.append(len(u.trajectory))
    if len({file_sha256(p) for p in trajectories}) != len(trajectories):
        raise ValueError(f"{receptor}: two replicate files are identical")
    n = min(counts)
    print(f"{receptor} GPCRmd {cfg['gpcrmd']['record']}: {len(group)} atoms, "
          f"{len(group.select_atoms('name CA'))} C-alphas, frames={counts}, using {n}/replicate", flush=True)
    if check_only:
        return dest

    stage = dest.parent / f".incomplete_{receptor}"
    if dest.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite {dest} or {stage}; delete it to rebuild")
    stage.mkdir(parents=True)
    universe.trajectory[0]
    group.write(str(stage / "minimal.pdb"))
    written = mda.Universe(str(stage / "minimal.pdb"))
    if not (np.array_equal(written.atoms.resids, group.resids) and np.array_equal(written.atoms.names, group.names)):
        raise ValueError(f"{receptor}: written PDB does not preserve receptor atom identities")
    records = []
    for i, (raw, count) in enumerate(zip(trajectories, counts), start=1):
        u = mda.Universe(str(topology), str(raw))
        out = stage / f"replica_{i:02d}.xtc"
        with mda.Writer(str(out), n_atoms=len(group)) as writer:
            for frame in range(n):
                ts = u.trajectory[frame]
                ts.time, ts.dt = frame * dt, dt
                writer.write(u.atoms[group.indices])
        check = mda.Universe(str(stage / "minimal.pdb"), str(out))
        if len(check.trajectory) != n or (n > 1 and abs(check.trajectory.dt - dt) > 0.1):
            raise ValueError(f"{out}: wrong frame count or spacing")
        records.append({"filename": raw.name, "sha256": file_sha256(raw), "source_frames": count,
                        "prepared": out.name, "retained_frames": n})
    (stage / "provenance.json").write_text(json.dumps({
        "receptor": receptor, "gpcrmd_record": int(cfg["gpcrmd"]["record"]), "md_replicas_mode": mode,
        "frame_dt_ps": dt, "replica_policy": REPLICA_POLICY,
        "receptor_ca": len(group.select_atoms("name CA")), "receptor_atoms": len(group),
        "topology_sha256": file_sha256(topology), "model_pdb_sha256": file_sha256(model),
        "fasta_sha256": file_sha256(layout.fasta(receptor)),
        "replicas": records,
    }, indent=2) + "\n")
    os.rename(stage, dest)
    print(f"PREPARED {receptor}: {dest}", flush=True)
    return dest


def prepared_inputs(layout: Layout, receptor: str, mode: str) -> tuple[Path, list[Path], dict]:
    d = layout.prepared(receptor, mode)
    prov = d / "provenance.json"
    if not prov.is_file():
        raise FileNotFoundError(f"{receptor}: not prepared for --md-replicas {mode}; run `gpcr-af2md prepare`")
    meta = json.loads(prov.read_text())
    trajectories = [d / r["prepared"] for r in meta["replicas"]]
    top = d / "minimal.pdb"
    for p in (top, *trajectories):
        if not p.is_file():
            raise FileNotFoundError(p)
    return top, trajectories, meta
