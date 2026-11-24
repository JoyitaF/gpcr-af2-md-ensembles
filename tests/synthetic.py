"""Build a small synthetic GPCRmd-like dataset for software tests.

The structures are idealised backbones built from phi/psi angles. They test the
software path (file handling, masks, numerics, provenance), not any biology.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

AA1_TO_3 = dict(zip("ARNDCEQGHILKMFPSTWYV",
                    "ALA ARG ASN ASP CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()))
LENGTH = 200
IONIC_POS, IONIC_NEG, NPXXY = 40, 150, 180
ICL3 = (100, 112)
KEEP = (5, 195)
RECORD = 999


def toy_sequence(seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    letters = np.array(list("ACDFGHIKLMNPQSTVW"))  # no R/E/Y except where placed below
    seq = list(rng.choice(letters, LENGTH))
    seq[IONIC_POS - 1], seq[IONIC_NEG - 1], seq[NPXXY - 1] = "R", "E", "Y"
    return "".join(seq)


def _place(a, b, c, bond, angle, torsion):
    """NeRF: position of atom d given a-b-c, |cd|, angle bcd and torsion abcd (radians). Vectorised over frames."""
    bc = c - b
    bc /= np.linalg.norm(bc, axis=-1, keepdims=True)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    m = np.cross(n, bc)
    torsion = np.broadcast_to(np.asarray(torsion, dtype=float), c.shape[:-1])
    d2 = np.stack([np.full_like(torsion, -bond * np.cos(angle)),
                   bond * np.sin(angle) * np.cos(torsion),
                   bond * np.sin(angle) * np.sin(torsion)], axis=-1)
    return c + d2[..., 0:1] * bc + d2[..., 1:2] * m + d2[..., 2:3] * n


def build(phi: np.ndarray, psi: np.ndarray, seq: str) -> tuple[list[tuple[int, str, str]], np.ndarray]:
    """phi, psi: (frames, residues) in degrees. Returns atom list and (frames, atoms, 3) coordinates."""
    F, L = phi.shape
    rad = np.deg2rad
    N = np.zeros((F, L, 3)); CA = np.zeros((F, L, 3)); C = np.zeros((F, L, 3))
    N[:, 0] = [0.0, 1.458, 0.0]
    CA[:, 0] = [0.0, 0.0, 0.0]
    C[:, 0] = [1.525, 0.0, 0.0]
    for i in range(1, L):
        N[:, i] = _place(N[:, i - 1], CA[:, i - 1], C[:, i - 1], 1.329, rad(116.2), rad(psi[:, i - 1]))
        CA[:, i] = _place(CA[:, i - 1], C[:, i - 1], N[:, i], 1.458, rad(121.7), rad(180.0))
        C[:, i] = _place(C[:, i - 1], N[:, i], CA[:, i], 1.525, rad(111.2), rad(phi[:, i]))
    atoms, coords = [], []
    for i in range(L):
        r, aa = i + 1, seq[i]
        nxt = N[:, i + 1] if i + 1 < L else _place(N[:, i], CA[:, i], C[:, i], 1.329, rad(116.2), rad(psi[:, i]))
        O = _place(nxt, CA[:, i], C[:, i], 1.231, rad(120.5), rad(180.0))
        CB = _place(C[:, i], N[:, i], CA[:, i], 1.53, rad(110.5), rad(122.5))
        unit = (CB - CA[:, i]) / np.linalg.norm(CB - CA[:, i], axis=-1, keepdims=True)
        block = [("N", N[:, i]), ("CA", CA[:, i]), ("C", C[:, i]), ("O", O), ("CB", CB)]
        extra = {"R": ["NE", "NH1", "NH2"], "E": ["OE1", "OE2"], "Y": ["OH"]}.get(aa, [])
        for k, name in enumerate(extra):
            block.append((name, CB + unit * (1.5 + 1.2 * k) + np.array([0.3 * k, -0.2 * k, 0.1])))
        for name, xyz in block:
            atoms.append((r, AA1_TO_3[aa], name))
            coords.append(xyz)
    return atoms, np.stack(coords, axis=1)


def write_pdb(path: Path, atoms, xyz, segid="P0", bfactor=None, extra_lines: str = "") -> None:
    lines = []
    for k, ((r, resname, name), p) in enumerate(zip(atoms, xyz), start=1):
        b = 0.0 if bfactor is None else bfactor
        lines.append(f"ATOM  {k:5d} {name:^4s} {resname:>3s} A{r:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                     f"{1.0:6.2f}{b:6.2f}      {segid:<4s}{name[0]:>2s}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines) + extra_lines + "END\n")


def angles(frames: int, rng, state: float = 0.0, noise: float = 6.0):
    phi = np.full((frames, LENGTH), -62.0) + rng.normal(0, noise, (frames, LENGTH))
    psi = np.full((frames, LENGTH), -41.0) + rng.normal(0, noise, (frames, LENGTH))
    loop = np.r_[ICL3[0]:ICL3[1], 60:66, 130:136]
    phi[:, loop] = -80 + rng.normal(0, 3 * noise, (frames, len(loop))) + state
    psi[:, loop] = 140 + rng.normal(0, 3 * noise, (frames, len(loop))) - state
    return phi, psi


def make_dataset(root: Path, frames=(280, 280, 260), af2_labels=("64:640", "1:2"), seeds=(1, 2, 3, 4, 5, 6, 7, 8)) -> dict:
    """Create data/, config/ under `root`. Returns paths and the expected bad torsion."""
    rng = np.random.default_rng(11)
    seq = toy_sequence()
    data, cfgdir = root / "data", root / "config"
    (data / "fasta").mkdir(parents=True)
    (data / "fasta" / "TOY1.fasta").write_text(">sp|TOY1|synthetic\n" + seq + "\n")

    # GPCRmd-like full system: receptor segment P0 + an unrelated protein segment P9 + ions.
    phi0, psi0 = angles(1, rng)
    atoms, xyz0 = build(phi0, psi0, seq)
    other_seq = "".join(rng.choice(list("GSAP"), 40))
    o_atoms, o_xyz = build(np.full((1, 40), -60.0), np.full((1, 40), -40.0), other_seq.ljust(LENGTH, "G")[:40])
    o_xyz = o_xyz + np.array([40.0, 40.0, 0.0])
    ions = [(1, "SOD", "SOD"), (2, "CLA", "CLA")]
    raw = data / "raw" / "gpcrmd" / str(RECORD)
    raw.mkdir(parents=True)

    def system_pdb(path, rec_xyz):
        lines = []
        k = 0
        for (r, resname, name), p in zip(atoms, rec_xyz):
            k += 1
            lines.append(f"ATOM  {k:5d} {name:^4s} {resname:>3s} A{r:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00      P0  {name[0]:>2s}\n")
        for (r, resname, name), p in zip(o_atoms, o_xyz[0]):
            k += 1
            lines.append(f"ATOM  {k:5d} {name:^4s} {resname:>3s} B{r:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00      P9  {name[0]:>2s}\n")
        for (r, resname, name) in ions:
            k += 1
            lines.append(f"ATOM  {k:5d} {name:^4s} {resname:>3s} I{r:4d}    {60.0 + r:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00      ION {'X':>2s}\n")
        path.write_text("".join(lines) + "END\n")

    system_pdb(raw / f"1_dyn_{RECORD}.pdb", xyz0[0])
    shutil.copy2(raw / f"1_dyn_{RECORD}.pdb", raw / f"2_dyn_{RECORD}.pdb")

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    n_other = len(o_atoms) + len(ions)
    for rep, (nf, state) in enumerate(zip(frames, (0.0, 25.0, -25.0)), start=1):
        phi, psi = angles(nf, rng, state=state)
        _, xyz = build(phi, psi, seq)
        tail = np.concatenate([np.repeat(o_xyz, nf, axis=0),
                               np.tile(np.array([[61.0, 0, 0], [62.0, 0, 0]]), (nf, 1, 1))], axis=1)
        full = np.concatenate([xyz, tail], axis=1).astype(np.float32)
        u = mda.Universe(str(raw / f"1_dyn_{RECORD}.pdb"))
        u.load_new(full, format=MemoryReader)
        with mda.Writer(str(raw / f"{10 + rep}_trj_{RECORD}.xtc"), n_atoms=len(u.atoms)) as w:
            for i, ts in enumerate(u.trajectory):
                ts.time = i * 200000.0  # GPCRmd raw headers: 1000 x the physical 200 ps
                w.write(u.atoms)
        assert full.shape[1] == len(atoms) + n_other

    # Experimental endpoints in canonical numbering, missing a few residues.
    for state, s in (("inactive", -10.0), ("active", 10.0)):
        phi, psi = angles(1, rng, state=s, noise=2.0)
        e_atoms, e_xyz = build(phi, psi, seq)
        keep = [i for i, a in enumerate(e_atoms) if a[0] not in (3, 4, 197, 198)]
        write_pdb(data / "endpoints" / "TOY1" / f"{state}.pdb", [e_atoms[i] for i in keep], e_xyz[0][keep])

    # AF2-like ensembles. One 1:2 model gets a stretched C-N bond at residue 70 so the
    # torsion psi(69)/phi(70) must be excluded from the matched set.
    bad_feature = None
    for li, label in enumerate(af2_labels):
        for seed in seeds:
            for model in range(1, 6):
                phi, psi = angles(1, rng, state=5.0 * li, noise=4.0 + 12.0 * (label == "1:2"))
                a_atoms, a_xyz = build(phi, psi, seq)
                xyz = a_xyz[0].copy()
                if label == "1:2" and seed == seeds[0] and model == 1:
                    sel = np.array([a[0] >= 70 for a in a_atoms])
                    xyz[sel] += np.array([3.0, 0.0, 0.0])
                    bad_feature = (70, "phi")
                out = (data / "af2" / "TOY1" / label.replace(":", "x") / f"seed_{seed}" /
                       f"TOY1_unrelaxed_rank_{model:03d}_alphafold2_ptm_model_{model}_seed_{seed:03d}.pdb")
                write_pdb(out, a_atoms, xyz, segid="", bfactor=90.0 - 30 * (label == "1:2"))

    cfgdir.mkdir()
    repo_cfg = Path(__file__).resolve().parents[1] / "config"
    analysis = yaml.safe_load((repo_cfg / "analysis.yaml").read_text())
    # Short synthetic replicates: smaller MD blocks and fewer draws than the paper.
    analysis["bootstrap"] = {"reference": af2_labels[0], "comparators": list(af2_labels[1:]),
                             "block_ns": [5, 10], "draws": 10, "seed_root": 1}
    analysis["states"]["nearest_md_exclusion_ns"] = 5.0
    receptors = {"TOY1": {
        "name": "synthetic receptor", "uniprot": "TOY1", "sequence_length": LENGTH,
        "gpcrmd": {"record": RECORD, "start_pdb": "none", "start_state": "inactive",
                   "topology": {"file_id": 1, "filename": f"1_dyn_{RECORD}.pdb", "sha256": None},
                   "model": {"file_id": 2, "filename": f"2_dyn_{RECORD}.pdb", "sha256": None},
                   "replicates": [{"file_id": 10 + i, "filename": f"{10 + i}_trj_{RECORD}.xtc",
                                   "access": "public" if i == 1 else "account", "sha256": None} for i in (1, 2, 3)],
                   "receptor_segments": "auto"},
        "endpoints": {"inactive": "none", "active": "none"},
        "analysis_keep_range": list(KEEP), "icl3_range": list(ICL3),
        "ionic_lock": {"res_pos": IONIC_POS, "res_pos_name": "ARG", "res_neg": IONIC_NEG, "res_neg_name": "GLU",
                       "pos_atoms": ["NH1", "NH2", "NE"], "neg_atoms": ["OE1", "OE2"]},
        "npxxy": {"resid": NPXXY, "resname": "TYR"},
        "contact_label": "synthetic contact (A)"}}
    (cfgdir / "receptors.yaml").write_text(yaml.safe_dump(receptors, sort_keys=False))
    (cfgdir / "analysis.yaml").write_text(yaml.safe_dump(analysis, sort_keys=False))
    return {"data": data, "config": cfgdir, "results": root / "results", "labels": list(af2_labels),
            "models_per_depth": 5 * len(seeds), "bad_feature": bad_feature, "frames": frames}
