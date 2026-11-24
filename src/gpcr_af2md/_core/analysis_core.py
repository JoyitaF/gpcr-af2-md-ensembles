#!/usr/bin/env python3
# =============================================================================
# analysis_core.py
#
# Cartesian comparison of MD trajectories with AF2 MSA-depth ensembles.
#
# Two complementary analyses are applied to MD and to every AF2 MSA-depth
# ensemble:
#
#   PRIMARY — STATE IDENTIFICATION
#       Pairwise Cα/BB distance matrix → StandardScaler → sklearn PCA
#       → k-means clustering with elbow selection, free-energy surface,
#       per-source cluster populations, PC1/PC2 scatter.
#
#   SECONDARY — SUBSPACE COMPARISON
#       Kabsch superposition → Cα-coordinate covariance PCA → RMSIP /
#       cumulative overlap vs MD reference; ANM harmonic baseline;
#       per-residue RMSF; activation descriptors (TM3-TM6 ionic lock,
#       NPxxY Y7.53); endpoint coverage; ICL3 dedicated PCA;
#       convergence diagnostics (inter-replica RMSIP, PC1 cosine content).
#
# Called by gpcr_af2md.cartesian, which sets CONFIG per receptor. Standalone use:
#   python analysis_core.py --md rep1.pdb rep1.xtc --md rep2.pdb rep2.xtc \
#                           --inactive inactive.pdb --active active.pdb --out results
#
# deps: numpy scipy MDAnalysis prody matplotlib scikit-learn
# =============================================================================
import os, glob, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "md_skip_ns":  25.0,
    "md_stride":   1,

    # subspace comparison
    "n_modes":     10,
    "anm_cutoff":  15.0,
    "anm_gamma":   1.0,

    # residue numbering offsets (0 when all inputs use UniProt numbering)
    "md_resid_offset":   -1,
    "af2_resid_offset":  -1,
    "xtal_resid_offset":  0,

    # excluded from the common set (default; set per receptor by the caller)
    "exclude_resids": list(range(209, 219)) + list(range(1, 6)) + list(range(309, 413)),

    # activation descriptors (default; set per receptor by the caller)
    "ionic_lock": {
        "res_pos": 102, "res_pos_name": "ARG",
        "res_neg": 228, "res_neg_name": "GLU",
        "pos_atoms": ["NH1", "NH2", "NE"],
        "neg_atoms": ["OE1", "OE2"],
    },
    "npxxy": {"resid": 288, "resname": "TYR"},

    # AF2 MSA-depth ensembles
    "af2_ensembles": {
        "512:5120": "af2/512x5120/*_unrelaxed_rank_*.pdb",
        "256:2560": "af2/256x2560/*_unrelaxed_rank_*.pdb",
        "128:1280": "af2/128x1280/*_unrelaxed_rank_*.pdb",
        "64:640":   "af2/64x640/*_unrelaxed_rank_*.pdb",
        "32:320":   "af2/32x320/*_unrelaxed_rank_*.pdb",
        "16:160":   "af2/16x160/*_unrelaxed_rank_*.pdb",
        "8:80":     "af2/8x80/*_unrelaxed_rank_*.pdb",
        "1:2":      "af2/single seq/*_unrelaxed_rank_*.pdb",
    },

    # state identification (distance-PCA + k-means)
    "n_dist_pcs": 20,     # PCs kept from distance-matrix PCA
    "n_clusters":  5,     # k for final k-means — inspect elbow plot first
    "k_range":  [2, 15],  # k values tried in elbow scan
}


# =============================================================================
# PRIMARY: DISTANCE-MATRIX PCA + CLUSTERING
# =============================================================================

def dist_features(coords):
    """F×N×3 → F×(N*(N-1)/2) pairwise distances (Å)."""
    from scipy.spatial.distance import pdist
    return np.array([pdist(c) for c in coords], dtype=np.float32)

def dist_pca_fit(D, n=20):
    """Fit StandardScaler + PCA on distance matrix D (F×M).
    Returns (scaler, pca) — both sklearn objects, pickle-safe."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    Ds = scaler.fit_transform(D)
    n = min(n, D.shape[0] - 1, D.shape[1])
    pca = PCA(n_components=n, random_state=0)
    pca.fit(Ds)
    return scaler, pca

def dist_pca_transform(D, scaler, pca):
    return pca.transform(scaler.transform(D))

def free_energy_2d(x, y, bins=50):
    """2D free-energy surface in units of kT (up to a constant; minimum set to 0)."""
    H, xe, ye = np.histogram2d(x, y, bins=bins, density=True)
    F = -np.log(H + 1e-12)
    F -= F.min()
    return F, xe, ye

def elbow_kmeans(X, k_range):
    """k-means inertia scan; returns list of (k, inertia)."""
    from sklearn.cluster import KMeans
    out = []
    for k in range(k_range[0], k_range[1]):
        km = KMeans(n_clusters=k, n_init=10, random_state=1)
        km.fit(X)
        out.append((k, km.inertia_))
    return out

def fit_kmeans(X, k):
    """Fit k-means; returns (labels array, fitted KMeans object)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=1)
    return km.fit_predict(X), km

def cluster_populations(labels, k):
    return np.bincount(labels, minlength=k) / float(len(labels))


# =============================================================================
# SECONDARY: SUBSPACE COMPARISON MATH  (script)
# =============================================================================

def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ Pc.T).T + Q.mean(0)

def kabsch_transform(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t

def superpose_to(coords, ref):
    return np.stack([kabsch(c, ref) for c in coords])

def pca_modes(coords, n=10):
    F, N, _ = coords.shape
    X = coords.reshape(F, N * 3)
    mean = X.mean(0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (F - 1)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    return mean, v[:, idx[:n]], w[idx[:n]]

def anm_modes(ref_ca, n=10, cutoff=15.0, gamma=1.0):
    from prody import ANM
    anm = ANM("anm")
    anm.buildHessian(np.asarray(ref_ca, float), cutoff=cutoff, gamma=gamma)
    anm.calcModes(n_modes=n, zeros=False)
    return anm.getEigvecs()

def rmsip(V, W, k):
    A = V[:, :k].T @ W[:, :k]
    return float(np.sqrt((A ** 2).sum() / k))

def cumulative_overlap(V_ref, W, k):
    co = [np.sqrt(((V_ref[:, i] @ W[:, :k]) ** 2).sum()) for i in range(k)]
    return float(np.mean(co))

def rmsf_per_atom(coords, ref):
    sup = superpose_to(coords, ref)
    dev = sup - sup.mean(0)
    return np.sqrt((dev ** 2).sum(-1).mean(0))

def subspace_project(coords, mean, eigvecs, k, ref):
    sup = superpose_to(coords, ref)
    X = sup.reshape(sup.shape[0], -1) - mean
    return X @ eigvecs[:, :k]

def cosine_content(proj_1d, i=1):
    T = len(proj_1d)
    t = np.arange(T)
    num = (np.cos(i * np.pi * t / T) * proj_1d).sum()
    den = (proj_1d ** 2).sum()
    return float((2.0 / T) * num ** 2 / den) if den > 0 else float("nan")


# =============================================================================
# I/O HELPERS  (MDAnalysis)
# =============================================================================

def _resid_sel(resids, atom_name="CA"):
    return f"name {atom_name} and resid " + " ".join(str(r) for r in sorted(resids))

def _pick_one_atom(atomgroup):
    """Choose one deterministic conformer when a structure contains alternate locations."""
    if len(atomgroup) == 0:
        return None
    if len(atomgroup) == 1:
        return atomgroup[0]

    try:
        occ = np.asarray(atomgroup.occupancies, float)
    except Exception:
        occ = np.ones(len(atomgroup), dtype=float)
    occ = np.nan_to_num(occ, nan=-1.0)
    try:
        alts = [str(x).strip() for x in atomgroup.altLocs]
    except Exception:
        alts = [""] * len(atomgroup)

    # Highest occupancy first; on a tie prefer conventional altloc A, then blank.
    best = max(range(len(atomgroup)),
               key=lambda i: (occ[i], alts[i] == "A", alts[i] == "", -i))
    return atomgroup[best]

def structure_ca(pdb, resids, offset=0, atom_name="CA"):
    """Single structure -> exactly one requested atom per canonical residue.

    Crystal structures can contain alternate-location atoms.  MDAnalysis returns
    every altloc for a name/resid selection, which breaks one-to-one Kabsch
    pairing.  Collapse those alternatives deterministically without changing
    coordinates or residue numbering.
    """
    import MDAnalysis as mda
    u = mda.Universe(pdb)
    coords, canon = [], []
    for r in sorted(resids):
        ag = u.select_atoms(f"resid {r + offset} and name {atom_name}")
        atom = _pick_one_atom(ag)
        if atom is None:
            raise ValueError(f"{pdb}: missing {atom_name} for canonical residue {r} "
                             f"(structure resid {r + offset})")
        coords.append(atom.position.copy())
        canon.append(r)
    return np.asarray(coords), np.asarray(canon, dtype=int)

def ca_resids_present(top, offset=0, atom_name="CA"):
    import MDAnalysis as mda
    u = mda.Universe(top)
    ca = u.select_atoms(f"name {atom_name}")
    return set(int(r) - offset for r in ca.resids)

def traj_ca(top, traj, resids, offset=0, skip_ns=0.0, stride=1, atom_name="CA"):
    import MDAnalysis as mda
    u = mda.Universe(top, traj) if traj else mda.Universe(top)
    sel = u.select_atoms(_resid_sel([r + offset for r in resids], atom_name))
    order = np.argsort(sel.resids)
    dt_ps = float(getattr(u.trajectory, "dt", 0.0)) or 1.0
    start = int(round(skip_ns * 1000.0 / dt_ps)) if traj else 0
    coords = []
    for ts in u.trajectory[start::stride]:
        coords.append(sel.positions[order].copy())
    return np.asarray(coords)

def ensemble_ca(pdb_glob, resids, offset=0):
    import MDAnalysis as mda
    files = sorted(glob.glob(pdb_glob))
    if not files:
        return np.empty((0, len(resids), 3)), np.array([])
    coords, plddt = [], []
    sel_str = _resid_sel([r + offset for r in resids])
    for f in files:
        u = mda.Universe(f)
        sel = u.select_atoms(sel_str)
        order = np.argsort(sel.resids)
        coords.append(sel.positions[order].copy())
        bf = getattr(sel, "tempfactors", None)
        plddt.append(float(np.mean(bf)) if bf is not None and len(bf) else np.nan)
    return np.asarray(coords), np.asarray(plddt)


# =============================================================================
# BACKBONE I/O HELPERS  (protein-and-backbone: N, CA, C, O)
# =============================================================================

_BB_ATOM_ORDER = {"N": 0, "CA": 1, "C": 2, "O": 3}

def _backbone_sel(resids):
    """Selection string for N/CA/C/O of the given (already-offset) residue IDs."""
    return "(name N CA C O) and resid " + " ".join(str(r) for r in sorted(resids))

def _backbone_sort(sel):
    """Index array that orders atoms by (resid, N→CA→C→O) for consistent pairing."""
    return np.lexsort([
        [_BB_ATOM_ORDER.get(n, 99) for n in sel.names],
        sel.resids,
    ])

def traj_backbone(top, traj, resids, offset=0, skip_ns=0.0, stride=1):
    """Trajectory → backbone coords F×(4N)×3 on the common set."""
    import MDAnalysis as mda
    u = mda.Universe(top, traj) if traj else mda.Universe(top)
    sel = u.select_atoms(_backbone_sel([r + offset for r in resids]))
    order = _backbone_sort(sel)
    dt_ps = float(getattr(u.trajectory, "dt", 0.0)) or 1.0
    start = int(round(skip_ns * 1000.0 / dt_ps)) if traj else 0
    coords = []
    for ts in u.trajectory[start::stride]:
        coords.append(sel.positions[order].copy())
    return np.asarray(coords)   # F × (4N) × 3

def structure_backbone(pdb, resids, offset=0):
    """Single PDB -> backbone coords (4N)x3 with one conformer per atom name.

    This mirrors structure_ca's altloc handling so crystal endpoints always have
    exactly N/CA/C/O for every common residue.
    """
    import MDAnalysis as mda
    u = mda.Universe(pdb)
    coords = []
    for r in sorted(resids):
        for name in ("N", "CA", "C", "O"):
            ag = u.select_atoms(f"resid {r + offset} and name {name}")
            atom = _pick_one_atom(ag)
            if atom is None:
                raise ValueError(f"{pdb}: missing backbone atom {name} for canonical "
                                 f"residue {r} (structure resid {r + offset})")
            coords.append(atom.position.copy())
    return np.asarray(coords)

def ensemble_backbone(pdb_glob, resids, offset=0):
    """AF2-style PDB ensemble → M×(4N)×3 backbone coords."""
    import MDAnalysis as mda
    files = sorted(glob.glob(pdb_glob))
    if not files:
        return np.empty((0, 4 * len(resids), 3))
    coords = []
    for f in files:
        u = mda.Universe(f)
        sel = u.select_atoms(_backbone_sel([r + offset for r in resids]))
        coords.append(sel.positions[_backbone_sort(sel)].copy())
    return np.asarray(coords)


# =============================================================================
# ACTIVATION DESCRIPTORS
# =============================================================================

def _check_resname(u, resid, expected, label):
    res = u.select_atoms(f"resid {resid}").residues
    got = res.resnames[0] if len(res) else "??"
    if expected not in got:
        print(f"  [WARN] {label}: resid {resid} is {got}, expected {expected} "
              f"-- check *_resid_offset in CONFIG.")
    return got

def descriptors(top, traj, ref_inactive_ca, common_resids, offset=0, skip_ns=0.0, stride=1):
    import MDAnalysis as mda
    u = mda.Universe(top, traj) if traj else mda.Universe(top)
    il = CONFIG["ionic_lock"]; ny = CONFIG["npxxy"]
    rp  = il["res_pos"]  + offset
    rn  = il["res_neg"]  + offset
    yr  = ny["resid"]    + offset
    _check_resname(u, rp, il["res_pos_name"], "ionic-lock +")
    _check_resname(u, rn, il["res_neg_name"], "ionic-lock -")
    _check_resname(u, yr, ny["resname"],      "NPxxY Tyr")
    pos    = u.select_atoms(f"resid {rp} and name " + " ".join(il["pos_atoms"]))
    neg    = u.select_atoms(f"resid {rn} and name " + " ".join(il["neg_atoms"]))
    ca_pos = u.select_atoms(f"resid {rp} and name CA")
    ca_neg = u.select_atoms(f"resid {rn} and name CA")
    y_ca   = u.select_atoms(f"resid {yr} and name CA")
    common = u.select_atoms(_resid_sel([r + offset for r in common_resids]))
    corder = np.argsort(common.resids)
    y_ref  = _ref_ca_for_resid(ref_inactive_ca, common_resids, ny["resid"])
    dt_ps  = float(getattr(u.trajectory, "dt", 0.0)) or 1.0
    start  = int(round(skip_ns * 1000.0 / dt_ps)) if traj else 0
    ionic, tm36, ydisp = [], [], []
    for ts in u.trajectory[start::stride]:
        P, Nn = pos.positions, neg.positions
        ionic.append(
            float(np.min(np.linalg.norm(P[:, None, :] - Nn[None, :, :], axis=-1)))
            if len(P) and len(Nn) else np.nan)
        tm36.append(float(np.linalg.norm(ca_pos.positions[0] - ca_neg.positions[0])))
        R, t = kabsch_transform(common.positions[corder], ref_inactive_ca)
        y_now = y_ca.positions[0] @ R.T + t
        ydisp.append(float(np.linalg.norm(y_now - y_ref)) if y_ref is not None else np.nan)
    return {"ionic_lock": np.array(ionic),
            "tm3_tm6_ca": np.array(tm36),
            "y288_disp":  np.array(ydisp)}

def _ref_ca_for_resid(ref_ca, common_resids, target_resid):
    order = sorted(common_resids)
    return ref_ca[order.index(target_resid)] if target_resid in order else None


def _concat_descriptors(descriptor_dicts):
    """Pool descriptor arrays from multiple structures/trajectories.

    This does not change how any descriptor is calculated; it only concatenates
    the per-frame/per-structure values so the plotted distribution represents
    every retained MD frame or every AF2 model rather than a single source file.
    """
    if not descriptor_dicts:
        return {"ionic_lock": np.array([]),
                "tm3_tm6_ca": np.array([]),
                "y288_disp": np.array([])}
    keys = descriptor_dicts[0].keys()
    return {k: np.concatenate([np.asarray(d[k]).reshape(-1) for d in descriptor_dicts])
            for k in keys}


# =============================================================================
# COMMON RESIDUE SET
# =============================================================================

def build_common_set(inactive, active, md_top, af2_globs):
    excl = set(CONFIG["exclude_resids"])
    sets = []
    if inactive: sets.append(ca_resids_present(inactive, CONFIG["xtal_resid_offset"]))
    if active:   sets.append(ca_resids_present(active,   CONFIG["xtal_resid_offset"]))
    if md_top:   sets.append(ca_resids_present(md_top,   CONFIG["md_resid_offset"]))
    for g in af2_globs:
        fs = sorted(glob.glob(g))
        if fs: sets.append(ca_resids_present(fs[0], CONFIG["af2_resid_offset"]))
    if not sets:
        raise SystemExit("Cannot define a common residue set -- give at least endpoints + MD.")
    common = sorted(set.intersection(*sets) - excl)
    print(f"[common set] {len(common)} residues "
          f"(range {common[0]}-{common[-1]}, ICL3 + termini excluded)")
    return common


# =============================================================================
# ANALYSIS BLOCKS
# =============================================================================

def analyze_md_reference(args, common, ref_struct):
    print("\n=== MD reference ===")
    cfg = CONFIG
    per_replica = []
    for (top, xtc) in args.md:
        c = traj_ca(top, xtc, common, cfg["md_resid_offset"],
                    cfg["md_skip_ns"], cfg["md_stride"])
        print(f"  {os.path.basename(xtc)}: {c.shape[0]} frames "
              f"(after {cfg['md_skip_ns']} ns skip)")
        per_replica.append(c)
    pooled = np.concatenate(per_replica, axis=0)
    print(f"  pooled: {pooled.shape[0]} frames, {pooled.shape[1]} Cα")

    sup = superpose_to(pooled, ref_struct)
    mean, V_md, w_md = pca_modes(sup, cfg["n_modes"])
    rmsf_md = rmsf_per_atom(pooled, ref_struct)
    md_mean_struct = sup.mean(0).reshape(-1, 3)
    V_anm_md = anm_modes(md_mean_struct, cfg["n_modes"], cfg["anm_cutoff"], cfg["anm_gamma"])
    anm_md_baseline = rmsip(V_anm_md, V_md, cfg["n_modes"])
    print(f"  RMSIP(ANM_MD, PCA_MD) = {anm_md_baseline:.3f}  "
          f"(ceiling for any ANM recovering MD subspace)")

    inter = []
    if len(per_replica) > 1:
        Vs = [pca_modes(superpose_to(c, ref_struct), cfg["n_modes"])[1] for c in per_replica]
        for i in range(len(Vs)):
            for j in range(i + 1, len(Vs)):
                inter.append(rmsip(Vs[i], Vs[j], cfg["n_modes"]))
        print(f"  inter-replica RMSIP = {np.mean(inter):.3f} ± {np.std(inter):.3f}")
    cos = []
    for c in per_replica:
        p1 = subspace_project(c, mean, V_md, 1, ref_struct)[:, 0]
        cos.append(cosine_content(p1, 1))
    print(f"  PC1 cosine content = {np.mean(cos):.3f}  (→1 = diffusive/under-sampled)")

    # Activation descriptors: pooled over every retained frame of every MD replica.
    desc = _concat_descriptors([
        descriptors(top, xtc, ref_struct, common,
                    cfg["md_resid_offset"], cfg["md_skip_ns"], cfg["md_stride"])
        for (top, xtc) in args.md
    ])
    print(f"  descriptor samples pooled across {len(args.md)} MD replicas: "
          f"{len(desc['tm3_tm6_ca'])} frames")

    return dict(mean=mean, V=V_md, eigvals=w_md, rmsf=rmsf_md, ref=ref_struct,
                anm_baseline=anm_md_baseline, inter_replica=inter,
                pc1_cosine=cos, descriptors=desc, pooled=pooled)


def analyze_af2_depths(args, common, md, endpoints):
    """Returns (rows, desc_by_depth, subspace_proj_by_depth, coords_by_depth)."""
    cfg = CONFIG
    rows, desc_bd, sproj_bd, coords_bd = [], {}, {}, {}
    if not cfg["af2_ensembles"]:
        print("\n=== AF2 depths: none configured ===")
        return rows, desc_bd, sproj_bd, coords_bd
    print("\n=== AF2 MSA-depth ensembles ===")
    for label, gpat in cfg["af2_ensembles"].items():
        coords, plddt = ensemble_ca(gpat, common, cfg["af2_resid_offset"])
        if coords.shape[0] < 3:
            print(f"  {label}: <3 structures found -- skipping")
            continue
        sup = superpose_to(coords, md["ref"])
        mean_e, V_e, _ = pca_modes(sup, min(cfg["n_modes"], coords.shape[0] - 1))
        V_anm_e = anm_modes(sup.mean(0).reshape(-1, 3), cfg["n_modes"],
                            cfg["anm_cutoff"], cfg["anm_gamma"])
        k = min(cfg["n_modes"], V_e.shape[1])
        rip_pca = rmsip(V_e, md["V"], k)
        rip_anm = rmsip(V_anm_e, md["V"], cfg["n_modes"])
        rmsf_e  = rmsf_per_atom(coords, md["ref"])
        r_ens   = float(np.corrcoef(rmsf_e, md["rmsf"])[0, 1])
        half = coords.shape[0] // 2
        Va = pca_modes(superpose_to(coords[:half], md["ref"]), k)[1]
        Vb = pca_modes(superpose_to(coords[half:], md["ref"]), k)[1]
        split = rmsip(Va, Vb, k)
        cov = {}
        for name, eca in endpoints.items():
            eca_sup = kabsch(eca, md["ref"])
            d = np.sqrt(((sup - eca_sup) ** 2).sum(-1).mean(-1))
            cov[name] = float(d.min())
        sproj_bd[label] = subspace_project(coords, md["mean"], md["V"], 2, md["ref"])
        coords_bd[label] = coords
        # Activation descriptors: one value per AF2 model at this MSA depth, pooled.
        depth_pdbs = sorted(glob.glob(gpat))
        desc_bd[label] = _concat_descriptors([
            descriptors(pdb, None, md["ref"], common, cfg["af2_resid_offset"])
            for pdb in depth_pdbs
        ])
        print(f"           descriptor samples: {len(desc_bd[label]['tm3_tm6_ca'])} AF2 models")
        rows.append(dict(depth=label, n=int(coords.shape[0]),
                         mean_pLDDT=float(np.nanmean(plddt)),
                         RMSIP_pca=rip_pca, RMSIP_anm=rip_anm,
                         cum_overlap=cumulative_overlap(md["V"], V_e, k),
                         rmsf_corr=r_ens, split_half_RMSIP=split,
                         min_rmsd_inactive=cov.get("inactive", np.nan),
                         min_rmsd_active=cov.get("active", np.nan)))
        print(f"  {label:>9}: n={coords.shape[0]:3d}  pLDDT={np.nanmean(plddt):5.1f}  "
              f"RMSIP_pca={rip_pca:.3f}  RMSIP_anm={rip_anm:.3f}  "
              f"RMSFr={r_ens:+.3f}  split={split:.3f}  "
              f"d_act={cov.get('active', float('nan')):.2f}")
    return rows, desc_bd, sproj_bd, coords_bd


def analyze_conformational_states(common, md, coords_by_depth, endpoints):
    """
    Primary conformational analysis.

    Fits StandardScaler + PCA on the pairwise Cα distance matrix of the pooled
    AA-MD trajectory, then projects the AF2 ensembles and endpoints onto that space.
    k-means clustering and FES are computed on the AA-MD projection.
    """
    print("\n=== Conformational states (distance-PCA + k-means) ===")
    cfg = CONFIG

    print("  computing pairwise Cα distances for AA-MD ...")
    D_md = dist_features(md["pooled"])
    print(f"  distance feature dim: {D_md.shape[1]} pairs  "
          f"({md['pooled'].shape[1]} Cα, {md['pooled'].shape[0]} frames)")

    scaler, pca = dist_pca_fit(D_md, cfg["n_dist_pcs"])
    var = pca.explained_variance_ratio_
    print(f"  dist-PCA: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
          f"cumulative(5 PCs)={var[:5].sum()*100:.1f}%")
    X_md = dist_pca_transform(D_md, scaler, pca)

    # elbow scan and final k-means
    print(f"  elbow scan k={cfg['k_range'][0]}..{cfg['k_range'][1]-1} ...")
    inertias = elbow_kmeans(X_md[:, :3], cfg["k_range"])
    k = cfg["n_clusters"]
    labels_md, kmeans = fit_kmeans(X_md[:, :3], k)
    pop_md = cluster_populations(labels_md, k)
    print(f"  k={k} MD populations: " +
          "  ".join(f"C{i}={pop_md[i]:.3f}" for i in range(k)))

    def _project_and_assign(coords):
        if coords is None or len(coords) == 0:
            return None, None, None
        X = dist_pca_transform(dist_features(coords), scaler, pca)
        lbl = kmeans.predict(X[:, :3])
        return X, lbl, cluster_populations(lbl, k)

    X_af2, labels_af2, pop_af2 = {}, {}, {}
    for label, coords in coords_by_depth.items():
        X_af2[label], labels_af2[label], pop_af2[label] = _project_and_assign(coords)

    # endpoint projections (single structure each)
    X_ep = {}
    for name, eca in endpoints.items():
        X_ep[name] = dist_pca_transform(dist_features(eca[None]), scaler, pca)[0]

    F_md, xe, ye = free_energy_2d(X_md[:, 0], X_md[:, 1])

    return dict(
        scaler=scaler, pca=pca, kmeans=kmeans, k=k, var_ratio=var,
        X_md=X_md, labels_md=labels_md, pop_md=pop_md,
        X_af2=X_af2, labels_af2=labels_af2, pop_af2=pop_af2,
        X_ep=X_ep, inertias=inertias,
        F_md=F_md, fes_edges=(xe, ye),
    )


def analyze_backbone_states(args, common, endpoints):
    """
    State identification on full backbone (N/CA/C/O) pairwise distances.

    Uses the protein backbone as the feature space; applied to MD and AF2.
    """
    print("\n=== Backbone state identification (N/CA/C/O distance-PCA + k-means) ===")
    cfg = CONFIG

    # AA-MD backbone — pool all replicas
    per_replica_bb = []
    for top, xtc in args.md:
        c = traj_backbone(top, xtc, common, cfg["md_resid_offset"],
                          cfg["md_skip_ns"], cfg["md_stride"])
        n_atoms = c.shape[1]
        print(f"  {os.path.basename(xtc)}: {c.shape[0]} frames, "
              f"{n_atoms} backbone atoms ({n_atoms // 4} residues × 4)")
        per_replica_bb.append(c)
    pooled_bb = np.concatenate(per_replica_bb, axis=0)

    D_md = dist_features(pooled_bb)
    print(f"  backbone distance feature dim: {D_md.shape[1]} pairs")
    scaler, pca = dist_pca_fit(D_md, cfg["n_dist_pcs"])
    var = pca.explained_variance_ratio_
    print(f"  bb-PCA: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
          f"cumulative(5)={var[:5].sum()*100:.1f}%")
    X_md = dist_pca_transform(D_md, scaler, pca)

    print(f"  elbow scan k={cfg['k_range'][0]}..{cfg['k_range'][1]-1} ...")
    inertias = elbow_kmeans(X_md[:, :3], cfg["k_range"])
    k = cfg["n_clusters"]
    labels_md, kmeans = fit_kmeans(X_md[:, :3], k)
    pop_md = cluster_populations(labels_md, k)
    print(f"  k={k} MD populations: " +
          "  ".join(f"C{i}={pop_md[i]:.3f}" for i in range(k)))

    # AF2 backbone (per MSA depth)
    X_af2, labels_af2, pop_af2 = {}, {}, {}
    for label, gpat in cfg["af2_ensembles"].items():
        coords_bb = ensemble_backbone(gpat, common, cfg["af2_resid_offset"])
        if len(coords_bb) >= 3:
            X = dist_pca_transform(dist_features(coords_bb), scaler, pca)
            lbl = kmeans.predict(X[:, :3])
            X_af2[label]      = X
            labels_af2[label] = lbl
            pop_af2[label]    = cluster_populations(lbl, k)

    # endpoint projections
    X_ep = {}
    for name, pdb in [("inactive", args.inactive), ("active", args.active)]:
        if pdb:
            bb = structure_backbone(pdb, common, cfg["xtal_resid_offset"])
            X_ep[name] = dist_pca_transform(dist_features(bb[None]), scaler, pca)[0]

    F_md, xe, ye = free_energy_2d(X_md[:, 0], X_md[:, 1])

    return dict(
        scaler=scaler, pca=pca, kmeans=kmeans, k=k, var_ratio=var,
        X_md=X_md, labels_md=labels_md, pop_md=pop_md,
        X_af2=X_af2, labels_af2=labels_af2, pop_af2=pop_af2,
        X_ep=X_ep, inertias=inertias,
        F_md=F_md, fes_edges=(xe, ye),
    )


# =============================================================================
# PLOTTING
# =============================================================================

# shared colour / marker conventions
_SRC_COLORS  = {"aa": "#2176AE"}
_EP_COLORS   = {"inactive": "#C0392B", "active": "#F39C12"}
_MK          = {"aa": "o", "af2": "D", "xtal": "*"}


def _af2_palette(labels):
    return {lbl: plt.cm.RdPu(0.35 + 0.60 * i / max(len(labels) - 1, 1))
            for i, lbl in enumerate(labels)}


def _plot_source_panels(out, prefix, pc_label, k, sources):
    """Per-source raw scatter and single-source population bar chart.

    One scatter and one population bar chart for every source supplied.
    sources is a list of
    (name, X, labels, pop, color); entries with X=None are silently skipped.
    """
    CMAP = plt.cm.tab10.colors[:k]
    for src_name, X, labels, pop, color in sources:
        if X is None:
            continue
        safe = src_name.replace(":", "-").replace(" ", "_").replace("\n", "_")

        # raw PC1/PC2 scatter — one per source, single colour
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.3, color=color, rasterized=True)
        ax.set_xlabel(f"{pc_label}1"); ax.set_ylabel(f"{pc_label}2")
        ax.set_title(f"{src_name}: {pc_label} projection")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{prefix}_raw_{safe}.png"), dpi=150); plt.close()

        # single-source population bar chart
        if pop is not None:
            fig, ax = plt.subplots(figsize=(max(4, k * 0.8), 4))
            ax.bar(range(k), pop, color=CMAP[:k], edgecolor="k", linewidth=0.4)
            ax.set_xticks(range(k))
            ax.set_xticklabels([f"C{i}" for i in range(k)])
            ax.set_ylabel("population fraction"); ax.set_ylim(0, 1)
            ax.set_title(f"{src_name}: cluster populations (k={k})")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"{prefix}_pop_{safe}.png"), dpi=150)
            plt.close()


def plot_conformational_states(out, states):
    """Primary plots: elbow, scree, FES, cluster scatter, population bar chart."""
    k = states["k"]
    CMAP = plt.cm.tab10.colors[:k]

    # 1. elbow
    fig, ax = plt.subplots(figsize=(5, 4))
    ks, ine = zip(*states["inertias"])
    ax.plot(ks, ine, "o-")
    ax.axvline(k, ls="--", c="red", label=f"chosen k={k}")
    ax.set_xlabel("k"); ax.set_ylabel("inertia (within-cluster SS)")
    ax.set_title("k-means elbow (dist-PCA, first 3 PCs)"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_state_elbow.png"), dpi=150); plt.close()

    # 2. scree
    var = states["var_ratio"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(1, len(var) + 1), var * 100, color="steelblue", edgecolor="k", linewidth=0.4)
    ax.plot(range(1, len(var) + 1), np.cumsum(var) * 100, "o-", color="red", label="cumulative")
    ax.set_xlabel("dist-PC"); ax.set_ylabel("% variance explained")
    ax.set_title("Distance-PCA scree plot"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_state_scree.png"), dpi=150); plt.close()

    # 3. FES (AA-MD)
    F, xe, ye = states["F_md"], *states["fes_edges"]
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(xc, yc, F.T, levels=20, cmap="RdYlBu_r")
    plt.colorbar(cf, ax=ax, label="ΔF (kT)")
    for name, x_ep in states["X_ep"].items():
        ax.scatter(x_ep[0], x_ep[1], marker="*", s=350,
                   color=_EP_COLORS.get(name, "#555"), edgecolor="k", zorder=5,
                   label=f"{name} (xtal)")
    ax.set_xlabel("dist-PC1"); ax.set_ylabel("dist-PC2")
    ax.set_title("AA-MD free-energy surface"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_state_fes_md.png"), dpi=150); plt.close()

    # 4. cluster-coloured scatter: MD + endpoints
    fig, ax = plt.subplots(figsize=(8, 6))
    X_md = states["X_md"]
    for i in range(k):
        mask = states["labels_md"] == i
        ax.scatter(X_md[mask, 0], X_md[mask, 1], s=4, alpha=0.35,
                   color=CMAP[i], rasterized=True,
                   label=f"MD C{i} ({states['pop_md'][i]:.2f})")
    for name, x_ep in states["X_ep"].items():
        ax.scatter(x_ep[0], x_ep[1], marker="*", s=350,
                   color=_EP_COLORS.get(name, "#555"), edgecolor="k", zorder=5,
                   label=f"{name} (xtal)")
    ax.set_xlabel("dist-PC1"); ax.set_ylabel("dist-PC2")
    ax.set_title(f"Conformational states (k={k}) in MD\n"
                 "○ MD  ★ crystal  (colour = cluster)")
    ax.legend(fontsize=7, ncol=2, markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_state_clusters.png"), dpi=150); plt.close()

    # 5. cluster population bar chart: MD and each AF2 setting
    src_pops = {"MD": states["pop_md"]}
    for lbl, pop in states["pop_af2"].items():
        if pop is not None:
            src_pops[f"AF2\n{lbl}"] = pop
    n_src = len(src_pops)
    src_names = list(src_pops.keys())
    x = np.arange(n_src)
    width = 0.8 / k
    fig, ax = plt.subplots(figsize=(max(6, n_src * 1.2 + 1), 4))
    for j in range(k):
        offset = (j - (k - 1) / 2) * width
        pops = [src_pops[s][j] for s in src_names]
        ax.bar(x + offset, pops, width=width, color=CMAP[j],
               edgecolor="k", linewidth=0.4, label=f"C{j}")
    ax.set_xticks(x)
    ax.set_xticklabels(src_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("population fraction")
    ax.set_title("Cluster populations by source")
    ax.legend(title="cluster", fontsize=8, ncol=min(k, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_state_populations.png"), dpi=150); plt.close()

    # per-source raw scatter + single-source population bars
    _af2_labels_all = list(CONFIG["af2_ensembles"].keys())
    AF2_PAL = _af2_palette(_af2_labels_all)
    src_panels = [
        ("MD",     states["X_md"],     states["labels_md"],     states["pop_md"],     _SRC_COLORS["aa"]),
    ]
    for lbl in _af2_labels_all:
        src_panels.append((f"AF2 {lbl}",
                           states["X_af2"].get(lbl),
                           states["labels_af2"].get(lbl),
                           states["pop_af2"].get(lbl),
                           AF2_PAL.get(lbl, "#888888")))
    _plot_source_panels(out, "fig_state", "dist-PC", k, src_panels)

    print(f"[state figures] written to {out}/")


def plot_backbone_states(out, states_bb):
    """Plots for the backbone (N/CA/C/O) state analysis.
    Mirrors plot_conformational_states with bb-prefixed filenames."""
    k = states_bb["k"]
    CMAP = plt.cm.tab10.colors[:k]

    # 1. elbow
    fig, ax = plt.subplots(figsize=(5, 4))
    ks, ine = zip(*states_bb["inertias"])
    ax.plot(ks, ine, "o-")
    ax.axvline(k, ls="--", c="red", label=f"chosen k={k}")
    ax.set_xlabel("k"); ax.set_ylabel("inertia (within-cluster SS)")
    ax.set_title("k-means elbow (backbone dist-PCA, first 3 PCs)"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_bb_state_elbow.png"), dpi=150); plt.close()

    # 2. scree
    var = states_bb["var_ratio"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(1, len(var) + 1), var * 100, color="steelblue", edgecolor="k", linewidth=0.4)
    ax.plot(range(1, len(var) + 1), np.cumsum(var) * 100, "o-", color="red", label="cumulative")
    ax.set_xlabel("bb-PC"); ax.set_ylabel("% variance explained")
    ax.set_title("Backbone distance-PCA scree"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_bb_state_scree.png"), dpi=150); plt.close()

    # 3. FES (AA-MD)
    F, xe, ye = states_bb["F_md"], *states_bb["fes_edges"]
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(xc, yc, F.T, levels=20, cmap="RdYlBu_r")
    plt.colorbar(cf, ax=ax, label="ΔF (kT)")
    for name, x_ep in states_bb["X_ep"].items():
        ax.scatter(x_ep[0], x_ep[1], marker="*", s=350,
                   color=_EP_COLORS.get(name, "#555"), edgecolor="k", zorder=5,
                   label=f"{name} (xtal)")
    ax.set_xlabel("bb-PC1"); ax.set_ylabel("bb-PC2")
    ax.set_title("AA-MD free-energy surface (backbone N/CA/C/O)"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_bb_state_fes_md.png"), dpi=150); plt.close()

    # 4. cluster-coloured scatter: MD + endpoints
    fig, ax = plt.subplots(figsize=(8, 6))
    X_md = states_bb["X_md"]
    for i in range(k):
        mask = states_bb["labels_md"] == i
        ax.scatter(X_md[mask, 0], X_md[mask, 1], s=4, alpha=0.35,
                   color=CMAP[i], rasterized=True,
                   label=f"MD C{i} ({states_bb['pop_md'][i]:.2f})")
    for name, x_ep in states_bb["X_ep"].items():
        ax.scatter(x_ep[0], x_ep[1], marker="*", s=350,
                   color=_EP_COLORS.get(name, "#555"), edgecolor="k", zorder=5,
                   label=f"{name} (xtal)")
    ax.set_xlabel("bb-PC1"); ax.set_ylabel("bb-PC2")
    ax.set_title(f"Backbone conformational states (k={k}) in MD\n"
                 "○ MD  ★ crystal  (colour = cluster)")
    ax.legend(fontsize=7, ncol=2, markerscale=2); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_bb_state_clusters.png"), dpi=150); plt.close()

    # 5. population bar chart: MD and each AF2 setting
    src_pops = {"MD": states_bb["pop_md"]}
    for lbl, pop in states_bb["pop_af2"].items():
        if pop is not None:
            src_pops[f"AF2\n{lbl}"] = pop
    n_src = len(src_pops)
    src_names = list(src_pops.keys())
    x = np.arange(n_src)
    width = 0.8 / k
    fig, ax = plt.subplots(figsize=(max(6, n_src * 1.2 + 1), 4))
    for j in range(k):
        offset = (j - (k - 1) / 2) * width
        ax.bar(x + offset, [src_pops[s][j] for s in src_names], width=width,
               color=CMAP[j], edgecolor="k", linewidth=0.4, label=f"C{j}")
    ax.set_xticks(x)
    ax.set_xticklabels(src_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("population fraction")
    ax.set_title("Backbone cluster populations by source")
    ax.legend(title="cluster", fontsize=8, ncol=min(k, 5)); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_bb_state_populations.png"), dpi=150); plt.close()

    # per-source raw scatter + single-source population bars
    _af2_labels_all = list(CONFIG["af2_ensembles"].keys())
    AF2_PAL = _af2_palette(_af2_labels_all)
    src_panels = [
        ("MD",            states_bb["X_md"],     states_bb["labels_md"],     states_bb["pop_md"],     _SRC_COLORS["aa"]),
    ]
    for lbl in _af2_labels_all:
        src_panels.append((f"AF2 {lbl}",
                           states_bb["X_af2"].get(lbl),
                           states_bb["labels_af2"].get(lbl),
                           states_bb["pop_af2"].get(lbl),
                           AF2_PAL.get(lbl, "#888888")))
    _plot_source_panels(out, "fig_bb_state", "bb-PC", k, src_panels)

    print(f"[backbone state figures] written to {out}/")


def plot_subspace(out, common, md, af2_rows, desc_bd, sproj_bd, endpoints):
    """Secondary subspace-comparison plots (RMSF, depth vs fidelity, descriptors,
    PC1/PC2 projected onto Cα-covariance subspace)."""
    resids = sorted(common)
    af2_labels = list(CONFIG["af2_ensembles"].keys())
    AF2_PAL = _af2_palette(af2_labels)

    # 1. RMSF profile
    plt.figure(figsize=(10, 4))
    plt.plot(resids, md["rmsf"], lw=2, color=_SRC_COLORS["aa"], label="AA-MD")
    plt.xlabel("residue (crystal numbering)"); plt.ylabel("Cα RMSF (Å)")
    plt.title("Per-residue flexibility"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_rmsf_profile.png"), dpi=150); plt.close()

    # 2. AF2 depth vs fidelity
    if af2_rows:
        labels = [r["depth"] for r in af2_rows]
        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(x, [r["RMSIP_pca"] for r in af2_rows], "o-", label="RMSIP (PCA)")
        ax1.plot(x, [r["RMSIP_anm"] for r in af2_rows], "s-", label="RMSIP (ANM)")
        ax1.plot(x, [r["rmsf_corr"] for r in af2_rows], "^-", label="RMSF corr")
        ax1.plot(x, [r["cum_overlap"] for r in af2_rows], "d-", label="cum. overlap")
        ax1.axhline(md["anm_baseline"], ls="--", c="grey",
                    label=f"ANM_MD ceiling ({md['anm_baseline']:.2f})")
        ax1.set_ylabel("agreement with MD subspace"); ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.plot(x, [r["mean_pLDDT"] for r in af2_rows], "x:", c="black", label="pLDDT")
        ax2.set_ylabel("mean pLDDT")
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_xlabel("MSA depth"); ax1.legend(loc="lower left", fontsize=8)
        plt.title("AF2 fidelity to MD vs MSA depth"); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_depth_vs_fidelity.png"), dpi=150); plt.close()

        plt.figure(figsize=(9, 4))
        plt.plot(x, [r["min_rmsd_inactive"] for r in af2_rows], "o-", label="→ inactive")
        plt.plot(x, [r["min_rmsd_active"]   for r in af2_rows], "s-", label="→ active")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.xlabel("MSA depth"); plt.ylabel("min Cα-RMSD to endpoint (Å)")
        plt.title("Endpoint coverage vs MSA depth"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_endpoint_coverage.png"), dpi=150); plt.close()

    # 3. Cα-subspace PC1/PC2 projection (MD, AF2 and endpoints)
    def _traj_scatter(ax, coords_2d, src, label, s=5, alpha=0.3):
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], color=_SRC_COLORS[src],
                   marker=_MK[src], s=s, alpha=alpha, rasterized=True, zorder=2, linewidths=0)
        ax.scatter([], [], color=_SRC_COLORS[src], marker=_MK[src], s=40, label=label)

    def _zoom_save(ax, ep_pts, af2_pts_list, path):
        if not ep_pts:
            return
        ref = np.array(ep_pts)
        if af2_pts_list:
            ref = np.vstack([ref, np.concatenate(af2_pts_list, axis=0)])
        pad = max(np.ptp(ref[:, 0]), np.ptp(ref[:, 1])) * 0.35 + 2.0
        ax.set_xlim(ref[:, 0].min() - pad, ref[:, 0].max() + pad)
        ax.set_ylim(ref[:, 1].min() - pad, ref[:, 1].max() + pad)
        plt.savefig(path, dpi=150)

    fig, ax = plt.subplots(figsize=(9, 7))
    pmd = subspace_project(md["pooled"], md["mean"], md["V"], 2, md["ref"])
    _traj_scatter(ax, pmd, "aa", "AA-MD  ○")
    af2_pts = []
    for label, p in sproj_bd.items():
        ax.scatter(p[:, 0], p[:, 1], marker=_MK["af2"], s=30, alpha=0.85,
                   color=AF2_PAL[label], label=f"AF2 {label}",
                   edgecolor="k", linewidth=0.4, zorder=4)
        af2_pts.append(p)
    ep_pts = []
    for name, eca in endpoints.items():
        pe = subspace_project(kabsch(eca, md["ref"])[None], md["mean"], md["V"], 2, md["ref"])[0]
        ax.scatter(pe[0], pe[1], marker=_MK["xtal"], s=300,
                   color=_EP_COLORS.get(name, "#555"), edgecolor="k",
                   label=f"{name} (xtal)", zorder=5)
        ep_pts.append(pe)
    ax.set_xlabel("MD PC1"); ax.set_ylabel("MD PC2")
    ax.set_title("All ensembles projected onto AA-MD essential plane\n"
                 "○ AA-MD  ◇ AF2  ★ crystal")
    ax.legend(fontsize=7, loc="best", ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_subspace_projection.png"), dpi=150)
    _zoom_save(ax, ep_pts, af2_pts, os.path.join(out, "fig_subspace_projection_zoom.png"))
    plt.close()

    # 4. activation-descriptor distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(axes[0], "ionic_lock",  CONFIG.get("contact_label", "TM3-TM6 ionic lock (Å)")),
                           (axes[1], "tm3_tm6_ca",  "TM3-TM6 Cα distance (Å)")]:
        md_vals = md["descriptors"][key]
        md_vals = md_vals[~np.isnan(md_vals)]
        if len(md_vals):
            ax.hist(md_vals, bins=40, density=True, alpha=0.5, label="MD")
        for lbl, d in desc_bd.items():
            vals = d[key][~np.isnan(d[key])]
            if len(vals):
                ax.hist(vals, bins=20, density=True, histtype="step", alpha=0.8, label=lbl)
        ax.set_xlabel(title); ax.set_ylabel("density"); ax.legend(fontsize=7)
    plt.suptitle("Activation-descriptor distributions"); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_descriptors.png"), dpi=150); plt.close()
    print(f"[subspace figures] written to {out}/")


def plot_icl_pca(out, args, icl_lo=202, icl_hi=225):
    """Dedicated distance-PCA + subspace PCA for the receptor's ICL3 residues.
    Loads the ICL range fresh, bypassing the global exclude_resids list."""
    cfg = CONFIG
    # ICL3 needs its own common residue set: intersect MD, the inactive reference
    # and every AF2 depth, so an endpoint with unresolved loop residues cannot
    # produce mismatched atom counts during Kabsch fitting.
    # The active endpoint is only a marker and therefore does not constrain the
    # PCA set; it is plotted only when it contains the full ICL common set.
    icl_sets = [ca_resids_present(args.md[0][0], cfg["md_resid_offset"])]
    if args.inactive:
        icl_sets.append(ca_resids_present(args.inactive, cfg["xtal_resid_offset"]))
    for gpat in cfg["af2_ensembles"].values():
        fs = sorted(glob.glob(gpat))
        if fs:
            icl_sets.append(ca_resids_present(fs[0], cfg["af2_resid_offset"]))

    icl_common = set.intersection(*icl_sets)
    icl_resids = sorted(r for r in icl_common if icl_lo <= r <= icl_hi)
    if len(icl_resids) < 3:
        print(f"\n[ICL PCA] fewer than 3 common residues in {icl_lo}-{icl_hi} -- skipping")
        return
    print(f"\n=== ICL PCA (common residues {icl_lo}-{icl_hi}, n={len(icl_resids)}) ===")

    per_replica = [traj_ca(top, xtc, icl_resids, cfg["md_resid_offset"],
                           cfg["md_skip_ns"], cfg["md_stride"])
                   for top, xtc in args.md]
    pooled_icl = np.concatenate(per_replica, axis=0)

    ref_icl = (structure_ca(args.inactive, icl_resids, cfg["xtal_resid_offset"])[0]
               if args.inactive else pooled_icl.mean(0))

    sup_icl = superpose_to(pooled_icl, ref_icl)
    k = min(10, sup_icl.shape[0] - 1)
    mean_icl, V_icl, w_icl = pca_modes(sup_icl, k)
    var_frac = w_icl / w_icl.sum() * 100
    print(f"  PC1={var_frac[0]:.1f}%  PC2={var_frac[1]:.1f}%  PC3={var_frac[2]:.1f}%")
    pmd = subspace_project(pooled_icl, mean_icl, V_icl, 2, ref_icl)

    # AF2 projections
    icl_proj_bd = {}
    for label, gpat in cfg["af2_ensembles"].items():
        coords, _ = ensemble_ca(gpat, icl_resids, cfg["af2_resid_offset"])
        if coords.shape[0] >= 2:
            icl_proj_bd[label] = subspace_project(coords, mean_icl, V_icl, 2, ref_icl)

    icl_ep = {}
    if args.inactive:
        icl_ep["inactive"], _ = structure_ca(args.inactive, icl_resids, cfg["xtal_resid_offset"])
    if args.active:
        active_present = ca_resids_present(args.active, cfg["xtal_resid_offset"])
        missing_active = sorted(set(icl_resids) - active_present)
        if missing_active:
            print(f"  [ICL PCA] active endpoint missing {len(missing_active)} of "
                  f"{len(icl_resids)} common ICL residues; active marker skipped")
        else:
            icl_ep["active"], _ = structure_ca(args.active, icl_resids, cfg["xtal_resid_offset"])

    af2_labels = list(cfg["af2_ensembles"].keys())
    AF2_PAL = _af2_palette(af2_labels)

    def _zoom_save(ax, ep_pts, af2_pts_list, path, ci=(0, 1)):
        if not ep_pts:
            return
        i, j = ci
        ref = np.array(ep_pts)[:, [i, j]]
        if af2_pts_list:
            ref = np.vstack([ref, np.concatenate(af2_pts_list, axis=0)[:, [i, j]]])
        pad = max(np.ptp(ref[:, 0]), np.ptp(ref[:, 1])) * 0.35 + 2.0
        ax.set_xlim(ref[:, 0].min() - pad, ref[:, 0].max() + pad)
        ax.set_ylim(ref[:, 1].min() - pad, ref[:, 1].max() + pad)
        plt.savefig(path, dpi=150)

    def _src_scatter(ax, x, y, src, label, s=5, alpha=0.35, zorder=2):
        ax.scatter(x, y, color=_SRC_COLORS[src], marker=_MK[src],
                   s=s, alpha=alpha, zorder=zorder, rasterized=True, linewidths=0)
        ax.scatter([], [], color=_SRC_COLORS[src], marker=_MK[src], s=40, label=label)

    subtitle = "○ AA-MD  ◇ AF2  ★ crystal"

    # PC1 vs PC2
    fig, ax = plt.subplots(figsize=(9, 6))
    _src_scatter(ax, pmd[:, 0], pmd[:, 1], "aa", "AA-MD  ○")
    af2_pts12 = []
    for label, p in icl_proj_bd.items():
        ax.scatter(p[:, 0], p[:, 1], marker=_MK["af2"], s=45, alpha=0.9,
                   color=AF2_PAL[label], label=f"AF2 {label}",
                   edgecolor="k", linewidth=0.4, zorder=4)
        af2_pts12.append(p)
    ep_pts12 = []
    for name, eca_icl in icl_ep.items():
        pe = subspace_project(kabsch(eca_icl, ref_icl)[None], mean_icl, V_icl, 2, ref_icl)[0]
        ax.scatter(pe[0], pe[1], marker=_MK["xtal"], s=320,
                   color=_EP_COLORS.get(name, "#555"), edgecolor="k",
                   label=f"{name} (xtal)", zorder=5)
        ep_pts12.append(pe)
    ax.set_xlabel(f"ICL PC1 ({var_frac[0]:.1f}%)")
    ax.set_ylabel(f"ICL PC2 ({var_frac[1]:.1f}%)")
    ax.set_title(f"ICL {icl_lo}–{icl_hi}: PC1 vs PC2\n{subtitle}")
    ax.legend(fontsize=7, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_icl_pc12.png"), dpi=150)
    _zoom_save(ax, ep_pts12, af2_pts12, os.path.join(out, "fig_icl_pc12_zoom.png"))
    plt.close()

    # PC1 vs PC3
    if V_icl.shape[1] >= 3:
        pmd3 = subspace_project(pooled_icl, mean_icl, V_icl, 3, ref_icl)
        fig, ax = plt.subplots(figsize=(9, 6))
        _src_scatter(ax, pmd3[:, 0], pmd3[:, 2], "aa", "AA-MD  ○")
        af2_pts13 = []
        for label, gpat in cfg["af2_ensembles"].items():
            coords_tmp, _ = ensemble_ca(gpat, icl_resids, cfg["af2_resid_offset"])
            if len(coords_tmp) < 2:
                continue
            p3 = subspace_project(coords_tmp, mean_icl, V_icl, 3, ref_icl)
            ax.scatter(p3[:, 0], p3[:, 2], marker=_MK["af2"], s=45, alpha=0.9,
                       color=AF2_PAL[label], label=f"AF2 {label}",
                       edgecolor="k", linewidth=0.4, zorder=4)
            af2_pts13.append(p3)
        ep_pts13 = []
        for name, eca_icl in icl_ep.items():
            pe3 = subspace_project(kabsch(eca_icl, ref_icl)[None], mean_icl, V_icl, 3, ref_icl)[0]
            ax.scatter(pe3[0], pe3[2], marker=_MK["xtal"], s=320,
                       color=_EP_COLORS.get(name, "#555"), edgecolor="k",
                       label=f"{name} (xtal)", zorder=5)
            ep_pts13.append(pe3)
        ax.set_xlabel(f"ICL PC1 ({var_frac[0]:.1f}%)")
        ax.set_ylabel(f"ICL PC3 ({var_frac[2]:.1f}%)")
        ax.set_title(f"ICL {icl_lo}–{icl_hi}: PC1 vs PC3\n{subtitle}")
        ax.legend(fontsize=7, ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_icl_pc13.png"), dpi=150)
        _zoom_save(ax, ep_pts13, af2_pts13, os.path.join(out, "fig_icl_pc13_zoom.png"), ci=(0, 2))
        plt.close()

    # PC1/PC2 time series
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for axi, col, ylabel in [(axes[0], 0, f"ICL PC1 ({var_frac[0]:.1f}%)"),
                              (axes[1], 1, f"ICL PC2 ({var_frac[1]:.1f}%)")]:
        axi.plot(np.linspace(0, 1, len(pmd)), pmd[:, col],
                 lw=0.5, alpha=0.8, color=_SRC_COLORS["aa"], label="AA-MD")
        for name, eca_icl in icl_ep.items():
            pe = subspace_project(kabsch(eca_icl, ref_icl)[None], mean_icl, V_icl, 2, ref_icl)[0]
            axi.axhline(pe[col], ls="--", lw=1.2,
                        color=_EP_COLORS.get(name, "#555"), label=f"{name} xtal")
        axi.set_ylabel(ylabel)
        axi.legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("normalised simulation time")
    plt.suptitle(f"ICL residues {icl_lo}–{icl_hi}: PC projections over time")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_icl_timeseries.png"), dpi=150); plt.close()

    # per-residue RMSF bar chart
    rmsf_icl = rmsf_per_atom(pooled_icl, ref_icl)
    sources_rmsf = [("AA-MD", rmsf_icl, "steelblue")]
    n = len(icl_resids)
    w = 0.8 / len(sources_rmsf)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), 4))
    for kk, (lbl, vals, col) in enumerate(sources_rmsf):
        off = (kk - (len(sources_rmsf) - 1) / 2) * w
        ax.bar(x + off, vals, width=w, color=col, edgecolor="k", linewidth=0.4, label=lbl)
    ax.legend(fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(icl_resids, rotation=60, ha="right", fontsize=9)
    ax.set_xlabel("residue (crystal numbering)"); ax.set_ylabel("RMSF (Å)")
    ax.set_title(f"ICL per-residue flexibility ({icl_lo}–{icl_hi})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_icl_rmsf.png"), dpi=150); plt.close()


# =============================================================================
# MAIN
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def save_cache(path, payload):
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    print(f"[cache] saved → {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

def load_cache(path):
    import pickle
    print(f"[cache] loading from {path} ...")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser(description="MD vs AF2 Cartesian analysis core")
    ap.add_argument("--md", nargs=2, action="append", metavar=("TPR", "XTC"),
                    required=True, help="MD replica (repeat for multiple replicas)")
    ap.add_argument("--inactive", help="inactive endpoint PDB")
    ap.add_argument("--active",   help="active endpoint PDB")
    ap.add_argument("--out",      default="results")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore existing cache and rerun all computation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache_path = os.path.join(args.out, "cache.pkl")

    if not args.recompute and os.path.exists(cache_path):
        payload       = load_cache(cache_path)
        common        = payload["common"]
        md            = payload["md"]
        af2_rows      = payload["af2_rows"]
        desc_bd       = payload["desc_bd"]
        sproj_bd      = payload["sproj_bd"]
        coords_bd     = payload["coords_bd"]
        endpoints     = payload["endpoints"]
        states        = payload["states"]
        states_bb     = payload["states_bb"]
        summary       = payload["summary"]
        print(f"[cache] {len(common)} residues, {md['pooled'].shape[0]} AA frames")
    else:
        af2_globs = list(CONFIG["af2_ensembles"].values())
        common = build_common_set(args.inactive, args.active, args.md[0][0], af2_globs)

        if args.inactive:
            ref, _ = structure_ca(args.inactive, common, CONFIG["xtal_resid_offset"])
            print("[ref] superposing onto inactive endpoint")
        else:
            c0  = traj_ca(args.md[0][0], args.md[0][1], common,
                          CONFIG["md_resid_offset"], CONFIG["md_skip_ns"], CONFIG["md_stride"])
            ref = c0.mean(0)
            print("[ref] no inactive endpoint -- superposing onto MD mean")

        endpoints = {}
        if args.inactive:
            endpoints["inactive"], _ = structure_ca(args.inactive, common,
                                                    CONFIG["xtal_resid_offset"])
        if args.active:
            endpoints["active"],   _ = structure_ca(args.active, common,
                                                    CONFIG["xtal_resid_offset"])

        md                            = analyze_md_reference(args, common, ref)
        af2_rows, desc_bd, sproj_bd, coords_bd = analyze_af2_depths(args, common, md, endpoints)
        states                        = analyze_conformational_states(
                                            common, md, coords_bd, endpoints)
        states_bb                     = analyze_backbone_states(args, common, endpoints)

        write_csv(os.path.join(args.out, "metrics_by_depth.csv"), af2_rows)
        summary = dict(
            n_common_residues=len(common),
            n_clusters=states["k"],
            ca_md_cluster_populations=states["pop_md"].tolist(),
            bb_md_cluster_populations=states_bb["pop_md"].tolist(),
            md_anm_baseline=md["anm_baseline"],
            md_pc1_cosine_mean=float(np.mean(md["pc1_cosine"])),
            md_inter_replica_rmsip=(float(np.mean(md["inter_replica"]))
                                    if md["inter_replica"] else None),
        )
        with open(os.path.join(args.out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        save_cache(cache_path, dict(
            common=common, md=md, af2_rows=af2_rows,
            desc_bd=desc_bd, sproj_bd=sproj_bd, coords_bd=coords_bd,
            endpoints=endpoints,
            states=states, states_bb=states_bb, summary=summary,
        ))

    plot_conformational_states(args.out, states)
    plot_backbone_states(args.out, states_bb)
    plot_subspace(args.out, common, md, af2_rows, desc_bd, sproj_bd, endpoints)
    plot_icl_pca(args.out, args)

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Results in {args.out}/")


if __name__ == "__main__":
    main()
