"""End-to-end run on synthetic data: prepare -> cartesian -> dpca -> summarize, both replicate modes."""
import json

import pandas as pd
import pytest

from gpcr_af2md.cli import main


def _run(info, command, mode, *extra):
    return main([command, "--config", str(info["config"]), "--data", str(info["data"]),
                 "--results", str(info["results"]), "--md-replicas", mode, *extra])


@pytest.mark.parametrize("mode", ["all", "public"])
def test_full_pipeline(synthetic, mode):
    info = synthetic
    subset = ["--depths", *info["labels"], "--af2-models-per-depth", str(info["models_per_depth"])]
    assert _run(info, "prepare", mode) == 0
    prov = json.loads((info["data"] / "prepared" / mode / "TOY1" / "provenance.json").read_text())
    n_rep = 3 if mode == "all" else 1
    shortest = min(info["frames"][:n_rep])
    assert len(prov["replicas"]) == n_rep
    assert all(r["retained_frames"] == shortest for r in prov["replicas"])

    assert _run(info, "cartesian", mode, *subset) == 0
    cart = info["results"] / mode / "cartesian" / "TOY1"
    summary = json.loads((cart / "summary.json").read_text())
    pops = summary["ca_md_cluster_populations"]
    skip_frames = 125  # 25 ns at 200 ps/frame
    assert round(sum(pops) * n_rep * (shortest - skip_frames)) == n_rep * (shortest - skip_frames)
    assert (summary["md_inter_replica_rmsip"] is None) == (mode == "public")
    assert json.loads((cart / "PROVENANCE.json").read_text())["skip_ns"] == 25.0

    assert _run(info, "dpca", mode) == 0
    d = info["results"] / mode / "dpca" / "TOY1"
    cov = pd.read_csv(d / "torsion_coverage.csv")
    bad = cov[(cov.canonical_residue == info["bad_feature"][0]) & (cov.torsion == info["bad_feature"][1])]
    assert bad.status.item() == "AF2_peptide_geometry" and bad.first_failing_msa_setting.item() == "1:2"
    metrics = pd.read_csv(d / "metrics_by_depth.csv")
    assert list(metrics.msa_depth) == info["labels"]
    assert metrics.js_divergence_8x8_bits.between(0, 1).all()

    assert _run(info, "states", mode) == 0
    st = info["results"] / mode / "states" / "TOY1"
    near = pd.read_csv(st / "nearest_md.csv")
    assert set(near.n_pc) == {3, 20} and near.af2_above_p95_pct.between(0, 100).all()
    assert len(pd.read_csv(st / "cluster_occupancy.csv")) == len(info["labels"])

    assert _run(info, "torsion-audit", mode) == 0
    audit = pd.read_csv(d / "af2_exclusions_by_setting.csv")
    assert audit["fails_only_at_1:2"].all() and (audit["1:2"] == 1).all()

    if mode == "all":  # the bootstrap resamples MD blocks within each of several replicates
        assert _run(info, "bootstrap", mode) == 0
        joint = pd.read_csv(info["results"] / mode / "bootstrap" / "bootstrap_joint.csv")
        assert len(joint) == 4 and (joint.ci_2p5 <= joint.ci_97p5).all()

    assert _run(info, "summarize", mode) == 0
    summary_dir = info["results"] / mode / "summary"
    assert (summary_dir / "jsd_reference_table.csv").is_file()
    assert (summary_dir / "nearest_md_table.csv").is_file()


def test_existing_results_are_not_overwritten(synthetic):
    info = synthetic
    cart = info["results"] / "all" / "cartesian" / "TOY1"
    if not cart.exists():
        pytest.skip("pipeline test did not run")
    before = (cart / "metrics_by_depth.csv").read_text()
    _run(info, "cartesian", "all", "--depths", *info["labels"], "--af2-models-per-depth", str(info["models_per_depth"]))
    assert (cart / "metrics_by_depth.csv").read_text() == before
