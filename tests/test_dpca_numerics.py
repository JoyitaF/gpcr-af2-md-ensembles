import numpy as np
import pytest

from gpcr_af2md import dpca


def test_dihedral_of_known_geometry():
    trans = np.array([[[1.0, 1, 0], [0, 0, 0], [1, 0, 0], [0, -1, 0]]])
    cis = np.array([[[0.0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]]])
    assert np.isclose(abs(dpca.dihedral_angles(trans)[0]), np.pi)
    assert np.isclose(dpca.dihedral_angles(cis)[0], 0.0)


def test_identical_distributions_have_zero_jsd():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 2))
    bins = tuple(np.quantile(x[:, a], np.arange(1, 8) / 8) for a in range(2))
    h = dpca.quantile_hist(x, bins)
    assert h.shape == (64,) and np.isclose(h.sum(), 1.0)
    m = dpca.score_metrics(x, x, bins, threshold=10.0)
    assert m["js_divergence_8x8_bits"] == pytest.approx(0.0, abs=1e-12)


def test_yaml_cannot_silently_change_hard_coded_numerics():
    cfg = dict(dpca.HARD_CODED, window_frames=100, random_draws=1000, seed=2026)
    dpca.check_settings(cfg)
    with pytest.raises(ValueError, match="pseudocount"):
        dpca.check_settings({**cfg, "pseudocount": 0.5})
