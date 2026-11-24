import re

from gpcr_af2md.cartesian import CORE_DIR, CORE_SHA256
from gpcr_af2md.config import Settings, file_sha256


def test_repository_config_is_consistent():
    s = Settings.load()
    assert set(s.receptors) == {"ADORA2A", "ADRB2", "CNR1", "CHRM2", "OPRD1"}
    for name, cfg in s.receptors.items():
        g = cfg["gpcrmd"]
        record = g["record"]
        files = [g["topology"], g["model"], *g["replicates"]]
        for entry in files:
            assert re.fullmatch(rf"{entry['file_id']}_(dyn|trj)_{record}\.(psf|pdb|xtc)", entry["filename"]), entry
        assert [r["access"] for r in g["replicates"]] == ["public", "account", "account"], name
        lo, hi = cfg["analysis_keep_range"]
        assert 1 <= lo < hi <= cfg["sequence_length"]
        assert lo <= cfg["ionic_lock"]["res_pos"] <= hi and lo <= cfg["npxxy"]["resid"] <= hi


def test_replicate_modes():
    s = Settings.load()
    assert len(s.replicates("OPRD1", "public")) == 1
    assert len(s.replicates("OPRD1", "all")) == 3


def test_analysis_core_is_the_checked_version():
    for name, expected in CORE_SHA256.items():
        assert file_sha256(CORE_DIR / name) == expected
