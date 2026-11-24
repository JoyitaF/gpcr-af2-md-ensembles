import pytest

from gpcr_af2md import download
from gpcr_af2md.config import Layout, Settings


def test_html_is_rejected(tmp_path):
    bad = tmp_path / "10713_trj_73.xtc"
    bad.write_bytes(b"<!DOCTYPE html><html>login</html>")
    with pytest.raises(ValueError, match="HTML"):
        download.check_file_kind(bad)


def test_checksum_mismatch_is_reported(tmp_path):
    f = tmp_path / "x.pdb"
    f.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    with pytest.raises(ValueError, match="does not match"):
        download.verify(f, "0" * 64)


def test_account_files_are_never_downloaded(tmp_path, monkeypatch):
    """In 'all' mode, missing replicates 2-3 produce instructions, not a download attempt."""
    calls = []

    def fake_fetch(url, target, **kw):
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".xtc":
            target.write_bytes((1995).to_bytes(4, "big") + b"\0" * 100)
        elif target.suffix == ".psf":
            target.write_text("PSF CMAP\n")
        else:
            target.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n")
        return target

    monkeypatch.setattr(download, "fetch_url", fake_fetch)
    s = Settings.load()
    for entry in (s.receptors["ADORA2A"]["gpcrmd"]["topology"], s.receptors["ADORA2A"]["gpcrmd"]["model"],
                  *s.receptors["ADORA2A"]["gpcrmd"]["replicates"]):
        entry["sha256"] = None  # fake files cannot match real checksums
    layout = Layout(tmp_path / "data", tmp_path / "results")
    download.fetch_gpcrmd(s, layout, "ADORA2A", "public")
    assert len(calls) == 3 and all("/dynadb/files/Dynamics/" in u for u in calls)
    with pytest.raises(download.MissingAccountFile, match="10505_trj_49.xtc"):
        download.fetch_gpcrmd(s, layout, "ADORA2A", "all")
    assert len(calls) == 3
