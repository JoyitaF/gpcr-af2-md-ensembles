"""Fetch inputs: GPCRmd files by file ID, and UniProt sequences.

Public GPCRmd files (topology, model and replicate 1) are downloaded directly.
Replicates 2-3 need a GPCRmd account, so this module never downloads them: it
checks that the user has placed them in the raw folder and verifies checksums.
"""
from __future__ import annotations

import shutil
import struct
import time
import urllib.request
from pathlib import Path

from .config import (GPCRMD_FILE_URL, GPCRMD_RECORD_URL, UNIPROT_FASTA_URL, Layout,
                     Settings, file_sha256)

from . import __version__

USER_AGENT = f"gpcr-af2md/{__version__} (research reproducibility)"
XTC_MAGIC = 1995


class MissingAccountFile(RuntimeError):
    """A replicate that needs a GPCRmd account is not in the raw folder yet."""


def _looks_like_html(head: bytes) -> bool:
    low = head.lower()
    return b"<html" in low or b"<!doctype" in low


def check_file_kind(path: Path) -> None:
    """Reject login pages, error pages and truncated downloads before they reach analysis."""
    with Path(path).open("rb") as fh:
        head = fh.read(2048)
    if not head:
        raise ValueError(f"{path}: empty file")
    if _looks_like_html(head):
        raise ValueError(f"{path}: received an HTML page instead of data (login or error page?)")
    suffix = Path(path).suffix.lower()
    if suffix == ".xtc" and struct.unpack(">i", head[:4])[0] != XTC_MAGIC:
        raise ValueError(f"{path}: not an XTC trajectory")
    if suffix == ".psf" and b"PSF" not in head[:200]:
        raise ValueError(f"{path}: not a PSF topology")
    if suffix == ".pdb" and not any(tag in head for tag in (b"ATOM", b"HETATM", b"CRYST1", b"REMARK", b"HEADER")):
        raise ValueError(f"{path}: not a PDB file")


def verify(path: Path, expected_sha256: str | None) -> str:
    check_file_kind(path)
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"{path}: SHA-256 {digest} does not match the recorded {expected_sha256}. "
            "Delete the file and download it again; if it still differs, GPCRmd may have replaced it.")
    return digest


def fetch_url(url: str, target: Path, retries: int = 3, timeout: int = 120) -> Path:
    """Download to `<target>.part`, then rename, so an interrupted download never looks complete."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, length=4 * 1024 * 1024)
            part.replace(target)
            return target
        except Exception as exc:  # network errors are retried, then reported
            last_error = exc
            if part.exists():
                part.unlink()
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _gpcrmd_file(layout: Layout, record: int, entry: dict, *, allow_download: bool) -> Path:
    path = layout.raw(record) / entry["filename"]
    if path.is_file():
        verify(path, entry.get("sha256"))
        return path
    if not allow_download:
        raise MissingAccountFile(str(path))
    print(f"[download] GPCRmd file {entry['file_id']} -> {path}", flush=True)
    fetch_url(GPCRMD_FILE_URL.format(filename=entry["filename"]), path)
    verify(path, entry.get("sha256"))
    return path


def fetch_gpcrmd(settings: Settings, layout: Layout, receptor: str, mode: str) -> list[Path]:
    """Download the public files for one receptor and check any account-only replicates."""
    cfg = settings.receptor(receptor)["gpcrmd"]
    record = int(cfg["record"])
    paths = [_gpcrmd_file(layout, record, cfg["topology"], allow_download=True),
             _gpcrmd_file(layout, record, cfg["model"], allow_download=True)]
    missing = []
    for entry in settings.replicates(receptor, mode):
        try:
            paths.append(_gpcrmd_file(layout, record, entry,
                                      allow_download=entry["access"] == "public"))
        except MissingAccountFile:
            missing.append(entry)
    if missing:
        names = "\n".join(f"  - {e['filename']} (file ID {e['file_id']})" for e in missing)
        raise MissingAccountFile(
            f"{receptor}: {len(missing)} replicate(s) need a GPCRmd account and are not in "
            f"{layout.raw(record)}:\n{names}\n"
            f"Sign in at https://www.gpcrmd.org, open {GPCRMD_RECORD_URL.format(record=record)}, "
            "download these files (or request them with the GPCRmd API downloader), and place "
            "them in the folder above with their original names. Then rerun this command.\n"
            "To run with public data only, use --md-replicas public.")
    return paths


def fetch_fasta(settings: Settings, layout: Layout, receptor: str) -> Path:
    cfg = settings.receptor(receptor)
    path = layout.fasta(receptor)
    if not path.is_file():
        print(f"[download] UniProt {cfg['uniprot']} -> {path}", flush=True)
        fetch_url(UNIPROT_FASTA_URL.format(accession=cfg["uniprot"]), path)
    seq = read_fasta(path)
    if len(seq) != int(cfg["sequence_length"]):
        raise ValueError(f"{path}: length {len(seq)} != {cfg['sequence_length']} in config. "
                         "UniProt may have revised the sequence; use the FASTA committed with the paper.")
    expected = cfg.get("sequence_sha256")
    if expected and file_sha256(path) != expected:
        raise ValueError(f"{path}: checksum differs from config/receptors.yaml")
    return path


def read_fasta(path: Path) -> str:
    seq = "".join(line.strip().upper() for line in Path(path).read_text().splitlines()
                  if line and not line.startswith(">"))
    if not seq or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
        raise ValueError(f"{path}: malformed FASTA")
    return seq


def record_checksums(settings: Settings, layout: Layout) -> dict[str, dict[str, str]]:
    """SHA-256 of every GPCRmd file present locally, for pasting into receptors.yaml."""
    found: dict[str, dict[str, str]] = {}
    for name, cfg in settings.receptors.items():
        g = cfg["gpcrmd"]
        for entry in (g["topology"], g["model"], *g["replicates"]):
            path = layout.raw(int(g["record"])) / entry["filename"]
            if path.is_file():
                found.setdefault(name, {})[entry["filename"]] = file_sha256(path)
    return found
