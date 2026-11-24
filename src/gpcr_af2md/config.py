"""Configuration loading and the on-disk layout of inputs and results."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

GPCRMD_FILE_URL = "https://www.gpcrmd.org/dynadb/files/Dynamics/{filename}"
GPCRMD_RECORD_URL = "https://www.gpcrmd.org/dynadb/dynamics/id/{record}/"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

REPLICA_MODES = ("public", "all")


def depth_dirname(label: str) -> str:
    """'512:5120' -> '512x5120' (the folder name ColabFold outputs are stored under)."""
    return label.replace(":", "x")


def file_sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Layout:
    """Where every input and output lives. Nothing outside these folders is written."""

    data: Path
    results: Path

    @classmethod
    def from_args(cls, data: str | Path | None, results: str | Path | None) -> "Layout":
        return cls(Path(data or REPO_ROOT / "data").resolve(),
                   Path(results or REPO_ROOT / "results").resolve())

    def raw(self, record: int) -> Path:
        return self.data / "raw" / "gpcrmd" / str(record)

    def prepared(self, receptor: str, mode: str) -> Path:
        return self.data / "prepared" / mode / receptor

    def fasta(self, receptor: str) -> Path:
        return self.data / "fasta" / f"{receptor}.fasta"

    def endpoints(self, receptor: str) -> Path:
        return self.data / "endpoints" / receptor

    def af2(self) -> Path:
        return self.data / "af2"

    def msa(self, receptor: str) -> Path:
        return self.data / "msas" / f"{receptor}.a3m"

    def cartesian(self, receptor: str, mode: str) -> Path:
        return self.results / mode / "cartesian" / receptor

    def dpca(self, receptor: str, mode: str) -> Path:
        return self.results / mode / "dpca" / receptor

    def summary(self, mode: str) -> Path:
        return self.results / mode / "summary"


@dataclass(frozen=True)
class Settings:
    receptors: dict[str, dict[str, Any]]
    analysis: dict[str, Any]
    receptors_sha256: str
    analysis_sha256: str

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> "Settings":
        root = Path(config_dir or DEFAULT_CONFIG_DIR)
        rec_path, ana_path = root / "receptors.yaml", root / "analysis.yaml"
        receptors = yaml.safe_load(rec_path.read_text())
        analysis = yaml.safe_load(ana_path.read_text())
        settings = cls(receptors, analysis, file_sha256(rec_path), file_sha256(ana_path))
        settings.validate()
        return settings

    def validate(self) -> None:
        for name, cfg in self.receptors.items():
            for key in ("uniprot", "sequence_length", "gpcrmd", "endpoints",
                        "analysis_keep_range", "icl3_range", "ionic_lock", "npxxy"):
                if key not in cfg:
                    raise ValueError(f"config/receptors.yaml: {name} lacks '{key}'")
            reps = cfg["gpcrmd"]["replicates"]
            if not reps or reps[0].get("access") != "public":
                raise ValueError(f"{name}: replicate 1 must be the public GPCRmd file")
        depths = self.analysis["af2"]["depths"]
        if not depths:
            raise ValueError("config/analysis.yaml: no AF2 depths")

    def receptor(self, name: str) -> dict[str, Any]:
        try:
            return self.receptors[name]
        except KeyError as exc:
            raise SystemExit(f"Unknown receptor {name!r}; choose from {', '.join(self.receptors)}") from exc

    def replicates(self, name: str, mode: str) -> list[dict[str, Any]]:
        """Replicates analysed in a mode: 'public' = replicate 1 only, 'all' = all three."""
        if mode not in REPLICA_MODES:
            raise ValueError(f"--md-replicas must be one of {REPLICA_MODES}")
        reps = self.receptor(name)["gpcrmd"]["replicates"]
        return reps[:1] if mode == "public" else list(reps)

    @property
    def depth_labels(self) -> list[str]:
        return list(self.analysis["af2"]["depths"])

    def core_config(self, name: str) -> dict[str, Any]:
        """Receptor entry in the shape the analysis core expects."""
        cfg = self.receptor(name)
        return {
            "sequence_length": int(cfg["sequence_length"]),
            "analysis_keep_range": list(cfg["analysis_keep_range"]),
            "icl3_range": list(cfg["icl3_range"]),
            "md_resid_offset": 0,
            "af2_resid_offset": 0,
            "xtal_resid_offset": 0,
            "frame_dt_ps": float(self.analysis["md"]["frame_dt_ps"]),
            "ionic_lock": dict(cfg["ionic_lock"]),
            "npxxy": dict(cfg["npxxy"]),
            "contact_label": cfg.get("contact_label", "TM3-TM6 ionic lock (A)"),
        }
