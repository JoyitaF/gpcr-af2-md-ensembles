"""Record the software environment of each run."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from . import __version__
from .config import REPO_ROOT

PACKAGES = ("numpy", "scipy", "scikit-learn", "MDAnalysis", "ProDy", "matplotlib", "biopython", "PyYAML")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode:
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def environment_record() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "gpcr_af2md": __version__,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
