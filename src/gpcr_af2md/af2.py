"""AlphaFold2 MSA-subsampling ladder with ColabFold.

The paper's ensembles were run on a GPU cluster with the ColabFold 1.6.1 container
(`workflow/af2/`). This module writes that task list and can run a subset of tasks
with a local `colabfold_batch` (used by the Colab reproducibility test).

Command per task (identical on the cluster):
    colabfold_batch <MSA.a3m> <out> --model-type alphafold2_ptm --num-models 5
        --num-recycle 3 --max-msa <max_seq>:<max_extra_seq> --random-seed <seed> --use-dropout
Templates are off (colabfold_batch default).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import Layout, Settings, depth_dirname


def task_rows(settings: Settings, receptors: list[str]) -> list[tuple]:
    af = settings.analysis["af2"]
    return [(r, depth_dirname(label), caps[0], caps[1], seed)
            for r in receptors for label, caps in af["depths"].items() for seed in af["seeds"]]


def write_tasks(settings: Settings, receptors: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("\t".join(map(str, row)) + "\n" for row in task_rows(settings, receptors)))
    print(f"{path}: {len(task_rows(settings, receptors))} tasks", flush=True)
    return path


def _colabfold() -> str:
    exe = shutil.which("colabfold_batch")
    if not exe:
        raise SystemExit("colabfold_batch not found. Install ColabFold (see notebooks/ or workflow/af2/).")
    return exe


def make_msa(settings: Settings, layout: Layout, receptor: str) -> Path:
    """Query the ColabFold MMseqs2 server once per receptor. Server databases change over
    time, so a new MSA can differ from the paper's; use the paper's .a3m to match its results."""
    target = layout.msa(receptor)
    if target.is_file():
        return target
    fasta = layout.fasta(receptor)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([_colabfold(), str(fasta), tmp, "--msa-only"], check=True)
        found = sorted(Path(tmp).glob("*.a3m"))
        if len(found) != 1:
            raise RuntimeError(f"Expected one .a3m from ColabFold, found {found}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], target)
    return target


def predict(settings: Settings, layout: Layout, receptor: str, labels: list[str], seeds: list[int]) -> None:
    """Run missing tasks; a finished task (DONE + 5 ranked PDBs) is never redone."""
    af = settings.analysis["af2"]
    msa = layout.msa(receptor)
    if not msa.is_file():
        raise FileNotFoundError(f"{msa}: run `gpcr-af2md af2-msa` or copy in the paper's MSA")
    for label in labels:
        max_seq, max_extra = af["depths"][label]
        for seed in seeds:
            dest = layout.af2() / receptor / depth_dirname(label) / f"seed_{seed}"
            done = dest / "DONE"
            if done.is_file() and len(list(dest.glob("*_unrelaxed_rank_*.pdb"))) == af["num_models"]:
                continue
            if dest.exists():
                raise FileExistsError(f"{dest} is incomplete; delete it to rerun this task")
            with tempfile.TemporaryDirectory() as tmp:
                cmd = [_colabfold(), str(msa), tmp, "--model-type", af["model_type"],
                       "--num-models", str(af["num_models"]), "--num-recycle", str(af["num_recycle"]),
                       "--max-msa", f"{max_seq}:{max_extra}", "--random-seed", str(seed)]
                if af["use_dropout"]:
                    cmd.append("--use-dropout")
                print(" ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
                models = sorted(Path(tmp).glob("*_unrelaxed_rank_*.pdb"))
                if len(models) != af["num_models"]:
                    raise RuntimeError(f"ColabFold returned {len(models)} models, expected {af['num_models']}")
                shutil.copytree(tmp, dest)
            done.touch()
