# AF2 MSA-subsampling ensembles vs GPCRmd molecular dynamics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joyitaf/gpcr-af2-md-ensembles/blob/main/notebooks/gpcr_af2_md_reproducibility_test.ipynb)
[![tests](https://github.com/joyitaf/gpcr-af2-md-ensembles/actions/workflows/tests.yml/badge.svg)](https://github.com/joyitaf/gpcr-af2-md-ensembles/actions/workflows/tests.yml)

Code for *AlphaFold2 Recovers Selected Motion Features but Misses Conformational Distributions in Five GPCRs* (Faruk, Cummins and Glidden-Handgis, bioRxiv, 2026).

We predict AlphaFold2 (AF2) ensembles for five class A GPCRs (ADORA2A, ADRB2, CNR1, CHRM2, OPRD1) at eight MSA subsampling settings, 20 seeds x 5 models each, and compare them with apo molecular dynamics from [GPCRmd](https://www.gpcrmd.org). The comparison covers Cartesian motion (PCA subspace overlap, RMSF, an ANM baseline), conformational states (distance-PCA clustering, nearest-MD-frame distances, activation descriptors) and backbone torsion distributions (dihedral PCA with a Jensen-Shannon divergence calibrated against MD subsamples).

## Contents

```
config/          receptors.yaml (GPCRmd file IDs, residue ranges), analysis.yaml (all settings)
src/gpcr_af2md/  the package; _core/ holds the Cartesian analysis core (checksum-locked)
data/endpoints/  experimental inactive/active structures, renumbered to UniProt positions
notebooks/       gpcr_af2_md_reproducibility_test.ipynb: one receptor, public data only
workflow/        Slurm scripts for the AF2 ladder (ColabFold 1.6.1) and the full analysis
tests/           unit tests and an end-to-end test on synthetic data
```

## Quick start

**In the browser.** Open the [Colab reproducibility test](https://colab.research.google.com/github/joyitaf/gpcr-af2-md-ensembles/blob/main/notebooks/gpcr_af2_md_reproducibility_test.ipynb) and run all cells on a T4 GPU runtime. It runs the full pipeline on one receptor with public GPCRmd data and a small AF2 ensemble, and finishes with PASS/FAIL checks against values recorded for the paper.

**Locally** (Python 3.10 or newer, Linux or macOS):

```bash
git clone https://github.com/joyitaf/gpcr-af2-md-ensembles.git
cd gpcr-af2-md-ensembles
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest                      # about 1 minute; synthetic data, no downloads
```

## Data

### MD trajectories: GPCRmd

All MD comes from the GPCRmd community simulations: apo receptor in a POPC bilayer with TIP3P water and Na+/Cl- ions, CHARMM36m, ACEMD, 4-fs time step, frames every 0.2 ns, three replicates of 500 ns per receptor.

| Receptor | GPCRmd record | Start structure | Replicate trajectories (file IDs) | Topology / model PDB |
|---|---|---|---|---|
| ADORA2A | [49](https://www.gpcrmd.org/dynadb/dynamics/id/49/) | 5IU4 (inactive) | 10504, 10505, 10506 | 10509 / 10510 |
| ADRB2 | [11](https://www.gpcrmd.org/dynadb/dynamics/id/11/) | 2RH1 (inactive) | 10185, 10186, 10187 | 10188 / 10189 |
| CNR1 | [163](https://www.gpcrmd.org/dynadb/dynamics/id/163/) | 5U09 (inactive) | 11516, 11517, 11518 | 11521 / 11520 |
| CHRM2 | [72](https://www.gpcrmd.org/dynadb/dynamics/id/72/) | 4MQT (active) | 10705, 10706, 10707 | 10710 / 10711 |
| OPRD1 | [73](https://www.gpcrmd.org/dynadb/dynamics/id/73/) | 4N6H (inactive) | 10713, 10714, 10715 | 10718 / 10719 |

**Access.** For each record, the topology, the model PDB and the **first** replicate are listed on the public record page and download without an account. Replicates 2 and 3 need a GPCRmd account; GPCRmd requires registration with an academic email for the complete database, limits each API request to 5 simulations, and applies a weekly per-user download quota (GPCRmd changelog, version 2.3, August 2026).

**This repository does not redistribute GPCRmd files.** `gpcr-af2md fetch` downloads the public files from GPCRmd itself. It never downloads account-only files: in `--md-replicas all` mode it checks that you have placed them in `data/raw/gpcrmd/<record>/` under their original names (for example `10714_trj_73.xtc`) and verifies their SHA-256 where one is recorded in `config/receptors.yaml`. If you use these data, cite GPCRmd: Rodríguez-Espigares I, Torrens-Fontanals M, Tiemann JKS, *et al.* GPCRmd uncovers the dynamics of the 3D-GPCRome. *Nat Methods* 17, 777-787 (2020). doi:[10.1038/s41592-020-0884-y](https://doi.org/10.1038/s41592-020-0884-y).

**Two run modes.**

| `--md-replicas` | MD used | Who can run it | Reproduces the paper? |
|---|---|---|---|
| `public` (default) | replicate 1 of each record | anyone | No: one replicate, so no MD-MD overlap and different MD references |
| `all` | replicates 1-3 | anyone with a GPCRmd account | Yes |

Results are written to `results/public/` or `results/all/`, so the two modes never mix.

**Processing.** `gpcr-af2md prepare` keeps receptor atoms only, checks every residue against the canonical UniProt sequence, trims unequal replicates to the shortest (CHRM2 replicate 3 has 2,250 frames, so all CHRM2 replicates use 2,250), and re-stamps frames at 200 ps (raw GPCRmd XTC headers store 1000 x the deposited interval; coordinates are not changed). Cartesian analyses skip the first 25 ns of each replicate; the dihedral analysis uses every frame. Both settings are in `config/analysis.yaml`.

### Sequences and experimental structures

Canonical sequences are fetched from UniProt (accessions in `config/receptors.yaml`) and checked against the expected length. Inactive and active endpoint structures (RCSB PDB) are included in `data/endpoints/`, with fusion partners removed and residues renumbered to UniProt positions; each file's header names its source entry and chain.

### AF2 ensembles

Predictions used ColabFold 1.6.1 (`alphafold2_ptm`, 5 models, 3 recycles, dropout on, templates off) with `--max-msa` set to 512:5120, 256:2560, 128:1280, 64:640, 32:320, 16:160, 8:80 and 1:2, and seeds 1-20: 800 tasks and 4,000 models. The MSAs and all 4,000 models will be deposited on Zenodo with the journal version of the paper; until then they are available from the authors on request. To use them, place them under `data/msas/` and `data/af2/`. To regenerate: `gpcr-af2md af2-tasks` writes the task list and `workflow/af2/predict_array.sbatch` runs it. A new MSA from the ColabFold server can differ from ours because the server databases change, so use the paper's MSAs to reproduce its numbers.

## Reproduce the paper

```bash
# 1. Public files (topology, model PDB, replicate 1) and UniProt sequences
gpcr-af2md fetch --md-replicas all     # stops with a list of the account-only files you still need
# 2. Place replicates 2-3 in data/raw/gpcrmd/<record>/, then:
gpcr-af2md fetch --md-replicas all     # verifies every file
# 3. AF2: put the MSAs in data/msas/ and the models in data/af2/
# 4. Prepare, analyse, summarise
gpcr-af2md run --md-replicas all       # or: sbatch workflow/slurm/analysis.sbatch
```

Stages can also be run one at a time: `fetch`, `prepare`, `cartesian`, `dpca`, `states` (cluster occupancy, model-index agreement, nearest-MD distances), `bootstrap` (paired seed/MD-block resampling of the 32:320 contrasts, 500 draws), `summarize`. `torsion-audit` counts, for every torsion excluded on AF2 grounds, how many models break it at each MSA setting. Add `--receptors OPRD1` to restrict any stage. Each stage writes to a temporary folder and renames it only on success; existing results are never overwritten (delete a folder to recompute it).

## Outputs

`results/<mode>/cartesian/<receptor>/`: `metrics_by_depth.csv` (RMSIP, RMSF correlation, pLDDT, minimum RMSD to each endpoint), `summary.json` (MD-ANM and MD-MD overlap, cluster populations), figures, and `PROVENANCE.json`.

`results/<mode>/dpca/<receptor>/`: `metrics_by_depth.csv` (JSD in bits, coverage), `window_reference_comparison.json` (random-frame and 20-ns-window references for the primary and sensitivity torsion sets), `torsion_coverage.csv` (why each torsion was kept or excluded, and the first MSA setting at which an AF2 model broke it), `dpca_scores.npz`, figures and `provenance.json`.

`results/<mode>/states/<receptor>/`: `cluster_occupancy.csv`, `model_index_agreement.csv`, `nearest_md.csv` (3 and 20 distance PCs; MD queries exclude frames within 25 ns in the same replicate).

`results/<mode>/bootstrap/`: per-receptor draws, `bootstrap_joint.csv` and `bootstrap_all_schemes.csv` (seed-only, MD-only and joint schemes; 25- and 50-ns blocks), and the resampling specification. With 50-ns blocks the last 125 frames of each replicate are not resampled but stay in the observed estimate.

`results/<mode>/summary/`: one CSV per manuscript table, JSD and coverage heatmaps, and `summary.md` with the panel-level numbers quoted in the paper.

## Reproducibility

* **Same numerics as the paper.** The Cartesian analyses run `src/gpcr_af2md/_core/analysis_core.py`, the code used for the paper (comments aside); the package refuses to run if the file's SHA-256 changes. The dihedral, nearest-MD and bootstrap code reproduce the paper's values exactly on its intermediate results.
* **Provenance.** Every result folder records input checksums, the configuration hashes, the package version and git commit, and the versions of Python and the scientific libraries.
* **Determinism.** Random steps use fixed seeds (k-means `random_state`, JSD reference draws seed 2026). AF2 seeds are 1-20.
* **Environment.** `pyproject.toml` gives minimum versions, and every result folder records the exact Python and package versions used.

## Citation

Please cite the preprint (see `CITATION.cff`) and GPCRmd. The AF2 alignments and predicted models are on Zenodo ([doi:10.5281/zenodo.24592084](https://doi.org/10.5281/zenodo.24592084)). If you use the AF2 ensembles, also cite ColabFold (Mirdita *et al.*, *Nat Methods* 19, 679-682, 2022) and AlphaFold2 (Jumper *et al.*, *Nature* 596, 583-589, 2021).

## License

Code: MIT (see `LICENSE`). Experimental structures in `data/endpoints/` derive from the RCSB PDB. GPCRmd data are not included and remain subject to GPCRmd's terms.

## Contact

Joyita F., Computational Molecular Biology Group, Chulalongkorn University. Open an issue for questions or problems. 
