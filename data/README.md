# data/

Only `endpoints/` is part of the repository. Everything else is created by the pipeline
or copied in by you, and is ignored by git:

| Folder | Contents | Created by |
|---|---|---|
| `endpoints/<R>/` | inactive.pdb, active.pdb (RCSB, renumbered to UniProt) | committed |
| `fasta/` | UniProt canonical sequences | `gpcr-af2md fetch` |
| `raw/gpcrmd/<record>/` | GPCRmd files under their original names | `fetch` (public files) or you (account-only replicates) |
| `prepared/<mode>/<R>/` | receptor-only topology and trajectories + provenance.json | `gpcr-af2md prepare` |
| `msas/` | one .a3m per receptor | [Zenodo](https://doi.org/10.5281/zenodo.24592084), or `gpcr-af2md af2-msa` |
| `af2/<R>/<caps>/seed_<n>/` | ColabFold ranked unrelaxed models | [Zenodo](https://doi.org/10.5281/zenodo.24592084), or the AF2 workflow |
