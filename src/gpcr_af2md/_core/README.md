# Analysis core

`analysis_core.py` holds the Cartesian analyses (PCA/RMSIP, RMSF, ANM, distance-PCA
states, activation descriptors, ICL3 PCA). `gpcr_af2md.cartesian` sets its
configuration for each receptor and checks the file's SHA-256 before every run, so an
unreviewed edit cannot change the results silently. To change the numerics, edit the
file, update `CORE_SHA256` in `cartesian.py`, and record the change in CHANGELOG.md.

The code is the version used for the paper's analyses; this copy differs from it only in
comments and docstrings (identical syntax tree otherwise).
