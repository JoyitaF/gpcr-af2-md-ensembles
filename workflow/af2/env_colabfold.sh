#!/usr/bin/env bash

module load Apptainer/1.3.2 2>/dev/null || true

export COLABFOLD_SIF="${COLABFOLD_SIF:-$HOME/apps/containers/colabfold_1.6.1-cuda12.sif}"

# Writable cache/tmp dirs on the host
export COLABFOLD_HOST_CACHE="${COLABFOLD_HOST_CACHE:-$HOME/colabfold_cache}"
export COLABFOLD_HOST_TMP="${COLABFOLD_HOST_TMP:-$HOME/colabfold_tmp}"

mkdir -p "$COLABFOLD_HOST_CACHE/colabfold"
mkdir -p "$COLABFOLD_HOST_CACHE/matplotlib"
mkdir -p "$COLABFOLD_HOST_TMP"

# Environment variables visible inside Apptainer
export APPTAINERENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export APPTAINERENV_MPLCONFIGDIR=/cache/matplotlib
export APPTAINERENV_TMPDIR=/tmp
export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_MPLCONFIGDIR=/cache/matplotlib
export SINGULARITYENV_TMPDIR=/tmp

if command -v apptainer >/dev/null 2>&1; then
  colabfold_batch() {
    apptainer exec --nv \
      -B "$HOME:$HOME" \
      -B "$PWD:$PWD" \
      -B "$COLABFOLD_HOST_CACHE:/cache" \
      -B "$COLABFOLD_HOST_TMP:/tmp" \
      --pwd "$PWD" \
      "$COLABFOLD_SIF" colabfold_batch "$@"
  }
  export -f colabfold_batch
elif command -v singularity >/dev/null 2>&1; then
  colabfold_batch() {
    singularity exec --nv \
      -B "$HOME:$HOME" \
      -B "$PWD:$PWD" \
      -B "$COLABFOLD_HOST_CACHE:/cache" \
      -B "$COLABFOLD_HOST_TMP:/tmp" \
      --pwd "$PWD" \
      "$COLABFOLD_SIF" colabfold_batch "$@"
  }
  export -f colabfold_batch
else
  echo "ERROR: no apptainer/singularity found."
  return 1 2>/dev/null || exit 1
fi
