#!/usr/bin/env bash
set -euo pipefail

module load Apptainer/1.3.2 2>/dev/null || true

mkdir -p "$HOME/apps/containers"

SIF="${COLABFOLD_SIF:-$HOME/apps/containers/colabfold_1.6.1-cuda12.sif}"
IMAGE="docker://ghcr.io/sokrypton/colabfold:1.6.1-cuda12"

if [ -s "$SIF" ]; then
  echo "Container already exists: $SIF"
  exit 0
fi

if command -v apptainer >/dev/null 2>&1; then
  apptainer pull "$SIF" "$IMAGE"
elif command -v singularity >/dev/null 2>&1; then
  singularity pull "$SIF" "$IMAGE"
else
  echo "ERROR: neither apptainer nor singularity found."
  exit 1
fi

echo "Pulled: $SIF"
