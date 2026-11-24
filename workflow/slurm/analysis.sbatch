#!/usr/bin/env bash
# Analysis for all five receptors with all three GPCRmd replicates (paper setting).
# Inputs: data/raw/gpcrmd/<record>/ (all files), data/af2/ (800 tasks), data/fasta/.
#SBATCH --job-name=gpcr_af2md
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gpcr_af2md.%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
       MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
gpcr-af2md run --md-replicas all
