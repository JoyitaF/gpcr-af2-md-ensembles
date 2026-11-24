#!/usr/bin/env bash
# One AF2 task (receptor x MSA caps x seed) per array element, as run for the paper.
# Edit the partition/QoS/GPU lines for your cluster.
#
#   gpcr-af2md af2-tasks --md-replicas all --out workflow/af2/tasks.tsv   # 800 tasks for 5 receptors
#   sbatch --array=1-800%20 workflow/af2/predict_array.sbatch
#
#SBATCH --job-name=gpcr_af2
#SBATCH --output=logs/gpcr_af2.%A_%a.out
#SBATCH --error=logs/gpcr_af2.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source workflow/af2/env_colabfold.sh          # ColabFold 1.6.1 (Apptainer image)

TASKS_FILE="${TASKS_FILE:-workflow/af2/tasks.tsv}"
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")"
[[ -n "$LINE" ]] || { echo "No task line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
IFS=$'\t' read -r RID DEPTH_ID MAX_SEQ MAX_EXTRA SEED <<< "$LINE"

MSA_FILE="data/msas/${RID}.a3m"
OUTDIR="data/af2/${RID}/${DEPTH_ID}/seed_${SEED}"
[[ -s "$MSA_FILE" ]] || { echo "Missing MSA: $MSA_FILE" >&2; exit 1; }
if [[ -f "$OUTDIR/DONE" ]]; then echo "Already done: $OUTDIR"; exit 0; fi
mkdir -p "$OUTDIR"

colabfold_batch "$MSA_FILE" "$OUTDIR" \
  --model-type alphafold2_ptm \
  --num-models 5 \
  --num-recycle 3 \
  --max-msa "${MAX_SEQ}:${MAX_EXTRA}" \
  --random-seed "$SEED" \
  --use-dropout

touch "$OUTDIR/DONE"
echo "DONE: $RID $DEPTH_ID seed $SEED"
