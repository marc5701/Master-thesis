#!/bin/bash
#SBATCH --job-name=ukb_all
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --array=0-115%80
#SBATCH --output=logs/ukb_all_%A_%a.out
#SBATCH --error=logs/ukb_all_%A_%a.err

set -euo pipefail

cd /home/groups/mignot/mdige/projects/actigraphy-viz
source activate_py312.sh

FILE_LIST="results/ukb/ukb_cwa_gz_files.txt"
OUT_OK="results/ukb/preprocessed_h5"
OUT_EX="results/ukb/excluded_h5"

CHUNK=1000
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

# per-task fail log (prevents interleaving across tasks)
FAIL_LOG="results/ukb/failures_ukb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"

mkdir -p "$OUT_OK" "$OUT_EX" results/ukb logs

echo "[TASK] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} start=${START} end=${END} host=$(hostname)"

python code/ukb_preprocess_cwa_gz.py \
  --file_list "$FILE_LIST" \
  --output_dir "$OUT_OK" \
  --exclusion_dir "$OUT_EX" \
  --fs_out 30 \
  --start_idx "$START" --end_idx "$END" \
  --num_workers 2 \
  --fail_log "$FAIL_LOG"
