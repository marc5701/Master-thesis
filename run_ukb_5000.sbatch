#!/bin/bash
#SBATCH --job-name=ukb_5000
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/ukb_5000_%j.out
#SBATCH --error=logs/ukb_5000_%j.err

set -euo pipefail

cd /home/groups/mignot/mdige/projects/actigraphy-viz
source activate_py312.sh

FILE_LIST="results/ukb/ukb_cwa_gz_files.txt"
OUT_OK="results/ukb/preprocessed_h5"
OUT_EXCL="results/ukb/excluded_h5"
FAIL_LOG="results/ukb/failures.tsv"

mkdir -p logs "$OUT_OK" "$OUT_EXCL" "$(dirname "$FAIL_LOG")"

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
echo "JobID: ${SLURM_JOB_ID:-NA}"

python code/ukb_preprocess_cwa_gz.py \
  --file_list "$FILE_LIST" \
  --output_dir "$OUT_OK" \
  --exclusion_dir "$OUT_EXCL" \
  --fs_out 30 \
  --start_idx 0 \
  --end_idx 5000 \
  --num_workers 2 \
  --fail_log "$FAIL_LOG"

echo "DONE"
