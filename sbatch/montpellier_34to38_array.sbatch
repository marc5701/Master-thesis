#!/bin/bash
#SBATCH --job-name=montp_big
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=34-38
#SBATCH --output=logs/montpellier_big_%A_%a.out
#SBATCH --error=logs/montpellier_big_%A_%a.err

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

# --- paths ---
SCRIPT="/oak/stanford/groups/mignot/mdige/code/montpellier_preprocess.py"
DATA_DIR="/oak/stanford/groups/mignot/actigraphy/NT1"
POP_XLSX="/oak/stanford/groups/mignot/actigraphy/NT1/0_POPULATION_CHARACTERISTICS.xlsx"
OUT_OK="/oak/stanford/groups/mignot/mdige/results/montpellier/ok_v3"
OUT_EXCL="/oak/stanford/groups/mignot/mdige/results/montpellier/excluded_v3"

# --- activate your venv explicitly (critical in sbatch) ---
cd /home/groups/mignot/mdige/projects/actigraphy-viz
source activate_py312.sh

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
echo "Running ID=$ID"

python "$SCRIPT" \
  --data_dir "$DATA_DIR" \
  --population_xlsx "$POP_XLSX" \
  --output_dir "$OUT_OK" \
  --exclusion_dir "$OUT_EXCL" \
  --start_id "$ID" --end_id "$ID" \
  --verbose
