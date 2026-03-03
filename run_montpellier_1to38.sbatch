#!/bin/bash
#SBATCH --job-name=montpellier_1to38
#SBATCH --output=logs/montpellier_1to38_%j.out
#SBATCH --error=logs/montpellier_1to38_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# --- paths ---
SCRIPT="/oak/stanford/groups/mignot/mdige/code/montpellier_preprocess.py"
DATA_DIR="/oak/stanford/groups/mignot/actigraphy/NT1"
POP_XLSX="/oak/stanford/groups/mignot/actigraphy/NT1/0_POPULATION_CHARACTERISTICS.xlsx"

OUT_OK="/oak/stanford/groups/mignot/mdige/results/montpellier/ok_v3"
OUT_EXCL="/oak/stanford/groups/mignot/mdige/results/montpellier/excluded_v3"

cd /home/groups/mignot/mdige/projects/actigraphy-viz
source .venv/bin/activate

mkdir -p logs
mkdir -p "$OUT_OK" "$OUT_EXCL"

echo "Running on host: $(hostname)"
echo "Python: $(which python)"
python -V

python "$SCRIPT" \
  --data_dir "$DATA_DIR" \
  --population_xlsx "$POP_XLSX" \
  --output_dir "$OUT_OK" \
  --exclusion_dir "$OUT_EXCL" \
  --start_id 1 --end_id 38 \
  --verbose

echo "DONE"
