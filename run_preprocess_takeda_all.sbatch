#!/bin/bash
#SBATCH --job-name=takeda_preproc
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/takeda_preproc_%j.out
#SBATCH --error=logs/takeda_preproc_%j.err

set -euo pipefail

cd /home/groups/mignot/mdige/projects/actigraphy-viz
mkdir -p logs
mkdir -p results/takeda/preprocessed_h5 results/takeda/excluded_h5

# If you normally load modules on Sherlock, uncomment and adjust:
# module purge
# module load python/3.12

# Activate your venv (adjust if your venv path differs)
source .venv/bin/activate

python code/preprocess_takeda.py \
  --input_dir /oak/stanford/groups/mignot/actigraphy/TAK-994-0001 \
  --lights_csv /oak/stanford/groups/mignot/actigraphy/TAK-994-0001/lights_0001.csv \
  --alignment_csv /oak/stanford/groups/mignot/actigraphy/TAK-994-0001/TAK-994-0001_alignment_tab.csv \
  --output_dir results/takeda/preprocessed_h5 \
  --exclusion_dir results/takeda/excluded_h5 \
  --fs_out 30 \
  --start 0 \
  --end -1
