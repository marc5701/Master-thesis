#!/bin/bash
#SBATCH --job-name=montpellier_preproc
#SBATCH --output=/oak/stanford/groups/mignot/mdige/results/montpellier/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/mignot/mdige/results/montpellier/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/groups/mignot/mdige/projects/actigraphy-viz/
source activate_py312.sh
# Go to your code directory (adjust if needed)
cd /home/groups/mignot/mdige/projects/actigraphy-viz/code


python montpellier_preprocess.py \
  --data_dir /oak/stanford/groups/mignot/actigraphy/NT1 \
  --population_xlsx /oak/stanford/groups/mignot/actigraphy/NT1/0_POPULATION_CHARACTERISTICS.xlsx \
  --output_dir /oak/stanford/groups/mignot/mdige/results/montpellier/ok \
  --exclusion_dir /oak/stanford/groups/mignot/mdige/results/montpellier/excluded \
  --resample_freq 30 \
  --nonwear_patience 90m \
  --nonwear_window 10s \
  --nonwear_stdtol 0.013 \
  --missing_threshold 0.5 \
  --night_start_hour 21 \
  --night_end_hour 9 \
  --start_id 1 --end_id 38


#!/bin/bash
#SBATCH --job-name=montpellier_preproc
#SBATCH --output=/oak/stanford/groups/mignot/mdige/results/montpellier/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/mignot/mdige/results/montpellier/logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --array=1-38

set -euo pipefail
cd /home/groups/mignot/mdige/projects/actigraphy-viz/
source activate_py312.sh
cd /home/groups/mignot/mdige/projects/actigraphy-viz/code

ID="${SLURM_ARRAY_TASK_ID}"

python montpellier_preprocess.py \
  --data_dir /oak/stanford/groups/mignot/actigraphy/NT1 \
  --population_xlsx /oak/stanford/groups/mignot/actigraphy/NT1/0_POPULATION_CHARACTERISTICS.xlsx \
  --output_dir /oak/stanford/groups/mignot/mdige/results/montpellier/ok \
  --exclusion_dir /oak/stanford/groups/mignot/mdige/results/montpellier/excluded \
  --resample_freq 30 \
  --nonwear_patience 90m \
  --nonwear_window 10s \
  --nonwear_stdtol 0.013 \
  --missing_threshold 0.5 \
  --night_start_hour 21 \
  --night_end_hour 9 \
  --start_id "$ID" --end_id "$ID"
