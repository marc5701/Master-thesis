#!/bin/bash
#SBATCH --job-name=nhanes_pax80h
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7775%40
#SBATCH --output=logs/nhanes_pax80h_%A_%a.out
#SBATCH --error=logs/nhanes_pax80h_%A_%a.err

set -euo pipefail

# --- paths ---
PROJECT="/home/groups/mignot/mdige/projects/actigraphy-viz"
SCRIPT="${PROJECT}/code/nhanes_preprocess.py"

TAR_DIR="/oak/stanford/groups/mignot/actigraphy/NHANES/pax_h/raw_tar_bz2"
MANIFEST="${PROJECT}/manifests/nhanes_pax80h_2013_2014_tarlist.txt"

OUT_OK="/oak/stanford/groups/mignot/mdige/results/nhanes/ok"
OUT_EXCL="/oak/stanford/groups/mignot/mdige/results/nhanes/excluded"

# One shared report CSV for the whole run (append one row per SEQN)
REPORT_DIR="/oak/stanford/groups/mignot/mdige/results/nhanes"
REPORT_CSV="${REPORT_DIR}/run_report_pax80h.csv"

mkdir -p "${OUT_OK}" "${OUT_EXCL}" "${REPORT_DIR}"

# --- activate venv ---
cd /home/groups/mignot/mdige/projects/actigraphy-viz/
source activate_py312.sh
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "Python: $(which python)"
python -V

# --- get this task's tar path from manifest ---
TAR_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${MANIFEST}")"
if [[ -z "${TAR_PATH}" ]]; then
  echo "[FATAL] No tar path found for task id ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

SEQN="$(basename "${TAR_PATH}" .tar.bz2)"
echo "SEQN=${SEQN}"
echo "TAR=${TAR_PATH}"

# --- run one participant ---
python "${SCRIPT}" \
  --tar_dir "${TAR_DIR}" \
  --output_dir "${OUT_OK}" \
  --exclusion_dir "${OUT_EXCL}" \
  --only_seqn "${SEQN}" \
  --run_report_csv "${REPORT_CSV}" \
  --verbose

