#!/bin/bash
#SBATCH --job-name=nhanes_g_pax80h
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-39%40
#SBATCH --output=logs/nhanes_g_pax80h_%A_%a.out
#SBATCH --error=logs/nhanes_g_pax80h_%A_%a.err

set -euo pipefail

PROJECT="/home/groups/mignot/mdige/projects/actigraphy-viz"
SCRIPT="${PROJECT}/code/nhanes_preprocess.py"

# --- NHANES 2011-2012 tarballs ---
TAR_DIR="/oak/stanford/groups/mignot/actigraphy/NHANES_2011_2012/pax_g/raw_tar_bz2"
MANIFEST="${PROJECT}/manifests/nhanes_2011_2012_pax80h_tarlist.txt"

# --- outputs (separate from 2013-2014) ---
OUT_ROOT="/oak/stanford/groups/mignot/mdige/results/nhanes_2011_2012"
OUT_OK="${OUT_ROOT}/ok"
OUT_EXCL="${OUT_ROOT}/excluded"
REPORT_DIR="${OUT_ROOT}/run_reports"

mkdir -p "${OUT_OK}" "${OUT_EXCL}" "${REPORT_DIR}"

cd "${PROJECT}"
# safer if activate_py312.sh sometimes misbehaves:
# source "${PROJECT}/.venv/bin/activate"
source activate_py312.sh

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}  ArrayJob: ${SLURM_ARRAY_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "Python: $(which python)"
python -V

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[FATAL] Manifest not found: ${MANIFEST}"
  exit 2
fi

N=$(wc -l < "${MANIFEST}")
TASKS=40
TID=${SLURM_ARRAY_TASK_ID}

# Per-task report file (avoids concurrent appends)
REPORT_CSV="${REPORT_DIR}/run_report_${SLURM_ARRAY_JOB_ID}_${TID}.csv"

echo "Manifest N=${N}  TASKS=${TASKS}  TID=${TID}"
echo "Report dir: ${REPORT_DIR}"
echo "Report file: ${REPORT_CSV}"

# Loop over global indices assigned to this array task
for ((GLOBAL_IDX=TID; GLOBAL_IDX < N; GLOBAL_IDX+=TASKS)); do
  TAR_PATH="$(sed -n "$((GLOBAL_IDX+1))p" "${MANIFEST}")"
  if [[ -z "${TAR_PATH}" ]]; then
    echo "[WARN] Empty TAR_PATH at GLOBAL_IDX=${GLOBAL_IDX} (skipping)"
    continue
  fi

  SEQN="$(basename "${TAR_PATH}" .tar.bz2)"
  echo "---- GLOBAL_IDX=${GLOBAL_IDX}  SEQN=${SEQN} ----"

  # Process one participant; script itself should skip if output exists
  python "${SCRIPT}" \
    --tar_dir "${TAR_DIR}" \
    --output_dir "${OUT_OK}" \
    --exclusion_dir "${OUT_EXCL}" \
    --only_seqn "${SEQN}" \
    --report_dir "${REPORT_DIR}" \
    --verbose || echo "[ERROR] SEQN=${SEQN} failed (continuing)"
done

echo "DONE task ${TID}"
