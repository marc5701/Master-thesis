#!/bin/bash
#SBATCH --job-name=nhanes_pax80h
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-39%40
#SBATCH --output=logs/nhanes_pax80h_%A_%a.out
#SBATCH --error=logs/nhanes_pax80h_%A_%a.err

set -u -o pipefail

PROJECT="/home/groups/mignot/mdige/projects/actigraphy-viz"
SCRIPT="${PROJECT}/code/nhanes_preprocess.py"

TAR_DIR="/oak/stanford/groups/mignot/actigraphy/NHANES/pax_h/raw_tar_bz2"
MANIFEST="${PROJECT}/manifests/nhanes_pax80h_2013_2014_tarlist.txt"

OUT_OK="/oak/stanford/groups/mignot/mdige/results/nhanes/ok"
OUT_EXCL="/oak/stanford/groups/mignot/mdige/results/nhanes/excluded"

REPORT_BASE="/oak/stanford/groups/mignot/mdige/results/nhanes/run_reports"
RUN_DIR="${REPORT_BASE}/${SLURM_ARRAY_JOB_ID}"
REPORT_DIR="${RUN_DIR}/task_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${OUT_OK}" "${OUT_EXCL}" "${REPORT_DIR}" "${RUN_DIR}"

cd "${PROJECT}"
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

echo "Manifest N=${N}  TASKS=${TASKS}  TID=${TID}"
echo "Report dir: ${REPORT_DIR}"

# Loop over global indices assigned to this task (strided)
for ((GLOBAL_IDX=TID; GLOBAL_IDX < N; GLOBAL_IDX+=TASKS)); do
  TAR_PATH="$(sed -n "$((GLOBAL_IDX+1))p" "${MANIFEST}")"
  if [[ -z "${TAR_PATH}" ]]; then
    echo "[WARN] Empty TAR_PATH at GLOBAL_IDX=${GLOBAL_IDX} (skipping)"
    continue
  fi

  SEQN="$(basename "${TAR_PATH}" .tar.bz2)"
  echo "---- GLOBAL_IDX=${GLOBAL_IDX}  SEQN=${SEQN} ----"

  # Process exactly one participant (script should skip if output already exists)
  python "${SCRIPT}" \
    --tar_dir "${TAR_DIR}" \
    --output_dir "${OUT_OK}" \
    --exclusion_dir "${OUT_EXCL}" \
    --only_seqn "${SEQN}" \
    --report_dir "${REPORT_DIR}" \
    --verbose

  RC=$?
  if [[ "${RC}" -ne 0 ]]; then
    echo "[ERROR] SEQN=${SEQN} failed with exit code ${RC} (continuing)"
  fi
done

echo "DONE task ${TID}"
