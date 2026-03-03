# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**actigraphy-viz** preprocesses and visualizes raw accelerometer (actigraphy) data from multiple clinical datasets. It runs on Stanford's Sherlock HPC cluster using SLURM array jobs for parallel processing. Output is standardized HDF5 files resampled to 30 Hz.

### Datasets

| Dataset | Script | Raw format | SBATCH examples |
|---|---|---|---|
| Montpellier NT1 | `code/montpellier_preprocess.py` | CWA (via openmovement) | `sbatch/montpellier_34to38_array.sbatch` |
| NHANES (2011-2014) | `code/nhanes_preprocess.py` | tar.bz2 archives | `sbatch/nhanes_pax80h_strided_array.sbatch` |
| UK Biobank | `code/ukb_preprocess_cwa_gz.py` | CWA.GZ | `sbatch/run_ukb_all_array.sbatch` |
| Takeda | `code/preprocess_takeda.py` | EDF files | `run_preprocess_takeda_all.sbatch` |

## Storage Model

- **Project root:** `/home/groups/mignot/mdige/projects/actigraphy-viz/`
- **OAK (permanent):** `code/`, `data/`, `notebooks/`, `results/` are symlinks into `/oak/stanford/groups/mignot/mdige/`
- **SCRATCH (ephemeral, ~90-day purge):** Virtual environments and temporary intermediates live on `/scratch/users/mdige/actigraphy-viz/`
- **Rule:** Heavy/intermediate writes go to SCRATCH; final artifacts go to OAK via `results/`

## Environment Setup

Two Python virtual environments, both managed with `uv`:

```bash
# Primary (Python 3.12) - preprocessing & visualization
source activate_py312.sh

# Legacy (Python 3.9) - compatibility with actigraphy_fm
source activate_py39.sh
```

These scripts handle `module purge`, loading `devel`, `uv/0.9.5`, the correct Python, and activating the venv. Always use them instead of manual module/venv activation.

To rebuild a purged venv:
```bash
source activate_py312.sh   # loads modules
uv pip install -r requirements_py312_freeze.txt
```

## Running Jobs

All preprocessing is submitted via SLURM. The pattern is:

```bash
cd /home/groups/mignot/mdige/projects/actigraphy-viz
sbatch sbatch/<script>.sbatch
```

### Parallelization patterns

- **Chunked (UKB):** Array task N processes files `[N*1000, (N+1)*1000)` from a file list
- **Strided (NHANES):** 40 array tasks; task T processes manifest indices `[T, T+40, T+80, ...]` to avoid write conflicts on shared output dirs
- **Per-subject (Montpellier):** `--array=1-38`, each task processes one subject ID

### Common script flags

All preprocessing scripts accept: `--output_dir`, `--exclusion_dir`, `--fs_out 30`, `--verbose`. NHANES adds `--tar_dir`, `--only_seqn`, `--report_dir`. UKB adds `--file_list`, `--start_idx`, `--end_idx`, `--num_workers`, `--fail_log`. Montpellier adds `--data_dir`, `--population_xlsx`, `--start_id`, `--end_id`.

## Key Files

- `manifests/` - File lists used by SLURM array jobs to distribute work (NHANES tar paths)
- `sbatch/` - Production SLURM array job scripts
- `logs/` - SLURM stdout/stderr (pattern: `<dataset>_<jobid>_<taskid>.out/.err`)
- `backup_to_oak.sh` - rsync project to OAK (excludes venvs, data, caches)
- `run_claude_container.sh` - Launches Claude Code in a Singularity container with appropriate bind mounts

## Claude Code Container Context

When running inside the Singularity container (`run_claude_container.sh`):
- `/repo` is the project root (read-only)
- `/workspace` is a persistent writable workspace on OAK
- `/tmpwork` and `/cache` are writable scratch directories
- `/oak_mignot` provides read-only access to group data on OAK
- The `code/`, `data/`, `notebooks/`, `results/` symlinks will not resolve inside the container; use `/oak_mignot/mdige/` paths instead

## Dependencies

Defined in `pyproject.toml`, locked in `uv.lock`. Key libraries: numpy, pandas, scipy, matplotlib, plotly, seaborn, h5py, pyarrow, edfio, openmovement. No test framework is configured.
