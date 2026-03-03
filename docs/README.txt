
This project uses **two virtual environments** stored on **SCRATCH** for performance:

- **py3.12** (main preprocessing + visualization): `/scratch/users/mdige/actigraphy-viz/.venv`
- **py3.9** (compatibility with Niels' `actigraphy_fm`): `/scratch/users/mdige/actigraphy-viz/.venv39`

⚠️ **Scratch is purgeable** (e.g., ~90 days of inactivity). If either venv disappears, follow the rebuild steps below.

---

## Project root + storage model

Project root (control center):

- `/home/groups/mignot/mdige/projects/actigraphy-viz/`

Permanent storage (OAK) is symlinked into the repo:

- `code/`, `data/`, `notebooks/`, `results/` are symlinks to:
  - `/oak/stanford/groups/mignot/mdige/{code,data,notebooks,results}`

Performance rule:

- Heavy/intermediate writes → **SCRATCH**
- Final/reproducible artifacts → **OAK** (via `results/` etc.)

---

## Module prerequisites (Sherlock / Lmod)

On Sherlock, `uv` and `python/*` are under the `devel` category. A safe pattern is:

```bash
module --force purge
module load devel
module load uv/0.9.5