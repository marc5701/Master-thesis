#!/usr/bin/env bash
set -eo pipefail

# If already in another venv, deactivate it
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi

module --force purge
module load devel
module load uv/0.9.5
module load python/3.9.0

source /home/groups/mignot/mdige/projects/actigraphy-viz/.venv39/bin/activate

echo "uv: $(which uv) ($(uv --version))"
echo "python: $(which python) ($(python -V))"
