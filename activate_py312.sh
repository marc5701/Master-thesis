#!/usr/bin/env bash
set -eo pipefail

# If already in another venv, deactivate it
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi

module --force purge
module load devel
module load uv/0.9.5
module load python/3.12.1
module load java/17.0.4

source /home/groups/mignot/mdige/projects/actigraphy-viz/.venv/bin/activate

echo "uv: $(which uv) ($(uv --version))"
echo "python: $(which python) ($(python -V))"
