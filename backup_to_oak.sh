#!/usr/bin/env bash
set -euo pipefail

SRC="/home/groups/mignot/mdige/projects/actigraphy-viz/"
DST="/oak/stanford/groups/mignot/projects/marcus_actigraphy_viz/"

rsync -avhL --delete \
  --exclude '.venv' \
  --exclude '.venv_*' \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'data/' \
  --exclude '*.cwa' \
  --exclude '*.cwa.gz' \
  "$SRC" "$DST"

echo "Backup complete -> $DST"
