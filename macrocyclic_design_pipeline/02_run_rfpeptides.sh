#!/usr/bin/env bash
# 02_run_rfpeptides.sh
#
# Runs every per-target script produced by 01_build_rfpeptides_runs.py.
# Sequential by default -- for real campaign sizes (10k-80k backbones per
# target, as in the paper) you almost certainly want to submit these to a
# GPU cluster instead (SLURM array, etc.) rather than looping locally.
#
# Usage:
#   ./02_run_rfpeptides.sh <rfpeptides_runs_dir>

set -euo pipefail

RUNS_DIR="${1:?Usage: 02_run_rfpeptides.sh <rfpeptides_runs_dir>}"

for script in "$RUNS_DIR"/run_*.sh; do
    echo "=== Running $script ==="
    bash "$script"
done

echo "All RFpeptides backbone-generation jobs finished."
echo "Each design's backbones are under $RUNS_DIR/<design_name>/"
