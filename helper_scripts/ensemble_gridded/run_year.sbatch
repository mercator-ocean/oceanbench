#!/bin/bash
#SBATCH --job-name=gloens-grid-year
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jseillade/probax/gridded/logs-year/start-%a.log

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

# Every path below comes from the environment, so any user can run the campaign from any
# tree. The #SBATCH log path cannot read one: slurm reads it before the shell does, so pass
# sbatch --output to move the logs.
SRC=${ENSEMBLE_CAMPAIGN_ROOT:?export ENSEMBLE_CAMPAIGN_ROOT, the directory holding src, output and logs}/src
VENV=${ENSEMBLE_VENV:?export ENSEMBLE_VENV, the python environment that carries oceanbench}
PYLIBS=${ENSEMBLE_PYTHON_LIBRARIES:?export ENSEMBLE_PYTHON_LIBRARIES, the pure python s3fs stack}
OUTPUT_ROOT=${ENSEMBLE_CAMPAIGN_ROOT}
# Task id 0..51 selects the 52 Thursday forecast starts of 2024, 2024-01-04 to 2024-12-26.
START=$(date -u -d "2024-01-04 + $((SLURM_ARRAY_TASK_ID * 7)) days" +%Y-%m-%d)
echo "task $SLURM_ARRAY_TASK_ID start $START"
cd "$SRC"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYLIBS:$SRC
"$VENV/bin/python" -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded.py \
  --output-root "$OUTPUT_ROOT/output-year" \
  score --start-date "$START" --references glorys glo12 \
  --first-lead-day 1 --last-lead-day 10 --write-maps
