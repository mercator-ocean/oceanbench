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
# Task id 0..51 selects the 52 Thursday forecast starts of 2024, 2024-01-04 to 2024-12-26.
START=$(date -u -d "2024-01-04 + $((SLURM_ARRAY_TASK_ID * 7)) days" +%Y-%m-%d)
echo "task $SLURM_ARRAY_TASK_ID start $START"
cd /scratch/jseillade/probax/gridded/src
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/jseillade/probax/gridded/pylibs:/scratch/jseillade/probax/gridded/src
/scratch/jseillade/oceanbench-venv-rebuild/bin/python -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded.py \
  --output-root /scratch/jseillade/probax/gridded/output-year \
  score --start-date "$START" --references glorys glo12 \
  --first-lead-day 1 --last-lead-day 10 --write-maps
