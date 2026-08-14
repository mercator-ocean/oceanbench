#!/bin/bash
# GloEns subsurface on the gridded axis against the staged GLORYS reference, over the 52
# Thursday forecast starts of 2024 and leads 1 to 10.
#
# The 3D stores chunk the whole 75-level column together, so one member of one day costs a
# 77 MB read whichever levels are wanted. A measured single lead of both variables took 307 s
# on its own, so a ten-lead start is about 51 minutes at full bandwidth and longer once eight
# workers share the CloudFerro egress cap; the limit below is sized from that measurement.
#SBATCH --job-name=gloens-depth-year
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jseillade/probax/gridded/logs-gloens-depth/task-%a.log

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

START=$(date -u -d "2024-01-04 + $((SLURM_ARRAY_TASK_ID * 7)) days" +%Y-%m-%d)
echo "task ${SLURM_ARRAY_TASK_ID} start ${START} on $(hostname)"

cd /scratch/jseillade/probax/gridded/src
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/jseillade/probax/gridded/pylibs:/scratch/jseillade/probax/gridded/src
/scratch/jseillade/oceanbench-venv-rebuild/bin/python -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root /scratch/jseillade/probax/gridded/output-year-gloens-depth \
  score --challenger gloens-depth --start-date "${START}" --references glorys \
  --first-lead-day 1 --last-lead-day 10
