#!/bin/bash
# glonet2-ens and glonet2-ens-icp on the gridded axis against the staged GLORYS reference.
# Task 0..51 is the base variant, 52..103 the initial-condition-perturbed variant, over the
# 52 Wednesday forecast starts of 2024. Concurrency is held at 8 because the CloudFerro
# egress cap is the binding constraint and more workers buy nothing.
#SBATCH --job-name=g2e-grid-year
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jseillade/probax/gridded/logs-g2e/task-%a.log

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

INDEX=${SLURM_ARRAY_TASK_ID}
if [ "${INDEX}" -lt 52 ]; then
  CHALLENGER=glonet2-ens
  OUTPUT=/scratch/jseillade/probax/gridded/output-year-g2e
else
  CHALLENGER=glonet2-ens-icp
  OUTPUT=/scratch/jseillade/probax/gridded/output-year-g2e-icp
  INDEX=$((INDEX - 52))
fi
START=$(date -u -d "2024-01-03 + $((INDEX * 7)) days" +%Y-%m-%d)
echo "task ${SLURM_ARRAY_TASK_ID} challenger ${CHALLENGER} start ${START} on $(hostname)"

set -a
. /scratch/jseillade/gpu2_archive/data/cloudferro-oceanbench.env
set +a

cd /scratch/jseillade/probax/gridded/src
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/jseillade/probax/gridded/pylibs:/scratch/jseillade/probax/gridded/src
/scratch/jseillade/oceanbench-venv-rebuild/bin/python -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "${OUTPUT}" \
  score --challenger "${CHALLENGER}" --start-date "${START}" --references glorys
