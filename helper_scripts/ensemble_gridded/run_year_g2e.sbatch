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

# Every path below comes from the environment, so any user can run the campaign from any
# tree. The #SBATCH log path cannot read one: slurm reads it before the shell does, so pass
# sbatch --output to move the logs.
SRC=${ENSEMBLE_CAMPAIGN_ROOT:?export ENSEMBLE_CAMPAIGN_ROOT, the directory holding src, output and logs}/src
VENV=${ENSEMBLE_VENV:?export ENSEMBLE_VENV, the python environment that carries oceanbench}
PYLIBS=${ENSEMBLE_PYTHON_LIBRARIES:?export ENSEMBLE_PYTHON_LIBRARIES, the pure python s3fs stack}
OUTPUT_ROOT=${ENSEMBLE_CAMPAIGN_ROOT}

INDEX=${SLURM_ARRAY_TASK_ID}
if [ "${INDEX}" -lt 52 ]; then
  CHALLENGER=glonet2-ens
  OUTPUT=$OUTPUT_ROOT/output-year-g2e
else
  CHALLENGER=glonet2-ens-icp
  OUTPUT=$OUTPUT_ROOT/output-year-g2e-icp
  INDEX=$((INDEX - 52))
fi
START=$(date -u -d "2024-01-03 + $((INDEX * 7)) days" +%Y-%m-%d)
echo "task ${SLURM_ARRAY_TASK_ID} challenger ${CHALLENGER} start ${START} on $(hostname)"

set -a
. "${ENSEMBLE_CREDENTIALS_FILE:?export ENSEMBLE_CREDENTIALS_FILE, the env file holding the object store keys}"
set +a

cd "$SRC"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYLIBS:$SRC
"$VENV/bin/python" -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "${OUTPUT}" \
  score --challenger "${CHALLENGER}" --start-date "${START}" --references glorys
