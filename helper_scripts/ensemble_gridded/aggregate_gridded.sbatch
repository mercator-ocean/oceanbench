#!/bin/bash
# Average the per-start score files of one output directory into scores.parquet and
# scores.csv. Chained behind its scoring array with afterany, so it runs once the array
# settles however it settles; the --expect-starts guard is what refuses to average a partial
# campaign, rather than relying on every task having succeeded.
#SBATCH --job-name=grid-aggregate
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jseillade/probax/gridded/logs-aggregate/%x-%j.log

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
OUTPUT=$1
EXPECT=$2

cd "$SRC"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYLIBS:$SRC
"$VENV/bin/python" -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "${OUTPUT}" aggregate --expect-starts "${EXPECT}"
