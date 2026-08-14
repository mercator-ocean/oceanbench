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

OUTPUT=$1
EXPECT=$2

cd /scratch/jseillade/probax/gridded/src
export PYTHONNOUSERSITE=1
export PYTHONPATH=/scratch/jseillade/probax/gridded/pylibs:/scratch/jseillade/probax/gridded/src
/scratch/jseillade/oceanbench-venv-rebuild/bin/python -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "${OUTPUT}" aggregate --expect-starts "${EXPECT}"
