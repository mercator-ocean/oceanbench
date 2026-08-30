#!/bin/bash
# Aggregate one challenger's per-start gridded fill scores once the array has finished.
# Submit with --export=ALL,CHALLENGER=...,EXPECT_STARTS=... and a dependency on the array.
# Writes year.done, whose AGGREGATE_EXIT line is the only reliable completion signal.
#SBATCH --job-name=obens-fill-agg
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jseillade/ensemble-eval-fills/logs/%x-%j.log

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -uo pipefail

# Every path below comes from the environment, so any user can run the campaign from any
# tree. The #SBATCH log path cannot read one: slurm reads it before the shell does, so pass
# sbatch --output to move the logs.
WORK=${ENSEMBLE_CAMPAIGN_ROOT:?export ENSEMBLE_CAMPAIGN_ROOT, the directory holding src, output and logs}
SRC=$WORK/src
VENV=${ENSEMBLE_VENV:?export ENSEMBLE_VENV, the python environment that carries oceanbench}
PYLIBS=${ENSEMBLE_PYTHON_LIBRARIES:?export ENSEMBLE_PYTHON_LIBRARIES, the pure python s3fs stack}
OUT=$WORK/output/$CHALLENGER
LOG=$WORK/logs/$CHALLENGER

export TMPDIR=$WORK/tmp/agg-$CHALLENGER
mkdir -p "$TMPDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SRC:$PYLIBS
export PYTHONPYCACHEPREFIX=$TMPDIR/pycache

set -a
. "${ENSEMBLE_CREDENTIALS_FILE:?export ENSEMBLE_CREDENTIALS_FILE, the env file holding the object store keys}"
set +a

rm -f "$LOG/year.done"

echo "=== markers ==="
TOTAL=$(ls "$LOG"/start-*.done 2>/dev/null | wc -l)
FAILED=$(grep -L "EXIT=0" "$LOG"/start-*.done 2>/dev/null | wc -l)
echo "markers=$TOTAL failed=$FAILED"
if [ "$FAILED" -ne 0 ]; then
  echo "failed starts:"
  grep -L "EXIT=0" "$LOG"/start-*.done 2>/dev/null
fi
echo "SCORING_MARKERS=$TOTAL" >> "$LOG/year.done"
echo "SCORING_FAILED=$FAILED" >> "$LOG/year.done"

echo "=== per start score files ==="
find "$OUT" -name 'scores-*.parquet' | wc -l

cd "$SRC"
"$VENV/bin/python" -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "$OUT" \
  aggregate --expect-starts "$EXPECT_STARTS"
RC=$?
echo "AGGREGATE_EXIT=$RC" >> "$LOG/year.done"
echo "=== aggregate rc=$RC ==="
ls -la "$OUT/scores.parquet" 2>&1
date -u
rm -rf "$TMPDIR"
exit $RC
