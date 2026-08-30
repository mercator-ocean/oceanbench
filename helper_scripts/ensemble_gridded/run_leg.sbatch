#!/bin/bash
# One leg of a gridded campaign as a single job holding a whole node, running its forecast
# starts concurrently inside rather than as an array of one job per start. The queue limits the
# jobs a user may run at once, not the cores, so an array of 52 one-start jobs runs four at a
# time while one exclusive node runs as many as the object store will feed.
#
# Concurrency is set by what the challenger store can serve, not by the node: CloudFerro
# saturates near eight readers, and GLORYS is read from the local stage and costs it nothing.
#
# The queue does cap the cores of a single job, at 64, so this asks for a share of a node
# rather than the whole one. Eight starts of four cores fit in half that: a start spends its
# time waiting on the network, four cores of one measured at 22 minutes of processor time
# over 54 minutes of wall clock, so cores are not what it is short of.
#
# Submit with --export=ALL,CHALLENGER=...,FIRST_START=...  and the paths below in the
# environment. Every start keeps its own done marker, so a resubmission skips what already
# succeeded, including starts an earlier array had finished.
#SBATCH --job-name=obens-ser-leg
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -uo pipefail

# Every path below comes from the environment, so any user can run the campaign from any
# tree. The #SBATCH log path cannot read one: slurm reads it before the shell does, so pass
# sbatch --output to move the logs.
export WORK=${ENSEMBLE_CAMPAIGN_ROOT:?export ENSEMBLE_CAMPAIGN_ROOT, the directory holding src, output and logs}
export SRC=$WORK/src
export VENV=${ENSEMBLE_VENV:?export ENSEMBLE_VENV, the python environment that carries oceanbench}
export PYLIBS=${ENSEMBLE_PYTHON_LIBRARIES:?export ENSEMBLE_PYTHON_LIBRARIES, the pure python s3fs stack}
export CHALLENGER=${CHALLENGER:?export CHALLENGER, the entry of the scorer to run}
export FIRST_START=${FIRST_START:?export FIRST_START, the first forecast start of the year}
export SCORER=${SCORER:-score_ensemble_gridded_multi.py}
export START_COUNT=${START_COUNT:-52}
export CONCURRENCY=${CONCURRENCY:-8}
export OUT=$WORK/output/$CHALLENGER
export LOG=$WORK/logs/$CHALLENGER
mkdir -p "$OUT" "$LOG" "$WORK/tmp"

set -a
. "${ENSEMBLE_CREDENTIALS_FILE:?export ENSEMBLE_CREDENTIALS_FILE, the env file holding the object store keys}"
set +a

export PYTHONNOUSERSITE=1
export PYTHONPATH=$SRC:$PYLIBS
# Four threads per start, the same as the array tasks this replaces, so the cores asked for
# cover the concurrency exactly.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "=== context ==="
date -u
hostname
cat "$WORK/VERSION"
echo "challenger=$CHALLENGER scorer=$SCORER first_start=$FIRST_START starts=$START_COUNT concurrency=$CONCURRENCY job=$SLURM_JOB_ID"
"$VENV/bin/python" -c "import oceanbench; print('IMPORTED_FROM=' + oceanbench.__file__)"

score_one_start() {
  local index=$1
  local start compact marker task_log started elapsed status
  start=$(date -u -d "$FIRST_START + $((index * 7)) days" +%Y-%m-%d)
  compact=${start//-/}
  marker=$LOG/start-$compact.done
  task_log=$LOG/start-$compact.log
  if [ -f "$marker" ] && grep -q "EXIT=0" "$marker"; then
    echo "skip $start, already scored"
    return 0
  fi
  rm -f "$marker"
  export TMPDIR=$WORK/tmp/$CHALLENGER-$compact
  export PYTHONPYCACHEPREFIX=$TMPDIR/pycache
  mkdir -p "$TMPDIR"
  started=$(date +%s)
  if [ "$SCORER" = "score_gloens_velocity_gridded.py" ]; then
    (cd "$SRC" && "$VENV/bin/python" -W ignore "helper_scripts/ensemble_gridded/$SCORER" \
      --output-root "$OUT" score --start-date "$start" --references glorys --last-lead-day 10) \
      > "$task_log" 2>&1
  else
    (cd "$SRC" && "$VENV/bin/python" -W ignore "helper_scripts/ensemble_gridded/$SCORER" \
      --output-root "$OUT" score --challenger "$CHALLENGER" --start-date "$start" --references glorys) \
      > "$task_log" 2>&1
  fi
  status=$?
  elapsed=$(($(date +%s) - started))
  echo "EXIT=$status SECONDS=$elapsed" > "$marker"
  echo "$start rc=$status seconds=$elapsed"
  rm -rf "$TMPDIR"
  return 0
}
export -f score_one_start

echo "=== scoring ==="
date -u
seq 0 $((START_COUNT - 1)) | xargs -P "$CONCURRENCY" -I{} bash -c 'score_one_start {}'
echo "=== scoring done ==="
date -u

echo "=== markers ==="
rm -f "$LOG/year.done"
TOTAL=$(ls "$LOG"/start-*.done 2>/dev/null | wc -l)
FAILED=$(grep -L "EXIT=0" "$LOG"/start-*.done 2>/dev/null | wc -l)
echo "markers=$TOTAL failed=$FAILED"
if [ "$FAILED" -ne 0 ]; then
  echo "failed starts:"
  grep -L "EXIT=0" "$LOG"/start-*.done 2>/dev/null
fi
echo "SCORING_MARKERS=$TOTAL" >> "$LOG/year.done"
echo "SCORING_FAILED=$FAILED" >> "$LOG/year.done"

echo "=== aggregate ==="
cd "$SRC" || exit 1
"$VENV/bin/python" -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "$OUT" \
  aggregate --expect-starts "$START_COUNT"
RC=$?
echo "AGGREGATE_EXIT=$RC" >> "$LOG/year.done"
echo "=== aggregate rc=$RC ==="
ls -la "$OUT/scores.parquet" 2>&1
date -u
exit $RC
