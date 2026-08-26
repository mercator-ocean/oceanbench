#!/bin/bash
# Gridded fill scoring of one ensemble challenger, one forecast start per array task.
# Submit with --export=ALL,CHALLENGER=...,FIRST_START=... and an array of 0-51.
# Each task writes its own done marker, so a resubmission skips what already succeeded.
#SBATCH --job-name=obens-fill
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jseillade/ensemble-eval-fills/logs/%x-%A_%a.log

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -uo pipefail

WORK=/scratch/jseillade/ensemble-eval-fills
SRC=$WORK/src
VENV=/scratch/jseillade/oceanbench-venv-050
# The read only campaign tree supplies the pure python s3fs stack and the staged GLORYS weeks.
PYLIBS=/scratch/jseillade/probax/gridded/pylibs
OUT=$WORK/output/$CHALLENGER
LOG=$WORK/logs/$CHALLENGER
mkdir -p "$OUT" "$LOG"

export TMPDIR=$WORK/tmp/$CHALLENGER-${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH=$SRC:$PYLIBS
export PYTHONPYCACHEPREFIX=$TMPDIR/pycache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

set -a
. /scratch/jseillade/gpu2_archive/data/cloudferro-oceanbench.env
set +a

START=$(date -u -d "$FIRST_START + $((SLURM_ARRAY_TASK_ID * 7)) days" +%Y-%m-%d)
COMPACT=$(echo "$START" | tr -d -)
MARKER=$LOG/start-$COMPACT.done

echo "=== context ==="
date -u
hostname
cat "$WORK/VERSION"
echo "challenger=$CHALLENGER start=$START task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID"

if [ -f "$MARKER" ] && grep -q "EXIT=0" "$MARKER"; then
  echo "skip $START, already scored"
  exit 0
fi
rm -f "$MARKER"

"$VENV/bin/python" -c "import oceanbench; print('IMPORTED_FROM=' + oceanbench.__file__)"

cd "$SRC"
time "$VENV/bin/python" -W ignore \
  helper_scripts/ensemble_gridded/score_ensemble_gridded_multi.py \
  --output-root "$OUT" \
  score --challenger "$CHALLENGER" --start-date "$START" --references glorys
RC=$?
echo "EXIT=$RC" > "$MARKER"
echo "=== $CHALLENGER $START rc=$RC ==="
date -u
rm -rf "$TMPDIR"
exit $RC
