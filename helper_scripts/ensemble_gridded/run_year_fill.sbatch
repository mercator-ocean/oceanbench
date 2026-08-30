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

# Every path below comes from the environment, so any user can run the campaign from any
# tree. The #SBATCH log path cannot read one: slurm reads it before the shell does, so pass
# sbatch --output to move the logs.
WORK=${ENSEMBLE_CAMPAIGN_ROOT:?export ENSEMBLE_CAMPAIGN_ROOT, the directory holding src, output and logs}
SRC=$WORK/src
VENV=${ENSEMBLE_VENV:?export ENSEMBLE_VENV, the python environment that carries oceanbench}
# The read only campaign tree supplies the pure python s3fs stack and the staged GLORYS weeks.
PYLIBS=${ENSEMBLE_PYTHON_LIBRARIES:?export ENSEMBLE_PYTHON_LIBRARIES, the pure python s3fs stack}
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
. "${ENSEMBLE_CREDENTIALS_FILE:?export ENSEMBLE_CREDENTIALS_FILE, the env file holding the object store keys}"
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
