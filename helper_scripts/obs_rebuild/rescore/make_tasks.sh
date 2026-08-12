#!/bin/bash

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Build the ladder task list. One line per rung x region x chunk.
set -euo pipefail
R=/scratch/jseillade/obs-rebuild/rescore
CHUNKS=${CHUNKS:-13}
OUT=$R/tasks.txt
: > "$OUT"
add() { # rung obsroot vars
  local rung=$1 root=$2 vars=$3 region
  for region in global ibi; do
    for ((c=0; c<CHUNKS; c++)); do
      echo "$rung $root $region $vars $c $CHUNKS" >> "$OUT"
    done
  done
}
add L0  /scratch/jseillade/obs-rebuild/store-legacy thetao,so,sla,uo,vo
add L1  /scratch/jseillade/obs-rebuild/views/L1     thetao,so,sla,uo,vo
add L3  /scratch/jseillade/obs-rebuild/views/L3     thetao,so,sla,uo,vo
add L2  /scratch/jseillade/obs-rebuild/views/L2     uo,vo
add L2b /scratch/jseillade/obs-rebuild/views/L2b    uo,vo
wc -l "$OUT"
