# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Compact per-run score parquets into the single public ``scores.parquet`` (contracts.md §3.1, §3.3).

Each scoring run writes ``runs/<challenger>/<year>/<region>/scores*.parquet`` (the same
schema as the public file). ``publish`` merges every run into one consolidated
``scores.parquet``:

- **schema-enforced**: the merged frame is reindexed onto the contract column order and
  dtypes (``oceanbench.runner.records.SCORE_COLUMNS``); a run missing or renaming a column
  is rejected.
- **newest run wins**: an upstream reprocessing re-scores the same key; the row from the
  most recent run supersedes the older one. Rows carrying a *different* ``challenger_version``
  coexist (old and new scores live side by side, §1), so the dedup key includes the version.
- **deterministic ordering**: the output is sorted by its full key so identical inputs
  always produce a byte-identical file.
- **stamped**: the compaction ``oceanbench_version`` is recorded in the parquet file metadata.
"""

from pathlib import Path

import pandas
import pyarrow
import pyarrow.parquet

from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.runner.records import SCORE_COLUMNS

SCORES_FILENAME = "scores.parquet"
_RUN_PARQUET_GLOB = "scores*.parquet"

# A row is one measurement; two rows are duplicates only if every identity field matches.
# challenger_version is part of the key so re-scores under a new version coexist rather than
# overwrite; start_date and lead_day make each per-start value its own row.
DEDUP_KEY_COLUMNS = [
    "challenger",
    "challenger_version",
    "year",
    "region",
    "metric",
    "reference",
    "variable",
    "depth",
    "lead_day",
    "start_date",
    "band",
    "polarity",
]


def discover_run_parquets(runs_root: str) -> list[str]:
    """Every run parquet under ``runs_root``, ordered oldest to newest by modification time.

    The order is the run-recency order the dedup relies on: later files win. Ties on
    modification time fall back to path order for determinism.
    """
    paths = list(Path(runs_root).rglob(_RUN_PARQUET_GLOB))
    return [str(path) for path in sorted(paths, key=lambda path: (path.stat().st_mtime, str(path)))]


def _enforce_schema(frame: pandas.DataFrame, source: str) -> pandas.DataFrame:
    missing = [column for column in SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Run parquet {source} is missing score columns: {missing}.")
    return frame[SCORE_COLUMNS]


def compact_run_frames(run_frames_oldest_first: list[pandas.DataFrame]) -> pandas.DataFrame:
    """Merge run frames (oldest first) into the deduplicated, deterministically ordered table."""
    if not run_frames_oldest_first:
        return pandas.DataFrame(columns=SCORE_COLUMNS)
    combined = pandas.concat(run_frames_oldest_first, ignore_index=True)
    deduplicated = combined.drop_duplicates(subset=DEDUP_KEY_COLUMNS, keep="last")
    ordered = deduplicated.sort_values(DEDUP_KEY_COLUMNS, na_position="last", kind="stable")
    return ordered.reset_index(drop=True)


def compact_scores(
    run_parquet_paths: list[str],
    output_path: str,
    *,
    oceanbench_version: str = OCEANBENCH_VERSION,
) -> str:
    """Compact the given run parquets (oldest first) into ``output_path`` and return it."""
    frames = [_enforce_schema(pandas.read_parquet(path), path) for path in run_parquet_paths]
    compacted = compact_run_frames(frames)
    output_path_object = Path(output_path)
    output_path_object.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.Table.from_pandas(compacted, preserve_index=False)
    stamped = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"oceanbench_version": oceanbench_version.encode("utf-8")}
    )
    pyarrow.parquet.write_table(stamped, output_path)
    return str(output_path_object)


def compact_runs_directory(
    runs_root: str,
    output_path: str,
    *,
    oceanbench_version: str = OCEANBENCH_VERSION,
) -> str:
    """Discover every run parquet under ``runs_root`` and compact them into ``output_path``."""
    return compact_scores(discover_run_parquets(runs_root), output_path, oceanbench_version=oceanbench_version)
