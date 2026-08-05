# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Merge a realism-only evaluation output into an already-scored unit output.

A unit scored before the realism battery worked on its grid carries every gridded, Class-4 and
Lagrangian record but no realism record. Rerunning the whole unit to recover the realism rows
would recompute hours of scores that are already correct, so realism is rerun on its own
(``evaluate --metrics realism``) into a separate directory and its rows are grafted here onto
the unit's canonical ``scores.parquet``.

The merge is idempotent: every realism row already present in the target is dropped before the
new rows are appended, so merging twice leaves the same parquet as merging once. The aggregated
artifacts are then regenerated through the very function ``evaluate`` uses
(:func:`oceanbench.packs.evaluate._write_scores_and_summary`), so the summary a merged unit
carries is the summary that unit would have carried had realism run in the first place. Realism
records are aggregates over the starts (``start_date`` is null, contracts.md §3.2), so they land
in the long-format parquet and leave the per-start summary numbers untouched.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas

from oceanbench.publish.aggregate import summary_to_json_records
from oceanbench.runner import records

REALISM_METRICS = frozenset(
    {
        records.METRIC_PSD_BAND_ENERGY_FRACTION,
        records.METRIC_EFFECTIVE_RESOLUTION_KILOMETRES,
        records.METRIC_ERROR_SPECTRUM_BAND_ENERGY,
        records.METRIC_ACTIVITY_RATIO,
        records.METRIC_EDDY_COUNT,
        records.METRIC_EDDY_HIT_RATE,
        records.METRIC_EDDY_MISS_RATE,
        records.METRIC_EDDY_MEAN_DISPLACEMENT_KILOMETRES,
    }
)

_SKILL_COLUMN_PREFIX = "skill_vs_"


@dataclass(frozen=True)
class MergeRealismResult:
    scores_path: str
    summary_path: str
    per_challenger_paths: list[str]
    realism_row_count: int
    total_row_count: int
    skill_baseline: str | None


def skill_baseline_from_summary(summary_path: Path) -> str | None:
    """Recover the baseline the unit's summary quotes skill against, from its ``skill_vs_*`` column."""
    if not summary_path.exists():
        return None
    summary_records = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in summary_records:
        for column in record:
            if column.startswith(_SKILL_COLUMN_PREFIX):
                return column[len(_SKILL_COLUMN_PREFIX) :]
    return None


def realism_rows(scores: pandas.DataFrame) -> pandas.DataFrame:
    """The realism-battery rows of a long-format scores frame."""
    if scores.empty:
        return scores
    return scores[scores["metric"].isin(REALISM_METRICS)].reset_index(drop=True)


def merge_realism_scores(
    unit_directory: str,
    realism_directory: str,
    *,
    skill_baseline: str | None = None,
) -> MergeRealismResult:
    """Append the realism-only run's rows to the unit's scores and regenerate its aggregates.

    ``unit_directory`` is an evaluation output directory holding ``scores.parquet`` and
    ``scores-summary.json``; ``realism_directory`` is the output directory of the realism-only
    rerun of the same (challenger, year, region). Any ``scores-<slug>.json`` the unit already
    carries is rewritten from the regenerated summary, the same content ``evaluate`` writes.
    """
    from oceanbench.packs.evaluate import (
        SCORES_FILENAME,
        SCORES_SUMMARY_FILENAME,
        _write_scores_and_summary,
    )

    unit_path = Path(unit_directory)
    scores_path = unit_path / SCORES_FILENAME
    summary_path = unit_path / SCORES_SUMMARY_FILENAME
    realism_scores_path = Path(realism_directory) / SCORES_FILENAME

    unit_scores = pandas.read_parquet(str(scores_path))
    appended = realism_rows(pandas.read_parquet(str(realism_scores_path)))
    if appended.empty:
        raise ValueError(f"{realism_scores_path} carries no realism record to merge.")

    kept = unit_scores[~unit_scores["metric"].isin(REALISM_METRICS)].reset_index(drop=True)
    merged = pandas.concat([kept, appended], ignore_index=True)[list(unit_scores.columns)]

    resolved_skill_baseline = (
        skill_baseline if skill_baseline is not None else skill_baseline_from_summary(summary_path)
    )
    summary = _write_scores_and_summary(
        merged,
        pandas.DataFrame(),
        skill_baseline=resolved_skill_baseline,
        scores_path=scores_path,
        summary_path=summary_path,
    )

    summary_json = json.dumps(summary_to_json_records(summary), sort_keys=True, indent=2, default=str)
    per_challenger_paths = [
        path for path in sorted(unit_path.glob("scores-*.json")) if path.name != SCORES_SUMMARY_FILENAME
    ]
    for path in per_challenger_paths:
        path.write_text(summary_json, encoding="utf-8")

    return MergeRealismResult(
        scores_path=str(scores_path),
        summary_path=str(summary_path),
        per_challenger_paths=[str(path) for path in per_challenger_paths],
        realism_row_count=len(appended),
        total_row_count=len(merged),
        skill_baseline=resolved_skill_baseline,
    )
