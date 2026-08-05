# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Grafting a realism-only rerun onto an already-scored unit output."""

import json
from pathlib import Path

import pandas
import pytest

from oceanbench.packs.evaluate import SCORES_FILENAME, SCORES_SUMMARY_FILENAME, _write_scores_and_summary
from oceanbench.publish.merge_realism import merge_realism_scores
from oceanbench.runner import records

_CHALLENGER = "glonet"
_BASELINE = "climatology"


def _per_start_record(challenger: str, start_date: str, value: float) -> dict:
    return {
        "challenger": challenger,
        "challenger_version": "test",
        "year": 2024,
        "region": "global",
        "metric": records.METRIC_ROOT_MEAN_SQUARE_DEVIATION,
        "reference": "glorys",
        "variable": "sea_surface_height_above_geoid",
        "depth": "surface",
        "lead_day": 1,
        "start_date": start_date,
        "band": None,
        "polarity": None,
        "value": value,
        "unit": "m",
        "n": None,
        "oceanbench_version": "test",
    }


def _realism_record(metric: str, value: float, band: str | None = None) -> dict:
    return {
        "challenger": _CHALLENGER,
        "challenger_version": "test",
        "year": 2024,
        "region": "global",
        "metric": metric,
        "reference": "glorys",
        "variable": "sea_surface_height_above_geoid",
        "depth": "surface",
        "lead_day": 1,
        "start_date": None,
        "band": band,
        "polarity": None,
        "value": value,
        "unit": None,
        "n": None,
        "oceanbench_version": "test",
    }


def _unit_directory(tmp_path: Path, *, with_realism: bool) -> Path:
    unit_path = tmp_path / "unit"
    unit_path.mkdir()
    per_start = [
        _per_start_record(challenger, start_date, value)
        for challenger, offset in ((_CHALLENGER, 0.0), (_BASELINE, 0.2))
        for start_date, value in (("2024-01-03", 0.10 + offset), ("2024-01-10", 0.12 + offset))
    ]
    stale_realism = [_realism_record(records.METRIC_ACTIVITY_RATIO, 9.99)] if with_realism else []
    _write_scores_and_summary(
        records.records_to_dataframe(per_start + stale_realism),
        pandas.DataFrame(),
        skill_baseline=_BASELINE,
        scores_path=unit_path / SCORES_FILENAME,
        summary_path=unit_path / SCORES_SUMMARY_FILENAME,
    )
    (unit_path / f"scores-{_CHALLENGER}.json").write_text("[]", encoding="utf-8")
    return unit_path


def _realism_directory(tmp_path: Path) -> Path:
    realism_path = tmp_path / "realism"
    realism_path.mkdir()
    realism_records = [
        _realism_record(records.METRIC_ACTIVITY_RATIO, 0.8),
        _realism_record(records.METRIC_PSD_BAND_ENERGY_FRACTION, 0.6, band="large"),
        _realism_record(records.METRIC_EDDY_COUNT, 12.0),
    ]
    records.records_to_dataframe(realism_records).to_parquet(str(realism_path / SCORES_FILENAME), index=False)
    return realism_path


def test_merge_appends_the_realism_rows_and_regenerates_the_summary(tmp_path: Path) -> None:
    unit_path = _unit_directory(tmp_path, with_realism=False)
    before = pandas.read_parquet(str(unit_path / SCORES_FILENAME))

    result = merge_realism_scores(str(unit_path), str(_realism_directory(tmp_path)))

    merged = pandas.read_parquet(result.scores_path)
    assert result.realism_row_count == 3
    assert len(merged) == len(before) + 3
    assert result.total_row_count == len(merged)
    assert list(merged.columns) == list(before.columns)
    assert merged[merged["start_date"].notna()].reset_index(drop=True).equals(before)

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary
    # Realism records carry no start distribution, so the per-start summary is untouched by them.
    assert {record["metric"] for record in summary} == {records.METRIC_ROOT_MEAN_SQUARE_DEVIATION}
    assert result.skill_baseline == _BASELINE
    assert any(column.startswith("skill_vs_") for column in summary[0])

    per_challenger = json.loads(Path(result.per_challenger_paths[0]).read_text(encoding="utf-8"))
    assert per_challenger == summary


def test_merging_twice_leaves_the_same_rows(tmp_path: Path) -> None:
    unit_path = _unit_directory(tmp_path, with_realism=False)
    realism_path = _realism_directory(tmp_path)

    first = merge_realism_scores(str(unit_path), str(realism_path))
    once = pandas.read_parquet(first.scores_path)
    second = merge_realism_scores(str(unit_path), str(realism_path))
    twice = pandas.read_parquet(second.scores_path)

    assert first.total_row_count == second.total_row_count
    assert once.equals(twice)


def test_a_stale_realism_row_is_replaced_not_duplicated(tmp_path: Path) -> None:
    unit_path = _unit_directory(tmp_path, with_realism=True)

    result = merge_realism_scores(str(unit_path), str(_realism_directory(tmp_path)))

    merged = pandas.read_parquet(result.scores_path)
    activity = merged[merged["metric"] == records.METRIC_ACTIVITY_RATIO]
    assert len(activity) == 1
    assert float(activity["value"].iloc[0]) == pytest.approx(0.8)


def test_a_realism_directory_without_realism_rows_is_refused(tmp_path: Path) -> None:
    unit_path = _unit_directory(tmp_path, with_realism=False)
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    records.records_to_dataframe([_per_start_record(_CHALLENGER, "2024-01-03", 0.1)]).to_parquet(
        str(empty_path / SCORES_FILENAME), index=False
    )

    with pytest.raises(ValueError):
        merge_realism_scores(str(unit_path), str(empty_path))
