# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Round-trip and dedup behaviour of the scores compaction (contracts.md §3.1, §3.3)."""

import os
import time

import pandas
import pyarrow.parquet
import pytest

from oceanbench.publish.compact import (
    compact_run_frames,
    compact_runs_directory,
    compact_scores,
    discover_run_parquets,
)
from oceanbench.runner.records import SCORE_COLUMNS, records_to_dataframe


def _record(**overrides) -> dict:
    record = {
        "challenger": "glonet_1_degree",
        "challenger_version": "0.2.1",
        "year": 2024,
        "region": "global",
        "metric": "rmsd",
        "reference": "glorys",
        "variable": "sea_surface_height_above_geoid",
        "depth": "surface",
        "lead_day": 1,
        "start_date": pandas.Timestamp("2024-01-03").date(),
        "band": None,
        "polarity": None,
        "value": 0.1,
        "unit": "m",
        "n": None,
        "oceanbench_version": "0.2.1",
    }
    record.update(overrides)
    return record


def _run_frame(records: list[dict]) -> pandas.DataFrame:
    return records_to_dataframe(records)


def _write_run(directory, challenger, records) -> str:
    run_directory = directory / challenger / "2024" / "global"
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / "scores.parquet"
    _run_frame(records).to_parquet(path, index=False)
    return str(path)


def test_round_trip_preserves_rows_and_schema(tmp_path):
    frame = _run_frame([_record(lead_day=lead) for lead in range(1, 11)])
    output = tmp_path / "scores.parquet"
    compact_scores(
        [_write_run(tmp_path / "runs", "glonet_1_degree", [_record(lead_day=lead) for lead in range(1, 11)])],
        str(output),
    )

    read_back = pandas.read_parquet(output)
    assert list(read_back.columns) == SCORE_COLUMNS
    assert len(read_back) == len(frame)


def test_newest_run_wins_on_duplicate_keys():
    old = _run_frame([_record(value=0.10)])
    new = _run_frame([_record(value=0.99)])
    compacted = compact_run_frames([old, new])  # oldest first
    assert len(compacted) == 1
    assert compacted["value"].iloc[0] == pytest.approx(0.99)


def test_rows_with_different_challenger_version_coexist():
    first = _run_frame([_record(challenger_version="0.2.1", value=0.10)])
    reprocessed = _run_frame([_record(challenger_version="0.3.0", value=0.20)])
    compacted = compact_run_frames([first, reprocessed])
    assert len(compacted) == 2
    assert set(compacted["challenger_version"]) == {"0.2.1", "0.3.0"}


def test_ordering_is_deterministic_regardless_of_input_order():
    frame_a = _run_frame([_record(lead_day=2, value=0.2), _record(lead_day=1, value=0.1)])
    frame_b = _run_frame([_record(lead_day=1, value=0.1), _record(lead_day=2, value=0.2)])
    assert compact_run_frames([frame_a]).equals(compact_run_frames([frame_b]))


def test_missing_column_is_rejected(tmp_path):
    truncated = _run_frame([_record()]).drop(columns=["value"])
    path = tmp_path / "bad.parquet"
    truncated.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing score columns"):
        compact_scores([str(path)], str(tmp_path / "out.parquet"))


def test_stamped_with_oceanbench_version(tmp_path):
    output = tmp_path / "scores.parquet"
    compact_scores(
        [_write_run(tmp_path / "runs", "glonet_1_degree", [_record()])],
        str(output),
        oceanbench_version="9.9.9",
    )
    metadata = pyarrow.parquet.read_metadata(output).metadata
    assert metadata[b"oceanbench_version"] == b"9.9.9"


def test_discovery_orders_by_modification_time_and_compacts_directory(tmp_path):
    runs = tmp_path / "runs"
    old_path = _write_run(runs, "glonet_1_degree", [_record(value=0.10)])
    time.sleep(0.01)
    # A second run over the same key, written later, must win.
    new_records = [_record(value=0.55)]
    later_directory = runs / "glonet_1_degree" / "2024" / "global"
    later_path = later_directory / "scores-later.parquet"
    _run_frame(new_records).to_parquet(later_path, index=False)
    os.utime(later_path, (time.time() + 10, time.time() + 10))

    discovered = discover_run_parquets(str(runs))
    assert discovered[-1] == str(later_path)
    assert old_path in discovered

    output = tmp_path / "scores.parquet"
    compact_runs_directory(str(runs), str(output))
    compacted = pandas.read_parquet(output)
    assert len(compacted) == 1
    assert compacted["value"].iloc[0] == pytest.approx(0.55)
