# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Match-up parquet serving layout: row-group packing, sort order, statistics, footer weight."""

import numpy
import pandas
import pyarrow.parquet
import pytest

from oceanbench.publish import viewer_artifacts

_BLOCKS = (
    ("eastward_sea_water_velocity", "15m"),
    ("northward_sea_water_velocity", "15m"),
    ("sea_surface_height_above_geoid", "surface"),
    ("sea_water_potential_temperature", "0-5m"),
    ("sea_water_potential_temperature", "100-300m"),
    ("sea_water_salinity", "0-5m"),
    ("sea_water_salinity", "100-300m"),
)
_STATISTICS_COLUMNS = ("variable", "depth_bin", "lead_day", "start_date")
_VALUE_COLUMNS = ("latitude", "longitude", "observation_value", "model_value")


def _matchups(*, starts, leads, rows_per_block, blocks=_BLOCKS, seed=0) -> pandas.DataFrame:
    generator = numpy.random.default_rng(seed)
    frames = []
    for start_date in starts:
        for lead_day in leads:
            for variable, depth_bin in blocks:
                frames.append(
                    pandas.DataFrame(
                        {
                            "variable": variable,
                            "depth_bin": depth_bin,
                            "lead_day": numpy.int16(lead_day),
                            "start_date": start_date,
                            "latitude": generator.uniform(-80, 85, rows_per_block),
                            "longitude": generator.uniform(-180, 180, rows_per_block),
                            "observation_value": generator.uniform(0, 30, rows_per_block),
                            "model_value": generator.uniform(0, 30, rows_per_block),
                        }
                    )
                )
    return pandas.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _starts(count):
    return [str(numpy.datetime64("2024-01-01") + numpy.timedelta64(index * 7, "D")) for index in range(count)]


def _row_group_keys(path):
    metadata = pyarrow.parquet.ParquetFile(path).metadata
    names = [metadata.schema.column(index).name for index in range(metadata.num_columns)]
    keys = []
    for index in range(metadata.num_row_groups):
        row_group = metadata.row_group(index)
        statistics = {name: row_group.column(names.index(name)).statistics for name in _STATISTICS_COLUMNS}
        keys.append((statistics, row_group.num_rows))
    return keys


def test_row_groups_pack_whole_blocks_of_one_pair_and_never_straddle_a_pair(tmp_path):
    path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(_matchups(starts=_starts(3), leads=(1, 2), rows_per_block=1000), path)

    keys = _row_group_keys(path)
    assert len(keys) == 6
    for statistics, row_count in keys:
        assert str(statistics["start_date"].min) == str(statistics["start_date"].max)
        assert statistics["lead_day"].min == statistics["lead_day"].max
        assert row_count == len(_BLOCKS) * 1000
    pairs = [(str(statistics["start_date"].min), statistics["lead_day"].min) for statistics, _ in keys]
    assert pairs == sorted(pairs)


def test_the_written_file_round_trips_every_value_dtype_and_the_sort_order(tmp_path):
    path = str(tmp_path / "class4-matchups.parquet")
    source = _matchups(starts=_starts(2), leads=(1, 3), rows_per_block=200)
    viewer_artifacts.write_matchup_parquet(source, path)

    table = pyarrow.parquet.read_table(path)
    assert table.schema.names == viewer_artifacts._MATCHUP_TARGET_SCHEMA.names
    assert table.schema.types == viewer_artifacts._MATCHUP_TARGET_SCHEMA.types
    assert table.num_rows == len(source)

    frame = table.to_pandas()
    keys = ["start_date", "lead_day", "variable", "depth_bin"]
    assert frame[keys].equals(frame[keys].sort_values(keys, kind="stable").reset_index(drop=True))
    for column in _VALUE_COLUMNS:
        assert numpy.allclose(
            numpy.sort(frame[column].to_numpy()),
            numpy.sort(source[column].to_numpy().astype(numpy.float32)),
        )


def test_statistics_are_written_only_on_the_grouping_columns(tmp_path):
    path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(_matchups(starts=_starts(1), leads=(1,), rows_per_block=100), path)

    metadata = pyarrow.parquet.ParquetFile(path).metadata
    names = [metadata.schema.column(index).name for index in range(metadata.num_columns)]
    row_group = metadata.row_group(0)
    for name in _STATISTICS_COLUMNS:
        assert row_group.column(names.index(name)).statistics is not None
    for name in _VALUE_COLUMNS:
        assert row_group.column(names.index(name)).statistics is None


def test_a_block_above_the_hard_cap_is_split_across_consecutive_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer_artifacts, "TARGET_ROW_GROUP_ROWS", 400)
    monkeypatch.setattr(viewer_artifacts, "MAXIMUM_ROW_GROUP_ROWS", 400)
    path = str(tmp_path / "class4-matchups.parquet")
    blocks = (("sea_surface_height_above_geoid", "surface"),)
    viewer_artifacts.write_matchup_parquet(
        _matchups(starts=_starts(1), leads=(1,), rows_per_block=1000, blocks=blocks), path
    )

    keys = _row_group_keys(path)
    assert [row_count for _, row_count in keys] == [400, 400, 200]


def test_packing_shrinks_the_footer_by_the_block_count_and_stays_under_the_budget(tmp_path, monkeypatch):
    frame = _matchups(starts=_starts(30), leads=range(1, 11), rows_per_block=40)
    packed_path = str(tmp_path / "packed.parquet")
    viewer_artifacts.write_matchup_parquet(frame, packed_path)

    per_block_path = str(tmp_path / "per-block.parquet")
    monkeypatch.setattr(viewer_artifacts, "TARGET_ROW_GROUP_ROWS", 1)
    viewer_artifacts.write_matchup_parquet(frame, per_block_path)

    packed = pyarrow.parquet.ParquetFile(packed_path).metadata
    per_block = pyarrow.parquet.ParquetFile(per_block_path).metadata
    assert packed.num_row_groups == 300
    assert per_block.num_row_groups == 300 * len(_BLOCKS)
    assert packed.serialized_size < per_block.serialized_size / (len(_BLOCKS) - 1)

    # The footer is linear in the column-chunk count, so the published global year (about 1 040
    # row groups over 8 columns) follows from the per-group cost measured here.
    footer_per_row_group = packed.serialized_size / packed.num_row_groups
    assert 1_040 * footer_per_row_group < 1_000_000


def test_verify_rejects_a_row_group_straddling_two_pairs(tmp_path):
    path = str(tmp_path / "straddling.parquet")
    projected = viewer_artifacts._projected_sorted_partition(
        _matchups(starts=_starts(2), leads=(1,), rows_per_block=50)
    )
    writer = pyarrow.parquet.ParquetWriter(
        path,
        projected.schema,
        compression="zstd",
        write_statistics=viewer_artifacts._MATCHUP_STATISTICS_COLUMNS,
    )
    writer.write_table(projected)
    writer.close()

    with pytest.raises(ValueError, match="mixes more than one"):
        viewer_artifacts.verify_matchup_parquet(path)


def test_streamed_and_whole_frame_writers_produce_the_same_layout(tmp_path):
    frame = _matchups(starts=_starts(3), leads=(1, 2), rows_per_block=300)
    whole_path = str(tmp_path / "whole.parquet")
    streamed_path = str(tmp_path / "streamed.parquet")
    viewer_artifacts.write_matchup_parquet(frame, whole_path)
    viewer_artifacts.write_matchup_parquet_streamed(
        (partition for _, partition in frame.groupby("start_date", sort=True)), streamed_path
    )

    whole = pyarrow.parquet.ParquetFile(whole_path).metadata
    streamed = pyarrow.parquet.ParquetFile(streamed_path).metadata
    assert whole.num_row_groups == streamed.num_row_groups
    assert whole.num_rows == streamed.num_rows
    assert [whole.row_group(index).num_rows for index in range(whole.num_row_groups)] == [
        streamed.row_group(index).num_rows for index in range(streamed.num_row_groups)
    ]
