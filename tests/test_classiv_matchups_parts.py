# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import numpy
import pandas
import pytest

from oceanbench.core import classIV
from oceanbench.core.classIV import (
    class4_rmsd_and_matchups,
    class4_rmsd_and_matchups_by_start,
    rmsd_class4_validation,
)
from oceanbench.core.classIV_matchups import Class4Matchups
from oceanbench.core.dataset_utils import Dimension
from oceanbench.publish import viewer_artifacts
from tests import synthetic_class4
from tests.test_viewer_artifacts import _SALINITY_KEY, _SEA_SURFACE_HEIGHT_KEY, _TEMPERATURE_KEY, _frame

_BLOCKS = [
    (_SEA_SURFACE_HEIGHT_KEY, "surface"),
    (_TEMPERATURE_KEY, "surface"),
    (_TEMPERATURE_KEY, "5-100m"),
    (_SALINITY_KEY, "0-5m"),
    (_SALINITY_KEY, "100-300m"),
]
_START_DATES = ["2024-01-03", "2024-01-10", "2024-01-17"]


def _table(sorted_by_start: bool) -> pandas.DataFrame:
    """A match-up table as Class IV writes it: one variable after the other, the rows of each shuffled."""
    frame = _frame(_BLOCKS, _START_DATES, [0, 1, 2], 40, seed=11, bias=0.1)
    frame.loc[frame.sample(frac=0.1, random_state=2).index, "model_value"] = numpy.nan
    variables = [_SEA_SURFACE_HEIGHT_KEY, _TEMPERATURE_KEY, _SALINITY_KEY]
    parts = [frame[frame["variable"] == variable] for variable in variables]
    if sorted_by_start:
        parts = [part.sort_values("first_day", kind="stable") for part in parts]
    return pandas.concat(parts, ignore_index=True)


def _on_disk(table: pandas.DataFrame) -> Class4Matchups:
    matchups = Class4Matchups.on_disk()
    for variable, rows in table.groupby("variable", sort=False):
        matchups.add(variable, rows)
    return matchups


@pytest.mark.parametrize("sorted_by_start", [True, False])
def test_matchups_on_disk_read_back_the_parts_of_the_table(sorted_by_start) -> None:
    table = _table(sorted_by_start)
    in_memory, on_disk = Class4Matchups.in_memory(table), _on_disk(table)

    assert len(on_disk) == len(table)
    assert on_disk.variables() == in_memory.variables()
    assert on_disk.first_days() == in_memory.first_days()
    assert on_disk.scored_first_days() == in_memory.scored_first_days()
    for variable in in_memory.variables():
        assert on_disk.lead_days(variable) == in_memory.lead_days(variable)
        for lead_day in in_memory.lead_days(variable):
            for first_day in [None, *in_memory.first_days()]:
                pandas.testing.assert_frame_equal(
                    on_disk.part(variable, lead_day, first_day),
                    in_memory.part(variable, lead_day, first_day).reset_index(drop=True),
                    check_exact=True,
                )


def test_matchups_on_disk_are_removed_with_their_object() -> None:
    on_disk = _on_disk(_table(True))
    directory = on_disk._directory
    assert any(directory.iterdir())

    del on_disk

    assert not directory.exists()


def test_viewer_files_from_the_matchups_on_disk_are_those_of_the_table(tmp_path) -> None:
    table = _table(True)
    on_disk = _on_disk(table)
    grid = viewer_artifacts.YEAR_GRIDS["global"]

    rows_by_start = viewer_artifacts.class4_rmsd_rows_by_start(table)
    pandas.testing.assert_frame_equal(
        viewer_artifacts.class4_rmsd_rows_by_start(on_disk), rows_by_start, check_exact=True
    )
    for name, function in [
        ("rmsd_by_depth", lambda matchups: viewer_artifacts.rmsd_by_depth(matchups)),
        ("geography", lambda matchups: viewer_artifacts.year_error_geography(matchups, grid, 3)),
        ("by_start", lambda matchups: viewer_artifacts.year_rmsd_by_start(matchups, rows_by_start, grid, 3)),
    ]:
        assert json.dumps(function(on_disk)) == json.dumps(function(table)), name
    viewer_artifacts.write_matchup_parquet(table, tmp_path / "table.parquet")
    viewer_artifacts.write_matchup_parquet(on_disk, tmp_path / "parts.parquet")
    assert (tmp_path / "table.parquet").read_bytes() == (tmp_path / "parts.parquet").read_bytes()


def _class4_arguments_sorted_by_start() -> tuple:
    observations = synthetic_class4.observations_dataset()
    order = numpy.argsort(observations[Dimension.FIRST_DAY_DATETIME.key()].values, kind="stable")
    return (
        synthetic_class4.challenger_dataset(),
        observations.isel({observations[Dimension.TIME.key()].dims[0]: order}),
        synthetic_class4.ocean_mask(),
        synthetic_class4.VARIABLES,
    )


def test_class4_one_start_at_a_time_gives_the_table_and_matchups_of_all_starts_at_once(monkeypatch) -> None:
    arguments = _class4_arguments_sorted_by_start()
    table_by_start, matchups_by_start = class4_rmsd_and_matchups(*arguments, keep_matchups=True)
    monkeypatch.setattr(classIV, "_start_observation_slices", lambda observations: None)
    _, matchups_by_variable = class4_rmsd_and_matchups(*arguments, keep_matchups=True)

    assert len(matchups_by_start._files) > 0 and not matchups_by_start._in_memory
    pandas.testing.assert_frame_equal(table_by_start, rmsd_class4_validation(*arguments), check_exact=True)
    pandas.testing.assert_frame_equal(
        matchups_by_start.to_dataframe(), matchups_by_variable.to_dataframe(), check_exact=True
    )


def test_class4_given_the_observations_one_start_at_a_time_gives_the_table_and_matchups_of_all_starts() -> None:
    challenger, observations, mask, variables = _class4_arguments_sorted_by_start()
    start_observations = (
        observations.isel({observations[Dimension.TIME.key()].dims[0]: start_slice})
        for start_slice in classIV._start_observation_slices(observations)
    )
    table, matchups = class4_rmsd_and_matchups(challenger, observations, mask, variables, keep_matchups=True)
    table_by_start, matchups_by_start = class4_rmsd_and_matchups_by_start(
        challenger, start_observations, mask, variables, keep_matchups=True
    )

    pandas.testing.assert_frame_equal(table_by_start, table, check_exact=True)
    pandas.testing.assert_frame_equal(matchups_by_start.to_dataframe(), matchups.to_dataframe(), check_exact=True)
    assert class4_rmsd_and_matchups_by_start(challenger, iter([]), mask, variables, keep_matchups=False)[1] is None


def test_matchups_added_out_of_start_date_order_after_parts_on_disk_are_refused() -> None:
    table = _table(True)
    first_days = sorted(table["first_day"].unique())
    on_disk = Class4Matchups.on_disk()
    on_disk.add(_SEA_SURFACE_HEIGHT_KEY, table[table["first_day"] == first_days[1]])

    with pytest.raises(ValueError):
        on_disk.add(_SEA_SURFACE_HEIGHT_KEY, table[table["first_day"] == first_days[0]])
