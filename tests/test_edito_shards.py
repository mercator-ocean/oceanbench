# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import importlib.util
import pickle
from pathlib import Path

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.ocean_mask import OCEAN_MASK_DEPTHS, OCEAN_MASK_STANDARD_DEPTHS
from oceanbench.core.rmsd import rmsd
from oceanbench.core.scores import score_rows

_SHARDS_PATH = Path(__file__).resolve().parents[1] / "edito" / "process" / "shards.py"
_METRIC = "rmsd_of_variables_compared_to_glorys_reanalysis"
_LABELS = ["Salinity (PSU) [sea_water_salinity]{surface}", "Temperature (°C) [sea_water_potential_temperature]{50m}"]


@pytest.fixture(scope="module")
def shards():
    specification = importlib.util.spec_from_file_location("edito_shards", _SHARDS_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _start_table(values) -> pandas.DataFrame:
    return pandas.DataFrame(values, index=_LABELS, columns=lead_day_labels(1, 2))


class _FakeShard:
    def __init__(self, start_dates, start_tables) -> None:
        self.start_dates = start_dates
        self.start_tables = start_tables


def test_merged_gridded_rows_are_the_mean_over_the_starts_of_every_shard(shards) -> None:
    first = [_start_table([[1.0, 2.0], [3.0, numpy.nan]]), _start_table([[3.0, 4.0], [5.0, numpy.nan]])]
    second = [_start_table([[5.0, 9.0], [7.0, 1.0]])]

    rows = shards._gridded_rows(_METRIC, [first, second])

    expected = score_rows(_METRIC, _start_table([[3.0, 5.0], [5.0, 1.0]]))
    assert [(row["standard_name"], row["depth"], row["lead_day"]) for row in rows] == [
        (row["standard_name"], row["depth"], row["lead_day"]) for row in expected
    ]
    assert [row["value"] for row in rows] == pytest.approx([row["value"] for row in expected])


def test_a_metric_held_by_some_shards_only_is_refused(shards) -> None:
    with pytest.raises(ValueError, match="some shards only"):
        shards._merged_start_tables(
            [_FakeShard([], {_METRIC: [_start_table([[1.0, 2.0], [3.0, 4.0]])]}), _FakeShard([], {})]
        )


@pytest.mark.parametrize(
    "shard_start_dates",
    [
        [["2024-01-10"], ["2024-01-03"]],
        [["2024-01-03"], ["2024-01-17"]],
        [["2024-01-03", "2024-01-10"], ["2024-01-10", "2024-01-17"]],
    ],
)
def test_shards_out_of_order_with_a_gap_or_an_overlap_are_refused(shards, shard_start_dates) -> None:
    with pytest.raises(ValueError, match="start date order"):
        shards._check_start_dates(
            [_FakeShard(dates, {}) for dates in shard_start_dates], ["2024-01-03", "2024-01-10", "2024-01-17"]
        )


def _random_temperature_dataset(start_count: int, start_dates_per_chunk: int) -> xarray.Dataset:
    generator = numpy.random.default_rng(0)
    latitudes = numpy.linspace(-70.0, 70.0, 24, dtype="float32")
    longitudes = numpy.linspace(-180.0, 170.0, 36, dtype="float32")
    shape = (start_count, 3, len(OCEAN_MASK_STANDARD_DEPTHS), len(latitudes), len(longitudes))
    values = generator.normal(10.0, 5.0, shape).astype("float32")
    values[generator.random(shape) < 0.1] = numpy.nan
    dimensions = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    start_dates = numpy.datetime64("2023-01-04") + 7 * numpy.arange(start_count).astype("timedelta64[D]")
    return xarray.Dataset(
        {Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (dimensions, values)},
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): start_dates.astype("datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0, 1, 2],
            Dimension.DEPTH.key(): OCEAN_MASK_STANDARD_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    ).chunk({Dimension.FIRST_DAY_DATETIME.key(): start_dates_per_chunk, Dimension.LEAD_DAY_INDEX.key(): 1})


def _random_ocean_mask(dataset: xarray.Dataset) -> xarray.DataArray:
    latitudes = dataset[Dimension.LATITUDE.key()].values
    longitudes = dataset[Dimension.LONGITUDE.key()].values
    values = numpy.random.default_rng(1).random((len(OCEAN_MASK_DEPTHS), len(latitudes), len(longitudes))) < 0.8
    return xarray.DataArray(
        values,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


@pytest.mark.parametrize(("start_dates_per_chunk", "shard_sizes"), [(1, [5, 4, 3]), (2, [4, 6, 2]), (2, [6, 5])])
def test_merged_gridded_rows_are_bit_for_bit_the_ones_of_one_evaluation(
    shards, start_dates_per_chunk, shard_sizes
) -> None:
    challenger_dataset = _random_temperature_dataset(sum(shard_sizes), start_dates_per_chunk)
    reference_dataset = _random_temperature_dataset(sum(shard_sizes), start_dates_per_chunk) * 0.5 + 4.0
    arguments = {
        "variables": [Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
        "ocean_mask": _random_ocean_mask(challenger_dataset),
        "keep_start_tables": True,
    }
    table, _ = rmsd(challenger_dataset=challenger_dataset, reference_dataset=reference_dataset, **arguments)
    shard_start_tables = []
    first_start = 0
    for size in shard_sizes:
        starts = {Dimension.FIRST_DAY_DATETIME.key(): slice(first_start, first_start + size)}
        _, start_tables = rmsd(
            challenger_dataset=challenger_dataset.isel(starts), reference_dataset=reference_dataset.isel(starts), **arguments
        )
        shard_start_tables.append(pickle.loads(pickle.dumps(start_tables)))
        first_start += size

    rows = shards._gridded_rows(_METRIC, shard_start_tables)

    assert rows == score_rows(_METRIC, table)
    for start_table in [table for tables in shard_start_tables for table in tables]:
        assert start_table[shards.MISSING_FRACTION_COLUMN].tolist() == table[shards.MISSING_FRACTION_COLUMN].tolist()


@pytest.mark.parametrize("shard_sizes", [[13, 13, 13, 13], [14, 13, 12, 13]])
def test_shards_with_an_odd_number_of_start_dates_before_the_last_are_refused(shards, shard_sizes) -> None:
    with pytest.raises(ValueError, match="every shard but the last"):
        shards._check_shard_sizes([_FakeShard([None] * size, {}) for size in shard_sizes])


def test_shards_with_an_even_number_of_start_dates_before_the_last_are_accepted(shards) -> None:
    shards._check_shard_sizes([_FakeShard([None] * size, {}) for size in [14, 14, 12, 11]])


def test_the_job_script_and_the_helper_offer_the_same_shard_modes(shards) -> None:
    commands = shards._parser()._subparsers._group_actions[0].choices
    assert set(commands) == {"evaluate-shard", "merge", "eddy-census", "eddy-merge"}
    script = (_SHARDS_PATH.parent / "run-evaluation.sh").read_text()
    for command in commands:
        assert f'"$SHARDS_HELPER" {command}' in script
