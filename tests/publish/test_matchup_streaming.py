# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The parallel, streamed match-up path must reproduce the serial whole-frame path exactly.

The parallelisation splits the year across forecast start dates and the write streams one start
at a time; because ``start_date`` is the leading sort key and the starts are disjoint, the served
parquet must be identical in row content, row order and row-group layout to the previous
whole-frame :func:`write_matchup_parquet`. These tests pin that equality (not merely an equal
RMSD) on a synthetic multi-start dataset, exercising both the serial (one worker) and the forked
parallel path.
"""

import numpy
import pandas
import pyarrow.parquet
import pytest

import oceanbench.core.classIV_support as classIV_support
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.publish import viewer_artifacts
from oceanbench.runner import matchups
from oceanbench.runner.records import RunContext

_FIRST_DAYS = numpy.array(["2024-01-03", "2024-01-10", "2024-01-17"], dtype="datetime64[ns]")


def _context() -> RunContext:
    return RunContext(
        challenger="glonet_1_degree",
        challenger_version="0.2.1",
        year=2024,
        region="global",
        oceanbench_version="0.2.1",
    )


def _challenger() -> xarray.Dataset:
    lead_days = numpy.array([0, 1, 2])
    latitudes = numpy.array([0.0, 1.0, 2.0, 3.0])
    longitudes = numpy.array([10.0, 11.0, 12.0, 13.0])
    generator = numpy.random.default_rng(0)
    shape = (len(_FIRST_DAYS), len(lead_days), len(latitudes), len(longitudes))
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                generator.normal(size=shape),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): _FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _observations() -> xarray.Dataset:
    generator = numpy.random.default_rng(1)
    observation_dimension = "obs"
    times = []
    first_days = []
    latitudes = []
    longitudes = []
    values = []
    # Several observations per (start, lead) so that within a (start, lead, variable, depth_bin)
    # group there are ties whose relative order must survive the parallel/streamed reshaping.
    for first_day in _FIRST_DAYS:
        for lead in (0, 1, 2):
            for _ in range(5):
                times.append(pandas.Timestamp(first_day) + pandas.Timedelta(days=int(lead)))
                first_days.append(first_day)
                latitudes.append(float(generator.uniform(0.0, 3.0)))
                longitudes.append(float(generator.uniform(10.0, 13.0)))
                values.append(float(generator.normal()))
    return xarray.Dataset(
        {
            Dimension.TIME.key(): (observation_dimension, numpy.array(times, dtype="datetime64[ns]")),
            Dimension.LATITUDE.key(): (observation_dimension, numpy.array(latitudes)),
            Dimension.LONGITUDE.key(): (observation_dimension, numpy.array(longitudes)),
            Dimension.DEPTH.key(): (observation_dimension, numpy.zeros(len(values))),
            Dimension.FIRST_DAY_DATETIME.key(): (observation_dimension, numpy.array(first_days, dtype="datetime64[ns]")),
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (observation_dimension, numpy.array(values)),
        }
    )


def _patch_sea_level_conversion(monkeypatch) -> None:
    monkeypatch.setattr(classIV_support, "get_dataset_resolution", lambda dataset: "native")
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda resolution: xarray.DataArray(0.0))


def _read(path: str) -> pandas.DataFrame:
    return pyarrow.parquet.ParquetFile(path).read().to_pandas()


def _row_group_row_counts(path: str) -> list[int]:
    metadata = pyarrow.parquet.ParquetFile(path).metadata
    return [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]


@pytest.mark.parametrize("max_workers", [1, 2])
def test_streamed_parallel_matches_serial_whole_frame(tmp_path, monkeypatch, max_workers) -> None:
    _patch_sea_level_conversion(monkeypatch)
    challenger = _challenger()
    observations = _observations()
    variables = [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID]
    context = _context()

    serial_frame = matchups.class4_matchups(challenger, observations, variables, context=context)
    serial_path = str(tmp_path / "serial.parquet")
    viewer_artifacts.write_matchup_parquet(serial_frame, serial_path)

    streamed_path = str(tmp_path / "streamed.parquet")
    partitions = matchups.iter_class4_matchups_by_start(
        challenger, observations, variables, context=context, max_workers=max_workers
    )
    viewer_artifacts.write_matchup_parquet_streamed(partitions, streamed_path)

    pandas.testing.assert_frame_equal(_read(streamed_path), _read(serial_path))
    assert _row_group_row_counts(streamed_path) == _row_group_row_counts(serial_path)
    # Guard the decomposition really did split the work into independent starts.
    produced = list(
        matchups.iter_class4_matchups_by_start(
            challenger, observations, variables, context=context, max_workers=max_workers
        )
    )
    assert len(produced) == len(_FIRST_DAYS)


def test_streamed_write_empty_produces_valid_empty_parquet(tmp_path) -> None:
    output_path = str(tmp_path / "empty.parquet")
    viewer_artifacts.write_matchup_parquet_streamed(iter(()), output_path)
    parquet_file = pyarrow.parquet.ParquetFile(output_path)
    assert parquet_file.schema_arrow.names == viewer_artifacts._MATCHUP_TARGET_SCHEMA.names
    assert parquet_file.metadata.num_rows == 0
