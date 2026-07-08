# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The parallel, streamed match-up path must reproduce the serial whole-frame path exactly.

The parallelisation splits the year across forecast start dates and the write streams one start
at a time; because ``start_date`` is the leading sort key and the starts are disjoint, the served
parquet must be identical in row content, row order and row-group layout to the previous
whole-frame :func:`write_matchup_parquet`. These tests pin that equality (not merely an equal
RMSD) on a synthetic multi-start dataset, exercising both the serial (one worker) and the SPAWNED
parallel path.

The synthetic datasets use sea-water potential temperature at the surface: its match-up value is a
purely numerical interpolation of the model field with no dependency on external reference data (no
mean-dynamic-topography lookup, unlike sea-surface height), so a spawned worker — which re-imports
modules from scratch and cannot inherit a monkeypatch — computes exactly the same values as the
in-process serial path. That equivalence is precisely what lets ``max_workers=2`` prove the spawned
re-open path reproduces the serial path byte for byte.
"""

import numpy
import pandas
import pyarrow.parquet
import pytest

import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.publish import viewer_artifacts
from oceanbench.runner import matchups
from oceanbench.runner.records import RunContext

_FIRST_DAYS = numpy.array(["2024-01-03", "2024-01-10", "2024-01-17"], dtype="datetime64[ns]")
_MATCHUP_VARIABLE = Variable.SEA_WATER_POTENTIAL_TEMPERATURE


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
    depths = numpy.array([0.0])
    latitudes = numpy.array([0.0, 1.0, 2.0, 3.0])
    longitudes = numpy.array([10.0, 11.0, 12.0, 13.0])
    generator = numpy.random.default_rng(0)
    shape = (len(_FIRST_DAYS), len(lead_days), len(depths), len(latitudes), len(longitudes))
    return xarray.Dataset(
        {
            _MATCHUP_VARIABLE.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                generator.normal(size=shape),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): _FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.DEPTH.key(): depths,
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
            _MATCHUP_VARIABLE.key(): (observation_dimension, numpy.array(values)),
        }
    )


def _read(path: str) -> pandas.DataFrame:
    return pyarrow.parquet.ParquetFile(path).read().to_pandas()


def _row_group_row_counts(path: str) -> list[int]:
    metadata = pyarrow.parquet.ParquetFile(path).metadata
    return [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]


@pytest.mark.parametrize("max_workers", [1, 2])
def test_streamed_parallel_matches_serial_whole_frame(tmp_path, max_workers) -> None:
    challenger = _challenger()
    observations = _observations()
    variables = [_MATCHUP_VARIABLE]
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


def test_parallel_path_uses_spawn_not_fork(monkeypatch) -> None:
    challenger = _challenger()
    observations = _observations()
    variables = [_MATCHUP_VARIABLE]
    context = _context()

    requested_start_methods = []
    real_get_context = matchups.multiprocessing.get_context

    def recording_get_context(method=None):
        requested_start_methods.append(method)
        return real_get_context(method)

    monkeypatch.setattr(matchups.multiprocessing, "get_context", recording_get_context)

    produced = list(
        matchups.iter_class4_matchups_by_start(
            challenger, observations, variables, context=context, max_workers=2
        )
    )
    assert len(produced) == len(_FIRST_DAYS)
    # The parallel executor must be built on the spawn context; fork must never be requested.
    assert "spawn" in requested_start_methods
    assert "fork" not in requested_start_methods


def test_store_backed_datasets_reconstruct_without_temp_copy(tmp_path, monkeypatch) -> None:
    """A store-backed dataset must take the reconstructable-store spec, never a temp materialisation.

    The datasets are written once to zarr stores and re-opened through ``open_zarr`` (the same shape
    as the production remote-zarr opens, which record ``encoding['source']``), then subset with
    ``isel`` exactly like ``subset_dataset_to_region``/start limits do. The parallel path must
    reconstruct the workers' datasets from those ORIGINAL stores: materialising is forbidden
    outright by the spy, and the spawned output must still equal the serial output exactly.
    """
    challenger = _challenger()
    observations = _observations().assign_coords(obs=numpy.arange(45))
    challenger_store = str(tmp_path / "challenger-source.zarr")
    observation_store = str(tmp_path / "observations-source.zarr")
    challenger.to_zarr(challenger_store, consolidated=True)
    observations.to_zarr(observation_store, consolidated=True)
    # Re-open from the stores and subset like production does (region box / start limit indexing).
    stored_challenger = xarray.open_zarr(challenger_store).isel({Dimension.LATITUDE.key(): slice(0, 4)})
    stored_observations = xarray.open_zarr(observation_store).isel(obs=slice(0, 45))
    variables = [_MATCHUP_VARIABLE]
    context = _context()

    def forbidden_materialise(dataset, directory, name):
        raise AssertionError("store-backed datasets must not be materialised to a temporary copy")

    monkeypatch.setattr(matchups, "_materialise_dataset_spec", forbidden_materialise)

    # Both datasets must yield a validated reconstructable spec pointing at the ORIGINAL stores.
    challenger_spec = matchups._store_backed_spec(stored_challenger)
    observation_spec = matchups._store_backed_spec(stored_observations)
    assert challenger_spec is not None and challenger_spec.source == challenger_store
    assert observation_spec is not None and observation_spec.source == observation_store

    # Record the executor context so a silent degrade-to-serial cannot fake a pass.
    requested_start_methods = []
    real_get_context = matchups.multiprocessing.get_context

    def recording_get_context(method=None):
        requested_start_methods.append(method)
        return real_get_context(method)

    monkeypatch.setattr(matchups.multiprocessing, "get_context", recording_get_context)

    serial = list(
        matchups.iter_class4_matchups_by_start(
            stored_challenger, stored_observations, variables, context=context, max_workers=1
        )
    )
    parallel = list(
        matchups.iter_class4_matchups_by_start(
            stored_challenger, stored_observations, variables, context=context, max_workers=2
        )
    )
    assert "spawn" in requested_start_methods
    assert len(parallel) == len(serial) == len(_FIRST_DAYS)
    for serial_frame, parallel_frame in zip(serial, parallel):
        pandas.testing.assert_frame_equal(parallel_frame, serial_frame)


def test_oversized_unreconstructable_dataset_degrades_to_serial(monkeypatch) -> None:
    challenger = _challenger()
    observations = _observations()
    variables = [_MATCHUP_VARIABLE]
    context = _context()
    # In-memory datasets have no store to reconstruct from; above the guard they must not be
    # copied to a temporary zarr but must still produce the frames (serially).
    monkeypatch.setattr(matchups, "_MATERIALISE_MAXIMUM_BYTES", 0)

    def forbidden_materialise(dataset, directory, name):
        raise AssertionError("an oversized dataset must never be materialised")

    monkeypatch.setattr(matchups, "_materialise_dataset_spec", forbidden_materialise)
    produced = list(
        matchups.iter_class4_matchups_by_start(
            challenger, observations, variables, context=context, max_workers=2
        )
    )
    assert len(produced) == len(_FIRST_DAYS)


def test_streamed_write_empty_produces_valid_empty_parquet(tmp_path) -> None:
    output_path = str(tmp_path / "empty.parquet")
    viewer_artifacts.write_matchup_parquet_streamed(iter(()), output_path)
    parquet_file = pyarrow.parquet.ParquetFile(output_path)
    assert parquet_file.schema_arrow.names == viewer_artifacts._MATCHUP_TARGET_SCHEMA.names
    assert parquet_file.metadata.num_rows == 0
