# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from pathlib import Path
from collections.abc import Callable

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_source import DatasetSource, get_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.local_stage import (
    local_stage_directory,
    open_or_create_local_stage_dataset,
    should_stage_locally,
    write_dataset_to_local_stage,
)

LAGRANGIAN_ROW_LABEL = "Lagrangian trajectory deviation (km) []{surface}"


def mean_weekly_lagrangian_deviations(weekly_deviations: list[numpy.ndarray]) -> numpy.ndarray:
    return pandas.concat(map(pandas.Series, weekly_deviations), axis=1).mean(axis=1).values


def surface_current_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    current_dataset = rename_dataset_with_standard_names(dataset)[
        [
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        ]
    ]
    if Dimension.DEPTH.key() in current_dataset.dims:
        return current_dataset.isel({Dimension.DEPTH.key(): 0}, drop=True)
    return current_dataset


def _first_day_datetime_of_week_dataset(dataset: xarray.Dataset) -> str:
    return str(dataset[Dimension.TIME.key()].values[0])


def _lagrangian_stage_sources(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> tuple[DatasetSource, DatasetSource] | None:
    if not should_stage_locally("challenger") and not should_stage_locally("references"):
        return None
    challenger_source = get_dataset_source(challenger_dataset)
    reference_source = get_dataset_source(reference_dataset)
    if challenger_source is None or reference_source is None:
        return None
    return challenger_source, reference_source


def _lagrangian_stage_directory(dataset_source: DatasetSource, lead_days_count: int) -> Path:
    resolution_suffix = "" if dataset_source.resolution is None else f"-{dataset_source.resolution}"
    return local_stage_directory() / (
        f"lagrangian-{dataset_source.kind}-{dataset_source.name}{resolution_suffix}-{lead_days_count}d"
    )


def _lagrangian_stage_path(
    dataset_source: DatasetSource,
    lead_days_count: int,
    first_day_datetime: str,
) -> Path:
    first_day = datetime.fromisoformat(first_day_datetime).strftime("%Y%m%d")
    return _lagrangian_stage_directory(dataset_source, lead_days_count) / f"{first_day}.zarr"


def _open_staged_lagrangian_dataset(stage_path: Path) -> xarray.Dataset:
    return xarray.open_dataset(stage_path, engine="zarr")


def _open_or_stage_lagrangian_dataset(
    dataset: xarray.Dataset,
    dataset_source: DatasetSource,
    first_day_datetime: str,
) -> xarray.Dataset:
    lead_days_count = dataset.sizes[Dimension.TIME.key()]
    stage_path = _lagrangian_stage_path(dataset_source, lead_days_count, first_day_datetime)
    return open_or_create_local_stage_dataset(
        stage_path,
        open_staged_dataset=_open_staged_lagrangian_dataset,
        build_stage=lambda resolved_stage_path: write_dataset_to_local_stage(
            dataset,
            resolved_stage_path,
            prepare_dataset=surface_current_dataset,
            load_before_write=True,
            clear_chunk_encoding=True,
        ),
    )


def _staged_weekly_lagrangian_deviation(
    challenger_week_dataset: xarray.Dataset,
    reference_week_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    staged_sources,
    compute_weekly_deviation: Callable[[xarray.Dataset, xarray.Dataset, numpy.ndarray, numpy.ndarray], numpy.ndarray],
) -> numpy.ndarray:
    first_day_datetime = _first_day_datetime_of_week_dataset(challenger_week_dataset)
    staged_challenger_week_dataset = _open_or_stage_lagrangian_dataset(
        dataset=challenger_week_dataset,
        dataset_source=staged_sources[0],
        first_day_datetime=first_day_datetime,
    )
    staged_reference_week_dataset = _open_or_stage_lagrangian_dataset(
        dataset=reference_week_dataset,
        dataset_source=staged_sources[1],
        first_day_datetime=first_day_datetime,
    )
    try:
        return compute_weekly_deviation(
            staged_challenger_week_dataset,
            staged_reference_week_dataset,
            latitudes,
            longitudes,
        )
    finally:
        staged_challenger_week_dataset.close()
        staged_reference_week_dataset.close()


def all_weekly_lagrangian_deviations(
    challenger_week_datasets: list[xarray.Dataset],
    reference_week_datasets: list[xarray.Dataset],
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    compute_weekly_deviation: Callable[[xarray.Dataset, xarray.Dataset, numpy.ndarray, numpy.ndarray], numpy.ndarray],
) -> list[numpy.ndarray]:
    if not challenger_week_datasets:
        return []
    staged_sources = _lagrangian_stage_sources(challenger_week_datasets[0], reference_week_datasets[0])
    if staged_sources is None:
        return [
            compute_weekly_deviation(challenger_week_dataset, reference_week_dataset, latitudes, longitudes)
            for challenger_week_dataset, reference_week_dataset in zip(
                challenger_week_datasets,
                reference_week_datasets,
                strict=True,
            )
        ]
    return [
        _staged_weekly_lagrangian_deviation(
            challenger_week_dataset,
            reference_week_dataset,
            latitudes,
            longitudes,
            staged_sources,
            compute_weekly_deviation,
        )
        for challenger_week_dataset, reference_week_dataset in zip(
            challenger_week_datasets,
            reference_week_datasets,
            strict=True,
        )
    ]
