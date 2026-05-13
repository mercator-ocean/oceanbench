# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import hashlib
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
LAGRANGIAN_DOMAIN_GRID_ROUNDING_DECIMALS = 6


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


def _normalised_domain_coordinate_values(coordinate: xarray.DataArray) -> tuple[tuple[int, ...], numpy.ndarray]:
    coordinate_values = numpy.asarray(coordinate.values, dtype=numpy.float64)
    if coordinate_values.size == 0:
        raise ValueError(f"Cannot stage lagrangian data without {coordinate.name} coordinates.")
    if not numpy.all(numpy.isfinite(coordinate_values)):
        raise ValueError(f"Cannot stage lagrangian data with non-finite {coordinate.name} coordinates.")
    rounded_coordinate_values = numpy.round(
        coordinate_values.ravel(),
        LAGRANGIAN_DOMAIN_GRID_ROUNDING_DECIMALS,
    ).astype("<f8", copy=False)
    return coordinate_values.shape, rounded_coordinate_values


def _update_domain_hash(
    domain_hash,
    coordinate_name: str,
    coordinate_shape: tuple[int, ...],
    coordinate_values: numpy.ndarray,
) -> None:
    domain_hash.update(coordinate_name.encode("ascii"))
    domain_hash.update(str(coordinate_shape).encode("ascii"))
    domain_hash.update(coordinate_values.tobytes())


def _lagrangian_domain_stage_variant(dataset: xarray.Dataset) -> str:
    standard_dataset = rename_dataset_with_standard_names(dataset)
    latitude_shape, latitude_values = _normalised_domain_coordinate_values(standard_dataset[Dimension.LATITUDE.key()])
    longitude_shape, longitude_values = _normalised_domain_coordinate_values(
        standard_dataset[Dimension.LONGITUDE.key()]
    )

    domain_hash = hashlib.sha256()
    _update_domain_hash(domain_hash, Dimension.LATITUDE.key(), latitude_shape, latitude_values)
    _update_domain_hash(domain_hash, Dimension.LONGITUDE.key(), longitude_shape, longitude_values)
    return f"domain-{latitude_values.size}x{longitude_values.size}-{domain_hash.hexdigest()[:12]}"


def _lagrangian_stage_directory(
    dataset_source: DatasetSource,
    lead_days_count: int,
    domain_variant: str | None = None,
) -> Path:
    resolution_suffix = "" if dataset_source.resolution is None else f"-{dataset_source.resolution}"
    variant_suffix = "" if dataset_source.variant is None else f"-{dataset_source.variant}"
    domain_variant_suffix = "" if domain_variant is None else f"-{domain_variant}"
    return local_stage_directory() / (
        f"lagrangian-{dataset_source.kind}-{dataset_source.name}"
        f"{resolution_suffix}{variant_suffix}{domain_variant_suffix}-{lead_days_count}d"
    )


def _lagrangian_stage_path(
    dataset_source: DatasetSource,
    lead_days_count: int,
    first_day_datetime: str,
    domain_variant: str | None = None,
) -> Path:
    first_day = datetime.fromisoformat(first_day_datetime).strftime("%Y%m%d")
    return _lagrangian_stage_directory(dataset_source, lead_days_count, domain_variant) / f"{first_day}.zarr"


def _open_staged_lagrangian_dataset(stage_path: Path) -> xarray.Dataset:
    return xarray.open_dataset(stage_path, engine="zarr")


def _open_or_stage_lagrangian_dataset(
    dataset: xarray.Dataset,
    dataset_source: DatasetSource,
    first_day_datetime: str,
) -> xarray.Dataset:
    lead_days_count = dataset.sizes[Dimension.TIME.key()]
    domain_variant = _lagrangian_domain_stage_variant(dataset)
    stage_path = _lagrangian_stage_path(
        dataset_source,
        lead_days_count,
        first_day_datetime,
        domain_variant,
    )
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
