# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from pathlib import Path

import dask
import numpy
import xarray
import zarr

from oceanbench.core.attribution import copernicus_marine_attribution_attrs
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.interpolate import apply_one_degree_interpolation, one_degree_target_grid
from oceanbench.pyramids.builder import LEAD_DAY_DIMENSION, START_DATE_DIMENSION, deflate_compressor
from oceanbench.pyramids.quantization import Quantization, quantization_for_range, zarr_encoding

DEPTH_DIMENSION = Dimension.DEPTH.key()

COLUMN_STORE_SUFFIX = ".columns.zarr"

# The water-column store serves the profile and section views: the full depth axis of temperature
# and salinity on the one-degree grid, quantized like the pyramid.
COLUMN_VARIABLES = (
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    Variable.SEA_WATER_SALINITY.key(),
)

# A chunk holds one start, all lead days, the full depth axis and a 64 x 64 tile, so a profile click is
# one fetch of about 1 MB and a global year stays near 2,000 objects per challenger.
DEFAULT_LATITUDE_TILE_SIZE = 64
DEFAULT_LONGITUDE_TILE_SIZE = 64


@dataclass(frozen=True)
class ColumnStoreResult:
    zarr_path: str
    variables: tuple[str, ...]
    chunk_shape: dict[str, int]
    object_count: int


def _on_one_degree_grid(selected: xarray.Dataset) -> xarray.Dataset:
    """The column variables on the grid of the ``*_1_degree`` challengers, interpolated only when needed."""
    latitude, longitude = one_degree_target_grid(selected)
    already_one_degree = numpy.array_equal(selected[Dimension.LATITUDE.key()].values, latitude) and numpy.array_equal(
        selected[Dimension.LONGITUDE.key()].values, longitude
    )
    if already_one_degree:
        return selected
    one_start_and_lead_chunks = {Dimension.FIRST_DAY_DATETIME.key(): 1, Dimension.LEAD_DAY_INDEX.key(): 1}
    return apply_one_degree_interpolation(selected.chunk(one_start_and_lead_chunks), latitude, longitude)


def _column_variables(dataset: xarray.Dataset) -> xarray.Dataset:
    """Temperature and salinity on the 1-degree grid, with the viewer ``start_date`` and 1-based ``lead_day``."""
    standardised = rename_dataset_with_standard_names(dataset)
    present = [name for name in COLUMN_VARIABLES if name in standardised.data_vars]
    if not present:
        raise ValueError(
            f"dataset carries none of the column-store variables {list(COLUMN_VARIABLES)}; "
            f"has {sorted(standardised.data_vars)}"
        )
    for name in present:
        if DEPTH_DIMENSION not in standardised[name].dims:
            raise ValueError(f"column-store variable {name!r} has no depth axis; got dims {standardised[name].dims}")
    selected = _on_one_degree_grid(standardised[present])
    lead_days = selected[Dimension.LEAD_DAY_INDEX.key()].values + 1
    return (
        selected.rename(
            {
                Dimension.FIRST_DAY_DATETIME.key(): START_DATE_DIMENSION,
                Dimension.LEAD_DAY_INDEX.key(): LEAD_DAY_DIMENSION,
            }
        )
        .assign_coords({LEAD_DAY_DIMENSION: lead_days})
        .reset_coords(drop=True)
    )


def _quantizations(columns: xarray.Dataset) -> dict[str, Quantization]:
    minimums, maximums = dask.compute(columns.min(skipna=True), columns.max(skipna=True))
    return {
        name: quantization_for_range(float(minimums[name].values), float(maximums[name].values))
        for name in columns.data_vars
    }


def _chunk_sizes(columns: xarray.Dataset, latitude_tile_size: int, longitude_tile_size: int, pack_leads: bool) -> dict:
    lead_size = int(columns.sizes[LEAD_DAY_DIMENSION])
    depth_size = int(columns.sizes[DEPTH_DIMENSION])
    latitude_size = int(columns.sizes[Dimension.LATITUDE.key()])
    longitude_size = int(columns.sizes[Dimension.LONGITUDE.key()])
    return {
        START_DATE_DIMENSION: 1,
        LEAD_DAY_DIMENSION: lead_size if pack_leads else 1,
        DEPTH_DIMENSION: depth_size,
        Dimension.LATITUDE.key(): min(latitude_tile_size, latitude_size),
        Dimension.LONGITUDE.key(): min(longitude_tile_size, longitude_size),
    }


def _tiles_along(size: int, tile: int) -> int:
    return -(-size // tile)


def object_count(columns: xarray.Dataset, chunk_sizes: dict) -> int:
    start_chunks = _tiles_along(int(columns.sizes[START_DATE_DIMENSION]), chunk_sizes[START_DATE_DIMENSION])
    lead_chunks = _tiles_along(int(columns.sizes[LEAD_DAY_DIMENSION]), chunk_sizes[LEAD_DAY_DIMENSION])
    depth_chunks = _tiles_along(int(columns.sizes[DEPTH_DIMENSION]), chunk_sizes[DEPTH_DIMENSION])
    latitude_chunks = _tiles_along(int(columns.sizes[Dimension.LATITUDE.key()]), chunk_sizes[Dimension.LATITUDE.key()])
    longitude_chunks = _tiles_along(
        int(columns.sizes[Dimension.LONGITUDE.key()]), chunk_sizes[Dimension.LONGITUDE.key()]
    )
    chunks_per_variable = start_chunks * lead_chunks * depth_chunks * latitude_chunks * longitude_chunks
    return chunks_per_variable * len(columns.data_vars)


def _encoding(columns: xarray.Dataset, quantizations: dict[str, Quantization], chunk_sizes: dict) -> dict:
    compressor = deflate_compressor()
    chunk_shape = tuple(chunk_sizes[dimension] for dimension in columns[next(iter(columns.data_vars))].dims)
    encoding = {}
    for name in columns.data_vars:
        encoding[name] = {**zarr_encoding(quantizations[name], compressor), "chunks": chunk_shape}
    for coordinate_name in set(columns.variables) - set(columns.data_vars):
        encoding[coordinate_name] = {"compressor": compressor}
    return encoding


def _write_root_attributes(store_path: Path, dataset_slug: str | None, year: int | None, chunk_sizes: dict) -> None:
    root_group = zarr.open_group(str(store_path), mode="a")
    root_group.attrs.update(copernicus_marine_attribution_attrs())
    root_group.attrs["chunk_sizes"] = {key: int(value) for key, value in chunk_sizes.items()}
    if dataset_slug is not None:
        root_group.attrs["dataset"] = dataset_slug
    if year is not None:
        root_group.attrs["year"] = int(year)


def build_column_store(
    dataset: xarray.Dataset,
    *,
    output_path: str,
    dataset_slug: str | None = None,
    year: int | None = None,
    latitude_tile_size: int = DEFAULT_LATITUDE_TILE_SIZE,
    longitude_tile_size: int = DEFAULT_LONGITUDE_TILE_SIZE,
    pack_leads: bool = True,
) -> ColumnStoreResult:
    """
    Write the one-degree temperature and salinity store, one forecast start at a time.

    The store is consolidated zarr v2 with one uint16 array per variable of dims
    ``(start_date, lead_day, depth, latitude, longitude)``, depth contiguous in every chunk.
    """
    columns = _column_variables(dataset)
    chunk_sizes = _chunk_sizes(columns, latitude_tile_size, longitude_tile_size, pack_leads)
    quantizations = _quantizations(columns)
    encoding = _encoding(columns, quantizations, chunk_sizes)

    store_path = Path(output_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    start_count = int(columns.sizes[START_DATE_DIMENSION])
    for start_index in range(start_count):
        start_slice = columns.isel({START_DATE_DIMENSION: [start_index]}).chunk(chunk_sizes)
        for variable_name in start_slice.variables:
            start_slice[variable_name].encoding = {}
        if start_index == 0:
            start_slice.to_zarr(store_path, mode="w", encoding=encoding, consolidated=False)
        else:
            start_slice.to_zarr(store_path, mode="a", append_dim=START_DATE_DIMENSION, consolidated=False)

    _write_root_attributes(store_path, dataset_slug, year, chunk_sizes)
    zarr.consolidate_metadata(str(store_path))

    return ColumnStoreResult(
        zarr_path=str(store_path),
        variables=tuple(columns.data_vars),
        chunk_shape=chunk_sizes,
        object_count=object_count(columns, chunk_sizes),
    )
