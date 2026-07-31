# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""One-degree water-column store for the viewer (contracts.md §6).

The field pyramid (``oceanbench.pyramids``) serves *maps*: one surface (or 15 m)
layer per fetch, coarsened for pan/zoom. It deliberately drops the depth axis. The
column store is its complement — it serves *profiles and sections*: the full water
column of temperature and salinity, so a click on the map reads a whole vertical
profile (all depths at one point) or a section without any server-side compute. Only
``sea_water_potential_temperature`` and ``sea_water_salinity`` are stored; every other
variable is skipped.

Horizontal grid: every dataset is written on the 1-degree grid, whatever its native
resolution. A profile is a vertical read, not a map: the extra horizontal detail of a
native grid buys the profile view nothing while multiplying the object count and the
per-click download by two orders of magnitude. The regrid reuses the same
``oceanbench.core.interpolate`` target grid and interpolation the ``*_1_degree``
challengers are built from, so a native store and its 1-degree variant land on the
identical grid; a dataset already on that grid is written through untouched. The full
native depth axis is kept.

Layout: ``<slug>.columns.zarr`` sits beside the ``<slug>.zarr`` pyramid, is zarr v2
with consolidated metadata (the flavour the viewer's ``zarr.js`` already reads), and
holds one array per variable with dims ``(start_date, lead_day, depth, latitude,
longitude)``. Each variable is the same quantized ``uint16`` (per-variable
``scale_factor`` / ``add_offset``, land/missing = ``_FillValue`` 65535, DEFLATE) as the
pyramid, so round-trip error stays at or below half a quantization step and the browser
decodes tiles with the platform ``DecompressionStream('deflate')`` — no wasm codec.

Chunking arithmetic
-------------------
The 1-degree global grid is ``start_date=52, lead_day=10, depth=50, latitude=170,
longitude=360``. Two hard targets pull against each other:

* a single profile click (1 start, 1 lead, all depths, 1 point, 1 variable) must
  download at most ~1.5 MB compressed — the client fetches whole chunks, so the
  click cost is the compressed size of the one chunk covering that point;
* the total object (chunk file) count per challenger must stay at or below ~300k,
  because an object store charges per object and the browser lists them.

Depth must be contiguous (a profile is one axis read), so the depth chunk is the full
depth axis — never split. That fixes ``depth`` in every chunk. The two remaining levers
are the horizontal tile ``(latitude, longitude)`` and whether the 10 lead days are
packed into one chunk:

* NOT packing leads (lead chunk = 1) keeps a click small but multiplies the object
  count by 10.
* Packing all 10 leads (lead chunk = 10) cuts the object count 10x. A click then
  downloads all leads of the point — which is exactly what a lead-scrubbing profile
  view wants.

So leads are packed and the chunk is ``[start=1, lead=10, depth=50, lat=64, lon=64]``,
the layout the already-published 1-degree stores use:

* Object count = ``ceil(170/64) x ceil(360/64)`` tiles ``= 3 x 6 = 18`` per
  ``(start, variable)``; ``x 52 starts x 2 variables = 1,872`` chunk objects, far under
  the 300k budget.
* Click chunk raw size = ``1 x 10 x 50 x 64 x 64 x 2 bytes = 4,096,000 B ~= 3.9 MiB``.
  Temperature and salinity are smooth, and every chunk carries land/missing
  ``_FillValue`` runs, so DEFLATE on the quantized ``uint16`` reaches well past 2.6x on
  realistic fields, bringing the compressed chunk under the ~1.5 MB budget. The exact
  ratio is data-dependent, so ``tests/publish/test_column_store.py`` measures it on a
  realistic synthetic column field and asserts the click cost, rather than trusting an
  assumed ratio.

Both levers are parameters (``latitude_tile_size``, ``longitude_tile_size``,
``pack_leads``) so a different grid can be retuned without touching the call site.

Streaming: the year is never materialised. The per-variable quantization range comes
from a single lazy ``dask`` min/max pass (no data pulled into memory), then the store
is written one forecast start at a time — the first start creates the store, each later
start is appended along ``start_date`` — so peak memory is one start's chunks. The
write is I/O bound, so a serial per-start loop is enough; it consumes the same lazy
multi-store challenger dataset the wave's pyramid step does.
"""

from dataclasses import dataclass
from pathlib import Path

import dask
import numpy
import xarray

from oceanbench.core.attribution import copernicus_marine_attribution_attrs
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.interpolate import apply_one_degree_interpolation, one_degree_target_grid
from oceanbench.pyramids.builder import LEAD_DAY_DIMENSION, START_DATE_DIMENSION
from oceanbench.pyramids.quantization import Quantization, quantization_for_range, zarr_encoding

DEPTH_DIMENSION = Dimension.DEPTH.key()

COLUMN_STORE_SUFFIX = ".columns.zarr"

# Temperature and salinity only; every other model variable is intentionally skipped.
COLUMN_VARIABLES = (
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    Variable.SEA_WATER_SALINITY.key(),
)

DEFAULT_LATITUDE_TILE_SIZE = 64
DEFAULT_LONGITUDE_TILE_SIZE = 64


@dataclass(frozen=True)
class ColumnStoreResult:
    zarr_path: str
    variables: tuple[str, ...]
    chunk_shape: dict[str, int]
    object_count: int


def _on_one_degree_grid(selected: xarray.Dataset) -> xarray.Dataset:
    """The two column variables on the 1-degree grid, interpolated only when they are not already.

    Uses the same target grid and interpolation as the ``*_1_degree`` challengers
    (``oceanbench.core.interpolate``), so a native dataset and its 1-degree variant produce the
    same column-store grid. A dataset already on that grid is returned untouched.
    """
    latitude, longitude = one_degree_target_grid(selected)
    already_one_degree = numpy.array_equal(selected[Dimension.LATITUDE.key()].values, latitude) and numpy.array_equal(
        selected[Dimension.LONGITUDE.key()].values, longitude
    )
    if already_one_degree:
        return selected
    return apply_one_degree_interpolation(selected, latitude, longitude)


def _column_variables(dataset: xarray.Dataset) -> xarray.Dataset:
    """Temperature and salinity on the 1-degree grid, renamed to viewer coordinates.

    The forecast dataset carries dims ``(first_day_datetime, lead_day_index, depth,
    latitude, longitude)``; this keeps the full depth axis, coarsens the horizontal grid to
    1 degree and renames the two forecast dimensions to the viewer's ``start_date`` and 1-based
    ``lead_day`` (matching the pyramid layer coordinates).
    """
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
    """Per-variable uint16 quantization from a single lazy min/max pass over the year.

    ``dask.compute`` streams the reduction (identical to the pyramid's range scan), so
    the whole year is never held in memory just to pick the scale/offset.
    """
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
    """Number of chunk objects the store writes: one per chunk per variable."""
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
    from oceanbench.pyramids.builder import _compressor

    compressor = _compressor()
    chunk_shape = tuple(chunk_sizes[dimension] for dimension in columns[next(iter(columns.data_vars))].dims)
    encoding = {}
    for name in columns.data_vars:
        encoding[name] = {**zarr_encoding(quantizations[name], compressor), "chunks": chunk_shape}
    for coordinate_name in set(columns.variables) - set(columns.data_vars):
        encoding[coordinate_name] = {"compressor": compressor}
    return encoding


def _write_root_attributes(store_path: Path, dataset_slug: str | None, year: int | None, chunk_sizes: dict) -> None:
    import zarr

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
    """Write the 1-degree temperature/salinity column store and return its handle.

    ``dataset`` is a forecast dataset (dims ``first_day_datetime, lead_day_index, depth,
    latitude, longitude``); only temperature and salinity are stored, on the 1-degree
    horizontal grid, quantized to ``uint16`` with depth contiguous in every chunk. The
    store is written one forecast start at a time (never materialising the year) as a
    consolidated zarr v2 store at ``output_path`` (conventionally ``<slug>.columns.zarr``).
    """
    import zarr

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
