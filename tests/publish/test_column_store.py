# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Column-store writer: quantization round-trip, chunk shape, click cost, readability."""

import numpy
import pytest
import xarray

from oceanbench.core.interpolate import apply_one_degree_interpolation, one_degree_target_grid
from oceanbench.publish.column_store import (
    COLUMN_STORE_SUFFIX,
    DEFAULT_LATITUDE_TILE_SIZE,
    DEFAULT_LONGITUDE_TILE_SIZE,
    build_column_store,
    object_count,
)
from oceanbench.pyramids.builder import _compressor
from oceanbench.pyramids.quantization import quantization_for_range

try:
    import numcodecs
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

_TEMPERATURE = "sea_water_potential_temperature"
_SALINITY = "sea_water_salinity"
_START = "first_day_datetime"
_LEAD = "lead_day_index"


def _realistic_column_field(
    *, starts: int, leads: int, depths: int, latitudes: numpy.ndarray, longitudes: numpy.ndarray, generator, floor
):
    """A smooth, stratified temperature-like field with land NaNs — realistically compressible.

    Large-scale latitude gradient, a warm/cold depth stratification, a few mesoscale wiggles and a
    little noise, then a coastline of NaNs. This is representative of what temperature and salinity
    look like, so the DEFLATE ratio it produces stands in for the real click cost.
    """
    latitude_grid = latitudes[:, None]
    longitude_grid = longitudes[None, :]
    surface = (
        floor
        + 18.0 * numpy.cos(numpy.radians(latitude_grid))
        + 2.0 * numpy.sin(numpy.radians(3 * longitude_grid))
        + 1.5 * numpy.sin(numpy.radians(5 * latitude_grid + 2 * longitude_grid))
    )
    depth_profile = numpy.exp(-numpy.arange(depths) / 6.0)[:, None, None]
    # Spatially coherent mesoscale structure (random-phase sinusoids), not white noise: real native
    # fields are smooth cell-to-cell, which is exactly what governs the DEFLATE ratio.
    wavenumbers = generator.uniform(4.0, 12.0, size=(3, 2))
    phases = generator.uniform(0.0, 2 * numpy.pi, size=3)
    mesoscale = sum(
        0.4
        * numpy.sin(numpy.radians(wavenumbers[k, 0] * latitude_grid) + phases[k])
        * numpy.cos(numpy.radians(wavenumbers[k, 1] * longitude_grid))
        for k in range(3)
    )
    field = numpy.empty((starts, leads, depths, latitudes.size, longitudes.size), dtype=numpy.float32)
    for start_index in range(starts):
        for lead_index in range(leads):
            drift = 0.3 * start_index + 0.1 * lead_index
            column = (surface[None, :, :] + mesoscale[None, :, :] + drift) * depth_profile
            field[start_index, lead_index] = column
    field[..., :3, :4] = numpy.nan
    return field


def _synthetic_challenger(
    *, starts=4, leads=10, depths=50, latitude_size=160, longitude_size=200, seed=7
) -> xarray.Dataset:
    generator = numpy.random.default_rng(seed)
    latitudes = numpy.linspace(-80.0, 90.0, latitude_size).astype("float32")
    longitudes = numpy.linspace(-180.0, 179.9, longitude_size).astype("float32")
    depth = numpy.linspace(0.5, 5700.0, depths).astype("float32")
    dims = (_START, _LEAD, "depth", "latitude", "longitude")
    temperature = _realistic_column_field(
        starts=starts,
        leads=leads,
        depths=depths,
        latitudes=latitudes,
        longitudes=longitudes,
        generator=generator,
        floor=2.0,
    )
    salinity = _realistic_column_field(
        starts=starts,
        leads=leads,
        depths=depths,
        latitudes=latitudes,
        longitudes=longitudes,
        generator=generator,
        floor=34.0,
    )
    coordinates = {
        _START: numpy.array(["2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24"][:starts], dtype="datetime64[ns]"),
        _LEAD: numpy.arange(leads),
        "depth": depth,
        "latitude": latitudes,
        "longitude": longitudes,
    }
    return xarray.Dataset(
        {
            _TEMPERATURE: (dims, temperature),
            _SALINITY: (dims, salinity),
            # A surface-only and a 3D variable that must both be skipped by the writer.
            "sea_surface_height_above_geoid": ((_START, _LEAD, "latitude", "longitude"), temperature[:, :, 0]),
            "eastward_sea_water_velocity": (dims, salinity),
        },
        coords=coordinates,
    ).chunk({_START: 1, _LEAD: leads, "depth": depths, "latitude": 80, "longitude": 100})


def test_quantization_round_trip_within_half_step():
    generator = numpy.random.default_rng(3)
    values = generator.normal(15.0, 5.0, size=200_000).astype("float64")
    quantization = quantization_for_range(float(values.min()), float(values.max()))
    stored = numpy.round((values - quantization.add_offset) / quantization.scale_factor).astype("uint16")
    decoded = stored.astype("float64") * quantization.scale_factor + quantization.add_offset
    assert numpy.max(numpy.abs(decoded - values)) <= quantization.quantization_step / 2 + 1e-9


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_writes_only_temperature_and_salinity_on_the_one_degree_grid(tmp_path):
    dataset = _synthetic_challenger()
    result = build_column_store(dataset, output_path=str(tmp_path / f"synthetic{COLUMN_STORE_SUFFIX}"))
    assert set(result.variables) == {_TEMPERATURE, _SALINITY}
    store = xarray.open_zarr(result.zarr_path, consolidated=True)
    assert set(store.data_vars) == {_TEMPERATURE, _SALINITY}
    expected_latitude, expected_longitude = one_degree_target_grid(dataset)
    assert numpy.array_equal(store["latitude"].values, expected_latitude)
    assert numpy.array_equal(store["longitude"].values, expected_longitude)
    # The native depth axis is kept whole; only the horizontal grid is coarsened.
    assert store.sizes["depth"] == dataset.sizes["depth"]
    assert numpy.array_equal(store["depth"].values, dataset["depth"].values)
    assert store.sizes["start_date"] == dataset.sizes[_START]
    assert list(store["lead_day"].values) == list(range(1, dataset.sizes[_LEAD] + 1))


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_dataset_already_at_one_degree_is_written_through_unchanged(tmp_path):
    dataset = _synthetic_challenger(starts=1, leads=2, depths=4, latitude_size=8, longitude_size=8)
    latitude = numpy.arange(-3.5, 4.5, 1.0)
    longitude = numpy.arange(-3.5, 4.5, 1.0)
    dataset = dataset.assign_coords(latitude=latitude, longitude=longitude)
    result = build_column_store(dataset, output_path=str(tmp_path / f"onedeg{COLUMN_STORE_SUFFIX}"))
    store = xarray.open_zarr(result.zarr_path, consolidated=True)
    assert numpy.array_equal(store["latitude"].values, latitude)
    assert numpy.array_equal(store["longitude"].values, longitude)
    # No interpolation ran, so the values are the source values to within the quantization step.
    original = dataset[_TEMPERATURE].isel({_START: 0, _LEAD: 0}).values
    decoded = store[_TEMPERATURE].isel(start_date=0, lead_day=0).values
    step = float(store[_TEMPERATURE].encoding["scale_factor"])
    finite = numpy.isfinite(original)
    assert numpy.all(numpy.abs(decoded[finite] - original[finite]) <= step)


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_chunk_shape_depth_contiguous_and_leads_packed(tmp_path):
    dataset = _synthetic_challenger()
    result = build_column_store(dataset, output_path=str(tmp_path / f"synthetic{COLUMN_STORE_SUFFIX}"))
    array = zarr.open_group(result.zarr_path, mode="r")[_TEMPERATURE]
    start_chunk, lead_chunk, depth_chunk, latitude_chunk, longitude_chunk = array.chunks
    assert start_chunk == 1
    assert lead_chunk == dataset.sizes[_LEAD]  # all leads packed together
    assert depth_chunk == dataset.sizes["depth"]  # depth contiguous: one chunk covers the column
    assert latitude_chunk == DEFAULT_LATITUDE_TILE_SIZE
    assert longitude_chunk == DEFAULT_LONGITUDE_TILE_SIZE


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_round_trip_through_store_within_half_step(tmp_path):
    dataset = _synthetic_challenger(starts=2, depths=20, latitude_size=80, longitude_size=100)
    result = build_column_store(dataset, output_path=str(tmp_path / f"synthetic{COLUMN_STORE_SUFFIX}"))
    store = xarray.open_zarr(result.zarr_path, consolidated=True)
    latitude, longitude = one_degree_target_grid(dataset)
    interpolated = apply_one_degree_interpolation(dataset[[_TEMPERATURE, _SALINITY]], latitude, longitude)
    for name in (_TEMPERATURE, _SALINITY):
        # The store is written on the 1-degree grid, so the round-trip bound is measured against
        # the interpolated field, not the native one.
        original = interpolated[name].rename({_START: "start_date", _LEAD: "lead_day"}).values
        decoded = store[name].values
        finite = numpy.isfinite(original)
        step = float(store[name].encoding["scale_factor"])
        # Half a quantization step is the ideal float64 bound (proven exactly in
        # test_quantization_round_trip_within_half_step). xarray's CF encoder evaluates
        # round((value - add_offset) / scale_factor) in the source float32, so the store adds the
        # float32 rounding of that expression, bounded by the data span times float32 eps.
        span = float(original[finite].max() - original[finite].min())
        float32_slack = span * numpy.finfo(numpy.float32).eps * 2
        assert numpy.all(numpy.abs(decoded[finite] - original[finite]) <= step / 2 + float32_slack + 1e-9)
        # Land NaNs survive the fill-value round trip.
        assert numpy.all(numpy.isnan(decoded[~finite]))


def test_object_count_and_click_cost_arithmetic_meets_targets():
    """The global 1-degree grid stays within both hard targets by construction."""
    sizes = {"start_date": 52, "lead_day": 10, "depth": 50, "latitude": 170, "longitude": 360}
    chunk_sizes = {"start_date": 1, "lead_day": 10, "depth": 50, "latitude": 64, "longitude": 64}
    latitude_tiles = -(-sizes["latitude"] // chunk_sizes["latitude"])
    longitude_tiles = -(-sizes["longitude"] // chunk_sizes["longitude"])
    count = sizes["start_date"] * 1 * 1 * latitude_tiles * longitude_tiles * 2
    assert (latitude_tiles, longitude_tiles) == (3, 6)
    assert count == 1_872
    assert count <= 300_000
    # Uncompressed click chunk (pre-DEFLATE) — the compressed number is measured in the next test.
    raw_bytes = chunk_sizes["lead_day"] * chunk_sizes["depth"] * chunk_sizes["latitude"] * chunk_sizes["longitude"] * 2
    assert raw_bytes == 4_096_000


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_object_count_matches_written_store(tmp_path):
    dataset = _synthetic_challenger()
    result = build_column_store(dataset, output_path=str(tmp_path / f"synthetic{COLUMN_STORE_SUFFIX}"))
    store = xarray.open_zarr(result.zarr_path, consolidated=True)
    assert result.object_count == object_count(store, result.chunk_shape)
    # The 160x200 native grid becomes 170x360 at 1 degree:
    # 4 starts x (ceil(170/64)=3 x ceil(360/64)=6) tiles x 2 vars = 144.
    assert result.object_count == 4 * 3 * 6 * 2


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_profile_click_cost_under_budget_on_realistic_field():
    """Measure the actual compressed size of one native click chunk on a realistic field.

    The click chunk is [start=1, lead=10, depth=50, lat=64, lon=64]. We quantize a realistic
    temperature field to uint16 (as the writer does) and DEFLATE exactly that chunk, then assert
    the compressed size is under the ~1.5 MB single-click budget.
    """
    generator = numpy.random.default_rng(11)
    latitudes = numpy.linspace(20.0, 36.0, 64).astype("float32")
    longitudes = numpy.linspace(-40.0, -24.0, 64).astype("float32")
    field = _realistic_column_field(
        starts=1, leads=10, depths=50, latitudes=latitudes, longitudes=longitudes, generator=generator, floor=2.0
    )[0]
    quantization = quantization_for_range(float(numpy.nanmin(field)), float(numpy.nanmax(field)))
    stored = numpy.where(
        numpy.isfinite(field),
        numpy.round((field - quantization.add_offset) / quantization.scale_factor),
        quantization.fill_value,
    ).astype("uint16")
    compressed = _compressor().encode(stored)
    compressed_megabytes = len(compressed) / 1e6
    assert stored.nbytes == 4_096_000  # raw chunk size, matches the docstring arithmetic
    assert compressed_megabytes <= 1.5, f"click chunk compressed to {compressed_megabytes:.2f} MB, over budget"


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_store_is_browser_flavour_consolidated_and_plain_zarr_readable(tmp_path):
    dataset = _synthetic_challenger(starts=2, depths=10, latitude_size=80, longitude_size=100)
    result = build_column_store(
        dataset, output_path=str(tmp_path / f"synthetic{COLUMN_STORE_SUFFIX}"), dataset_slug="synthetic", year=2024
    )
    # Consolidated metadata present (viewer opens the store from .zmetadata).
    group = zarr.open_consolidated(result.zarr_path)
    assert set(group.array_keys()) >= {_TEMPERATURE, _SALINITY}
    assert group.attrs["dataset"] == "synthetic"
    assert group.attrs["year"] == 2024
    # zarr v2 (the flavour the viewer's zarr.js reads), DEFLATE-compressed uint16.
    temperature = group[_TEMPERATURE]
    assert temperature.dtype == numpy.dtype("uint16")
    assert isinstance(temperature.compressor, numcodecs.Zlib)
    # Plain xarray open (no special codecs) decodes the quantized values back to float.
    store = xarray.open_zarr(result.zarr_path, consolidated=True)
    assert store[_TEMPERATURE].dtype.kind == "f"
