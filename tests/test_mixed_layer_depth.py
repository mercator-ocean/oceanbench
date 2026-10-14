# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import gsw
import numpy
import pytest
import xarray

import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core import mixed_layer_depth
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references import mixed_layer_depth_reference
from oceanbench.core.runtime_configuration import RuntimeConfiguration

GLORYS_DEPTHS = [
    0.494, 1.541, 2.646, 3.819, 5.078, 6.441, 7.93, 9.573, 11.405, 13.467, 15.81, 18.496, 21.599, 25.211, 29.445,
    34.434, 40.344, 47.374, 55.764, 65.807, 77.854, 92.326, 109.729, 130.666, 155.851, 186.126, 222.475, 266.04,
    318.127, 380.213, 453.938, 541.089, 643.567, 763.333, 902.339, 1062.44, 1245.291, 1452.251, 1684.284, 1941.893,
    2225.078, 2533.336, 2865.703, 3220.82, 3597.032, 3992.484, 4405.224, 4833.291, 5274.784, 5727.917,
]  # fmt: skip
COARSE_DEPTHS = [0.494, 47.374, 92.326, 155.851, 222.475, 318.127, 380.213, 453.938, 541.089]
TWELFTH_DEGREE = 1.0 / 12.0


def _profile_dataset(depths: list[float], sigma0_values: list[float]) -> xarray.Dataset:
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    shape = (1, 1, len(depths), 1, 1)
    sigma0 = numpy.array(sigma0_values, dtype=float).reshape(shape)
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                dimension_names,
                numpy.where(numpy.isnan(sigma0), sigma0, 10.0),
            ),
            Variable.SEA_WATER_SALINITY.key(): (dimension_names, numpy.where(numpy.isnan(sigma0), sigma0, 35.0)),
            "sigma0": (dimension_names, sigma0),
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): [0.0],
            Dimension.LONGITUDE.key(): [0.0],
        },
    )


def _mld_value(dataset: xarray.Dataset, monkeypatch) -> float:
    monkeypatch.setattr(
        mixed_layer_depth,
        "_compute_potential_density_anomaly",
        lambda _salinity, _temperature, depth, _longitude, _latitude: dataset["sigma0"].sel(
            {Dimension.DEPTH.key(): depth}
        ),
    )
    mixed_layer_depth_dataset = mixed_layer_depth.compute_mixed_layer_depth(dataset)
    return float(mixed_layer_depth_dataset[Variable.MIXED_LAYER_DEPTH.key()].values.squeeze())


def _linear_sigma0(depths: list[float], gradient: float) -> list[float]:
    return [25.0 + gradient * depth for depth in depths]


@pytest.mark.parametrize("depths", [[0.5 + level for level in range(200)], COARSE_DEPTHS], ids=["fine", "coarse"])
def test_mixed_layer_depth_interpolates_the_analytic_crossing_depth(depths, monkeypatch) -> None:
    # sigma0 = 25 + 0.001 z: reference 25.01 at 10 m, threshold 25.04 reached at 40 m exactly.
    dataset = _profile_dataset(depths, _linear_sigma0(depths, gradient=0.001))

    assert _mld_value(dataset, monkeypatch) == pytest.approx(40.0)


def test_mixed_layer_depth_accepts_chunked_data(monkeypatch) -> None:
    dataset = _profile_dataset(COARSE_DEPTHS, _linear_sigma0(COARSE_DEPTHS, gradient=0.001)).chunk(
        {Dimension.DEPTH.key(): 2}
    )

    assert _mld_value(dataset, monkeypatch) == pytest.approx(40.0)


def test_mixed_layer_depth_reference_density_is_interpolated_at_10_meters(monkeypatch) -> None:
    # Reference 25.01 interpolated between 5 m and 15 m, threshold 25.04 between 30 m (25.03) and 60 m (25.10).
    dataset = _profile_dataset([5.0, 15.0, 30.0, 60.0], [25.00, 25.02, 25.03, 25.10])

    assert _mld_value(dataset, monkeypatch) == pytest.approx(30.0 + 30.0 * 0.01 / 0.07)


def test_mixed_layer_depth_uses_deepest_valid_level_when_threshold_is_never_crossed(monkeypatch) -> None:
    sigma0 = [25.0 if depth < 100.0 else numpy.nan for depth in GLORYS_DEPTHS]

    assert _mld_value(_profile_dataset(GLORYS_DEPTHS, sigma0), monkeypatch) == pytest.approx(92.326)


def test_mixed_layer_depth_uses_deepest_valid_level_of_columns_shallower_than_10_meters(monkeypatch) -> None:
    sigma0 = [25.0 + depth if depth < 6.0 else numpy.nan for depth in GLORYS_DEPTHS]

    assert _mld_value(_profile_dataset(GLORYS_DEPTHS, sigma0), monkeypatch) == pytest.approx(5.078)


def test_mixed_layer_depth_masks_land_points(monkeypatch) -> None:
    dataset = _profile_dataset(COARSE_DEPTHS, [numpy.nan] * len(COARSE_DEPTHS))

    assert numpy.isnan(_mld_value(dataset, monkeypatch))


def test_mixed_layer_depth_is_capped_at_600_meters_when_the_crossing_is_deeper(monkeypatch) -> None:
    # Crossing at 10 + 0.03 / 0.00004 = 760 m.
    dataset = _profile_dataset(GLORYS_DEPTHS, _linear_sigma0(GLORYS_DEPTHS, gradient=0.00004))

    assert _mld_value(dataset, monkeypatch) == 600.0


def test_mixed_layer_depth_interpolates_across_600_meters_with_the_first_level_below(monkeypatch) -> None:
    # Crossing at 10 + 0.03 / 0.00005 = 610 m, between 541.089 m and 643.567 m, then capped.
    dataset = _profile_dataset(GLORYS_DEPTHS, _linear_sigma0(GLORYS_DEPTHS, gradient=0.00005))

    assert _mld_value(dataset, monkeypatch) == 600.0

    # Crossing at 10 + 0.03 / 0.000054 = 565.6 m, found through the 643.567 m level.
    dataset = _profile_dataset(GLORYS_DEPTHS, _linear_sigma0(GLORYS_DEPTHS, gradient=0.000054))

    assert _mld_value(dataset, monkeypatch) == pytest.approx(10.0 + 0.03 / 0.000054)


def test_mixed_layer_depth_of_a_challenger_ending_at_541_meters_is_not_extended(monkeypatch) -> None:
    dataset = _profile_dataset(COARSE_DEPTHS, _linear_sigma0(COARSE_DEPTHS, gradient=0.00004))

    assert _mld_value(dataset, monkeypatch) == pytest.approx(541.089)


def test_mixed_layer_depth_of_a_step_profile_on_coarse_levels_is_shallow(monkeypatch) -> None:
    # Mixed down to 30 m, then 1 kg/m3 denser. With nothing between 0.494 m and 47.374 m the 10 m density
    # is interpolated across the step, and the crossing lands at 10 + 0.03 * (47.374 - 0.494) / 1, about 11 m.
    sigma0 = [25.0 if depth <= 30.0 else 26.0 for depth in COARSE_DEPTHS]

    assert _mld_value(_profile_dataset(COARSE_DEPTHS, sigma0), monkeypatch) == pytest.approx(10.0 + 0.03 * 46.88)


def test_potential_density_anomaly_uses_conservative_temperature_from_potential_temperature() -> None:
    depths = numpy.array([0.494, 47.374, 222.475, 541.089, 1245.291])
    potential_temperature = numpy.array([18.0, 15.0, 10.0, 6.0, 3.5])
    practical_salinity = numpy.array([35.5, 35.4, 35.1, 34.8, 34.7])
    latitude, longitude = 40.0, -30.0
    depth = xarray.DataArray(depths, dims=[Dimension.DEPTH.key()])

    sigma0 = mixed_layer_depth._compute_potential_density_anomaly(
        xarray.DataArray(practical_salinity, dims=[Dimension.DEPTH.key()]),
        xarray.DataArray(potential_temperature, dims=[Dimension.DEPTH.key()]),
        depth,
        xarray.DataArray(longitude),
        xarray.DataArray(latitude),
    )

    pressure = gsw.p_from_z(-depths, latitude)
    absolute_salinity = gsw.SA_from_SP(practical_salinity, pressure, longitude, latitude)
    expected = gsw.sigma0(absolute_salinity, gsw.CT_from_pt(absolute_salinity, potential_temperature))
    numpy.testing.assert_allclose(sigma0.values, expected, rtol=0, atol=1e-10)


def _native_week(first_day_datetime, days_count: int, points_count: int = 4) -> xarray.Dataset:
    depths = numpy.array(GLORYS_DEPTHS)
    latitudes = numpy.arange(points_count) * TWELFTH_DEGREE
    longitudes = numpy.arange(points_count) * TWELFTH_DEGREE
    shape = (days_count, depths.size, latitudes.size, longitudes.size)
    gradients = 0.0005 * (1 + numpy.arange(points_count * points_count)).reshape(points_count, points_count)
    temperature = numpy.broadcast_to(20.0 - depths[None, :, None, None] * gradients[None, None], shape).copy()
    temperature[:, :, 0, 0] = numpy.nan
    salinity = numpy.where(numpy.isnan(temperature), numpy.nan, 35.0)
    dimensions = ["time", "depth", "latitude", "longitude"]
    return xarray.Dataset(
        {
            "thetao": (dimensions, temperature, {"standard_name": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()}),
            "so": (dimensions, salinity, {"standard_name": Variable.SEA_WATER_SALINITY.key()}),
        },
        coords={
            "time": numpy.datetime64(first_day_datetime, "D") + numpy.arange(days_count),
            "depth": depths,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


def _quarter_degree_grid() -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            "lat": ("lat", [0.0, 0.25], {"standard_name": "latitude"}),
            "lon": ("lon", [0.0, 0.25], {"standard_name": "longitude"}),
        }
    )


def _challenger(depths: list[float], lead_days_count: int = 2) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count),
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): numpy.arange(4) * TWELFTH_DEGREE,
            Dimension.LONGITUDE.key(): numpy.arange(4) * TWELFTH_DEGREE,
        }
    )


def _use_runtime_configuration(monkeypatch, **configuration) -> None:
    monkeypatch.setattr(runtime_configuration, "_runtime_configuration", RuntimeConfiguration(**configuration))


def _write_reference_store(root, resolution: str = "twelfth_degree") -> xarray.Dataset:
    store = mixed_layer_depth_reference.build_reference_mixed_layer_depth_week(
        _native_week(numpy.datetime64("2024-01-03"), 3)
    )
    store_name = mixed_layer_depth_reference.REFERENCE_MIXED_LAYER_DEPTH_STORE_NAME.format(
        dataset_name="glorys", resolution=resolution
    )
    store.to_zarr(root / store_name / "20240103.zarr")
    return store


def _forbid_remote_reads(monkeypatch) -> None:
    def fail_remote_read(*_args, **_kwargs):
        raise AssertionError("The reference mixed layer depth must not be read remotely when it is staged.")

    monkeypatch.setattr(mixed_layer_depth_reference, "REFERENCE_MIXED_LAYER_DEPTH_ROOT_URL", "https://example.invalid")
    monkeypatch.setattr(mixed_layer_depth_reference, "open_remote_zarr", fail_remote_read)
    monkeypatch.setattr(mixed_layer_depth_reference, "open_remote_multizarr", fail_remote_read)


def test_reference_builder_matches_the_on_the_fly_mixed_layer_depth() -> None:
    native = _native_week(numpy.datetime64("2024-01-03"), 2)
    on_the_fly = mixed_layer_depth.compute_mixed_layer_depth(native)[Variable.MIXED_LAYER_DEPTH.key()]

    built = mixed_layer_depth_reference.build_reference_mixed_layer_depth_week(native)[Variable.MIXED_LAYER_DEPTH.key()]

    assert built.dtype == numpy.float32
    assert numpy.isnan(built.values[:, 0, 0]).all()
    assert numpy.isfinite(built.values[:, 1:, 1:]).all()
    numpy.testing.assert_allclose(built.values, on_the_fly.values, rtol=1e-6)


def test_quarter_degree_reference_takes_the_coincident_twelfth_degree_points() -> None:
    native = _native_week(numpy.datetime64("2024-01-03"), 1)
    on_the_fly = mixed_layer_depth.compute_mixed_layer_depth(native)[Variable.MIXED_LAYER_DEPTH.key()]

    built = mixed_layer_depth_reference.build_reference_mixed_layer_depth_week(native, _quarter_degree_grid())

    assert built["latitude"].values.tolist() == [0.0, 0.25]
    assert built["longitude"].values.tolist() == [0.0, 0.25]
    numpy.testing.assert_allclose(
        built[Variable.MIXED_LAYER_DEPTH.key()].values,
        on_the_fly.isel(latitude=[0, 3], longitude=[0, 3]).values,
        rtol=1e-6,
    )


def test_reference_mixed_layer_depth_is_read_from_the_precomputed_store(monkeypatch, tmp_path) -> None:
    _use_runtime_configuration(monkeypatch)
    monkeypatch.setattr(mixed_layer_depth_reference, "REFERENCE_MIXED_LAYER_DEPTH_ROOT_URL", str(tmp_path))
    store = _write_reference_store(tmp_path)

    reference = mixed_layer_depth_reference.glorys_reanalysis_mixed_layer_depth_dataset(_challenger(COARSE_DEPTHS))

    assert reference[Variable.MIXED_LAYER_DEPTH.key()].dims == (
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    )
    numpy.testing.assert_array_equal(
        reference[Variable.MIXED_LAYER_DEPTH.key()].values[0],
        store[Variable.MIXED_LAYER_DEPTH.key()].values[:2],
    )


def test_reference_mixed_layer_depth_does_not_depend_on_the_challenger_depth_levels(monkeypatch, tmp_path) -> None:
    _use_runtime_configuration(monkeypatch)
    monkeypatch.setattr(mixed_layer_depth_reference, "REFERENCE_MIXED_LAYER_DEPTH_ROOT_URL", str(tmp_path))
    _write_reference_store(tmp_path)

    references = [
        mixed_layer_depth_reference.glorys_reanalysis_mixed_layer_depth_dataset(_challenger(depths))
        for depths in (COARSE_DEPTHS, GLORYS_DEPTHS)
    ]

    xarray.testing.assert_identical(references[0], references[1])


def test_reference_mixed_layer_depth_is_staged_once_then_read_without_network(monkeypatch, tmp_path) -> None:
    stage_directory = tmp_path / "stage"
    _use_runtime_configuration(monkeypatch, staged_components=("all",), stage_directory=str(stage_directory))
    monkeypatch.setattr(mixed_layer_depth_reference, "REFERENCE_MIXED_LAYER_DEPTH_ROOT_URL", str(tmp_path / "remote"))
    store = _write_reference_store(tmp_path / "remote")

    staged = mixed_layer_depth_reference.glorys_reanalysis_mixed_layer_depth_dataset(_challenger(COARSE_DEPTHS))

    assert (stage_directory / "reference-glorys-twelfth_degree-mixed-layer-depth-2d" / "20240103.zarr").is_dir()
    _forbid_remote_reads(monkeypatch)

    reused = mixed_layer_depth_reference.glorys_reanalysis_mixed_layer_depth_dataset(_challenger(COARSE_DEPTHS))

    xarray.testing.assert_equal(reused, staged)
    numpy.testing.assert_array_equal(
        reused[Variable.MIXED_LAYER_DEPTH.key()].values[0],
        store[Variable.MIXED_LAYER_DEPTH.key()].values[:2],
    )
