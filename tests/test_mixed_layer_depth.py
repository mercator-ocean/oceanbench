# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core import mixed_layer_depth
from oceanbench.core.dataset_utils import Dimension, Variable


def _dataset(temperature_values: list[float], density_values: list[float]) -> xarray.Dataset:
    depths = [0.5, 47.0, 600.0, 700.0]
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.DEPTH.key(): depths,
        Dimension.LATITUDE.key(): [0.0],
        Dimension.LONGITUDE.key(): [0.0],
    }
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    shape = (1, 1, len(depths), 1, 1)
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                dimension_names,
                numpy.array(temperature_values).reshape(shape),
            ),
            Variable.SEA_WATER_SALINITY.key(): (dimension_names, numpy.zeros(shape)),
            "potential_density": (dimension_names, numpy.array(density_values).reshape(shape)),
        },
        coords=coordinates,
    )


def _mld_value(dataset: xarray.Dataset, monkeypatch) -> float:
    monkeypatch.setattr(
        mixed_layer_depth,
        "_compute_absolute_salinity",
        lambda salinity, _depth, _longitude, _latitude: salinity,
    )
    monkeypatch.setattr(
        mixed_layer_depth,
        "_compute_potential_density",
        lambda _absolute_salinity, _temperature, depth: dataset["potential_density"].sel(
            {Dimension.DEPTH.key(): depth}
        ),
    )

    mixed_layer_depth_dataset = mixed_layer_depth.compute_mixed_layer_depth(dataset)

    return float(mixed_layer_depth_dataset[Variable.MIXED_LAYER_DEPTH.key()].values.squeeze())


def test_mixed_layer_depth_keeps_native_first_threshold_depth(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.04, 1000.05, 1000.06],
    )

    assert _mld_value(dataset, monkeypatch) == 47.0


def test_mixed_layer_depth_accepts_chunked_data(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.04, 1000.05, 1000.06],
    ).chunk({Dimension.DEPTH.key(): 2})

    assert _mld_value(dataset, monkeypatch) == 47.0


def test_mixed_layer_depth_ignores_threshold_crossings_below_600_meters(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.01, 1000.02, 1000.05],
    )

    assert _mld_value(dataset, monkeypatch) == 600.0


def test_mixed_layer_depth_caps_depth_before_density_computation(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.01, 1000.02, 1000.05],
    )
    density_depths = []

    monkeypatch.setattr(
        mixed_layer_depth,
        "_compute_absolute_salinity",
        lambda salinity, _depth, _longitude, _latitude: salinity,
    )

    def compute_potential_density(_absolute_salinity, _temperature, depth):
        density_depths.extend(depth.values.tolist())
        return dataset["potential_density"].sel({Dimension.DEPTH.key(): depth})

    monkeypatch.setattr(mixed_layer_depth, "_compute_potential_density", compute_potential_density)

    mixed_layer_depth.compute_mixed_layer_depth(dataset)

    assert density_depths == [0.5, 47.0, 600.0]


def test_depth_cap_keeps_surface_variables_without_depth_dimension() -> None:
    surface_variable_dimensions = (
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    )
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.01, 1000.02, 1000.05],
    ).assign(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                surface_variable_dimensions,
                numpy.zeros((1, 1, 1, 1)),
            )
        }
    )

    capped_dataset = mixed_layer_depth._cap_depth(dataset)

    assert capped_dataset[Dimension.DEPTH.key()].values.tolist() == [0.5, 47.0, 600.0]
    assert capped_dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].dims == surface_variable_dimensions


def test_mixed_layer_depth_uses_deepest_valid_capped_depth_when_threshold_is_never_crossed(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, numpy.nan, numpy.nan],
        density_values=[1000.0, 1000.01, numpy.nan, numpy.nan],
    )

    assert _mld_value(dataset, monkeypatch) == 47.0


def test_mixed_layer_depth_masks_land_points(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[numpy.nan, numpy.nan, numpy.nan, numpy.nan],
        density_values=[numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    )

    assert numpy.isnan(_mld_value(dataset, monkeypatch))


def _realistic_dataset() -> xarray.Dataset:
    depths = [0.5, 47.0, 92.0, 222.0, 318.0, 541.0]
    latitudes = [-20.0, -0.25, 0.25, 20.0]
    longitudes = [10.0, 11.0, 12.0]
    shape = (2, 3, len(depths), len(latitudes), len(longitudes))
    generator = numpy.random.default_rng(20260801)
    temperature = (20.0 - 0.02 * numpy.array(depths)[:, None, None] + generator.normal(0, 0.1, shape)).astype("float32")
    salinity = (35.0 + 0.001 * numpy.array(depths)[:, None, None] + generator.normal(0, 0.01, shape)).astype("float32")
    temperature[:, :, :, 0, 0] = numpy.nan
    salinity[:, :, :, 0, 0] = numpy.nan
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                temperature,
            ),
            Variable.SEA_WATER_SALINITY.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                salinity,
            ),
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0, 1, 2],
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def test_bounded_density_blocks_do_not_change_the_mixed_layer_depth(monkeypatch) -> None:
    dataset = _realistic_dataset()

    bounded = mixed_layer_depth.compute_mixed_layer_depth(dataset)[Variable.MIXED_LAYER_DEPTH.key()].values

    monkeypatch.setattr(mixed_layer_depth, "_as_density_blocks", lambda data_array: data_array)
    unbounded = mixed_layer_depth.compute_mixed_layer_depth(dataset)[Variable.MIXED_LAYER_DEPTH.key()].values

    numpy.testing.assert_array_equal(bounded, unbounded)


def test_density_blocks_are_bounded_per_start_and_lead_day() -> None:
    temperature = _realistic_dataset()[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]

    blocks = mixed_layer_depth._as_density_blocks(temperature)

    assert blocks.chunksizes[Dimension.FIRST_DAY_DATETIME.key()] == (1, 1)
    assert blocks.chunksizes[Dimension.LEAD_DAY_INDEX.key()] == (1, 1, 1)
    assert blocks.chunksizes[Dimension.DEPTH.key()] == (6,)
