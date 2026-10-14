# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import gsw
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


PROFILE_DEPTHS = [0.494, 5.078, 9.573, 18.496, 29.445, 47.374, 77.854, 130.666, 222.475, 380.213, 541.089]
PROFILE_POTENTIAL_TEMPERATURE = [18.0, 18.0, 18.0, 17.99, 17.98, 16.5, 15.0, 13.0, 11.0, 9.0, 7.5]
PROFILE_SALINITY = [36.2, 36.2, 36.2, 36.2, 36.2, 36.1, 36.0, 35.8, 35.6, 35.3, 35.1]
PROFILE_LATITUDE = 30.0
PROFILE_LONGITUDE = -40.0


def _profile_dataset() -> xarray.Dataset:
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    shape = (1, 1, len(PROFILE_DEPTHS), 1, 1)
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                dimension_names,
                numpy.array(PROFILE_POTENTIAL_TEMPERATURE).reshape(shape),
            ),
            Variable.SEA_WATER_SALINITY.key(): (dimension_names, numpy.array(PROFILE_SALINITY).reshape(shape)),
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.DEPTH.key(): PROFILE_DEPTHS,
            Dimension.LATITUDE.key(): [PROFILE_LATITUDE],
            Dimension.LONGITUDE.key(): [PROFILE_LONGITUDE],
        },
    )


def test_density_is_sigma0_from_conservative_temperature() -> None:
    dataset = _profile_dataset()
    depth = dataset[Dimension.DEPTH.key()]
    absolute_salinity = mixed_layer_depth._compute_absolute_salinity(
        dataset[Variable.SEA_WATER_SALINITY.key()],
        depth,
        dataset[Dimension.LONGITUDE.key()],
        dataset[Dimension.LATITUDE.key()],
    )
    density = mixed_layer_depth._compute_potential_density(
        absolute_salinity, dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()], depth
    )

    depths = numpy.array(PROFILE_DEPTHS)
    expected_absolute_salinity = gsw.SA_from_SP(
        numpy.array(PROFILE_SALINITY), gsw.p_from_z(-depths, PROFILE_LATITUDE), PROFILE_LONGITUDE, PROFILE_LATITUDE
    )
    expected_density = gsw.sigma0(
        expected_absolute_salinity,
        gsw.CT_from_pt(expected_absolute_salinity, numpy.array(PROFILE_POTENTIAL_TEMPERATURE)),
    )

    numpy.testing.assert_allclose(density.values.squeeze(), expected_density, rtol=0, atol=1e-10)


def test_mixed_layer_depth_keeps_the_crossing_level_of_the_previous_density() -> None:
    depths = numpy.array(PROFILE_DEPTHS)
    previous_density = gsw.pot_rho_t_exact(
        gsw.SA_from_SP(numpy.array(PROFILE_SALINITY), depths, PROFILE_LONGITUDE, PROFILE_LATITUDE),
        numpy.array(PROFILE_POTENTIAL_TEMPERATURE),
        depths,
        0,
    )
    previous_crossing_depth = depths[numpy.argmax(previous_density - previous_density[0] >= 0.03)]

    mixed_layer_depth_dataset = mixed_layer_depth.compute_mixed_layer_depth(_profile_dataset())

    assert previous_crossing_depth == 47.374
    assert float(mixed_layer_depth_dataset[Variable.MIXED_LAYER_DEPTH.key()].values.squeeze()) == 47.374
