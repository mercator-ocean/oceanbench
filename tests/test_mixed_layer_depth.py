# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core import mixed_layer_depth
from oceanbench.core.dataset_utils import Dimension, Variable

DEPTHS = [0.5, 47.0, 600.0, 700.0]


def _dataset(
    temperature_values: list[float],
    density_values: list[float],
    depths: list[float] = DEPTHS,
) -> xarray.Dataset:
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.DEPTH.key(): depths,
        Dimension.LATITUDE.key(): [30.0],
        Dimension.LONGITUDE.key(): [-30.0],
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
                numpy.array(temperature_values, dtype=float).reshape(shape),
            ),
            Variable.SEA_WATER_SALINITY.key(): (dimension_names, numpy.full(shape, 35.0)),
            "potential_density": (dimension_names, numpy.array(density_values, dtype=float).reshape(shape)),
        },
        coords=coordinates,
    )


def _linear_density_dataset(depths: list[float]) -> xarray.Dataset:
    return _dataset(
        temperature_values=[10.0] * len(depths),
        density_values=[25.0 + 0.001 * depth for depth in depths],
        depths=depths,
    )


def _mld_value(dataset: xarray.Dataset, monkeypatch) -> float:
    monkeypatch.setattr(
        mixed_layer_depth,
        "_compute_potential_density_anomaly",
        lambda _salinity, _temperature, depth, _longitude, _latitude: dataset["potential_density"].sel(
            {Dimension.DEPTH.key(): depth}
        ),
    )

    mixed_layer_depth_dataset = mixed_layer_depth.compute_mixed_layer_depth(dataset)

    return float(mixed_layer_depth_dataset[Variable.MIXED_LAYER_DEPTH.key()].values.squeeze())


@pytest.mark.parametrize(
    "depths",
    [
        [0.5, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
        [0.5, 1.5, 2.6, 3.8, 5.1, 6.4, 7.9, 9.6, 11.4, 13.5, 15.8, 18.5, 21.6, 25.2, 29.4, 34.4, 40.3, 47.4],
        [1.0, 10.0, 100.0],
    ],
)
def test_mixed_layer_depth_is_the_interpolated_crossing_below_ten_meters_on_any_grid(depths, monkeypatch) -> None:
    assert _mld_value(_linear_density_dataset(depths), monkeypatch) == pytest.approx(40.0)


def test_mixed_layer_depth_references_the_density_interpolated_at_ten_meters(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[25.0, 25.093, 25.1, 25.2],
    )

    assert _mld_value(dataset, monkeypatch) == pytest.approx(25.0)


def test_mixed_layer_depth_accepts_chunked_data(monkeypatch) -> None:
    dataset = _linear_density_dataset([0.5, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0]).chunk({Dimension.DEPTH.key(): 2})

    assert _mld_value(dataset, monkeypatch) == pytest.approx(40.0)


def test_mixed_layer_depth_takes_the_first_crossing_of_a_profile_with_an_inversion(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0] * 5,
        density_values=[25.0, 25.0, 25.04, 25.02, 25.08],
        depths=[0.5, 10.0, 20.0, 30.0, 40.0],
    )

    assert _mld_value(dataset, monkeypatch) == pytest.approx(17.5)


def test_mixed_layer_depth_ignores_threshold_crossings_below_600_meters(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[25.0, 25.01, 25.02, 25.05],
    )

    assert _mld_value(dataset, monkeypatch) == 600.0


def test_mixed_layer_depth_caps_depth_before_density_computation(monkeypatch) -> None:
    dataset = _linear_density_dataset(DEPTHS)
    density_depths = []

    def compute_potential_density_anomaly(_salinity, _temperature, depth, _longitude, _latitude):
        density_depths.extend(depth.values.tolist())
        return dataset["potential_density"].sel({Dimension.DEPTH.key(): depth})

    monkeypatch.setattr(mixed_layer_depth, "_compute_potential_density_anomaly", compute_potential_density_anomaly)

    mixed_layer_depth.compute_mixed_layer_depth(dataset)

    assert density_depths == [0.5, 47.0, 600.0]


def test_depth_cap_keeps_surface_variables_without_depth_dimension() -> None:
    surface_variable_dimensions = (
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    )
    dataset = _linear_density_dataset(DEPTHS).assign(
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
        density_values=[25.0, 25.01, numpy.nan, numpy.nan],
    )

    assert _mld_value(dataset, monkeypatch) == 47.0


def test_mixed_layer_depth_masks_land_points(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[numpy.nan, numpy.nan, numpy.nan, numpy.nan],
        density_values=[numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    )

    assert numpy.isnan(_mld_value(dataset, monkeypatch))


def test_potential_density_takes_potential_temperature_and_is_independent_of_depth() -> None:
    dataset = _dataset(temperature_values=[20.0, 20.0], density_values=[0.0, 0.0], depths=[0.5, 500.0])

    potential_density = mixed_layer_depth._compute_potential_density_anomaly(
        dataset[Variable.SEA_WATER_SALINITY.key()],
        dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()],
        dataset[Dimension.DEPTH.key()],
        dataset[Dimension.LONGITUDE.key()],
        dataset[Dimension.LATITUDE.key()],
    ).values.squeeze()

    assert potential_density[0] == pytest.approx(24.7656, abs=1e-4)
    assert potential_density[1] == pytest.approx(potential_density[0], abs=2e-3)
