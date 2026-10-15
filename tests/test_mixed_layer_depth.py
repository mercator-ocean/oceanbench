# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import gsw
import numpy
import pytest
import xarray

from oceanbench.core import metrics, mixed_layer_depth
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


STANDARD_50_DEPTHS = [
    0.494, 1.541, 2.646, 3.819, 5.078, 6.441, 7.93, 9.573, 11.405, 13.467,
    15.81, 18.496, 21.599, 25.211, 29.445, 34.434, 40.344, 47.374, 55.764, 65.807,
    77.854, 92.326, 109.729, 130.666, 155.851, 186.126, 222.475, 266.04, 318.127, 380.213,
    453.938, 541.089, 643.567, 763.333, 902.339, 1062.44, 1245.291, 1452.251, 1684.284, 1941.893,
    2225.078, 2533.336, 2865.703, 3220.82, 3597.032, 3992.484, 4405.224, 4833.291, 5274.784, 5727.917,
]  # fmt: skip
NINE_CHALLENGER_DEPTHS = [0.494, 47.374, 92.326, 155.851, 222.475, 318.127, 380.213, 453.938, 541.089]


def _stratified_dataset(depths: list[float]) -> xarray.Dataset:
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    depth_values = numpy.array(depths)
    shape = (1, 1, len(depths), 1, 1)
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                dimension_names,
                (20.0 - 0.02 * depth_values).reshape(shape),
            ),
            Variable.SEA_WATER_SALINITY.key(): (dimension_names, numpy.full(shape, 35.0)),
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): [PROFILE_LATITUDE],
            Dimension.LONGITUDE.key(): [PROFILE_LONGITUDE],
        },
    )


@pytest.mark.parametrize(
    ("metric", "reference_loader_name"),
    [
        (metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis, "glorys_reanalysis_dataset"),
        (metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis, "glo12_analysis_dataset"),
    ],
)
def test_reference_mixed_layer_depth_is_computed_at_the_challenger_depth_levels(
    monkeypatch, metric, reference_loader_name
) -> None:
    reference_dataset = _stratified_dataset(STANDARD_50_DEPTHS)
    challenger_dataset = _stratified_dataset(NINE_CHALLENGER_DEPTHS)
    captured = {}

    def fake_rmsd(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(metrics, reference_loader_name, lambda _challenger_dataset: reference_dataset)
    monkeypatch.setattr(metrics, "ocean_mask", lambda: None)
    monkeypatch.setattr(metrics, "rmsd", fake_rmsd)

    metric(challenger_dataset)

    mixed_layer_depth_key = Variable.MIXED_LAYER_DEPTH.key()
    reference_mixed_layer_depth = captured["reference_dataset"][mixed_layer_depth_key]
    restricted_reference = reference_dataset.sel({Dimension.DEPTH.key(): NINE_CHALLENGER_DEPTHS})
    expected = mixed_layer_depth.compute_mixed_layer_depth(restricted_reference)[mixed_layer_depth_key]
    full_reference = mixed_layer_depth.compute_mixed_layer_depth(reference_dataset)[mixed_layer_depth_key]

    assert float(full_reference.values.squeeze()) == 6.441
    assert float(expected.values.squeeze()) == 47.374
    xarray.testing.assert_identical(reference_mixed_layer_depth.compute(), expected.compute())
