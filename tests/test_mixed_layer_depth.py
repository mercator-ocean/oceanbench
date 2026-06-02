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
        lambda _absolute_salinity, _temperature, _depth: dataset["potential_density"],
    )

    mixed_layer_depth_dataset = mixed_layer_depth.compute_mixed_layer_depth(dataset)

    return float(mixed_layer_depth_dataset[Variable.MIXED_LAYER_DEPTH.key()].values.squeeze())


def test_mixed_layer_depth_keeps_native_first_threshold_depth(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.04, 1000.05, 1000.06],
    )

    assert _mld_value(dataset, monkeypatch) == 47.0


def test_mixed_layer_depth_ignores_threshold_crossings_below_600_meters(monkeypatch) -> None:
    dataset = _dataset(
        temperature_values=[10.0, 10.0, 10.0, 10.0],
        density_values=[1000.0, 1000.01, 1000.02, 1000.05],
    )

    assert _mld_value(dataset, monkeypatch) == 600.0


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
