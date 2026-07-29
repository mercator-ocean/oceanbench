# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.rmsd import _rmsd


def test_rmsd_uses_area_weights_without_land_in_denominator() -> None:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    values = numpy.array(
        [
            [
                [
                    [1.0, numpy.nan],
                    [3.0, 5.0],
                ]
            ],
            [
                [
                    [2.0, 4.0],
                    [numpy.nan, 6.0],
                ]
            ],
        ]
    )
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.LATITUDE.key(): [0.0, 60.0],
        Dimension.LONGITUDE.key(): [10.0, 11.0],
    }
    challenger_dataset = xarray.Dataset(
        {
            variable_key: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords=coordinates,
    )
    reference_dataset = xarray.zeros_like(challenger_dataset)

    rmsd_dataset = _rmsd(challenger_dataset, reference_dataset)

    expected_first_day_rmsd = numpy.sqrt((1.0**2 * 1.0 + 3.0**2 * 0.5 + 5.0**2 * 0.5) / (1.0 + 0.5 + 0.5))
    expected_second_day_rmsd = numpy.sqrt((2.0**2 * 1.0 + 4.0**2 * 1.0 + 6.0**2 * 0.5) / (1.0 + 1.0 + 0.5))
    expected_rmsd = (expected_first_day_rmsd + expected_second_day_rmsd) / 2.0
    naive_land_weighted_rmsd = numpy.sqrt((1.0**2 * 1.0 + 3.0**2 * 0.5 + 5.0**2 * 0.5) / (1.0 + 1.0 + 0.5 + 0.5))
    actual_rmsd = float(rmsd_dataset[variable_key].sel({Dimension.LEAD_DAY_INDEX.key(): 0}))

    assert numpy.isclose(actual_rmsd, expected_rmsd)
    assert not numpy.isclose(actual_rmsd, naive_land_weighted_rmsd)


def test_rmsd_snaps_nearly_matching_spatial_coordinates_before_xarray_alignment() -> None:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    challenger_latitudes = numpy.array([0.0, 1.00001, 2.0], dtype=numpy.float32)
    challenger_longitudes = numpy.array([10.0, 11.00001, 12.0], dtype=numpy.float32)
    reference_latitudes = numpy.array([0.0, 1.0, 2.0], dtype=numpy.float32)
    reference_longitudes = numpy.array([10.0, 11.0, 12.0], dtype=numpy.float32)
    challenger_values = numpy.array(
        [
            [
                [
                    [1.0, 10.0, 2.0],
                    [100.0, 200.0, 300.0],
                    [3.0, 400.0, 4.0],
                ]
            ]
        ]
    )
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    base_coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
    }
    challenger_dataset = xarray.Dataset(
        {
            variable_key: (
                dimension_names,
                challenger_values,
            )
        },
        coords={
            **base_coordinates,
            Dimension.LATITUDE.key(): challenger_latitudes,
            Dimension.LONGITUDE.key(): challenger_longitudes,
        },
    )
    reference_dataset = xarray.Dataset(
        {
            variable_key: (
                dimension_names,
                numpy.zeros_like(challenger_values),
            )
        },
        coords={
            **base_coordinates,
            Dimension.LATITUDE.key(): reference_latitudes,
            Dimension.LONGITUDE.key(): reference_longitudes,
        },
    )

    rmsd_dataset = _rmsd(challenger_dataset, reference_dataset)

    latitude_weights = numpy.cos(numpy.deg2rad(reference_latitudes))[:, numpy.newaxis]
    expected_rmsd = numpy.sqrt(
        numpy.sum(challenger_values[0, 0] ** 2 * latitude_weights)
        / numpy.sum(numpy.ones_like(challenger_values[0, 0]) * latitude_weights)
    )
    legacy_inner_join_values = challenger_values[0, 0][[0, 2]][:, [0, 2]]
    legacy_inner_join_latitudes = reference_latitudes[[0, 2]]
    legacy_latitude_weights = numpy.cos(numpy.deg2rad(legacy_inner_join_latitudes))[:, numpy.newaxis]
    legacy_inner_join_rmsd = numpy.sqrt(
        numpy.sum(legacy_inner_join_values**2 * legacy_latitude_weights)
        / numpy.sum(numpy.ones_like(legacy_inner_join_values) * legacy_latitude_weights)
    )
    actual_rmsd = float(rmsd_dataset[variable_key].sel({Dimension.LEAD_DAY_INDEX.key(): 0}))

    assert numpy.isclose(actual_rmsd, expected_rmsd)
    assert not numpy.isclose(actual_rmsd, legacy_inner_join_rmsd)
