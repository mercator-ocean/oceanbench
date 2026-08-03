# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_source import DatasetSource, get_dataset_source, with_dataset_source
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.interpolate import interpolate_1_degree


def _dataset() -> xarray.Dataset:
    return xarray.Dataset(
        {
            "thetao": (
                ["time", "lat", "lon"],
                numpy.arange(18, dtype=float).reshape(2, 3, 3),
                {"standard_name": "sea_water_potential_temperature"},
            )
        },
        coords={
            "time": numpy.array(["2024-01-03", "2024-01-04"], dtype="datetime64[ns]"),
            "lat": xarray.DataArray(
                [-1.0, 0.0, 1.0],
                dims=["lat"],
                attrs={"standard_name": "latitude"},
            ),
            "lon": xarray.DataArray(
                [10.0, 11.0, 12.0],
                dims=["lon"],
                attrs={"standard_name": "longitude"},
            ),
        },
    )


def test_interpolate_1_degree_marks_dataset_source_as_one_degree() -> None:
    dataset = with_dataset_source(_dataset(), kind="challenger", name="glonet")

    interpolated = interpolate_1_degree(dataset)

    assert get_dataset_source(interpolated) == DatasetSource(
        kind="challenger",
        name="glonet",
        resolution="one_degree",
    )


def test_interpolate_1_degree_preserves_missing_dataset_source() -> None:
    interpolated = interpolate_1_degree(_dataset())

    assert get_dataset_source(interpolated) is None


def test_interpolation_bounds_the_start_and_lead_day_chunks() -> None:
    from oceanbench.core.dataset_utils import Dimension
    from oceanbench.core.interpolate import apply_one_degree_interpolation, one_degree_target_grid

    dataset = xarray.Dataset(
        {
            "thetao": (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    "lat",
                    "lon",
                ],
                numpy.zeros((3, 4, 3, 3)),
                {"standard_name": "sea_water_potential_temperature"},
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(
                ["2024-01-03", "2024-01-10", "2024-01-17"], dtype="datetime64[ns]"
            ),
            Dimension.LEAD_DAY_INDEX.key(): [0, 1, 2, 3],
            "lat": xarray.DataArray([-1.0, 0.0, 1.0], dims=["lat"], attrs={"standard_name": "latitude"}),
            "lon": xarray.DataArray([10.0, 11.0, 12.0], dims=["lon"], attrs={"standard_name": "longitude"}),
        },
    )
    new_latitude, new_longitude = one_degree_target_grid(rename_dataset_with_standard_names(dataset))

    interpolated = apply_one_degree_interpolation(dataset, new_latitude, new_longitude)

    chunk_sizes = interpolated["sea_water_potential_temperature"].chunksizes
    assert chunk_sizes[Dimension.FIRST_DAY_DATETIME.key()] == (1, 1, 1)
    assert chunk_sizes[Dimension.LEAD_DAY_INDEX.key()] == (1, 1, 1, 1)
