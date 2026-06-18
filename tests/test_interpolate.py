# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

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


def test_interpolate_1_degree_produces_a_one_degree_grid() -> None:
    interpolated = interpolate_1_degree(_dataset())

    assert interpolated["latitude"].values.tolist() == [-0.5, 0.5]
    assert interpolated["longitude"].values.tolist() == [10.5, 11.5]
