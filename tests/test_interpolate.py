# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_source import DatasetSource, get_dataset_source, with_dataset_source
from oceanbench.core.interpolate import interpolate_1_degree


def _dataset(values: numpy.ndarray) -> xarray.Dataset:
    return xarray.Dataset(
        {"thetao": (["time", "lat", "lon"], values, {"standard_name": "sea_water_potential_temperature"})},
        coords={
            "time": numpy.array(["2024-01-03", "2024-01-04"], dtype="datetime64[ns]"),
            "lat": xarray.DataArray([-1.0, 0.0, 1.0], dims=["lat"], attrs={"standard_name": "latitude"}),
            "lon": xarray.DataArray([10.0, 11.0, 12.0], dims=["lon"], attrs={"standard_name": "longitude"}),
        },
    )


def test_interpolate_1_degree_targets_half_degree_centres_inside_the_grid_and_interpolates_linearly() -> None:
    # thetao = 3 * latitude_index + longitude_index (+ 9 on the second day): a plane, so linear
    # interpolation at the half-degree centres is exact and can be written down by hand.
    interpolated = interpolate_1_degree(_dataset(numpy.arange(18, dtype=float).reshape(2, 3, 3)))

    assert interpolated["latitude"].values.tolist() == [-0.5, 0.5]
    assert interpolated["longitude"].values.tolist() == [10.5, 11.5]
    numpy.testing.assert_allclose(
        interpolated["sea_water_potential_temperature"].values,
        [[[2.0, 3.0], [5.0, 6.0]], [[11.0, 12.0], [14.0, 15.0]]],
    )


def test_interpolate_1_degree_marks_dataset_source_as_one_degree() -> None:
    dataset = with_dataset_source(_dataset(numpy.zeros((2, 3, 3))), kind="challenger", name="glonet")

    interpolated = interpolate_1_degree(dataset)

    assert get_dataset_source(interpolated) == DatasetSource(
        kind="challenger",
        name="glonet",
        resolution="one_degree",
    )


def test_interpolate_1_degree_preserves_missing_dataset_source() -> None:
    interpolated = interpolate_1_degree(_dataset(numpy.zeros((2, 3, 3))))

    assert get_dataset_source(interpolated) is None
