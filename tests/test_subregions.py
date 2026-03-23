# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench import subregions


def test_subset_gridded_dataset_to_ibi():
    dataset = xarray.Dataset(
        data_vars={
            "thetao": (
                ["latitude", "longitude"],
                numpy.arange(16).reshape(4, 4),
                {"standard_name": "sea_water_potential_temperature"},
            )
        },
        coords={
            "latitude": ("latitude", [20.0, 30.0, 40.0, 60.0], {"standard_name": "latitude"}),
            "longitude": ("longitude", [-30.0, -10.0, 0.0, 10.0], {"standard_name": "longitude"}),
        },
    )

    subset = subregions.subset(dataset, "ibi")

    assert subset["latitude"].values.tolist() == [30.0, 40.0]
    assert subset["longitude"].values.tolist() == [-10.0, 0.0]


def test_subset_point_dataset_to_ibi():
    dataset = xarray.Dataset(
        data_vars={
            "latitude": ("observations", [20.0, 30.0, 50.0], {"standard_name": "latitude"}),
            "longitude": ("observations", [-30.0, -10.0, 1.0], {"standard_name": "longitude"}),
            "thetao": (
                "observations",
                [10.0, 11.0, 12.0],
                {"standard_name": "sea_water_potential_temperature"},
            ),
        }
    )

    subset = subregions.subset(dataset, "IBI")

    assert subset.sizes["observations"] == 2
    assert subset["thetao"].values.tolist() == [11.0, 12.0]


def test_resolve_unknown_sub_region_raises_value_error():
    try:
        subregions.resolve_sub_region("unknown")
    except ValueError as error:
        assert "Unsupported sub-region" in str(error)
    else:
        raise AssertionError("resolve_sub_region should reject unknown sub-regions")
