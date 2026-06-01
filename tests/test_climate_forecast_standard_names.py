# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable


def test_rename_dataset_with_standard_names_preserves_unsupported_elevation_standard_name() -> None:
    dataset = xarray.Dataset(
        {
            "thetao": (
                ("time", "depth", "lat", "lon"),
                numpy.zeros((1, 1, 1, 1)),
                {"standard_name": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()},
            )
        },
        coords={
            "time": ("time", [0], {"standard_name": Dimension.TIME.key()}),
            "depth": ("depth", [0.5], {"standard_name": "elevation"}),
            "lat": ("lat", [0.0], {"standard_name": Dimension.LATITUDE.key()}),
            "lon": ("lon", [10.0], {"standard_name": Dimension.LONGITUDE.key()}),
        },
    )

    renamed_dataset = rename_dataset_with_standard_names(dataset)

    assert Dimension.DEPTH.key() in renamed_dataset.dims
    assert "elevation" not in renamed_dataset.dims
    assert Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key() in renamed_dataset
    assert Dimension.LATITUDE.key() in renamed_dataset.dims
    assert Dimension.LONGITUDE.key() in renamed_dataset.dims
