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
