# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.pyramids import levels as level_planning


def _grid(cell_size_deg: float, span_deg: float = 16.0) -> xarray.Dataset:
    latitudes = numpy.arange(-span_deg, span_deg, cell_size_deg)
    longitudes = numpy.arange(-span_deg, span_deg, cell_size_deg)
    field = numpy.ones((latitudes.size, longitudes.size))
    return xarray.Dataset(
        {"field": (("latitude", "longitude"), field)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )


def test_one_degree_dataset_is_a_single_level():
    plans = level_planning.plan_levels(_grid(1.0))
    assert [plan.level for plan in plans] == [0]


def test_fine_grid_halves_up_to_about_one_degree():
    plans = level_planning.plan_levels(_grid(0.25))
    assert [plan.level for plan in plans] == [0, 1, 2]
    assert [round(plan.cell_size_deg, 3) for plan in plans] == [0.25, 0.5, 1.0]
    assert plans[-1].cell_size_deg >= 1.0


def test_each_level_halves_the_spatial_sizes():
    plans = level_planning.plan_levels(_grid(0.25))
    assert [plan.latitude_size for plan in plans] == [128, 64, 32]
    assert [plan.longitude_size for plan in plans] == [128, 64, 32]


def test_coarsen_by_two_ignores_land():
    latitudes = numpy.arange(-1.0, 1.0, 0.5)
    longitudes = numpy.arange(-1.0, 1.0, 0.5)
    values = numpy.array(
        [
            [1.0, numpy.nan, 2.0, 2.0],
            [3.0, 3.0, 2.0, 2.0],
            [4.0, 4.0, 6.0, 6.0],
            [4.0, 4.0, 6.0, 6.0],
        ]
    )
    dataset = xarray.Dataset(
        {"field": (("latitude", "longitude"), values)},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    coarse = level_planning.coarsen_by_two(dataset)["field"].values
    assert coarse.shape == (2, 2)
    assert numpy.isclose(coarse[0, 0], numpy.nanmean([1.0, 3.0, 3.0]))
    assert numpy.isclose(coarse[1, 1], 6.0)
