# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
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


def test_coarse_levels_of_an_odd_grid_stay_on_a_regular_grid_centred_on_their_blocks():
    latitudes = numpy.linspace(-80.0, 90.0, 681)
    longitudes = numpy.arange(1440) * 0.25 - 180.0
    current = xarray.Dataset(
        {"field": (("latitude", "longitude"), numpy.ones((latitudes.size, longitudes.size)))},
        coords={"latitude": latitudes, "longitude": longitudes},
    )
    fine_latitude_groups = [latitudes[index : index + 1] for index in range(latitudes.size)]
    for plan in level_planning.plan_levels(current)[1:]:
        current = level_planning.coarsen_by_two(current)
        fine_latitude_groups = [
            numpy.concatenate(fine_latitude_groups[index : index + 2])
            for index in range(0, len(fine_latitude_groups), 2)
        ]
        coarse_latitudes = current["latitude"].values
        assert numpy.allclose(numpy.diff(coarse_latitudes), plan.cell_size_deg)
        assert numpy.allclose(numpy.diff(current["longitude"].values), plan.cell_size_deg)
        full_blocks = [group.size == 2**plan.level for group in fine_latitude_groups]
        block_centroids = numpy.array([group.mean() for group in fine_latitude_groups])
        assert numpy.allclose(coarse_latitudes[full_blocks], block_centroids[full_blocks])
        assert numpy.isclose(coarse_latitudes[0] - plan.cell_size_deg / 2.0, latitudes[0] - 0.125)


def test_coarsen_by_two_rejects_a_single_point_axis_and_an_irregular_grid():
    single = _grid(1.0).isel(latitude=slice(0, 1))
    with pytest.raises(ValueError, match="at least 2 points"):
        level_planning.coarsen_by_two(single)
    irregular = _grid(1.0)
    latitudes = irregular["latitude"].values.copy()
    latitudes[5:] += 0.5
    with pytest.raises(ValueError, match="regular grid"):
        level_planning.coarsen_by_two(irregular.assign_coords(latitude=latitudes))
