# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Pyramid level planning and spatial coarsening (contracts.md §6).

Level 0 is the native grid; each further level halves the spatial resolution
(a 2x2 block mean over ocean cells) up to about one degree. A dataset already at
about one degree collapses to a single level, the degenerate case is handled by
producing only ``level/0``.
"""

from dataclasses import dataclass

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension

_TARGET_COARSEST_CELL_DEGREES = 1.0
_MINIMUM_LEVEL_CELLS = 2


@dataclass(frozen=True)
class LevelPlan:
    level: int
    cell_size_deg: float
    latitude_size: int
    longitude_size: int


def native_cell_size_degrees(dataset: xarray.Dataset) -> float:
    """Median absolute latitude spacing of the native grid, in degrees."""
    latitudes = numpy.asarray(dataset[Dimension.LATITUDE.key()].values, dtype=float)
    if latitudes.size < 2:
        raise ValueError("Need at least two latitudes to infer the native cell size.")
    return float(numpy.median(numpy.abs(numpy.diff(latitudes))))


def plan_levels(dataset: xarray.Dataset) -> list[LevelPlan]:
    """Level plan from native up to the first level at or coarser than ~1 degree.

    Always contains at least ``level/0``. Coarsening stops once a level reaches the
    target cell size or a spatial dimension would fall below two cells.
    """
    native_cell = native_cell_size_degrees(dataset)
    latitude_size = int(dataset.sizes[Dimension.LATITUDE.key()])
    longitude_size = int(dataset.sizes[Dimension.LONGITUDE.key()])
    plans = [LevelPlan(0, native_cell, latitude_size, longitude_size)]
    while (
        plans[-1].cell_size_deg < _TARGET_COARSEST_CELL_DEGREES
        and plans[-1].latitude_size // 2 >= _MINIMUM_LEVEL_CELLS
        and plans[-1].longitude_size // 2 >= _MINIMUM_LEVEL_CELLS
    ):
        previous = plans[-1]
        plans.append(
            LevelPlan(
                level=previous.level + 1,
                cell_size_deg=previous.cell_size_deg * 2.0,
                latitude_size=(previous.latitude_size + 1) // 2,
                longitude_size=(previous.longitude_size + 1) // 2,
            )
        )
    return plans


def coarsen_by_two(layers: xarray.Dataset) -> xarray.Dataset:
    """2x2 block mean that ignores land (NaN), padding odd edges with NaN."""
    return layers.coarsen(
        {Dimension.LATITUDE.key(): 2, Dimension.LONGITUDE.key(): 2},
        boundary="pad",
    ).mean(skipna=True)
