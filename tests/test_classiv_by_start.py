# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas

from oceanbench.core.classIV import rmsd_class4_validation, rmsd_class4_validation_by_start
from oceanbench.core.dataset_utils import Dimension
from tests import synthetic_class4


def _observations_by_start(observations):
    first_days = observations[Dimension.FIRST_DAY_DATETIME.key()].values
    order = numpy.argsort(first_days, kind="stable")
    observations = observations.isel(observation=order)
    first_days = first_days[order]
    return [observations.isel(observation=first_days == first_day) for first_day in numpy.unique(first_days)]


def test_class4_rmsd_by_start_is_the_class4_rmsd_of_all_starts() -> None:
    challenger = synthetic_class4.challenger_dataset()
    observations = synthetic_class4.observations_dataset()
    mask = synthetic_class4.ocean_mask()
    by_start = _observations_by_start(observations)
    every_start = xarray_concat(by_start)

    expected = rmsd_class4_validation(challenger, every_start, mask, synthetic_class4.VARIABLES)
    actual = rmsd_class4_validation_by_start(challenger, iter(by_start), mask, synthetic_class4.VARIABLES)

    pandas.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_a_start_without_observations_changes_nothing() -> None:
    challenger = synthetic_class4.challenger_dataset()
    mask = synthetic_class4.ocean_mask()
    by_start = _observations_by_start(synthetic_class4.observations_dataset())
    empty = by_start[0].isel(observation=slice(0, 0))

    expected = rmsd_class4_validation_by_start(challenger, iter(by_start), mask, synthetic_class4.VARIABLES)
    actual = rmsd_class4_validation_by_start(challenger, iter([empty, *by_start]), mask, synthetic_class4.VARIABLES)

    pandas.testing.assert_frame_equal(actual, expected, check_exact=True)


def xarray_concat(parts):
    import xarray

    return xarray.concat(parts, dim="observation")
