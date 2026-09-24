# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest

from oceanbench.core.dataset_utils import is_global_longitude_grid


@pytest.mark.parametrize("step", [1 / 12, 1 / 4, 1.0])
def test_a_global_longitude_grid_is_recognised(step: float) -> None:
    longitudes = numpy.arange(-180.0, 180.0, step).astype(numpy.float32)

    assert is_global_longitude_grid(longitudes)


def test_an_ibi_longitude_crop_is_not_global() -> None:
    longitudes = numpy.linspace(-19.0, 5.0, 865).astype(numpy.float32)

    assert not is_global_longitude_grid(longitudes)
