# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest

from oceanbench.pyramids.quantization import (
    QUANTIZED_FILL_VALUE,
    quantization_for_data,
    quantization_for_range,
)


def _round_trip(values: numpy.ndarray, quantization) -> numpy.ndarray:
    stored = numpy.round((values - quantization.add_offset) / quantization.scale_factor)
    return stored * quantization.scale_factor + quantization.add_offset


def test_round_trip_error_within_one_quantization_step():
    values = numpy.linspace(-2.0, 34.0, 5000)
    quantization = quantization_for_range(values.min(), values.max())
    decoded = _round_trip(values, quantization)
    assert numpy.abs(decoded - values).max() <= quantization.quantization_step


def test_stored_levels_stay_below_the_fill_sentinel():
    values = numpy.linspace(-2.0, 34.0, 5000)
    quantization = quantization_for_range(values.min(), values.max())
    stored = numpy.round((values - quantization.add_offset) / quantization.scale_factor)
    assert stored.min() >= 0
    assert stored.max() < QUANTIZED_FILL_VALUE


def test_constant_field_is_encoded_exactly():
    quantization = quantization_for_range(7.5, 7.5)
    decoded = _round_trip(numpy.full(10, 7.5), quantization)
    assert numpy.allclose(decoded, 7.5)


def test_quantization_ignores_non_finite_entries():
    values = numpy.array([1.0, numpy.nan, 3.0, numpy.inf])
    quantization = quantization_for_data(values)
    assert quantization.add_offset < 1.0
    assert quantization.scale_factor > 0


def test_all_non_finite_raises():
    with pytest.raises(ValueError):
        quantization_for_data(numpy.array([numpy.nan, numpy.inf]))
