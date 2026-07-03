# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""uint16 quantization for viewer pyramids (contracts.md §6).

Each variable is stored as ``uint16`` with a per-variable ``scale_factor`` and
``add_offset`` so the client decodes ``value = stored * scale_factor + add_offset``.
The scale is chosen from the variable's actual finite data range with a small
margin; land / missing cells store an explicit ``_FillValue`` sentinel. The
quantization step (``scale_factor``) is far below model error for every ocean
variable, so the display copy is visually lossless.
"""

from dataclasses import dataclass

import numpy

QUANTIZED_FILL_VALUE = 65535
_QUANTIZED_MAXIMUM_LEVEL = 65534
_RANGE_MARGIN_FRACTION = 0.01


@dataclass(frozen=True)
class Quantization:
    """uint16 encoding of one variable: ``value = stored * scale_factor + add_offset``."""

    scale_factor: float
    add_offset: float
    fill_value: int = QUANTIZED_FILL_VALUE

    @property
    def quantization_step(self) -> float:
        """Largest possible round-trip error is half of this; the test bound is this."""
        return self.scale_factor


def quantization_for_range(minimum_value: float, maximum_value: float) -> Quantization:
    """Choose a uint16 quantization spanning ``[minimum_value, maximum_value]`` with margin.

    A constant field (``minimum == maximum``) gets a unit scale so decoding is exact.
    """
    if not numpy.isfinite(minimum_value) or not numpy.isfinite(maximum_value):
        raise ValueError("Cannot quantize a variable with no finite values.")
    value_span = maximum_value - minimum_value
    if value_span <= 0:
        return Quantization(scale_factor=1.0, add_offset=float(minimum_value))
    margin = _RANGE_MARGIN_FRACTION * value_span
    lower = minimum_value - margin
    upper = maximum_value + margin
    scale_factor = (upper - lower) / _QUANTIZED_MAXIMUM_LEVEL
    return Quantization(scale_factor=float(scale_factor), add_offset=float(lower))


def quantization_for_data(values: numpy.ndarray) -> Quantization:
    """Quantization derived from the finite entries of an array."""
    finite_values = values[numpy.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot quantize an array with no finite values.")
    return quantization_for_range(float(finite_values.min()), float(finite_values.max()))


def zarr_encoding(quantization: Quantization, compressor: object) -> dict:
    """CF encoding that makes xarray write the variable as quantized uint16.

    xarray stores ``round((value - add_offset) / scale_factor)`` as ``uint16`` and
    maps NaN to ``_FillValue``; reading back with default decoding reverses it. The
    zarr chunk shape is adopted from the (already tiled) dask array, so no explicit
    ``chunks`` is set here.
    """
    return {
        "dtype": "uint16",
        "scale_factor": quantization.scale_factor,
        "add_offset": quantization.add_offset,
        "_FillValue": quantization.fill_value,
        "compressor": compressor,
    }
