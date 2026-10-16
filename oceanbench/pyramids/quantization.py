# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass

import numpy

# Viewer fields are stored as uint16 with a per-variable scale and offset, land as an explicit fill value.
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
