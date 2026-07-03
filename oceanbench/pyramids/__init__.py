# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Viewer zarr pyramid builder (contracts.md §6)."""

from oceanbench.pyramids.builder import (
    DEFAULT_TILE_SIZE,
    PyramidResult,
    VariableSpec,
    build_pyramid,
)
from oceanbench.pyramids.challenger_layers import viewer_layers

__all__ = [
    "DEFAULT_TILE_SIZE",
    "PyramidResult",
    "VariableSpec",
    "build_pyramid",
    "viewer_layers",
]
