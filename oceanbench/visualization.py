# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the functions to generate evaluation visualizations.
"""

from collections.abc import Mapping, Sequence

import xarray

from oceanbench.core.dataset_utils import Variable
from oceanbench.core import visualization


def plot_surface_comparison_explorer(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    variables: Sequence[Variable | str] = visualization.DEFAULT_SURFACE_COMPARISON_VARIABLES,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] | None = None,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    challenger_name: str = "Challenger",
    maximum_map_cells: int = visualization.DEFAULT_EXPLORER_MAXIMUM_MAP_CELLS,
    height_pixels: int = visualization.DEFAULT_EXPLORER_HEIGHT_PIXELS,
):
    """
    Display a browser-side explorer for challenger, reference, and error surface maps.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    reference_dataset : xarray.Dataset
        The reference dataset.
    reference_name : str
        The display name of the reference dataset.
    variables : sequence of Variable or str, optional
        The variables to display.
    first_day_index : int, optional
        The forecast run index to display.
    lead_day_indices : sequence of int, optional
        The lead day indices to include. Defaults to all available lead days.
    depth_selectors : mapping, optional
        Optional single depth selector per variable. If omitted, all available
        depths are exposed for depth-resolved variables.
    challenger_name : str, optional
        The display name of the challenger dataset.
    maximum_map_cells : int, optional
        Maximum number of displayed cells per map before stride-based thinning.
    height_pixels : int, optional
        The iframe height used by notebook and website renderers.
    """

    return visualization.plot_surface_comparison_explorer(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        reference_name=reference_name,
        variables=variables,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        depth_selectors=depth_selectors,
        challenger_name=challenger_name,
        maximum_map_cells=maximum_map_cells,
        height_pixels=height_pixels,
    )


__all__ = [
    "plot_surface_comparison_explorer",
]
