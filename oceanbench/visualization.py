# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the functions to generate evaluation visualizations.
"""

from collections.abc import Mapping, Sequence

import pandas
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
    title: str = "Forecast comparison maps",
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
        Optional single depth selector per variable. If omitted, depth-resolved
        variables expose the OceanBench demo depth set: surface, 50 m, 100 m,
        200 m, 300 m, and 500 m, selected on the nearest available model depth.
    challenger_name : str, optional
        The display name of the challenger dataset.
    maximum_map_cells : int, optional
        Maximum number of displayed cells per map before stride-based thinning.
    height_pixels : int, optional
        The iframe height used by notebook and website renderers.
    title : str, optional
        The display title shown above the interactive map.
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
        title=title,
    )


def plot_multi_reference_surface_comparison_explorer(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variables: Sequence[Variable | str] = visualization.DEFAULT_SURFACE_COMPARISON_VARIABLES,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] | None = None,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    challenger_name: str = "Challenger",
    maximum_map_cells: int = visualization.DEFAULT_EXPLORER_MAXIMUM_MAP_CELLS,
    height_pixels: int = visualization.DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Forecast comparison maps",
):
    """
    Display one browser-side explorer comparing a challenger to several references.

    The challenger layer is embedded once, while the reference, error, absolute
    error, and RMSE layers remain available for each reference.
    """

    return visualization.plot_multi_reference_surface_comparison_explorer(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        variables=variables,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        depth_selectors=depth_selectors,
        challenger_name=challenger_name,
        maximum_map_cells=maximum_map_cells,
        height_pixels=height_pixels,
        title=title,
    )


def plot_multi_reference_zonal_psd_comparison(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variables: Sequence[Variable | str] = visualization.DEFAULT_ZONAL_PSD_VARIABLES,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] = visualization.DEFAULT_ZONAL_PSD_LEAD_DAY_INDICES,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    challenger_name: str = "Challenger",
):
    """
    Plot compact zonal power spectral density comparisons against several references.
    """

    return visualization.plot_multi_reference_zonal_psd_comparison(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        variables=variables,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        depth_selectors=depth_selectors,
        challenger_name=challenger_name,
    )


def plot_multi_reference_lagrangian_trajectory_explorer(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    first_day_index: int = 0,
    particle_count: int = visualization.DEFAULT_LAGRANGIAN_PARTICLE_COUNT,
    seed: int = 123,
    challenger_name: str = "Challenger",
    height_pixels: int = visualization.DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Lagrangian trajectory divergence",
):
    """
    Display an animated browser-side explorer for sampled Lagrangian trajectory divergence.
    """

    return visualization.plot_multi_reference_lagrangian_trajectory_explorer(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        first_day_index=first_day_index,
        particle_count=particle_count,
        seed=seed,
        challenger_name=challenger_name,
        height_pixels=height_pixels,
        title=title,
    )


def plot_class4_observation_error_explorer(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    variables: Sequence[Variable | str] = visualization.DEFAULT_SURFACE_COMPARISON_VARIABLES,
    first_day_index: int = 0,
    maximum_points_per_frame: int = visualization.DEFAULT_CLASS4_MAXIMUM_POINTS_PER_FRAME,
    height_pixels: int = visualization.DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Class IV observation error maps",
    comparison_dataframe: pandas.DataFrame | None = None,
):
    """
    Display a browser-side explorer for challenger errors against Class IV observations.
    """

    return visualization.plot_class4_observation_error_explorer(
        challenger_dataset=challenger_dataset,
        observation_dataset=observation_dataset,
        variables=variables,
        first_day_index=first_day_index,
        maximum_points_per_frame=maximum_points_per_frame,
        height_pixels=height_pixels,
        title=title,
        comparison_dataframe=comparison_dataframe,
    )


def plot_multi_reference_eddy_matching_explorer(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] | None = None,
    maximum_contour_points: int = visualization.DEFAULT_EDDY_MAXIMUM_CONTOUR_POINTS,
    height_pixels: int = visualization.DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Mesoscale eddy matching",
):
    """
    Display a browser-side explorer for SSH mesoscale eddy detections and matches.
    """

    return visualization.plot_multi_reference_eddy_matching_explorer(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        maximum_contour_points=maximum_contour_points,
        height_pixels=height_pixels,
        title=title,
    )


__all__ = [
    "plot_class4_observation_error_explorer",
    "plot_multi_reference_eddy_matching_explorer",
    "plot_multi_reference_lagrangian_trajectory_explorer",
    "plot_multi_reference_surface_comparison_explorer",
    "plot_multi_reference_zonal_psd_comparison",
    "plot_surface_comparison_explorer",
]
