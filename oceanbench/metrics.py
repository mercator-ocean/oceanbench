# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the functions to compute metrics.
"""

import xarray

from pandas import DataFrame

from oceanbench.core import metrics
from oceanbench.regions import RegionLike


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to observations.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_observations(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the deviation of Lagrangian trajectories compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the deviation of Lagrangian trajectories compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def marine_heatwave_diagnostics_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
    climatology_path: str | None = None,
) -> DataFrame:
    """
    Compute marine heatwave diagnostics compared to GLORYS reanalysis.

    Marine heatwaves (Hobday et al. 2016) are detected with heatwave3 in the surface temperature
    of each forecast window and of the GLORYS reanalysis window, after prepending the same seven days of
    GLORYS reanalysis before the forecast start, against one heatwave3 climatology. The scores are the
    probability of detection, false alarm ratio and critical success index of marine heatwave
    days and of days at or above categories II, III and IV, the category accuracy, the
    temperature RMSD on days where both, one or neither flag a marine heatwave, and the
    temperature bias.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.
    climatology_path : str, optional
        Path or URL of the heatwave3 climatology covering the region. Required until a global
        climatology is published.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.marine_heatwave_diagnostics_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
        climatology_path=climatology_path,
    )


def marine_heatwave_diagnostics_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
    climatology_path: str | None = None,
) -> DataFrame:
    """
    Compute marine heatwave diagnostics compared to GLO12 analysis.

    Marine heatwaves (Hobday et al. 2016) are detected with heatwave3 in the surface temperature
    of each forecast window and of the GLO12 analysis window, after prepending the same seven days of
    GLO12 analysis before the forecast start, against one heatwave3 climatology. The scores are the
    probability of detection, false alarm ratio and critical success index of marine heatwave
    days and of days at or above categories II, III and IV, the category accuracy, the
    temperature RMSD on days where both, one or neither flag a marine heatwave, and the
    temperature bias.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.
    climatology_path : str, optional
        Path or URL of the heatwave3 climatology covering the region. Required until a global
        climatology is published.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.marine_heatwave_diagnostics_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
        climatology_path=climatology_path,
    )
