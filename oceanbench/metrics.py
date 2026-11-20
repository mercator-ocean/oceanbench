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


def smoothing_controlled_rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
    climatology_dataset: xarray.Dataset | None = None,
) -> DataFrame:
    """
    Compute the smoothing-controlled Root Mean Square Deviation (RMSD) of variables compared to observations.

    The RMSD a challenger reaches after the best Gaussian smoothing (width 0, 15, 30, 45, 60 or 90 km) and the best
    rescaling of its anomaly against a monthly climatology, on the Class IV observations. The width and the slope are
    fitted on every other forecast initialization day and scored on the others (Murphy, 1988, potential skill made
    scale aware). Smoothing or damping a forecast cannot lower this score, unlike the RMSD.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.
    climatology_dataset : xarray.Dataset, optional
        A monthly climatology with one time step per calendar month. Defaults to the GLORYS12 monthly climatology.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores, shaped like the RMSD compared to observations. The fitted smoothing
        width (km) and slope of each score are in ``attrs["row_diagnostics"]``.
    """

    return metrics.smoothing_controlled_rmsd_of_variables_compared_to_observations(
        challenger_dataset=challenger_dataset,
        region=region,
        climatology_dataset=climatology_dataset,
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
