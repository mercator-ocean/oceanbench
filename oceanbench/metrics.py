# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the functions to compute metrics.
"""

import xarray

from pandas import DataFrame

from oceanbench.core import metrics


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_glorys_reanalysis(challenger_dataset=challenger_dataset)


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(challenger_dataset=challenger_dataset)


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(challenger_dataset=challenger_dataset)


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the deviation of Lagrangian trajectories compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_glo12_analysis(challenger_dataset=challenger_dataset)


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(challenger_dataset=challenger_dataset)


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(challenger_dataset=challenger_dataset)


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the deviation of Lagrangian trajectories compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset
    )


def rmsd_of_variables_compared_to_class4_observations(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to Class4 observations.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_variables_compared_to_class4_observations(challenger_dataset=challenger_dataset)
