# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the reference datasets used in OceanBench for OceanBench challenger to explore.
"""

import xarray
from oceanbench.core.references import glorys, glo12


def glorys_reanalysis_2024() -> xarray.Dataset:
    """
    Open 2024 GLORYS reanalysis with as an `xarray.Dataset` with 1/12° resolution.

    Returns
    -------
    Dataset
        GLORYS reanalysis of 2024.
    """

    return glorys.glorys_reanalysis_2024()


def glorys_reanalysis_2024_1_4() -> xarray.Dataset:
    """
    Open 2024 GLORYS reanalysis with as an `xarray.Dataset` with 1/4° resolution.

    Returns
    -------
    Dataset
        GLORYS reanalysis of 2024.
    """

    return glorys.glorys_reanalysis_2024_1_4()


def glo12_analysis_2024() -> xarray.Dataset:
    """
    Open 2024 GLO12 analysis with as an `xarray.Dataset` with 1/12° resolution.

    Returns
    -------
    Dataset
        GLO12 analysis of 2024.
    """

    return glo12.glo12_analysis_2024()


def glo12_analysis_2024_1_4() -> xarray.Dataset:
    """
    Open 2024 GLO12 analysis with as an `xarray.Dataset` with 1/4° resolution.

    Returns
    -------
    Dataset
        GLO12 analysis of 2024.
    """

    return glo12.glo12_analysis_2024_1_4()
