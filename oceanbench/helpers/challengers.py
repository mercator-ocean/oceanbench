# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the API that brings challengers on other grids onto the GLO12 grid.

It needs the optional regridding dependencies: ``pip install "oceanbench[regridding]"``.
"""

from pathlib import Path

import xarray

try:
    from oceanbench.core.glo12_regridding import regrid_to_glo12_grid
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        'oceanbench.helpers.challengers needs pyresample: pip install "oceanbench[regridding]"'
    ) from error
from oceanbench.core.challenger_format import to_challenger_format
from oceanbench.core.croco_conversion import croco_to_standard_dataset

__all__ = [
    "croco_to_glo12_challenger",
]


def croco_to_glo12_challenger(croco_path: str | Path) -> xarray.Dataset:
    """
    Turn one CROCO forecast into an OceanBench challenger dataset on the GLO12 1/12 degree grid.

    The CROCO history file must use the Vtransform 2 s-coordinate, store time in seconds since
    2000-01-01, and hold snapshots regularly spaced from 00:00 of the first day to 00:00 after the last day.

    Land points are set to NaN and the outermost rho rows and columns are dropped.
    S-levels are linearly interpolated to the 50 GLO12 depths (top value copied above the shallowest level,
    NaN below the deepest level), velocities are moved to rho points and rotated to eastward and northward
    components, and daily means use the trapezoidal rule over the snapshots.

    The daily means are then regridded bilinearly onto the GLO12 points inside the CROCO latitude and
    longitude range. Corners that are NaN (land, below the seabed, outside the domain) are dropped and the
    remaining weights renormalised, so a target point is NaN only when all four corners are.
    The latitude and longitude are bit-identical to the GLO12 coordinates.

    Parameters
    ----------
    croco_path : str or Path
        Path to a CROCO history file with temp, salt, u, v and zeta.

    Returns
    -------
    xarray.Dataset
        Daily thetao, so, uo, vo and zos with dimensions ``first_day_datetime``, ``lead_day_index``,
        ``depth`` (except zos), ``latitude`` and ``longitude``.

    Examples
    --------
    >>> from oceanbench.helpers import challengers
    >>> challenger = challengers.croco_to_glo12_challenger("croco_his.nc")  # doctest: +SKIP
    >>> challenger.to_netcdf("croco_challenger.nc")  # doctest: +SKIP
    """
    with xarray.open_dataset(croco_path, decode_times=False) as croco_dataset:
        return to_challenger_format(regrid_to_glo12_grid(croco_to_standard_dataset(croco_dataset))).load()
