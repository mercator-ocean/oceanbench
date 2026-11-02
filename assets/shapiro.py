# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2
#
# Source: https://gitlab.mercator-ocean.fr/internal-shared-group/validation/eddy-tracking-intercomparison
# File: eddy_load/my_functions.py
# Commit: db0261e51d10afbd0a5d80d14984ead0e8578052

import numpy as np


def shapiro_1d(rla_varin, id_np, missval, ln_depth):
    """
    Apply a 1D Shapiro filter (3 points filter) horizontally to a 2D field
    for a regional domain (embedded in a global one).

    Parameters:
    rla_varin : numpy.ndarray
        Input variable to filter (3D or 4D array).
    id_np : int
        Number of Shapiro filter iterations.
    missval : float
        Missing value.
    ln_depth : boolean for nb of dim:
               - True => 4D input array with depth dim
               - False => 3D input array without depth dim

    Returns:
    rla_varout : numpy.ndarray
        Output filtered variable.
    """
    rl_alpha = 1.0 / 2.0
    if ln_depth:
        jpi = rla_varin.shape[3]
        jpj = rla_varin.shape[2]
        print("------------------------------------------")
        print("            Shapiro Filter                ")
        print("------------------------------------------")
        print("")
        print("- nb of passages           = ", id_np)
        print("- nb of X grid point :jpi  = ", jpi)
        print("- nb of Y grid point :jpj = ", jpj)
        print("- ln_depth = ", ln_depth)
        print("")
        print("------------------------------------------")
        rla_varout = np.zeros((1, 1, jpj, jpi))
        rla_varout_tmp = np.zeros((1, 1, jpj, jpi))
        rla_mask = np.zeros((1, 1, jpj, jpi))
        for ji in range(0, jpi):
            for jj in range(0, jpj):
                if np.isnan(rla_varin[0, 0, jj, ji]):
                    rla_mask[0, 0, jj, ji] = 0.0
                else:
                    rla_mask[0, 0, jj, ji] = 1.0
        rla_varout_tmp[0, 0, :, :] = rla_varin[0, 0, :, :]
        for jn in range(id_np):
            for ji in range(1, jpi - 1):
                for jj in range(1, jpj - 1):
                    test1 = rla_mask[0, 0, jj, ji - 1]
                    test2 = rla_mask[0, 0, jj, ji]
                    test3 = rla_mask[0, 0, jj, ji + 1]
                    test4 = rla_mask[0, 0, jj - 1, ji]
                    test5 = rla_mask[0, 0, jj + 1, ji]
                    if test1 == 1.0 and test2 == 1.0 and test3 == 1.0 and test4 == 1.0 and test5 == 1.0:
                        znum = (
                            rla_varout_tmp[0, 0, jj, ji]
                            - 0.25
                            * rl_alpha
                            * (
                                rla_mask[0, 0, jj, ji - 1]
                                + rla_mask[0, 0, jj, ji + 1]
                                + rla_mask[0, 0, jj - 1, ji]
                                + rla_mask[0, 0, jj + 1, ji]
                            )
                            * rla_varout_tmp[0, 0, jj, ji]
                        )
                        znum += (
                            0.25
                            * rl_alpha
                            * (
                                rla_varout_tmp[0, 0, jj, ji - 1] * rla_mask[0, 0, jj, ji - 1]
                                + rla_varout_tmp[0, 0, jj, ji + 1] * rla_mask[0, 0, jj, ji + 1]
                                + rla_varout_tmp[0, 0, jj - 1, ji] * rla_mask[0, 0, jj - 1, ji]
                                + rla_varout_tmp[0, 0, jj + 1, ji] * rla_mask[0, 0, jj + 1, ji]
                            )
                        )
                        rla_varout[0, 0, jj, ji] = znum
                    else:
                        rla_varout[0, 0, jj, ji] = rla_varin[0, 0, jj, ji]
            rla_varout[0, 0, 0, :] = rla_varout[0, 0, 1, :]
            rla_varout[0, 0, jpj - 1, :] = rla_varout[0, 0, jpj - 2, :]
            rla_varout[0, 0, :, 0] = rla_varout[0, 0, :, 1]
            rla_varout[0, 0, :, jpi - 1] = rla_varout[0, 0, :, jpi - 2]
            rla_varout_tmp[0, 0, :, :] = rla_varout[0, 0, :, :]
    else:
        jpi = rla_varin.shape[2]
        jpj = rla_varin.shape[1]
        print("------------------------------------------")
        print("            Shapiro Filter                ")
        print("------------------------------------------")
        print("")
        print("- nb of passages           = ", id_np)
        print("- nb of X grid point :jpi  = ", jpi)
        print("- nb of Y grid point :jpj = ", jpj)
        print("- ln_depth = ", ln_depth)
        print("")
        print("------------------------------------------")
        rla_varout = np.zeros((1, jpj, jpi))
        rla_varout_tmp = np.zeros((1, jpj, jpi))
        rla_mask = np.zeros((1, jpj, jpi))
        for ji in range(0, jpi):
            for jj in range(0, jpj):
                if np.isnan(rla_varin[0, jj, ji]):
                    rla_mask[0, jj, ji] = 0.0
                else:
                    rla_mask[0, jj, ji] = 1.0
        rla_varout_tmp[0, :, :] = rla_varin[0, :, :]
        for jn in range(id_np):
            for ji in range(1, jpi - 1):
                for jj in range(1, jpj - 1):
                    test1 = rla_mask[0, jj, ji - 1]
                    test2 = rla_mask[0, jj, ji]
                    test3 = rla_mask[0, jj, ji + 1]
                    test4 = rla_mask[0, jj - 1, ji]
                    test5 = rla_mask[0, jj + 1, ji]
                    if test1 == 1.0 and test2 == 1.0 and test3 == 1.0 and test4 == 1.0 and test5 == 1.0:
                        znum = (
                            rla_varout_tmp[0, jj, ji]
                            - 0.25
                            * rl_alpha
                            * (
                                rla_mask[0, jj, ji - 1]
                                + rla_mask[0, jj, ji + 1]
                                + rla_mask[0, jj - 1, ji]
                                + rla_mask[0, jj + 1, ji]
                            )
                            * rla_varout_tmp[0, jj, ji]
                        )
                        znum += (
                            0.25
                            * rl_alpha
                            * (
                                rla_varout_tmp[0, jj, ji - 1] * rla_mask[0, jj, ji - 1]
                                + rla_varout_tmp[0, jj, ji + 1] * rla_mask[0, jj, ji + 1]
                                + rla_varout_tmp[0, jj - 1, ji] * rla_mask[0, jj - 1, ji]
                                + rla_varout_tmp[0, jj + 1, ji] * rla_mask[0, jj + 1, ji]
                            )
                        )
                        rla_varout[0, jj, ji] = znum
                    else:
                        rla_varout[0, jj, ji] = rla_varin[0, jj, ji]
            rla_varout[0, 0, :] = rla_varout[0, 1, :]
            rla_varout[0, jpj - 1, :] = rla_varout[0, jpj - 2, :]
            rla_varout[0, :, 0] = rla_varout[0, :, 1]
            rla_varout[0, :, jpi - 1] = rla_varout[0, :, jpi - 2]
            rla_varout_tmp[0, :, :] = rla_varout[0, :, :]
    return rla_varout
