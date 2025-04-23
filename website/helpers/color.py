# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import matplotlib as mpl

CMAP = mpl.colormaps["bwr"]
MAX_DIFF_DISTANCE = 1


def get_color(reference_value: float, compared_value: float):
    # constraint the difference between -max_diff_distance and max_diff_distance
    constraint_diff = max(
        -MAX_DIFF_DISTANCE,
        min(MAX_DIFF_DISTANCE, compared_value - reference_value),
    )
    # normalize the difference between 0 and 1
    normalized_diff = (constraint_diff + MAX_DIFF_DISTANCE) / (MAX_DIFF_DISTANCE * 2)
    return mpl.colors.rgb2hex(CMAP(normalized_diff))
