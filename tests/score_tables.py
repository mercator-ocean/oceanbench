# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas

from oceanbench.core.classIV_support import format_class4_results
from oceanbench.core.lagrangian_support import LAGRANGIAN_ROW_LABEL
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.rmsd import MISSING_FRACTION_COLUMN, _variable_depth_label

LEAD_DAYS_COUNT = 3


def gridded_table() -> pandas.DataFrame:
    labels = [
        _variable_depth_label(None, variable, depth)
        for variable, depth in [
            ("sea_surface_height_above_geoid", "surface"),
            ("sea_water_potential_temperature", "surface"),
            ("sea_water_potential_temperature", "50m"),
            ("northward_sea_water_velocity", "500m"),
        ]
    ]
    table = pandas.DataFrame(
        numpy.random.default_rng(1).random((len(labels), LEAD_DAYS_COUNT)),
        index=labels,
        columns=lead_day_labels(1, LEAD_DAYS_COUNT),
    )
    table[MISSING_FRACTION_COLUMN] = [0.000671, 0.0, 0.0125, 0.5]
    return table


def class4_table() -> pandas.DataFrame:
    results = pandas.DataFrame(
        [
            (variable, depth_bin, lead_day, 0.1 + lead_day / 7 + index / 3, 1000 + index, index)
            for index, (variable, depth_bin) in enumerate(
                [
                    ("sea_water_potential_temperature", "surface"),
                    ("sea_water_salinity", "0-5m"),
                    ("sea_surface_height_above_geoid", "surface"),
                    ("eastward_sea_water_velocity", "15m"),
                ]
            )
            for lead_day in range(LEAD_DAYS_COUNT)
        ],
        columns=["variable", "depth_bin", "lead_day", "rmsd", "count", "missing"],
    )
    return format_class4_results(results, LEAD_DAYS_COUNT)


def lagrangian_table() -> pandas.DataFrame:
    return pandas.DataFrame({LAGRANGIAN_ROW_LABEL: [11.027288, 21.413128]}, index=lead_day_labels(2, 3)).T


TABLES = {
    "rmsd_of_variables_compared_to_glorys_reanalysis": gridded_table(),
    "rmsd_of_variables_compared_to_observations": class4_table(),
    "deviation_of_lagrangian_trajectories_compared_to_glo12_analysis": lagrangian_table(),
}
