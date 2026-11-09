# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

import numpy
import pandas

from oceanbench.core.classIV_support import format_class4_results
from oceanbench.core.lagrangian_support import LAGRANGIAN_ROW_LABEL
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.metrics import MESSAGE_COLUMN
from oceanbench.core.rmsd import MISSING_FRACTION_COLUMN, _variable_depth_label
from oceanbench.core.scores import score_rows, write_scores_file

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.notebook_score_parser import get_all_model_scores_from_notebook  # noqa: E402

LEAD_DAYS_COUNT = 3


def _gridded_table() -> pandas.DataFrame:
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


def _class4_table() -> pandas.DataFrame:
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


def _lagrangian_table() -> pandas.DataFrame:
    return pandas.DataFrame({LAGRANGIAN_ROW_LABEL: [11.027288, 21.413128]}, index=lead_day_labels(2, 3)).T


TABLES = {
    "rmsd_of_variables_compared_to_glorys_reanalysis": _gridded_table(),
    "rmsd_of_variables_compared_to_observations": _class4_table(),
    "deviation_of_lagrangian_trajectories_compared_to_glo12_analysis": _lagrangian_table(),
}

_PARSER_METRIC_KEYS = {
    ("rmsd_of_variables", "glorys_reanalysis"): "rmsd_variables_glorys",
    ("rmsd_of_variables", "observations"): "rmsd_variables_observations",
    ("deviation_of_lagrangian_trajectories", "glo12_analysis"): "lagrangian_glo12",
}


def _notebook_with_tables(path: Path) -> None:
    cells = [
        {
            "cell_type": "code",
            "source": f"oceanbench.metrics.{metric_function_name}(\n    challenger_dataset,\n    region=region,\n)",
            "outputs": [{"data": {"text/html": table.to_html()}}],
        }
        for metric_function_name, table in TABLES.items()
    ]
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")


def test_score_rows_match_the_website_notebook_parser_at_printed_precision(tmp_path):
    notebook_path = tmp_path / "challenger.global.report.ipynb"
    _notebook_with_tables(notebook_path)
    parsed_scores = get_all_model_scores_from_notebook(str(notebook_path), "challenger")
    parsed_values = {
        (metric_key, depth, variable_name, variable.unit, int(lead_day)): value
        for metric_key, score in parsed_scores.items()
        for depth, depth_scores in score.depths.items()
        for variable_name, variable in depth_scores.variables.items()
        for lead_day, value in variable.data.items()
    }

    row_values = {
        (
            _PARSER_METRIC_KEYS[row["metric"], row["reference"]],
            row["depth"].capitalize() if row["metric"] == "rmsd_of_variables" else "flat",
            row["variable"],
            row["unit"],
            row["lead_day"],
        ): round(row["value"], 6)
        for metric_function_name, table in TABLES.items()
        for row in score_rows(metric_function_name, table)
    }

    assert len(row_values) == 4 * 3 + 4 * 3 + 2
    assert row_values == parsed_values


def test_score_rows_keep_full_precision_and_the_row_columns():
    gridded_rows = score_rows("rmsd_of_variables_compared_to_glorys_reanalysis", TABLES[next(iter(TABLES))])
    class4_rows = score_rows("rmsd_of_variables_compared_to_observations", _class4_table())
    lagrangian_rows = score_rows("deviation_of_lagrangian_trajectories_compared_to_glo12_analysis", _lagrangian_table())

    assert gridded_rows[0] == {
        "metric": "rmsd_of_variables",
        "reference": "glorys_reanalysis",
        "variable": "sea surface height",
        "standard_name": "sea_surface_height_above_geoid",
        "depth": "surface",
        "unit": "m",
        "lead_day": 1,
        "value": float(_gridded_table().iloc[0, 0]),
        "missing_fraction": 0.000671,
    }
    sea_level_anomaly_row = next(row for row in class4_rows if row["variable"] == "sea level anomaly")
    assert sea_level_anomaly_row["standard_name"] == "sea_surface_height_above_geoid"
    assert (sea_level_anomaly_row["observations"], sea_level_anomaly_row["missing"]) == (1002, 2)
    assert type(sea_level_anomaly_row["observations"]) is int
    assert [(row["lead_day"], row["standard_name"], row["unit"]) for row in lagrangian_rows] == [
        (2, None, "km"),
        (3, None, "km"),
    ]


def test_a_metric_that_returned_a_message_has_no_rows():
    message_table = pandas.DataFrame({MESSAGE_COLUMN: ["Observation-based Class IV scores were not computed"]})

    assert score_rows("rmsd_of_variables_compared_to_observations", message_table) == []


def test_write_scores_file_writes_missing_values_as_null(tmp_path):
    table = _lagrangian_table()
    table.iloc[0, 1] = numpy.nan
    path = tmp_path / "challenger.global.scores.json"

    write_scores_file(
        {"rows": score_rows("deviation_of_lagrangian_trajectories_compared_to_glo12_analysis", table)}, path
    )

    assert [row["value"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"]] == [11.027288, None]
