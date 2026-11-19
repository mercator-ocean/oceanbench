# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path

import jsonschema
import numpy
import pandas
import pytest

from oceanbench.core.metrics import MESSAGE_COLUMN
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.core.scores import SCORES_SCHEMA_VERSION, score_rows, write_scores_file

from score_tables import TABLES, class4_table, gridded_table, lagrangian_table


def test_score_rows_keep_full_precision_and_the_row_columns():
    gridded_rows = score_rows("rmsd_of_variables_compared_to_glorys_reanalysis", TABLES[next(iter(TABLES))])
    class4_rows = score_rows("rmsd_of_variables_compared_to_observations", class4_table())
    lagrangian_rows = score_rows("deviation_of_lagrangian_trajectories_compared_to_glo12_analysis", lagrangian_table())

    assert gridded_rows[0] == {
        "metric": "rmsd_of_variables",
        "reference": "glorys_reanalysis",
        "variable": "sea surface height",
        "standard_name": "sea_surface_height_above_geoid",
        "depth": "surface",
        "unit": "m",
        "lead_day": 1,
        "value": float(gridded_table().iloc[0, 0]),
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
    table = lagrangian_table()
    table.iloc[0, 1] = numpy.nan
    path = tmp_path / "challenger.global.scores.json"

    write_scores_file(
        {"rows": score_rows("deviation_of_lagrangian_trajectories_compared_to_glo12_analysis", table)}, path
    )

    assert [row["value"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"]] == [11.027288, None]


SAMPLE_SCORES_FILES = sorted(Path(__file__).parent.parent.joinpath("assets").glob("*.scores.json"))


@pytest.mark.parametrize("path", SAMPLE_SCORES_FILES, ids=lambda path: path.name)
def test_sample_scores_files_follow_the_scores_schema(path):
    validate_against_schema(json.loads(path.read_text(encoding="utf-8")), f"scores-v{SCORES_SCHEMA_VERSION}")


def test_scores_schema_rejects_an_unknown_schema_version():
    document = json.loads(SAMPLE_SCORES_FILES[0].read_text(encoding="utf-8")) | {"schema_version": 2}

    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(document, f"scores-v{SCORES_SCHEMA_VERSION}")
