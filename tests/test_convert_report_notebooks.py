# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

import jsonschema
from oceanbench.core.scores import score_rows
from score_tables import TABLES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "helper_scripts"))

from convert_report_notebooks import scores_document_from_notebook  # noqa: E402

NOTEBOOK_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "glonet_sample.global.report.ipynb"
SAMPLE_SCORES_FILE = PROJECT_ROOT / "assets" / "glonet_sample.global.scores.json"
SCHEMA = json.loads((PROJECT_ROOT / "oceanbench" / "schemas" / "scores-v1.schema.json").read_text(encoding="utf-8"))
NOTEBOOK_METADATA = {"papermill": {"end_time": "2026-01-01T00:00:00.000000"}}


def _document_from_fixture() -> dict:
    notebook = json.loads(NOTEBOOK_FIXTURE.read_text(encoding="utf-8"))
    return scores_document_from_notebook(notebook, "glonet_sample", "global", "0.6.0")


def _score_table(rows: list[tuple[str, float, float]]) -> str:
    body = "\n".join(
        f"<tr><th>{label}</th><td>{lead_day_1}</td><td>{lead_day_2}</td></tr>" for label, lead_day_1, lead_day_2 in rows
    )
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th><th>Lead day 2</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def _score_table_with_count_columns(rows: list[tuple[str, float, float, int, int, float]]) -> str:
    body = "\n".join(
        f"<tr><th>{label}</th><td>{lead_day_1}</td><td>{lead_day_2}</td>"
        f"<td>{count}</td><td>{missing}</td><td>{missing_fraction}</td></tr>"
        for label, lead_day_1, lead_day_2, count, missing, missing_fraction in rows
    )
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th><th>Lead day 2</th>"
        "<th>Observations</th><th>Missing</th><th>Missing fraction</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def _metric_cell(metric_call: str, html_table: str) -> dict:
    return {
        "cell_type": "code",
        "source": [f"{metric_call}(challenger_dataset)"],
        "outputs": [{"data": {"text/html": [html_table]}}],
    }


def _document_from_cells(cells: list[dict]) -> dict:
    return scores_document_from_notebook({"cells": cells, "metadata": NOTEBOOK_METADATA}, "glo12", "global", "0.6.0")


def _lead_day_values(document: dict) -> dict:
    return {
        (row["metric"], row["reference"], row["variable"], row["depth"]): {
            lead_day_row["lead_day"]: lead_day_row["value"]
            for lead_day_row in document["rows"]
            if (lead_day_row["metric"], lead_day_row["reference"], lead_day_row["variable"], lead_day_row["depth"])
            == (row["metric"], row["reference"], row["variable"], row["depth"])
        }
        for row in document["rows"]
    }


def test_a_converted_report_notebook_is_a_valid_scores_file() -> None:
    jsonschema.validate(_document_from_fixture(), SCHEMA)


def test_a_converted_report_notebook_has_the_scores_of_the_sample_scores_file() -> None:
    sample_scores = json.loads(SAMPLE_SCORES_FILE.read_text(encoding="utf-8"))

    assert _lead_day_values(_document_from_fixture()) == _lead_day_values(sample_scores)


def test_the_scores_of_each_metric_are_extracted_from_the_notebook_tables() -> None:
    document = _document_from_cells(
        [
            _metric_cell(
                "oceanbench.metrics.rmsd_of_variables_compared_to_observations",
                _score_table(
                    [
                        ("Temperature (C) [sea_water_potential_temperature]{surface}", 0.1, 0.2),
                        ("Salinity (PSU) [sea_water_salinity]{0-5m}", 0.3, 0.4),
                        ("Sea level anomaly (m) [sea_surface_height_above_geoid]{surface}", 0.5, 0.6),
                        ("Zonal current (m/s) [eastward_sea_water_velocity]{15m}", 0.7, 0.8),
                        ("Meridional current (m/s) [northward_sea_water_velocity]{15m}", 0.9, 1.0),
                    ]
                ),
            ),
            _metric_cell(
                "oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis",
                _score_table([("Temperature (C) [sea_water_potential_temperature]{100m}", 1.1, 1.2)]),
            ),
            _metric_cell(
                "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis",
                _score_table([("Mixed Layer Depth (m) [mixed_layer_depth]", 1.3, 1.4)]),
            ),
            _metric_cell(
                "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis",
                _score_table(
                    [
                        ("Zonal geostrophic current (m/s) [geostrophic_eastward_sea_water_velocity]", 1.5, 1.6),
                        ("Meridional geostrophic current (m/s) [geostrophic_northward_sea_water_velocity]", 1.7, 1.8),
                    ]
                ),
            ),
            _metric_cell(
                "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis",
                _score_table([("Lagrangian trajectory deviation (km) []", 2.1, 2.2)]),
            ),
        ]
    )

    values = _lead_day_values(document)
    glorys_temperature = next(row for row in document["rows"] if row["depth"] == "100m")
    lagrangian_deviation = next(row for row in document["rows"] if row["metric"].startswith("deviation_of"))

    assert values == {
        ("rmsd_of_variables", "observations", "temperature", "surface"): {1: 0.1, 2: 0.2},
        ("rmsd_of_variables", "observations", "salinity", "0-5m"): {1: 0.3, 2: 0.4},
        ("rmsd_of_variables", "observations", "sea level anomaly", "surface"): {1: 0.5, 2: 0.6},
        ("rmsd_of_variables", "observations", "zonal current", "15m"): {1: 0.7, 2: 0.8},
        ("rmsd_of_variables", "observations", "meridional current", "15m"): {1: 0.9, 2: 1.0},
        ("rmsd_of_variables", "glorys_reanalysis", "temperature", "100m"): {1: 1.1, 2: 1.2},
        ("rmsd_of_mixed_layer_depth", "glorys_reanalysis", "mixed layer depth", "surface"): {1: 1.3, 2: 1.4},
        ("rmsd_of_geostrophic_currents", "glorys_reanalysis", "zonal geostrophic current", "surface"): {1: 1.5, 2: 1.6},
        ("rmsd_of_geostrophic_currents", "glorys_reanalysis", "meridional geostrophic current", "surface"): {
            1: 1.7,
            2: 1.8,
        },
        ("deviation_of_lagrangian_trajectories", "glorys_reanalysis", "lagrangian trajectory deviation", "surface"): {
            1: 2.1,
            2: 2.2,
        },
    }
    assert glorys_temperature["standard_name"] == "sea_water_potential_temperature"
    assert lagrangian_deviation["standard_name"] is None
    assert document["lead_days_count"] == 2


def test_the_count_columns_of_a_table_are_not_scores() -> None:
    document = _document_from_cells(
        [
            _metric_cell(
                "oceanbench.metrics.rmsd_of_variables_compared_to_observations",
                _score_table_with_count_columns(
                    [("Salinity (PSU) [sea_water_salinity]{0-5m}", 0.3, 0.4, 1234, 56, 0.0123)]
                ),
            )
        ]
    )

    assert _lead_day_values(document) == {
        ("rmsd_of_variables", "observations", "salinity", "0-5m"): {1: 0.3, 2: 0.4},
    }


def _notebook_with_tables() -> dict:
    cells = [
        {
            "cell_type": "code",
            "source": f"oceanbench.metrics.{metric_function_name}(\n    challenger_dataset,\n    region=region,\n)",
            "outputs": [{"data": {"text/html": table.to_html()}}],
        }
        for metric_function_name, table in TABLES.items()
    ]
    return {"cells": cells, "metadata": NOTEBOOK_METADATA}


def _row_key(row: dict) -> tuple:
    return row["metric"], row["reference"], row["depth"], row["variable"], row["unit"], row["lead_day"]


def test_converted_notebook_rows_match_the_score_rows_at_printed_precision() -> None:
    converted_values = {
        _row_key(row): row["value"]
        for row in scores_document_from_notebook(_notebook_with_tables(), "challenger", "global", "0.6.0")["rows"]
    }

    row_values = {
        _row_key(row): round(row["value"], 6)
        for metric_function_name, table in TABLES.items()
        for row in score_rows(metric_function_name, table)
    }

    assert len(row_values) == 4 * 3 + 4 * 3 + 2
    assert row_values == converted_values
