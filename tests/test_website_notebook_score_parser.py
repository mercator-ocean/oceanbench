# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.notebook_score_parser import get_all_model_scores_from_notebook  # noqa: E402
from helpers.notebook_score_parser import get_model_score_from_file  # noqa: E402
from helpers.notebook_score_parser import get_model_score_from_notebook  # noqa: E402


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


def _metric_cell(metric_call: str, html_table: str) -> dict:
    return {
        "cell_type": "code",
        "source": [f"{metric_call}(challenger_dataset)"],
        "outputs": [{"data": {"text/html": [html_table]}}],
    }


def _write_notebook(notebook_path: Path) -> None:
    notebook = {
        "cells": [
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
                _score_table(
                    [
                        ("Temperature (C) [sea_water_potential_temperature]{100m}", 1.1, 1.2),
                    ]
                ),
            ),
            _metric_cell(
                "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis",
                _score_table(
                    [
                        ("Mixed Layer Depth (m) [mixed_layer_depth]", 1.3, 1.4),
                    ]
                ),
            ),
            _metric_cell(
                "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis",
                _score_table(
                    [
                        ("Lagrangian trajectory deviation (km) []", 2.1, 2.2),
                    ]
                ),
            ),
        ]
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")


def test_parser_extracts_scores_from_local_report_notebook(tmp_path):
    notebook_path = tmp_path / "glo12.report.ipynb"
    _write_notebook(notebook_path)

    scores = get_all_model_scores_from_notebook(str(notebook_path), "glo12")

    assert set(scores) == {
        "rmsd_variables_observations",
        "rmsd_variables_glorys",
        "rmsd_mld_glorys",
        "lagrangian_glorys",
    }

    observation_score = scores["rmsd_variables_observations"]
    assert observation_score.depths["Surface"].variables["temperature"].data == {"1": 0.1, "2": 0.2}
    assert observation_score.depths["Surface"].variables["sea level anomaly"].data == {"1": 0.5, "2": 0.6}
    assert observation_score.depths["0-5m"].variables["salinity"].data == {"1": 0.3, "2": 0.4}
    assert observation_score.depths["15m"].variables["zonal current"].data == {"1": 0.7, "2": 0.8}
    assert observation_score.depths["15m"].variables["meridional current"].data == {"1": 0.9, "2": 1.0}

    glorys_score = scores["rmsd_variables_glorys"]
    assert glorys_score.depths["100m"].variables["temperature"].standard_name == "sea_water_potential_temperature"
    assert glorys_score.depths["100m"].variables["temperature"].data == {"1": 1.1, "2": 1.2}

    lagrangian_score = scores["lagrangian_glorys"]
    lagrangian_variable = lagrangian_score.depths["flat"].variables["lagrangian trajectory deviation"]
    assert lagrangian_variable.standard_name == ""
    assert lagrangian_variable.data == {"1": 2.1, "2": 2.2}


def test_track_parser_merges_metric_fragments(tmp_path):
    notebook_path = tmp_path / "glo12.report.ipynb"
    _write_notebook(notebook_path)

    score = get_model_score_from_notebook(str(notebook_path), "GLO12", "glorys_reanalysis")

    assert score.name == "GLO12"
    assert score.depths["100m"].variables["temperature"].data == {"1": 1.1, "2": 1.2}
    assert score.depths["flat"].variables["mixed layer depth"].data == {"1": 1.3, "2": 1.4}
    assert score.depths["flat"].variables["lagrangian trajectory deviation"].data == {"1": 2.1, "2": 2.2}


def test_model_score_roundtrip_from_file(tmp_path):
    score_path = tmp_path / "glo12.json"
    score_path.write_text(
        json.dumps(
            {
                "name": "GLO12",
                "depths": {
                    "Surface": {
                        "variables": {
                            "temperature": {
                                "standard_name": "sea_water_potential_temperature",
                                "unit": "C",
                                "data": {"1": 0.1},
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    score = get_model_score_from_file(str(score_path))

    assert score.name == "GLO12"
    assert score.depths["Surface"].variables["temperature"].data == {"1": 0.1}
