# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.live_validation_report import ForecastValidationMetadata, render_forecast_validation_page  # noqa: E402


def _score_table() -> str:
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th><th>Lead day 10</th></tr></thead>"
        "<tbody>"
        "<tr><th>Temperature (C) [sea_water_potential_temperature]{surface}</th><td>1.2</td><td>1.4</td></tr>"
        "<tr><th>Salinity (PSU) [sea_water_salinity]{0-5m}</th><td>0.3</td><td>0.4</td></tr>"
        "<tr><th>Zonal current (m/s) [eastward_sea_water_velocity]{15m}</th><td>0.2</td><td>0.25</td></tr>"
        "<tr><th>Meridional current (m/s) [northward_sea_water_velocity]{15m}</th><td>0.21</td><td>0.24</td></tr>"
        "</tbody>"
        "</table>"
    )


def _drifter_score_table() -> str:
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th><th>Lead day 10</th></tr></thead>"
        "<tbody>"
        "<tr><th>Class-4 drifter trajectory deviation mean (km)</th><td>12.3</td><td>45.6</td></tr>"
        "<tr><th>Class-4 matched drifter count</th><td>9</td><td>5</td></tr>"
        "</tbody>"
        "</table>"
    )


def _write_notebook(notebook_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_observation.rmsd",
                "outputs": [{"data": {"text/html": _score_table()}}],
            },
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_drifter_trajectory_deviation",
                "outputs": [{"data": {"text/html": _drifter_score_table()}}],
            },
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_drifter_trajectory_explorer",
                "outputs": [{"data": {"text/html": '<iframe class="drifter-widget"></iframe>'}}],
            },
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_observation_error_explorer",
                "outputs": [{"data": {"text/html": '<iframe class="class4-widget"></iframe>'}}],
            },
        ]
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")


def test_render_forecast_validation_page_uses_notebook_outputs_without_notebook_chrome(tmp_path: Path) -> None:
    notebook_path = tmp_path / "glonet.latest.global.report.ipynb"
    _write_notebook(notebook_path)

    html = render_forecast_validation_page(
        notebook_path,
        ForecastValidationMetadata(
            system_label="GLONET",
            forecast_init="2026-05-13",
            validated_lead_days="1-10 days",
            observation_cutoff="2026-05-23",
            status="Complete",
            note="Demonstration only: regenerated forecast.",
        ),
    )

    assert "Class IV validation is complete" in html
    assert "GLONET" in html
    assert "2026-05-13" in html
    assert "2026-05-23" in html
    assert "Demonstration only: regenerated forecast." in html
    assert "validation-demo-note" in html
    assert "Temperature, surface" in html
    assert "Lead 10" in html
    assert 'data-tooltip="Lead 1: 1.200 C"' in html
    assert 'data-tooltip="Lead 10: 1.400 C"' in html
    assert "validation-sparkline-tooltip" in html
    assert 'class="validation-sparkline-point"' in html
    assert "Drifter trajectory scores" in html
    assert "Drifter trajectory divergence" in html
    assert "class-4 drifter trajectory deviation mean" in html
    assert "12.300 km" not in html
    assert "<td>12.300</td>" in html
    assert '<iframe class="drifter-widget"></iframe>' in html
    assert "<title>" not in html
    assert '<section class="validation-method-note">' in html
    assert "<details" not in html
    assert '<iframe class="class4-widget"></iframe>' in html
    assert html.index("Lead-time scores") < html.index("Detailed Class IV RMSD")
    assert html.index("Detailed Class IV RMSD") < html.index("Drifter trajectory scores")
    assert html.index("Drifter trajectory scores") < html.index("Observation error maps")
    assert html.index("Observation error maps") < html.index("Drifter trajectory divergence")
    assert html.index('<iframe class="class4-widget"></iframe>') < html.index(
        '<iframe class="drifter-widget"></iframe>'
    )
    assert "evaluation_report" not in html
    assert "cell" not in html.lower()
