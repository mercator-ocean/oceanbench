# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

import pytest  # noqa: E402

from helpers.live_validation_report import (
    ForecastValidationMetadata,
    render_forecast_validation_page,
)  # noqa: E402


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
        "<thead><tr><th></th><th>Lead day 1 (init)</th><th>Lead day 10</th></tr></thead>"
        "<tbody>"
        "<tr><th>Class-4 drifter trajectory deviation mean (km)</th><td>12.3</td><td>45.6</td></tr>"
        "<tr><th>Class-4 matched drifter count</th><td>9</td><td>5</td></tr>"
        "</tbody>"
        "</table>"
    )


def _full_score_preview() -> dict:
    # The variable-driven manifest preview the listing page renders; the report page reuses it
    # verbatim so the two stay consistent. Temperature values match the notebook table below.
    return {
        "metrics": [
            {
                "label": "Temperature, surface",
                "unit": "C",
                "lead_values": {"1": 1.2, "10": 1.4},
            },
            {
                "label": "Salinity, 0-5 m",
                "unit": "PSU",
                "lead_values": {"1": 0.3, "10": 0.4},
            },
            {
                "label": "Currents, 15 m",
                "unit": "m/s",
                "lead_values": {"1": 0.29, "10": 0.34},
            },
            {
                "label": "Sea level anomaly",
                "unit": "m",
                "lead_values": {"1": 0.05, "10": 0.07},
            },
            {
                "label": "Surface drifter deviation",
                "unit": "km",
                "lead_values": {"1": 12.3, "10": 45.6},
            },
        ]
    }


def _drifter_only_score_preview() -> dict:
    return {
        "metrics": [
            {
                "label": "Surface drifter deviation",
                "unit": "km",
                "lead_values": {"1": 12.3, "10": 45.6},
            },
        ]
    }


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
                "outputs": [
                    {"data": {"text/html": '<iframe class="drifter-widget"></iframe>'}}
                ],
            },
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_observation_error_explorer",
                "outputs": [
                    {"data": {"text/html": '<iframe class="class4-widget"></iframe>'}}
                ],
            },
        ]
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")


def test_render_forecast_validation_page_uses_notebook_outputs_without_notebook_chrome(
    tmp_path: Path,
) -> None:
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
            note="Forecast regenerated on demand.",
            system_id="octo-glonet-p1d",
            score_preview=_full_score_preview(),
        ),
    )

    assert "Class IV evaluation is complete" in html
    assert "GLONET" in html
    assert "2026-05-13" in html
    assert "2026-05-23" in html
    assert "Forecast regenerated on demand." in html
    assert "validation-note" in html
    # The representative-scores widget is the manifest preview (same renderer as the listing page).
    assert "Temperature, surface" in html
    assert "Sea level anomaly" in html
    assert "Surface drifter deviation" in html
    assert "Lead 10" in html
    assert 'data-tooltip="Lead 1: 1.200 C"' in html
    assert 'data-tooltip="Lead 10: 1.400 C"' in html
    assert "validation-sparkline-tooltip" in html
    assert 'class="validation-sparkline-point"' in html
    # The detailed RMSD + drifter tables still come from the executed notebook.
    assert "Drifter trajectory scores" in html
    assert "Drifter trajectory divergence" in html
    assert "Lead 1 (init)" in html
    assert "class-4 drifter trajectory deviation mean" in html
    assert "<td>12.300</td>" in html
    assert '<iframe class="drifter-widget"></iframe>' in html
    assert "<title>" not in html
    assert '<section class="validation-method-note">' in html
    assert "<details" not in html
    assert '<iframe class="class4-widget"></iframe>' in html
    assert "Representative lead-time scores" in html
    assert html.index("Representative lead-time scores") < html.index(
        "Detailed Class IV RMSD"
    )
    assert html.index("Detailed Class IV RMSD") < html.index(
        "Drifter trajectory scores"
    )
    assert html.index("Drifter trajectory scores") < html.index(
        "Observation error maps"
    )
    assert html.index("Observation error maps") < html.index(
        "Drifter trajectory divergence"
    )
    assert html.index('<iframe class="class4-widget"></iframe>') < html.index(
        '<iframe class="drifter-widget"></iframe>'
    )
    assert "evaluation_report" not in html
    assert "cell" not in html.lower()


def test_render_forecast_validation_page_supports_glonet2_ibi_system(
    tmp_path: Path,
) -> None:
    notebook_path = tmp_path / "glonet2-ibi-experimental.latest.ibi.report.ipynb"
    _write_notebook(notebook_path)

    html = render_forecast_validation_page(
        notebook_path,
        ForecastValidationMetadata(
            system_label="GLONET2 IBI (experimental)",
            forecast_init="2026-05-13",
            validated_lead_days="1-10 days",
            observation_cutoff="2026-05-23",
            status="Complete",
            system_id="octo-glonet2-ibi-p1d",
            score_preview=_full_score_preview(),
        ),
    )

    assert "Class IV evaluation is complete" in html
    assert "GLONET2 IBI (experimental)" in html
    assert "Temperature, surface" in html
    assert "Drifter trajectory scores" in html


def _write_surface_only_notebook(notebook_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_drifter_trajectory_deviation",
                "outputs": [{"data": {"text/html": _drifter_score_table()}}],
            },
            {
                "cell_type": "code",
                "source": "evaluation_report.class4_drifter_trajectory_explorer",
                "outputs": [
                    {"data": {"text/html": '<iframe class="drifter-widget"></iframe>'}}
                ],
            },
        ]
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")


def test_render_forecast_validation_page_self_suppresses_class4_rmsd_for_surface_only(
    tmp_path: Path,
) -> None:
    notebook_path = tmp_path / "glonet.latest.global.report.ipynb"
    _write_surface_only_notebook(notebook_path)

    html = render_forecast_validation_page(
        notebook_path,
        ForecastValidationMetadata(
            system_label="GLONET HR",
            forecast_init="2026-05-13",
            validated_lead_days="1-10 days",
            observation_cutoff="2026-05-23",
            status="Complete",
            system_id="octo-glonet-p1d",
            score_preview=_drifter_only_score_preview(),
        ),
    )

    # One self-suppressing path: the unified intro + the manifest preview (drifter only).
    assert "Class IV evaluation is complete" in html
    assert "Representative lead-time scores" in html
    assert "Surface drifter deviation" in html
    assert "Drifter trajectory scores" in html
    assert "Drifter trajectory divergence" in html
    assert "class-4 drifter trajectory deviation mean" in html
    assert '<iframe class="drifter-widget"></iframe>' in html
    # The gridded Class IV RMSD sections self-suppress (no scorable observation variables).
    assert "Detailed Class IV RMSD" not in html
    assert "Observation error maps" not in html
    assert "Temperature, surface" not in html
    assert "evaluation_report" not in html


def test_render_forecast_validation_page_rejects_unmapped_system(
    tmp_path: Path,
) -> None:
    notebook_path = tmp_path / "mystery.latest.global.report.ipynb"
    _write_notebook(notebook_path)

    with pytest.raises(ValueError, match="octo-mystery-p1d"):
        render_forecast_validation_page(
            notebook_path,
            ForecastValidationMetadata(
                system_label="Mystery system",
                forecast_init="2026-05-13",
                validated_lead_days="1-10 days",
                observation_cutoff="2026-05-23",
                status="Complete",
                system_id="octo-mystery-p1d",
            ),
        )
