# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.live_validation_manifest import (  # noqa: E402
    forecast_validation_metadata,
    render_live_validation_preview_panel,
    render_live_validation_summary,
    render_live_validation_table,
    report_notebook_path,
)


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-06-08T13:24:19Z",
                "evaluations": [
                    {
                        "system_id": "octo-glonet-p1d",
                        "system_label": "GLONET",
                        "forecast_lead_days": 10,
                        "forecast_init": "2026-05-13",
                        "validated_lead_days": "1-10 days",
                        "observation_cutoff": "2026-05-23",
                        "status": "Complete",
                        "report_notebook": "glonet.latest.global.report.ipynb",
                        "score_preview": {
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
                                    "label": "Zonal current, 15 m",
                                    "unit": "m/s",
                                    "lead_values": {"1": 0.2, "10": 0.25},
                                },
                                {
                                    "label": "Meridional current, 15 m",
                                    "unit": "m/s",
                                    "lead_values": {"1": 0.21, "10": 0.24},
                                },
                            ]
                        },
                        "note": "Forecast regenerated on demand.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_manifest_without_score_preview(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-06-08T13:24:19Z",
                "evaluations": [
                    {
                        "system_id": "octo-glonet-p1d",
                        "system_label": "GLONET",
                        "forecast_lead_days": 10,
                        "forecast_init": "2026-05-13",
                        "validated_lead_days": "1-10 days",
                        "observation_cutoff": "2026-05-23",
                        "status": "Complete",
                        "report_notebook": "glonet.latest.global.report.ipynb",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_live_validation_table_and_report_metadata_are_manifest_driven(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nrt-validation-manifest.json"
    _write_manifest(manifest_path)

    summary_html = render_live_validation_summary(manifest_path)
    html = render_live_validation_table(manifest_path)
    preview_html = render_live_validation_preview_panel(manifest_path)
    metadata = forecast_validation_metadata(manifest_path, "octo-glonet-p1d")

    assert "Forecast init" in summary_html
    assert "2026-05-13" in summary_html
    assert "Observation cutoff" in summary_html
    assert "2026-05-23" in summary_html
    assert "Completed systems" in summary_html
    assert "1 / 1" in summary_html
    assert "Last updated" in summary_html
    assert "2026-06-08 13:24:19 UTC" in summary_html
    assert "GLONET" in html
    assert "2026-05-13" not in html
    assert "2026-05-23" not in html
    assert "1-10 days" in html
    assert "Forecast horizon" in html
    assert "Evaluated leads" in html
    assert "10 days" in html
    assert "Score preview" not in html
    assert "data-live-preview-target" in html
    assert "live-preview-select active" in html
    assert "live-evaluation-row is-selected" in html
    assert "Report" in html
    assert "Open report" not in html
    assert "Challenger reports" not in html
    assert "live-status" not in html
    assert "glonet-forecast-validation.html" in html
    assert "Representative lead-time scores" in preview_html
    assert "Select a challenger to preview selected RMSD metrics" in preview_html
    assert "reports contain the full evaluation tables" in preview_html
    assert "live-preview-tab" not in preview_html
    assert "<h3>GLONET</h3>" not in preview_html
    assert "Temperature, surface" in preview_html
    assert "Salinity, 0-5 m" in preview_html
    assert "Zonal current, 15 m" in preview_html
    assert "Meridional current, 15 m" in preview_html
    assert "Lead 1" in preview_html
    assert "Lead 10" in preview_html
    assert "1.200 C" in preview_html
    assert "1.400 C" in preview_html
    assert "validation-score-grid" in preview_html
    assert "validation-sparkline-tooltip" in preview_html
    assert report_notebook_path(manifest_path, "octo-glonet-p1d") == "reports/glonet.latest.global.report.ipynb"
    assert metadata.note == "Forecast regenerated on demand."


def test_live_validation_preview_panel_shows_unavailable_without_manifest_scores(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nrt-validation-manifest.json"
    _write_manifest_without_score_preview(manifest_path)

    preview_html = render_live_validation_preview_panel(manifest_path)

    assert "Preview unavailable" in preview_html
    assert "validation-score-grid" not in preview_html
