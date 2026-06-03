# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path


WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"


def test_website_has_live_evaluations_nav_and_page() -> None:
    quarto_config = (WEBSITE_DIRECTORY / "_quarto.yml").read_text(encoding="utf-8")
    live_page = (WEBSITE_DIRECTORY / "live-evaluations.qmd").read_text(encoding="utf-8")
    clean_report_page = (WEBSITE_DIRECTORY / "glonet-forecast-validation.qmd").read_text(encoding="utf-8")

    assert "live-evaluations.qmd" in quarto_config
    assert "NRT forecast validation" in quarto_config
    assert "Near-real-time forecast validation" in live_page
    assert "render_live_validation_table" in live_page
    assert "reports/nrt-validation-manifest.json" in live_page
    assert "| GLONET |" not in live_page
    assert "2026-05-13" not in live_page
    assert "2026-05-23" not in live_page
    assert "reports/glonet.latest.global.report.html" not in live_page
    assert "render_forecast_validation_page" in clean_report_page
    assert "forecast_validation_metadata" in clean_report_page
    assert "report_notebook_path" in clean_report_page
    assert "reports/nrt-validation-manifest.json" in clean_report_page
    assert "2026-05-13" not in clean_report_page
