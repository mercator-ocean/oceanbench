# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"


def test_website_links_to_prebuilt_report_html() -> None:
    challengers_page = (WEBSITE_DIRECTORY / "challengers.qmd").read_text(encoding="utf-8")
    interactive_scores = (WEBSITE_DIRECTORY / "interactive-scores.js").read_text(encoding="utf-8")

    assert 'report_metadata("reports", challenger_name, region_id)["report_url"]' in challengers_page
    assert 'report_metadata("reports", challenger_name, region_id)["notebook_url"]' in challengers_page
    assert 'href="reports/{challenger_name}.{region_id}.report.html"' not in challengers_page
    assert 'href="reports/${name}.${regionId}.report.html"' not in interactive_scores
    assert "modelReportLink(name, regionId)" in interactive_scores
    assert "reportUrls = parsedData.report_urls || {}" in interactive_scores


def test_static_prerender_does_not_download_report_notebooks() -> None:
    download_reports = (WEBSITE_DIRECTORY / "download_reports.py").read_text(encoding="utf-8")

    assert "download_notebook" not in download_reports
    assert "download_scores" in download_reports
    assert "No complete prebuilt report packages were discovered." in download_reports
