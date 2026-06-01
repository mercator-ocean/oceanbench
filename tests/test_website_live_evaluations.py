# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path


WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"


def test_website_has_live_evaluations_nav_and_page() -> None:
    quarto_config = (WEBSITE_DIRECTORY / "_quarto.yml").read_text(encoding="utf-8")
    live_page = (WEBSITE_DIRECTORY / "live-evaluations.qmd").read_text(encoding="utf-8")

    assert "live-evaluations.qmd" in quarto_config
    assert "NRT forecast validation" in quarto_config
    assert "Near-real-time forecast validation" in live_page
    assert "| GLONET |" in live_page
    assert "Forecast init" in live_page
    assert "Validated lead days" in live_page
    assert "Observation cutoff" in live_page
    assert "Status" in live_page
    assert "2026-05-13" in live_page
    assert "2026-05-23" in live_page
    assert "1-10 days" in live_page
    assert "Complete" in live_page
    assert "2026-06-01" not in live_page
    assert "date remapping" not in live_page
    assert "reports/glonet.latest.global.report.html" in live_page
