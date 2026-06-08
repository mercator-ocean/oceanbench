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
    render_live_validation_table,
    report_notebook_path,
)


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluations": [
                    {
                        "system_id": "octo-glonet-p1d",
                        "system_label": "GLONET",
                        "forecast_init": "2026-05-13",
                        "validated_lead_days": "1-10 days",
                        "observation_cutoff": "2026-05-23",
                        "status": "Complete",
                        "report_notebook": "glonet.latest.global.report.ipynb",
                        "note": "Forecast regenerated on demand.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_live_validation_table_and_report_metadata_are_manifest_driven(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nrt-validation-manifest.json"
    _write_manifest(manifest_path)

    html = render_live_validation_table(manifest_path)
    metadata = forecast_validation_metadata(manifest_path, "octo-glonet-p1d")

    assert "GLONET" in html
    assert "2026-05-13" in html
    assert "glonet-forecast-validation.html" in html
    assert report_notebook_path(manifest_path, "octo-glonet-p1d") == "reports/glonet.latest.global.report.ipynb"
    assert metadata.note == "Forecast regenerated on demand."
