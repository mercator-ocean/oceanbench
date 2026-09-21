# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Challengers whose report notebooks ship with this branch instead of the bucket.

Preview branches carry a locally evaluated system so the website can show it next
to the officially published ones. Nothing here is uploaded and nothing here reaches
the public bucket; the notebooks are read straight from website/local_reports.
"""

import os

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_REPORTS_DIRECTORY = os.path.join(_REPOSITORY_ROOT, "local_reports")

# version -> challenger name -> report file name inside LOCAL_REPORTS_DIRECTORY/<version>
LOCAL_CHALLENGERS = {
    "0.5.0": {
        "hclimrep": ["global"],
    },
}


def local_challengers_for_version(version: str) -> list[str]:
    return list(LOCAL_CHALLENGERS.get(version, {}).keys())


def local_report_path(version: str, challenger_name: str, region_id: str) -> str | None:
    if region_id not in LOCAL_CHALLENGERS.get(version, {}).get(challenger_name, []):
        return None
    path = os.path.join(LOCAL_REPORTS_DIRECTORY, version, f"{challenger_name}.{region_id}.report.ipynb")
    return path if os.path.exists(path) else None


def local_reports(version: str) -> list[tuple[str, str, str]]:
    return [
        (challenger_name, region_id, path)
        for challenger_name, region_ids in LOCAL_CHALLENGERS.get(version, {}).items()
        for region_id in region_ids
        if (path := local_report_path(version, challenger_name, region_id))
    ]
