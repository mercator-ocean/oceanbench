# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import html
import os

from helpers.s3_discovery import default_version

OUTPUT_DIRECTORY = os.path.abspath(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
REPORTS_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "reports")
REPORT_PAGE_FILE_NAME = "index.html"


def _report_names(version_directory: str) -> list[str]:
    return sorted(
        report_name
        for report_name in os.listdir(version_directory)
        if os.path.isfile(os.path.join(version_directory, report_name, REPORT_PAGE_FILE_NAME))
    )


def _redirect_html(version: str, report_name: str) -> str:
    redirect_url = f"../{version}/{report_name}/"
    escaped_redirect_url = html.escape(redirect_url, quote=True)
    escaped_report_name = html.escape(report_name)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={escaped_redirect_url}">
  <link rel="canonical" href="{escaped_redirect_url}">
  <title>Redirecting to {escaped_report_name}</title>
  <script>location.replace("{escaped_redirect_url}");</script>
</head>
<body>
  <p>Redirecting to the latest report…</p>
</body>
</html>
"""


def _write_redirect(version: str, report_name: str) -> None:
    redirect_path = os.path.join(REPORTS_DIRECTORY, report_name, REPORT_PAGE_FILE_NAME)
    os.makedirs(os.path.dirname(redirect_path), exist_ok=True)
    with open(redirect_path, "w", encoding="utf-8") as file:
        file.write(_redirect_html(version, report_name))


def main() -> None:
    version = default_version()
    report_names = _report_names(os.path.join(REPORTS_DIRECTORY, version))
    for report_name in report_names:
        _write_redirect(version, report_name)
    print(f"Wrote {len(report_names)} report redirect stubs for version {version}.")


if __name__ == "__main__":
    main()
