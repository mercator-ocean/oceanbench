# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import html
import os

from helpers.s3_discovery import default_version

OUTPUT_DIRECTORY = os.path.abspath(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
REPORTS_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "reports")


def _report_file_names(version_directory: str) -> list[str]:
    return sorted(file_name for file_name in os.listdir(version_directory) if file_name.endswith(".report.html"))


def _redirect_html(version: str, file_name: str) -> str:
    redirect_url = f"./{version}/{file_name}"
    escaped_redirect_url = html.escape(redirect_url, quote=True)
    escaped_file_name = html.escape(file_name)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={escaped_redirect_url}">
  <link rel="canonical" href="{escaped_redirect_url}">
  <title>Redirecting to {escaped_file_name}</title>
  <script>location.replace("{escaped_redirect_url}");</script>
</head>
<body>
  <p>Redirecting to the latest report…</p>
</body>
</html>
"""


def _write_redirect(version: str, file_name: str) -> None:
    redirect_path = os.path.join(REPORTS_DIRECTORY, file_name)
    with open(redirect_path, "w", encoding="utf-8") as file:
        file.write(_redirect_html(version, file_name))


def main() -> None:
    version = default_version()
    report_file_names = _report_file_names(os.path.join(REPORTS_DIRECTORY, version))
    for file_name in report_file_names:
        _write_redirect(version, file_name)
    print(f"Wrote {len(report_file_names)} report redirect stubs for version {version}.")


if __name__ == "__main__":
    main()
