# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import html
import os

from helpers.s3_discovery import _version_sort_key, default_version

SCRIPT_DIRECTORY = os.path.dirname(__file__)


def _output_directory() -> str:
    configured_output_directory = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
    if configured_output_directory:
        return os.path.abspath(configured_output_directory)
    return os.path.join(SCRIPT_DIRECTORY, "_site")


def _highest_rendered_version(reports_directory: str) -> str:
    if not os.path.isdir(reports_directory):
        return ""

    version_names = [
        file_name
        for file_name in os.listdir(reports_directory)
        if os.path.isdir(os.path.join(reports_directory, file_name))
    ]
    sorted_versions = sorted(version_names, key=_version_sort_key, reverse=True)
    return sorted_versions[0] if sorted_versions else ""


def _default_rendered_version(reports_directory: str) -> str:
    try:
        version = default_version()
    except Exception as error:
        print(f"Warning: could not discover default report version: {error}")
        return _highest_rendered_version(reports_directory)

    return version or _highest_rendered_version(reports_directory)


def _report_file_names(version_directory: str) -> list[str]:
    if not os.path.isdir(version_directory):
        return []
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


def _write_redirect(reports_directory: str, version: str, file_name: str) -> None:
    redirect_path = os.path.join(reports_directory, file_name)
    with open(redirect_path, "w", encoding="utf-8") as file:
        file.write(_redirect_html(version, file_name))


def main() -> None:
    output_directory = _output_directory()
    reports_directory = os.path.join(output_directory, "reports")
    version = _default_rendered_version(reports_directory)

    if not version:
        print(f"Warning: no rendered report version found under {reports_directory}.")
        return

    report_file_names = _report_file_names(os.path.join(reports_directory, version))
    for file_name in report_file_names:
        _write_redirect(reports_directory, version, file_name)

    print(f"Wrote {len(report_file_names)} report redirect stubs for version {version}.")


if __name__ == "__main__":
    main()
