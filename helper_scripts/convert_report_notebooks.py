# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Convert the published report notebooks (up to version 0.6.0) to scores files, and optionally upload them.

Contributor housekeeping, run once through ``make convert-report-notebooks``. It does not depend on the
``oceanbench`` library nor on the website code.

Uploading uses the S3 variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_S3_ENDPOINT``).
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum

import boto3
import requests
from bs4 import BeautifulSoup

OUTPUT_BUCKET = "oceanbench-bucket"
REPORTS_ROOT_PREFIX = "public/evaluation-reports"
BUCKET_URL = f"https://s3.waw3-1.cloudferro.com/{OUTPUT_BUCKET}"
EVALUATION_INDEX_URL = f"{BUCKET_URL}/{REPORTS_ROOT_PREFIX}/index.json"
NOTEBOOK_SUFFIX = ".report.ipynb"
SCORES_FILE_SUFFIX = ".scores.json"
LAST_NOTEBOOK_VERSION = "0.6.0"
REGION_DISPLAY_NAMES = {"global": "Global", "ibi": "IBI"}
SCORES_SCHEMA_VERSION = 1
FLAT_DEPTH = "surface"
CHALLENGER_CODE_MARKER = "Open challenger dataset"
METRIC_FUNCTION_PREFIX = "oceanbench.metrics."
REFERENCE_SEPARATOR = "_compared_to_"
DEPTH_METRIC = "rmsd_of_variables"
METRIC_FUNCTIONS = [
    f"{metric}{REFERENCE_SEPARATOR}{reference}"
    for metric, references in (
        (DEPTH_METRIC, ("glorys_reanalysis", "glo12_analysis", "observations")),
        ("rmsd_of_mixed_layer_depth", ("glorys_reanalysis", "glo12_analysis")),
        ("rmsd_of_geostrophic_currents", ("glorys_reanalysis", "glo12_analysis")),
        ("deviation_of_lagrangian_trajectories", ("glorys_reanalysis", "glo12_analysis")),
    )
    for reference in references
]
MAXIMUM_PARALLEL_REQUESTS = 16
HTTP_OK = 200
REQUEST_TIMEOUT_SECONDS = 30

_VARIABLE_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]+)\) \[([^\]]*)\](?:\{([^}]+)\})?$")
_LEAD_DAY_NUMBER_PATTERN = re.compile(r"(\d+)$")
_NON_LEAD_DAY_HEADERS = ("observations", "missing", "missing fraction")
_DISPLAY_NAME_RENAMES = {
    "height": "sea surface height",
    "surface height": "sea surface height",
    "northward velocity": "meridional current",
    "eastward velocity": "zonal current",
    "northward geostrophic velocity": "meridional geostrophic current",
    "eastward geostrophic velocity": "zonal geostrophic current",
}


class UploadMode(Enum):
    UPLOAD = "upload"
    LOCAL_ONLY = "local-only"


def _parse_variable_label(label: str) -> tuple[str, str, str, str]:
    match = _VARIABLE_LABEL_PATTERN.match(label)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4) or ""
    return label, "", "unknown", ""


def _normalise_display_name(display_name: str) -> str:
    normalised_display_name = display_name.lower()
    return _DISPLAY_NAME_RENAMES.get(normalised_display_name, normalised_display_name)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def _cell_html_output(cell: dict) -> str | None:
    for output in cell.get("outputs", []):
        if "data" in output and "text/html" in output["data"]:
            html_parts = output["data"]["text/html"]
            if isinstance(html_parts, list):
                return "".join(line.removesuffix("\n") for line in html_parts)
            return html_parts
    return None


def _metric_tables(cells: list[dict]) -> dict[str, str]:
    return {
        metric_function: html
        for cell in cells
        for metric_function in METRIC_FUNCTIONS
        if f"{METRIC_FUNCTION_PREFIX}{metric_function}" in _cell_source(cell)
        if (html := _cell_html_output(cell))
    }


def _parse_cell_value(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _lead_day_number(header: str) -> int:
    match = _LEAD_DAY_NUMBER_PATTERN.search(header)
    return int(match.group(1) if match else header)


def _table_rows(html: str) -> list[tuple[str, dict[int, float | None]]]:
    soup = BeautifulSoup(html, features="html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    lead_days = [
        _lead_day_number(header) for header in headers[1:] if header.strip().lower() not in _NON_LEAD_DAY_HEADERS
    ]
    return [
        (
            row.find("th").get_text(strip=True),
            {day: _parse_cell_value(cell.get_text(strip=True)) for day, cell in zip(lead_days, row.find_all("td"))},
        )
        for row in soup.find("tbody").find_all("tr")
    ]


def _depth_and_variable(display_name: str, depth_label: str, metric_function: str) -> tuple[str, str] | None:
    if not metric_function.startswith(DEPTH_METRIC):
        return FLAT_DEPTH, _normalise_display_name(display_name)
    if not depth_label:
        return None
    return depth_label.lower(), _normalise_display_name(display_name.removeprefix(depth_label.capitalize() + " "))


def _variables_by_depth(html: str, metric_function: str) -> dict[str, dict[str, tuple[str, str, dict]]]:
    parsed_rows = [
        (depth, variable, standard_name, unit, data)
        for label, data in _table_rows(html)
        for display_name, unit, standard_name, depth_label in [_parse_variable_label(label)]
        if (placement := _depth_and_variable(display_name, depth_label, metric_function))
        for depth, variable in [placement]
    ]
    return {
        depth: {
            variable: (standard_name, unit, data)
            for row_depth, variable, standard_name, unit, data in parsed_rows
            if row_depth == depth
        }
        for depth in dict.fromkeys(depth for depth, *_ in parsed_rows)
    }


def _rows(metric_tables: dict[str, str]) -> list[dict]:
    return [
        {
            "metric": metric_function.split(REFERENCE_SEPARATOR)[0],
            "reference": metric_function.split(REFERENCE_SEPARATOR)[1],
            "variable": variable,
            "standard_name": standard_name or None,
            "depth": depth,
            "unit": unit,
            "lead_day": lead_day,
            "value": value,
        }
        for metric_function, html in metric_tables.items()
        for depth, variables in _variables_by_depth(html, metric_function).items()
        for variable, (standard_name, unit, data) in variables.items()
        for lead_day, value in data.items()
    ]


def _challenger_code(cells: list[dict]) -> str:
    return next(
        (
            _cell_source(code_cell)
            for marker_cell, code_cell in zip(cells, cells[1:])
            if CHALLENGER_CODE_MARKER in _cell_source(marker_cell) and code_cell["cell_type"] == "code"
        ),
        "",
    )


def _timings(cells: list[dict], metric_functions: list[str]) -> dict[str, float]:
    return {
        metric_function: cell["metadata"]["papermill"]["duration"]
        for metric_function in metric_functions
        for cell in cells
        if cell["cell_type"] == "code"
        and f"{METRIC_FUNCTION_PREFIX}{metric_function}(" in _cell_source(cell)
        and "duration" in cell.get("metadata", {}).get("papermill", {})
    }


def _created_at(notebook: dict) -> str:
    end_time = datetime.fromisoformat(notebook["metadata"]["papermill"]["end_time"])
    return end_time.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def scores_document_from_notebook(notebook: dict, challenger_name: str, region_id: str, version: str) -> dict:
    """The scores a report notebook shows, as a scores file.

    Its scores are those printed in the notebook, at the precision it prints.
    """
    metric_tables = _metric_tables(notebook["cells"])
    rows = _rows(metric_tables)
    return {
        "schema_version": SCORES_SCHEMA_VERSION,
        "oceanbench_version": version,
        "challenger": challenger_name,
        "region": {
            "id": region_id,
            "display_name": REGION_DISPLAY_NAMES[region_id],
            "official": True,
            "bounds": None,
        },
        "lead_days_count": len({row["lead_day"] for row in rows}),
        "challenger_code": _challenger_code(notebook["cells"]),
        "created_at": _created_at(notebook),
        "timings": _timings(notebook["cells"], list(metric_tables)),
        "rows": rows,
    }


def _version_key(version: str) -> list[int]:
    return [int(component) if component.isdigit() else -1 for component in version.split(".")]


def _evaluation_index() -> dict:
    response = requests.get(EVALUATION_INDEX_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _notebook_reports() -> list[tuple[str, str, str]]:
    return [
        (version, challenger_name, region_id)
        for version, version_index in _evaluation_index()["versions"].items()
        if _version_key(version) <= _version_key(LAST_NOTEBOOK_VERSION)
        for challenger_name in version_index["challengers"]
        for region_id in REGION_DISPLAY_NAMES
    ]


def _notebook(version: str, challenger_name: str, region_id: str) -> dict | None:
    url = f"{BUCKET_URL}/{REPORTS_ROOT_PREFIX}/{version}/{challenger_name}.{region_id}{NOTEBOOK_SUFFIX}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    return response.json() if response.status_code == HTTP_OK else None


def _write(document: dict, output_directory: str) -> str:
    path = os.path.join(
        output_directory,
        document["oceanbench_version"],
        f"{document['challenger']}.{document['region']['id']}{SCORES_FILE_SUFFIX}",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(document, file, separators=(",", ":"))
    return path


def _upload(path: str, output_directory: str) -> None:
    key = f"{REPORTS_ROOT_PREFIX}/{os.path.relpath(path, output_directory)}"
    endpoint_url = os.environ.get("BOTO3_ENDPOINT_URL", f"https://{os.environ['AWS_S3_ENDPOINT']}")
    boto3.client("s3", endpoint_url=endpoint_url).upload_file(
        path, OUTPUT_BUCKET, key, ExtraArgs={"ContentType": "application/json"}
    )
    print(f"Uploaded {key}")


def _convert(report: tuple[str, str, str], output_directory: str, upload_mode: UploadMode) -> str | None:
    notebook = _notebook(*report)
    if notebook is None:
        return None
    path = _write(scores_document_from_notebook(notebook, report[1], report[2], report[0]), output_directory)
    if upload_mode is UploadMode.UPLOAD:
        _upload(path, output_directory)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--upload", dest="upload_mode", action="store_const", const=UploadMode.UPLOAD)
    arguments = parser.parse_args()
    upload_mode = arguments.upload_mode or UploadMode.LOCAL_ONLY
    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        paths = list(
            pool.map(
                lambda report: _convert(report, arguments.output_directory, upload_mode),
                _notebook_reports(),
            )
        )
    print(f"Converted {sum(path is not None for path in paths)} report notebooks to {arguments.output_directory}.")


if __name__ == "__main__":
    main()
