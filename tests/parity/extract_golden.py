#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2
"""Parity golden dataset builder for the OceanBench v2 rebuild.

Parses every published 2024 evaluation-report notebook (version 0.2.1) into
LONG-FORMAT score records. Logic adapted faithfully from
website/helpers/notebook_score_parser.py + type.py (origin/main).

Outputs (in this script's directory):
  - golden_scores.parquet
  - golden_scores.json
  - metadata.json
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(HERE, "notebooks")
SOURCE_VERSION = "0.2.1"

# ---- parsing helpers (adapted verbatim from notebook_score_parser.py) --------

_VARIABLE_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]+)\) \[([^\]]*)\](?:\{([^}]+)\})?$")
_LEAD_DAY_NUMBER_PATTERN = re.compile(r"(\d+)$")
_DISPLAY_NAME_RENAMES = {
    "height": "sea surface height",
    "surface height": "sea surface height",
    "northward velocity": "meridional current",
    "eastward velocity": "zonal current",
    "northward geostrophic velocity": "meridional geostrophic current",
    "eastward geostrophic velocity": "zonal geostrophic current",
}


def _parse_variable_label(label):
    match = _VARIABLE_LABEL_PATTERN.match(label)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4) or ""
    return label, "", "unknown", ""


def _normalise_display_name(display_name):
    normalised = display_name.lower()
    return _DISPLAY_NAME_RENAMES.get(normalised, normalised)


_METRICS = [
    {"key": "rmsd_variables", "function": "rmsd_of_variables", "has_depths": True},
    {"key": "rmsd_mld", "function": "rmsd_of_mixed_layer_depth", "has_depths": False},
    {"key": "rmsd_geostrophic", "function": "rmsd_of_geostrophic_currents", "has_depths": False},
    {"key": "lagrangian", "function": "deviation_of_lagrangian_trajectories", "has_depths": False},
]
_REFERENCES = [
    {"suffix": "glorys", "function_suffix": "compared_to_glorys_reanalysis"},
    {"suffix": "glo12", "function_suffix": "compared_to_glo12_analysis"},
]
_OBSERVATIONS_METRIC_KEY = "rmsd_variables_observations"

_METRIC_PATTERNS = {
    f"{m['key']}_{r['suffix']}": f"oceanbench.metrics.{m['function']}_{r['function_suffix']}"
    for m in _METRICS
    for r in _REFERENCES
} | {_OBSERVATIONS_METRIC_KEY: "oceanbench.metrics.rmsd_of_variables_compared_to_observations"}

_DEPTH_VARIABLE_METRICS = {f"{m['key']}_{r['suffix']}" for m in _METRICS for r in _REFERENCES if m["has_depths"]} | {
    _OBSERVATIONS_METRIC_KEY
}

# the 9 canonical metric keys, in a stable order
METRIC_KEYS = list(_METRIC_PATTERNS.keys())


def _get_cell_source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def _get_cell_html_output(cell):
    for output in cell.get("outputs", []):
        if "data" in output and "text/html" in output["data"]:
            html_parts = output["data"]["text/html"]
            if isinstance(html_parts, list):
                return "".join(line.removesuffix("\n") for line in html_parts)
            return html_parts
    return None


def _get_all_metrics_from_notebook(raw_notebook):
    return {
        metric_key: html
        for cell in raw_notebook["cells"]
        for metric_key, pattern in _METRIC_PATTERNS.items()
        if pattern in _get_cell_source(cell)
        if (html := _get_cell_html_output(cell))
    }


def _parse_cell_value(text):
    try:
        return float(text)
    except ValueError:
        return None


def _extract_lead_day_number(header):
    match = _LEAD_DAY_NUMBER_PATTERN.search(header)
    return match.group(1) if match else header


def _parse_html_table_rows(raw_table):
    soup = BeautifulSoup(raw_table, features="html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    lead_days = [_extract_lead_day_number(h) for h in headers[1:]]
    rows = []
    for row in soup.find("tbody").find_all("tr"):
        label = row.find("th").get_text(strip=True)
        values = {day: _parse_cell_value(cell.get_text(strip=True)) for day, cell in zip(lead_days, row.find_all("td"))}
        rows.append({"label": label, "data": values})
    return rows


def _clean_value(v):
    """NaN / None -> None (null)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def records_from_table(raw_table, metric_key, challenger, region):
    """Emit long-format records from one metric HTML table."""
    is_depth = metric_key in _DEPTH_VARIABLE_METRICS
    records = []
    for row in _parse_html_table_rows(raw_table):
        display_name, unit, standard_name, depth_label = _parse_variable_label(row["label"])
        if is_depth:
            if not depth_label:
                continue  # depth-variable tables keep only rows carrying a depth
            depth = depth_label.capitalize()
            variable_name = _normalise_display_name(display_name.removeprefix(depth + " "))
            out_depth = depth
        else:
            variable_name = _normalise_display_name(display_name)
            out_depth = None  # flat metrics collapse to depth-agnostic in the website model
        for day, value in row["data"].items():
            records.append(
                {
                    "source_version": SOURCE_VERSION,
                    "challenger": challenger,
                    "region": region,
                    "metric_key": metric_key,
                    "variable_standard_name": standard_name,
                    "display_name": variable_name,
                    "unit": unit,
                    "depth_label": out_depth,
                    "lead_day": int(day) if str(day).isdigit() else None,
                    "value": _clean_value(value),
                }
            )
    return records


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


NOTEBOOK_FILE_PATTERN = re.compile(r"^(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)\.report\.ipynb$")


def main():
    files = sorted(f for f in os.listdir(NOTEBOOK_DIR) if f.endswith(".report.ipynb"))
    all_records = []
    per_notebook = []
    parse_failures = []
    metric_presence = {}  # (challenger, region) -> set(metric_key found)

    for fname in files:
        m = NOTEBOOK_FILE_PATTERN.match(fname)
        challenger, region = m.group("challenger"), m.group("region")
        path = os.path.join(NOTEBOOK_DIR, fname)
        with open(path) as fh:
            nb = json.load(fh)
        metrics_html = _get_all_metrics_from_notebook(nb)
        metric_presence[(challenger, region)] = set(metrics_html.keys())
        for metric_key, raw_table in metrics_html.items():
            try:
                all_records.extend(records_from_table(raw_table, metric_key, challenger, region))
            except Exception as e:  # noqa: BLE001
                parse_failures.append(f"{fname}:{metric_key}: {e}")
        per_notebook.append(
            {
                "file": fname,
                "challenger": challenger,
                "region": region,
                "bytes": os.path.getsize(path),
                "sha256": sha256_of(path),
                "metrics_found": sorted(metrics_html.keys()),
                "missing_metrics": sorted(set(METRIC_KEYS) - set(metrics_html.keys())),
            }
        )

    # ---- write JSON ----------------------------------------------------------
    json_path = os.path.join(HERE, "golden_scores.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=None)

    # ---- write Parquet -------------------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = [
        "source_version",
        "challenger",
        "region",
        "metric_key",
        "variable_standard_name",
        "display_name",
        "unit",
        "depth_label",
        "lead_day",
        "value",
    ]
    table = pa.table(
        {
            "source_version": pa.array([r["source_version"] for r in all_records], pa.string()),
            "challenger": pa.array([r["challenger"] for r in all_records], pa.string()),
            "region": pa.array([r["region"] for r in all_records], pa.string()),
            "metric_key": pa.array([r["metric_key"] for r in all_records], pa.string()),
            "variable_standard_name": pa.array([r["variable_standard_name"] for r in all_records], pa.string()),
            "display_name": pa.array([r["display_name"] for r in all_records], pa.string()),
            "unit": pa.array([r["unit"] for r in all_records], pa.string()),
            "depth_label": pa.array([r["depth_label"] for r in all_records], pa.string()),
            "lead_day": pa.array([r["lead_day"] for r in all_records], pa.int64()),
            "value": pa.array([r["value"] for r in all_records], pa.float64()),
        }
    ).select(cols)
    parquet_path = os.path.join(HERE, "golden_scores.parquet")
    pq.write_table(table, parquet_path)

    # ---- write metadata ------------------------------------------------------
    metadata = {
        "index_default_version": SOURCE_VERSION,
        "reports_root": "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/",
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(all_records),
        "metric_keys": METRIC_KEYS,
        "depth_variable_metrics": sorted(_DEPTH_VARIABLE_METRICS),
        "notebooks": per_notebook,
        "parse_failures": parse_failures,
    }
    with open(os.path.join(HERE, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- console sanity report ----------------------------------------------
    from collections import Counter, defaultdict

    print(f"notebooks parsed: {len(files)}")
    print(f"total records: {len(all_records)}")
    print(f"parse failures: {parse_failures or 'none'}")

    challengers = sorted({c for c, _ in metric_presence})
    regions = sorted({rg for _, rg in metric_presence})
    print(f"challengers ({len(challengers)}): {challengers}")
    print(f"regions ({len(regions)}): {regions}")

    print("\n-- regions found per challenger --")
    reg_by_ch = defaultdict(list)
    for c, rg in metric_presence:
        reg_by_ch[c].append(rg)
    for c in challengers:
        print(f"  {c}: {sorted(reg_by_ch[c])}")

    print("\n-- missing metrics per (challenger, region) --")
    any_missing = False
    for c, rg in sorted(metric_presence):
        missing = sorted(set(METRIC_KEYS) - metric_presence[(c, rg)])
        if missing:
            any_missing = True
            print(f"  {c}.{rg}: MISSING {missing}")
    if not any_missing:
        print("  none")

    print("\n-- row counts per (challenger, region, metric_key) --")
    counts = Counter((r["challenger"], r["region"], r["metric_key"]) for r in all_records)
    for c, rg, mk in sorted(counts):
        print(f"  {c:18s} {rg:7s} {mk:28s} {counts[(c, rg, mk)]}")

    print("\n-- null value count --")
    nulls = sum(1 for r in all_records if r["value"] is None)
    print(f"  {nulls} null values out of {len(all_records)}")

    print(f"\nwrote:\n  {parquet_path}\n  {json_path}\n  {os.path.join(HERE, 'metadata.json')}")


if __name__ == "__main__":
    main()
