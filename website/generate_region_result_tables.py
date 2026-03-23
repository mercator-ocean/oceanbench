#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
from pathlib import Path

from helpers.notebook_score_parser import TRACK_METRIC_CALLS
from helpers.notebook_score_parser import get_model_score_from_notebook


def _parse_named_report(named_report: str) -> tuple[str, str]:
    if "=" not in named_report:
        raise ValueError(f"Invalid report definition: {named_report}. Expected NAME=PATH.")
    model_name, notebook_path = named_report.split("=", 1)
    if model_name.strip() == "" or notebook_path.strip() == "":
        raise ValueError(f"Invalid report definition: {named_report}. Expected NAME=PATH.")
    return model_name.strip(), notebook_path.strip()


def _write_track_score_table(
    output_root: Path,
    region_identifier: str,
    track: str,
    model_name: str,
    notebook_path: str,
) -> None:
    model_score = get_model_score_from_notebook(notebook_path, model_name, track)
    track_output_directory = output_root / region_identifier / track
    track_output_directory.mkdir(parents=True, exist_ok=True)
    output_file = track_output_directory / f"{model_name.lower()}.json"
    output_file.write_text(model_score.model_dump_json(indent=2) + "\n", encoding="utf8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_region_result_tables.py",
        description="Generate Quarto result tables from OceanBench evaluation notebooks.",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Target sub-region identifier, for example ibi",
    )
    parser.add_argument(
        "--output-root",
        default="_result_tables",
        help="Directory where generated result tables are written",
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="Model report definitions in the form NAME=PATH_OR_URL",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root)
    for named_report in args.reports:
        model_name, notebook_path = _parse_named_report(named_report)
        for track in TRACK_METRIC_CALLS:
            _write_track_score_table(output_root, args.region, track, model_name, notebook_path)


if __name__ == "__main__":
    main()
