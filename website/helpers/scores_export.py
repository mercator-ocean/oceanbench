# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from enum import Enum
from pathlib import Path

import pandas
import xarray

SCORES_API_DIRECTORY = "api/scores"
SCORES_API_INDEX_FILE = "index.json"
ONE_DEGREE_SUFFIX = "_1_degree"
FLAT_DEPTH_KEY = "flat"
TABLE_COLUMNS = [
    "version",
    "region",
    "track",
    "challenger",
    "comparison",
    "metric",
    "depth",
    "variable",
    "standard_name",
    "unit",
    "lead_day",
    "value",
]


DIMENSIONS = ["challenger", "comparison", "metric", "depth", "variable", "lead_day"]
DESCRIPTOR_DIMENSIONS = ["comparison", "metric", "depth", "variable"]
DESCRIPTOR_COLUMNS = ["unit", "standard_name"]


class ExportFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    ZARR = "zarr"


class Track(Enum):
    HIGH_RESOLUTION = "high_resolution"
    ONE_DEGREE = "one_degree"


def _challenger_track(challenger_name: str) -> Track:
    return Track.ONE_DEGREE if challenger_name.endswith(ONE_DEGREE_SUFFIX) else Track.HIGH_RESOLUTION


def _comparison_by_metric(sections: dict[str, dict]) -> dict[str, str]:
    return {
        metric_key: comparison
        for comparison, section in sections.items()
        for metric_key in [section["depth_metric"], *section["flat_metrics"]]
    }


def _metric_rows(metric_key: str, metric_score: dict, comparison: str, metric_titles: dict[str, str]) -> list[dict]:
    return [
        {
            "challenger": metric_score["name"],
            "comparison": comparison,
            "metric": metric_titles[metric_key],
            "depth": "" if depth == FLAT_DEPTH_KEY else depth,
            "variable": variable,
            "standard_name": variable_score["standard_name"],
            "unit": variable_score["unit"],
            "lead_day": int(lead_day),
            "value": value,
        }
        for depth, depth_score in metric_score["depths"].items()
        for variable, variable_score in depth_score["variables"].items()
        for lead_day, value in variable_score["data"].items()
    ]


def scores_table(
    version: str,
    region_id: str,
    track: str,
    version_bundle: dict,
    metric_titles: dict[str, str],
    sections: dict[str, dict],
) -> pandas.DataFrame:
    """Scores of a version, region and track as one row per challenger, metric, depth, variable and lead day."""
    comparison_by_metric = _comparison_by_metric(sections)
    rows = [
        {"version": version, "region": region_id, "track": track} | row
        for challenger_name, metrics in version_bundle["regions"][region_id]["challengers"].items()
        if _challenger_track(challenger_name).value == track
        for metric_key, metric_score in metrics.items()
        for row in _metric_rows(metric_key, metric_score, comparison_by_metric[metric_key], metric_titles)
    ]
    return pandas.DataFrame(rows, columns=TABLE_COLUMNS).astype({"lead_day": "int64", "value": "float64"})


def scores_dataset(table: pandas.DataFrame) -> xarray.Dataset:
    """Scores as an array of values over the challenger, comparison, metric, depth, variable and lead day dimensions."""
    values = table.set_index(DIMENSIONS)[["value"]].to_xarray()
    descriptors = (
        table.drop_duplicates(DESCRIPTOR_DIMENSIONS)
        .set_index(DESCRIPTOR_DIMENSIONS)[DESCRIPTOR_COLUMNS]
        .to_xarray()
        .fillna("")
    )
    return values.merge(descriptors).assign_attrs(version=str(table["version"].iloc[0]))


def _write_table(table: pandas.DataFrame, path_without_extension: Path, export_format: ExportFormat) -> None:
    path = path_without_extension.with_suffix(f".{export_format.value}")
    {
        ExportFormat.CSV: lambda: table.to_csv(path, index=False),
        ExportFormat.PARQUET: lambda: table.to_parquet(path, index=False),
        ExportFormat.ZARR: lambda: scores_dataset(table).to_zarr(path, mode="w", consolidated=True),
    }[export_format]()


def _tracks_with_scores(table_by_track: dict[str, pandas.DataFrame]) -> dict[str, pandas.DataFrame]:
    return {track: table for track, table in table_by_track.items() if not table.empty}


def _region_tables(
    version: str, region_id: str, version_bundle: dict, metric_titles: dict[str, str], sections: dict[str, dict]
) -> dict[str, pandas.DataFrame]:
    return _tracks_with_scores(
        {
            track.value: scores_table(version, region_id, track.value, version_bundle, metric_titles, sections)
            for track in Track
        }
    )


def _export_track(table: pandas.DataFrame, track_directory_path: Path, relative_path: str) -> dict[str, str]:
    track_directory_path.parent.mkdir(parents=True, exist_ok=True)
    for export_format in ExportFormat:
        _write_table(table, track_directory_path, export_format)
    return {export_format.value: f"{relative_path}.{export_format.value}" for export_format in ExportFormat}


def _export_region(
    version: str,
    region_id: str,
    version_bundle: dict,
    metric_titles: dict[str, str],
    sections: dict[str, dict],
    output_directory: Path,
) -> dict[str, dict[str, str]]:
    return {
        track: _export_track(
            table,
            output_directory / version / region_id / track,
            f"{SCORES_API_DIRECTORY}/{version}/{region_id}/{track}",
        )
        for track, table in _region_tables(version, region_id, version_bundle, metric_titles, sections).items()
    }


def write_scores_exports(
    version_bundles: dict[str, dict],
    metric_titles: dict[str, str],
    sections: dict[str, dict],
    output_directory: Path,
) -> dict:
    """Write the scores of every version, region and track as csv, parquet and zarr, and return their index."""
    index = {
        "versions": {
            version: {
                region_id: _export_region(version, region_id, version_bundle, metric_titles, sections, output_directory)
                for region_id in version_bundle["region_order"]
            }
            for version, version_bundle in version_bundles.items()
        }
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / SCORES_API_INDEX_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index
