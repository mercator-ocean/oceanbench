# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Build a static local viewer containing a local forecast and remote official datasets."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urljoin

import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.remote_json import read_json_url
from oceanbench.publish import class4_overlays
from oceanbench.publish.viewer_artifacts import (
    EDDY_CENSUS_FILENAME,
    MATCHUP_PARQUET_FILENAME,
    RMSD_BY_DEPTH_FILENAME,
    YEAR_ERROR_GEOGRAPHY_FILENAME,
    YEAR_RMSD_BY_START_FILENAME,
)
from oceanbench.pyramids import build_pyramid, viewer_layers

# Keep aligned with website/viewer/config.js. Update both rebuild-preview values at release.
# CloudFerro is the only data origin; EDITO MinIO is retired.
OFFICIAL_PUBLISHED_BASE_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/benchmark/rebuild-preview/"
LOCAL_VIEWER_DIRECTORY = "viewer"
INSIGHTS_FILENAME = "insights.json"

# The insights index key each artifact of a dataset is published under.
_LOCAL_INSIGHT_FILENAMES = {
    "class4_matchups": MATCHUP_PARQUET_FILENAME,
    "eddies": EDDY_CENSUS_FILENAME,
    "spectra": "spectra.json",
    "rmsd_by_depth": RMSD_BY_DEPTH_FILENAME,
    "year_error_geography": YEAR_ERROR_GEOGRAPHY_FILENAME,
    "year_rmsd_by_start": YEAR_RMSD_BY_START_FILENAME,
}


def published_base_url() -> str:
    return os.environ.get("OCEANBENCH_PUBLISHED_BASE", OFFICIAL_PUBLISHED_BASE_URL).rstrip("/") + "/"


def official_data_base_url() -> str:
    return urljoin(published_base_url(), "viewer/data/")


@dataclass(frozen=True)
class LocalViewerResult:
    viewer_directory: str
    datasets_path: str
    zarr_path: str
    manifest_path: str


def _official_datasets() -> list[dict]:
    data_base_url = official_data_base_url()
    catalog = read_json_url(urljoin(data_base_url, "datasets.json"))
    return [
        {
            **entry,
            "store": urljoin(data_base_url, entry["store"].removeprefix("./data/")),
            "manifest": urljoin(data_base_url, entry["manifest"].removeprefix("./data/")),
        }
        for entry in catalog["datasets"]
    ]


def _absolutise(value, data_base_url: str):
    """Rewrite every ``./data/...`` reference of a published insights index to an absolute URL."""
    if isinstance(value, str) and value.startswith("./data/"):
        return urljoin(data_base_url, value.removeprefix("./data/"))
    if isinstance(value, dict):
        return {key: _absolutise(item, data_base_url) for key, item in value.items()}
    if isinstance(value, list):
        return [_absolutise(item, data_base_url) for item in value]
    return value


def _official_insights() -> dict:
    """The published insights index with every artifact reference pointing at the public bucket."""
    data_base_url = official_data_base_url()
    return _absolutise(read_json_url(urljoin(data_base_url, INSIGHTS_FILENAME)), data_base_url)


def _local_insights(data_directory: Path, dataset_slug: str) -> dict:
    """The insights entries of the locally evaluated dataset, as URLs relative to the data prefix.

    Reads whatever ``write_viewer_artifacts`` left under the viewer data prefix; an artifact that
    was skipped is simply absent, exactly as it is for a published dataset.
    """
    dataset_directory = data_directory / "insights" / dataset_slug
    if not dataset_directory.is_dir():
        return {}
    entries = {}
    for region_directory in sorted(path for path in dataset_directory.iterdir() if path.is_dir()):
        region = region_directory.name
        artifacts = {
            key: f"./data/insights/{dataset_slug}/{region}/{filename}"
            for key, filename in _LOCAL_INSIGHT_FILENAMES.items()
            if (region_directory / filename).is_file()
        }
        if (region_directory / class4_overlays.CLASS4_OVERLAY_DIRECTORY).is_dir():
            artifacts["class4_overlays"] = class4_overlays.overlay_index_entry(dataset_slug, region)
        if artifacts:
            entries[region] = artifacts
    return {dataset_slug: entries} if entries else {}


def _write_insights_index(data_directory: Path, dataset_slug: str) -> None:
    """Merge the local dataset's insights with the published ones into the viewer's index."""
    official = _official_insights()
    index = {
        **official,
        "datasets": {**official.get("datasets", {}), **_local_insights(data_directory, dataset_slug)},
    }
    (data_directory / INSIGHTS_FILENAME).write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")


def _write_viewer_application(viewer_directory: Path) -> None:
    """Copy the viewer single-page application next to its data so it opens over a plain file server."""
    source_viewer = Path(__file__).resolve().parents[2] / "website" / "viewer"
    # "qa" is the maintainers' Playwright harness and carries its own node_modules (tens of MB);
    # it is not part of the application a user opens.
    shutil.copytree(source_viewer, viewer_directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("data", "qa"))
    index_path = viewer_directory / "index.html"
    index_path.write_text(
        index_path.read_text(encoding="utf-8")
        .replace("../favicon-light.png", "./favicon-light.png")
        .replace(
            "</head>",
            '  <link rel="icon" href="data:image/svg+xml,' '<svg xmlns=%22http://www.w3.org/2000/svg%22/>">\n</head>',
        ),
        encoding="utf-8",
    )
    shutil.copyfile(source_viewer.parent / "favicon-light.png", viewer_directory / "favicon-light.png")


def build_local_viewer(
    forecast_dataset: xarray.Dataset,
    *,
    output_directory: str,
    year: int,
    dataset_slug: str = "your_model",
    label: str | None = None,
    pyramid_zarr_path: str | None = None,
    pyramid_manifest_path: str | None = None,
    starts_limit: int | None = None,
) -> LocalViewerResult:
    """Assemble a local viewer site: the field pyramid, the dataset catalog and the application.

    ``pyramid_zarr_path`` and ``pyramid_manifest_path`` adopt a pyramid built earlier in the same
    output directory (the serving artifacts write one to the very same place) instead of building
    a second identical one.
    """
    viewer_directory = Path(output_directory) / LOCAL_VIEWER_DIRECTORY
    _write_viewer_application(viewer_directory)
    data_directory = viewer_directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    _write_insights_index(data_directory, dataset_slug)

    if pyramid_zarr_path is None or pyramid_manifest_path is None:
        selected = (
            forecast_dataset
            if starts_limit is None
            else forecast_dataset.isel({Dimension.FIRST_DAY_DATETIME.key(): slice(0, starts_limit)})
        )
        layers, specs = viewer_layers(selected)
        if not layers.data_vars:
            raise ValueError("The forecast contains none of the variables supported by the viewer")
        pyramid = build_pyramid(
            layers,
            specs,
            output_path=str(data_directory / f"{dataset_slug}.zarr"),
            dataset_slug=dataset_slug,
            year=year,
        )
        pyramid_zarr_path = pyramid.zarr_path
        pyramid_manifest_path = pyramid.manifest_path

    datasets = [
        {
            "slug": dataset_slug,
            "label": label if label is not None else "Your model (local)",
            "store": f"./data/{dataset_slug}.zarr",
            "manifest": f"./data/{dataset_slug}.viewer-manifest.json",
        },
        *_official_datasets(),
    ]
    datasets_path = data_directory / "datasets.json"
    datasets_path.write_text(json.dumps({"datasets": datasets}, sort_keys=True, indent=2), encoding="utf-8")
    return LocalViewerResult(
        viewer_directory=str(viewer_directory),
        datasets_path=str(datasets_path),
        zarr_path=pyramid_zarr_path,
        manifest_path=pyramid_manifest_path,
    )
