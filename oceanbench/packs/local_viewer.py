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
from oceanbench.packs.fetch import read_json_url
from oceanbench.pyramids import build_pyramid, viewer_layers

# Keep aligned with website/viewer/config.js. Update both rebuild-preview values at release.
OFFICIAL_PUBLISHED_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench/dev/benchmark/rebuild-preview/"
LOCAL_VIEWER_DIRECTORY = "viewer"


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


def _write_viewer_application(viewer_directory: Path) -> None:
    """Copy the viewer single-page application next to its data so it opens over a plain file server."""
    source_viewer = Path(__file__).resolve().parents[2] / "website" / "viewer"
    shutil.copytree(source_viewer, viewer_directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("data"))
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
    insights_path = data_directory / "insights.json"
    if not insights_path.exists():
        insights_path.write_text(
            json.dumps({"datasets": {}, "regions": {}}, sort_keys=True, indent=2),
            encoding="utf-8",
        )

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
