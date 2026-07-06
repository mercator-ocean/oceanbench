# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Build a static local viewer containing a local forecast and remote official datasets."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from urllib.parse import urljoin
from urllib.request import urlopen

import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.pyramids import build_pyramid, viewer_layers

OFFICIAL_DATA_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench/dev/benchmark/rebuild-preview/viewer/data/"
LOCAL_VIEWER_DIRECTORY = "viewer"


@dataclass(frozen=True)
class LocalViewerResult:
    viewer_directory: str
    datasets_path: str
    zarr_path: str
    manifest_path: str


def _official_datasets() -> list[dict]:
    with urlopen(urljoin(OFFICIAL_DATA_BASE_URL, "datasets.json"), timeout=30) as response:  # noqa: S310
        catalog = json.load(response)
    return [
        {
            **entry,
            "store": urljoin(OFFICIAL_DATA_BASE_URL, entry["store"].removeprefix("./data/")),
            "manifest": urljoin(OFFICIAL_DATA_BASE_URL, entry["manifest"].removeprefix("./data/")),
        }
        for entry in catalog["datasets"]
    ]


def build_local_viewer(
    forecast_dataset: xarray.Dataset,
    *,
    output_directory: str,
    year: int,
    starts_limit: int | None = None,
) -> LocalViewerResult:
    """Build the local pyramid, merge its descriptor with the official catalog, and copy the SPA."""
    viewer_directory = Path(output_directory) / LOCAL_VIEWER_DIRECTORY
    source_viewer = Path(__file__).resolve().parents[2] / "website" / "viewer"
    shutil.copytree(source_viewer, viewer_directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("data"))
    data_directory = viewer_directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)

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
        output_path=str(data_directory / "your_model.zarr"),
        dataset_slug="your_model",
        year=year,
    )
    datasets = [
        {
            "slug": "your_model",
            "label": "Your model (local)",
            "store": "./data/your_model.zarr",
            "manifest": "./data/your_model.viewer-manifest.json",
        },
        *_official_datasets(),
    ]
    datasets_path = data_directory / "datasets.json"
    datasets_path.write_text(json.dumps({"datasets": datasets}, sort_keys=True, indent=2), encoding="utf-8")
    return LocalViewerResult(
        viewer_directory=str(viewer_directory),
        datasets_path=str(datasets_path),
        zarr_path=pyramid.zarr_path,
        manifest_path=pyramid.manifest_path,
    )
