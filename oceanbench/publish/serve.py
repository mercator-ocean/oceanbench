# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Serve the map viewer over a local directory of viewer artifacts, with no network.

The viewer is a static single-page application under ``website/viewer`` that reads its data
from a sibling ``./data/`` prefix or from whatever ``?data=`` points at. This server mounts the
application at ``/`` and the artifacts directory the caller names at ``/data/``, so a locally
produced artifact tree is browsable exactly as the published one is, without copying the
application next to the data first.

``website/viewer`` is located relative to this file, at the repository root
(``<repo>/oceanbench/publish/serve.py`` -> ``<repo>/website/viewer``), the same way
:mod:`oceanbench.packs.local_viewer` finds it. The viewer is not packaged as installed package
data, so serving from a wheel install without the repository checkout is not supported and
raises a clear error.
"""

from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

DATA_MOUNT_PATH = "/data"
DEFAULT_VIEWER_PORT = 8799
DEFAULT_VIEWER_HOST = "127.0.0.1"
VIEWER_DATASET_CATALOG_FILENAME = "datasets.json"
VIEWER_ARTIFACT_COMPANION_FILENAMES = (
    "insights.json",
    "scores-summary.json",
    "<dataset>.zarr",
    "<dataset>.viewer-manifest.json",
)


@dataclass(frozen=True)
class ViewerServer:
    server: ThreadingHTTPServer
    url: str
    viewer_directory: str
    artifacts_directory: str


def viewer_application_directory() -> Path:
    """Absolute path of the ``website/viewer`` single-page application."""
    directory = Path(__file__).resolve().parents[2] / "website" / "viewer"
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(
            f"the viewer application is not at {directory}; 'oceanbench view' serves "
            "website/viewer from the repository checkout and is not available from a wheel install"
        )
    return directory


def validate_viewer_artifacts_directory(artifacts_directory: str) -> Path:
    """Check ``artifacts_directory`` looks like a viewer data prefix, or explain what is missing."""
    directory = Path(artifacts_directory).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"{artifacts_directory} is not a directory")
    if not (directory / VIEWER_DATASET_CATALOG_FILENAME).is_file():
        companions = "\n".join(f"  - {name}" for name in VIEWER_ARTIFACT_COMPANION_FILENAMES)
        raise FileNotFoundError(
            f"{directory} has no {VIEWER_DATASET_CATALOG_FILENAME}, so it is not a viewer artifacts "
            f"directory. A viewer artifacts directory is the './data/' prefix of a viewer site and must "
            f"contain:\n  - {VIEWER_DATASET_CATALOG_FILENAME} (the dataset catalog, required)\n"
            f"and normally also:\n{companions}\n"
            "Produce one with 'oceanbench evaluate --viewer-artifacts' and point at its viewer/data directory."
        )
    return directory.resolve()


class _ViewerRequestHandler(SimpleHTTPRequestHandler):
    """Serve the viewer application at ``/`` and the artifacts tree at ``/data/``."""

    viewer_directory = ""
    artifacts_directory = ""

    def translate_path(self, path: str) -> str:
        request_path = urlparse(path).path
        if request_path == DATA_MOUNT_PATH or request_path.startswith(DATA_MOUNT_PATH + "/"):
            self.directory = self.artifacts_directory
            return super().translate_path(request_path[len(DATA_MOUNT_PATH) :] or "/")
        self.directory = self.viewer_directory
        return super().translate_path(path)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - signature fixed by the stdlib
        return


def build_viewer_server(
    artifacts_directory: str,
    *,
    port: int = DEFAULT_VIEWER_PORT,
    host: str = DEFAULT_VIEWER_HOST,
) -> ViewerServer:
    """Bind a viewer server without serving yet, so the caller decides when to block."""
    artifacts = validate_viewer_artifacts_directory(artifacts_directory)
    viewer = viewer_application_directory()

    handler = type(
        "OceanBenchViewerRequestHandler",
        (_ViewerRequestHandler,),
        {"viewer_directory": str(viewer), "artifacts_directory": str(artifacts)},
    )
    server = ThreadingHTTPServer((host, port), handler)
    bound_port = server.server_address[1]
    return ViewerServer(
        server=server,
        url=f"http://{host}:{bound_port}/?data={DATA_MOUNT_PATH}/",
        viewer_directory=str(viewer),
        artifacts_directory=str(artifacts),
    )
