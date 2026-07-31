# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Offline viewer server: the two mounts and the missing-datasets.json message."""

import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from oceanbench.publish.serve import build_viewer_server, validate_viewer_artifacts_directory


def _fake_artifacts_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "datasets.json").write_text(
        json.dumps({"datasets": [{"slug": "your_model", "store": "./data/your_model.zarr"}]}),
        encoding="utf-8",
    )
    (directory / "insights.json").write_text(json.dumps({"datasets": {}, "regions": {}}), encoding="utf-8")
    return directory


@pytest.fixture
def running_viewer(tmp_path: Path):
    viewer = build_viewer_server(str(_fake_artifacts_directory(tmp_path)), port=0)
    thread = threading.Thread(target=viewer.server.serve_forever, daemon=True)
    thread.start()
    try:
        yield viewer, f"http://127.0.0.1:{viewer.server.server_address[1]}"
    finally:
        viewer.server.shutdown()
        viewer.server.server_close()
        thread.join(timeout=5)


def test_root_serves_the_viewer_application(running_viewer):
    _, base_url = running_viewer
    with urlopen(f"{base_url}/") as response:
        body = response.read().decode("utf-8")
    assert response.status == 200
    assert "<html" in body.lower()


def test_data_mount_serves_the_artifacts_directory(running_viewer):
    _, base_url = running_viewer
    with urlopen(f"{base_url}/data/datasets.json") as response:
        catalog = json.load(response)
    assert response.status == 200
    assert catalog["datasets"][0]["slug"] == "your_model"


def test_missing_artifact_is_a_404(running_viewer):
    _, base_url = running_viewer
    with pytest.raises(HTTPError) as raised:
        urlopen(f"{base_url}/data/absent.json")
    assert raised.value.code == 404


def test_printed_url_carries_the_data_override(running_viewer):
    viewer, _ = running_viewer
    assert viewer.url.endswith("/?data=/data/")


def test_a_directory_without_datasets_json_is_rejected(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="datasets.json"):
        validate_viewer_artifacts_directory(str(tmp_path / "empty"))


def test_a_missing_directory_is_rejected(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        validate_viewer_artifacts_directory(str(tmp_path / "absent"))
