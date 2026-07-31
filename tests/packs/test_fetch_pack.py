# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Pack download tests, served from a throwaway local http.server (never a real bucket)."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from oceanbench.packs.fetch import (
    PACK_FILE_INDEX_FILENAME,
    default_pack_cache_directory,
    fetch_pack,
    pack_name_from_source,
    resolve_offline_references,
    write_pack_file_index,
)

_PACK_FILES = {
    "pack-manifest.json": '{"schema_version": "1.0"}',
    "README.md": "# pack\n",
    "references/glorys.zarr/.zattrs": '{"a": 1}',
    "references/glorys.zarr/0.0": "chunk-bytes",
    "observations/observations.zarr/.zattrs": '{"b": 2}',
}


def _write_published_pack(root: Path) -> Path:
    pack_directory = root / "pack-full-2024-global"
    for relative, content in _PACK_FILES.items():
        path = pack_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    write_pack_file_index(str(pack_directory))
    return pack_directory


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - signature fixed by the stdlib
        return


@pytest.fixture
def published_pack(tmp_path: Path):
    served_root = tmp_path / "served"
    served_root.mkdir()
    pack_directory = _write_published_pack(served_root)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(served_root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/{pack_directory.name}/"
    try:
        yield base_url, pack_directory
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_pack_file_index_lists_every_file_but_itself(tmp_path: Path):
    pack_directory = _write_published_pack(tmp_path)
    index = json.loads((pack_directory / PACK_FILE_INDEX_FILENAME).read_text(encoding="utf-8"))

    assert [entry["path"] for entry in index["files"]] == sorted(_PACK_FILES)
    assert index["file_count"] == len(_PACK_FILES)
    assert index["total_bytes"] == sum(len(content.encode()) for content in _PACK_FILES.values())
    assert index["pack_name"] == "pack-full-2024-global"


def test_fetch_pack_downloads_every_file(published_pack, tmp_path: Path):
    base_url, source_directory = published_pack
    destination = tmp_path / "downloaded"

    summary = fetch_pack(base_url, str(destination))

    assert summary.downloaded_count == len(_PACK_FILES)
    assert summary.skipped_count == 0
    assert summary.total_count == len(_PACK_FILES)
    assert summary.destination == str(destination)
    for relative, content in _PACK_FILES.items():
        assert (destination / relative).read_text(encoding="utf-8") == content
    assert (destination / PACK_FILE_INDEX_FILENAME).is_file()
    assert json.loads((source_directory / PACK_FILE_INDEX_FILENAME).read_text(encoding="utf-8"))["files"]


def test_fetch_pack_skips_files_whose_size_already_matches(published_pack, tmp_path: Path):
    base_url, _ = published_pack
    destination = tmp_path / "downloaded"

    fetch_pack(base_url, str(destination))
    second = fetch_pack(base_url, str(destination))

    assert second.downloaded_count == 0
    assert second.skipped_count == len(_PACK_FILES)
    assert second.downloaded_bytes == 0


def test_fetch_pack_redownloads_a_truncated_file(published_pack, tmp_path: Path):
    base_url, _ = published_pack
    destination = tmp_path / "downloaded"
    fetch_pack(base_url, str(destination))
    (destination / "references" / "glorys.zarr" / "0.0").write_text("short", encoding="utf-8")

    summary = fetch_pack(base_url, str(destination))

    assert summary.downloaded_count == 1
    assert (destination / "references" / "glorys.zarr" / "0.0").read_text(encoding="utf-8") == "chunk-bytes"


def test_fetch_pack_without_an_index_explains_itself(tmp_path: Path):
    served_root = tmp_path / "empty"
    (served_root / "pack-x").mkdir(parents=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(served_root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match=PACK_FILE_INDEX_FILENAME):
            fetch_pack(f"http://127.0.0.1:{server.server_address[1]}/pack-x/", str(tmp_path / "out"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fetch_pack_rejects_a_bare_name():
    with pytest.raises(ValueError, match="not a pack URL"):
        fetch_pack("pack-full-2024-global")


def test_pack_name_from_source():
    assert pack_name_from_source("https://example.test/packs/pack-full-2024-global/") == "pack-full-2024-global"
    assert pack_name_from_source("pack-quick-2024") == "pack-quick-2024"


def test_default_pack_cache_directory_honours_the_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OCEANBENCH_PACK_CACHE", str(tmp_path / "cache"))
    assert default_pack_cache_directory("pack-a") == tmp_path / "cache" / "pack-a"


def test_resolve_offline_references_passes_local_paths_through(tmp_path: Path):
    assert resolve_offline_references(None) is None
    assert resolve_offline_references(str(tmp_path)) == str(tmp_path)


def test_resolve_offline_references_fetches_https_into_the_cache(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_fetch_pack(source, destination=None):
        calls["source"] = source
        from oceanbench.packs.fetch import PackFetchSummary

        return PackFetchSummary(
            pack_name="pack-full-2024-global",
            destination=str(tmp_path / "cached"),
            downloaded_count=3,
            downloaded_bytes=30,
            skipped_count=0,
            total_count=3,
            total_bytes=30,
        )

    monkeypatch.setattr("oceanbench.packs.fetch.fetch_pack", fake_fetch_pack)
    resolved = resolve_offline_references("https://example.test/packs/pack-full-2024-global/")

    assert calls["source"] == "https://example.test/packs/pack-full-2024-global/"
    assert resolved == str(tmp_path / "cached")
