# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import hashlib

from oceanbench.publish import content_address


def _write(path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def test_content_addressed_name_is_sha256_with_suffix(tmp_path):
    source = _write(tmp_path / "class4-matchups.parquet", b"example bytes")
    name = content_address.content_addressed_name(source)
    assert name == f"{hashlib.sha256(b'example bytes').hexdigest()}.parquet"


def test_identical_content_addresses_to_the_same_name(tmp_path):
    first = _write(tmp_path / "a.json", b"same")
    second = _write(tmp_path / "b.json", b"same")
    assert content_address.content_addressed_name(first) == content_address.content_addressed_name(second)


def test_different_content_addresses_differently(tmp_path):
    first = _write(tmp_path / "a.json", b"one")
    second = _write(tmp_path / "b.json", b"two")
    assert content_address.content_addressed_name(first) != content_address.content_addressed_name(second)


def test_publish_blob_copies_under_content_addressed_name(tmp_path):
    source = _write(tmp_path / "spectra.json", b"payload")
    destination = tmp_path / "insights"
    blob = content_address.publish_blob(source, str(destination))
    assert (destination / blob.name).read_bytes() == b"payload"
    assert blob.bytes == len(b"payload")
    assert blob.name.endswith(".json")


def test_publish_blob_is_idempotent(tmp_path):
    source = _write(tmp_path / "spectra.json", b"payload")
    destination = tmp_path / "insights"
    first = content_address.publish_blob(source, str(destination))
    second = content_address.publish_blob(source, str(destination))
    assert first.name == second.name
    assert len(list(destination.iterdir())) == 1
