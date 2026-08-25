# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Tests for the gzip-aware JSON-over-HTTPS reader (urlopen is always stubbed)."""

from contextlib import contextmanager
import gzip
import io
import json

from oceanbench.core.remote_json import read_json_url


def _stub_urlopen(monkeypatch, body: bytes):
    @contextmanager
    def fake_urlopen(url, timeout=None):  # noqa: ARG001 - signature mirrors urllib
        yield io.BytesIO(body)

    monkeypatch.setattr("oceanbench.core.remote_json.urlopen", fake_urlopen)


def test_read_json_url_inflates_a_gzip_encoded_body(monkeypatch):
    payload = {"datasets": [{"name": "glorys"}]}
    _stub_urlopen(monkeypatch, gzip.compress(json.dumps(payload).encode("utf-8")))

    assert read_json_url("https://example.test/data/datasets.json") == payload


def test_read_json_url_reads_a_plain_body(monkeypatch):
    payload = {"datasets": []}
    _stub_urlopen(monkeypatch, json.dumps(payload).encode("utf-8"))

    assert read_json_url("https://example.test/data/datasets.json") == payload
