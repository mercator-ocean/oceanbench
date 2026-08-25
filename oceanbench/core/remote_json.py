# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Read a JSON document published over anonymous HTTPS."""

import gzip
import json
from urllib.request import urlopen


def read_json_url(url: str, *, timeout: int = 30) -> dict:
    """Read a JSON document over HTTPS, inflating a gzip-encoded body.

    Published JSON objects are stored gzip-compressed and served with
    ``Content-Encoding: gzip``. urllib does not act on that header, so the body
    arrives as gzip bytes and has to be inflated before it can be parsed.
    """
    with urlopen(url, timeout=timeout) as response:  # noqa: S310
        body = response.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return json.loads(body)
