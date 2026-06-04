# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from base64 import b64decode
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

DATA_IMAGE_URI_PATTERN = re.compile(r"data:image/(?P<format>png|jpeg|jpg|webp);base64,(?P<payload>[A-Za-z0-9+/=\s]*)")


@dataclass(frozen=True)
class WidgetAssetOutput:
    directory: Path
    reference_prefix: str


_WIDGET_ASSET_OUTPUT: WidgetAssetOutput | None = None


def configure_widget_asset_output(directory: str | Path, reference_prefix: str | None = None) -> None:
    global _WIDGET_ASSET_OUTPUT
    asset_directory = Path(directory)
    asset_reference_prefix = reference_prefix if reference_prefix is not None else f"{asset_directory.name}/"
    _WIDGET_ASSET_OUTPUT = WidgetAssetOutput(
        directory=asset_directory,
        reference_prefix=asset_reference_prefix,
    )


def clear_widget_asset_output() -> None:
    global _WIDGET_ASSET_OUTPUT
    _WIDGET_ASSET_OUTPUT = None


def _write_image_data_uri_asset(data_uri: str, asset_output: WidgetAssetOutput) -> str:
    asset_output.directory.mkdir(parents=True, exist_ok=True)
    match = DATA_IMAGE_URI_PATTERN.fullmatch(data_uri)
    if match is None:
        raise ValueError("Expected a PNG, JPEG, or WebP data URI.")
    image_format = match.group("format").lower()
    extension = "jpg" if image_format == "jpeg" else image_format
    content = b64decode(re.sub(r"\s+", "", match.group("payload")))
    digest = sha256(content).hexdigest()[:16]
    asset_file_name = f"image-{digest}.{extension}"
    (asset_output.directory / asset_file_name).write_bytes(content)
    return f"{asset_output.reference_prefix}{asset_file_name}"


def _externalize_payload_value(value, asset_output: WidgetAssetOutput):
    if isinstance(value, dict):
        return {key: _externalize_payload_value(nested_value, asset_output) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_externalize_payload_value(nested_value, asset_output) for nested_value in value]
    if isinstance(value, str) and DATA_IMAGE_URI_PATTERN.fullmatch(value):
        return _write_image_data_uri_asset(value, asset_output)
    return value


def _write_payload_json_asset(payload: dict[str, object], asset_output: WidgetAssetOutput) -> str:
    asset_output.directory.mkdir(parents=True, exist_ok=True)
    payload = _externalize_payload_value(payload, asset_output)
    content = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = sha256(content).hexdigest()[:16]
    asset_file_name = f"payload-{digest}.json"
    (asset_output.directory / asset_file_name).write_bytes(content)
    return f"{asset_output.reference_prefix}{asset_file_name}"


def _payload_loader_javascript(asset_reference: str) -> str:
    asset_reference_json = json.dumps(asset_reference)
    return (
        "(() => {"
        "const request = new XMLHttpRequest();"
        f'request.open("GET", {asset_reference_json}, false);'
        "request.send(null);"
        "if ((request.status >= 200 && request.status < 300) || request.status === 0) {"
        "return JSON.parse(request.responseText);"
        "}"
        f'throw new Error("Failed to load OceanBench widget payload: " + {asset_reference_json});'
        "})()"
    )


def widget_payload_javascript(payload: dict[str, object]) -> str:
    if _WIDGET_ASSET_OUTPUT is None:
        return json.dumps(payload, separators=(",", ":"))
    return _payload_loader_javascript(_write_payload_json_asset(payload, _WIDGET_ASSET_OUTPUT))
