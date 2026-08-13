# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Per-(start, lead, variable, depth) Class-4 overlay extracts (contracts.md §4, §6).

The Class-4 match-up parquet is the analysis artifact: it carries every match-up of the year at
full precision and is several gigabytes. The viewer does not need any of that to paint the
scatter overlay of one forecast start, one lead and one variable, so this module writes a tiny
companion file per selection: the ``.obx`` extract.

An extract holds the four arrays the overlay needs (``latitude``, ``longitude``,
``observation_value``, ``model_value``) quantized to ``uint16`` with the pyramid's
``scale_factor`` / ``add_offset`` convention, behind a JSON header that carries the selection
key, the quantizations and the honest observation counts. At the default display cap a full
extract is 400 KB and a median one is around 220 KB, against the 6 MB parquet footer plus
several megabytes of row groups a first interactive load costs today.

Counts are never silently dropped: ``observation_count`` is every row of the selection,
``matched_count`` the finite obs/model pairs of it, and ``displayed_count`` what the file
actually stores. A viewer that shows fewer points than ``matched_count`` has the numbers in
hand to say so.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy

from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.pyramids.quantization import Quantization, quantization_for_range

CLASS4_OVERLAY_DIRECTORY = "class4-overlays"
OVERLAY_MANIFEST_FILENAME = "manifest.json"
OVERLAY_FILE_SUFFIX = ".obx"
OVERLAY_MAGIC = b"OBX1"
OVERLAY_FORMAT = "obx"
OVERLAY_FORMAT_VERSION = 1

# 50 000 points is 400 KB of payload: the size budget the viewer's first paint is held to, and
# already far denser than a world map can resolve. Selections above it are subsampled.
DISPLAY_POINT_CAP = 50_000

OVERLAY_VALUE_COLUMNS = ("latitude", "longitude", "observation_value", "model_value")
_OVERLAY_KEY_COLUMNS = ("variable", "depth_bin", "start_date", "lead_day")
_HEADER_ALIGNMENT = 8
_QUANTIZED_MAXIMUM_CODE = 65534


@dataclass(frozen=True)
class OverlayEntry:
    """One written extract: its selection key, its counts and its relative path."""

    variable: str
    depth_bin: str
    start_date: str
    lead_day: int
    observation_count: int
    matched_count: int
    displayed_count: int
    byte_size: int
    relative_path: str


def overlay_relative_path(variable: str, depth_bin: str, start_date: str, lead_day: int) -> str:
    """Path of one extract relative to the ``class4-overlays`` directory."""
    return "/".join(
        (
            _path_component(variable),
            _path_component(depth_bin),
            f"{_path_component(start_date)}-lead{int(lead_day):02d}{OVERLAY_FILE_SUFFIX}",
        )
    )


def _path_component(value) -> str:
    text = str(value)
    if not text or "/" in text or "\\" in text or text in (".", ".."):
        raise ValueError(f"{value!r} cannot be part of an overlay path")
    return text


def _selection_seed(dataset_slug: str, region: str, variable: str, depth_bin: str, start_date: str, lead_day: int):
    """Deterministic subsampling seed, so a regen of the same selection picks the same points."""
    key = "|".join((dataset_slug, region, variable, depth_bin, start_date, str(int(lead_day))))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _selected_indices(matched_count: int, display_point_cap: int, seed: int) -> numpy.ndarray | None:
    """Indices kept for display, or ``None`` when the whole selection fits under the cap.

    A uniform sample without replacement, not a stride: match-ups arrive in observation-file
    order (satellite tracks, float profiles), and every fixed stride over that order risks
    sampling one track geometry rather than the ocean.
    """
    if display_point_cap <= 0 or matched_count <= display_point_cap:
        return None
    generator = numpy.random.default_rng(seed)
    return numpy.sort(generator.choice(matched_count, size=display_point_cap, replace=False))


def _quantize(values: numpy.ndarray) -> tuple[numpy.ndarray, Quantization]:
    if values.size == 0:
        return numpy.zeros(0, dtype=numpy.uint16), Quantization(scale_factor=1.0, add_offset=0.0)
    quantization = quantization_for_range(float(values.min()), float(values.max()))
    codes = numpy.rint((values - quantization.add_offset) / quantization.scale_factor)
    return numpy.clip(codes, 0, _QUANTIZED_MAXIMUM_CODE).astype(numpy.uint16), quantization


def _quantization_payload(quantization: Quantization) -> dict:
    return {"scale_factor": quantization.scale_factor, "add_offset": quantization.add_offset}


def encode_class4_overlay(
    latitude,
    longitude,
    observation_value,
    model_value,
    *,
    dataset_slug: str,
    region: str,
    variable: str,
    depth_bin: str,
    start_date: str,
    lead_day: int,
    observation_count: int | None = None,
    display_point_cap: int = DISPLAY_POINT_CAP,
) -> tuple[bytes, OverlayEntry]:
    """Encode one selection into ``.obx`` bytes and its manifest entry.

    The four inputs are the raw match-up arrays of a single
    ``(start_date, lead_day, variable, depth_bin)`` selection, in any order. Rows whose
    observation or model value is not finite are dropped before the display cap is applied;
    ``observation_count`` (default: the input length) records the selection before that drop.
    """
    latitude = numpy.asarray(latitude, dtype=numpy.float64)
    longitude = numpy.asarray(longitude, dtype=numpy.float64)
    observation_value = numpy.asarray(observation_value, dtype=numpy.float64)
    model_value = numpy.asarray(model_value, dtype=numpy.float64)
    total = int(latitude.size) if observation_count is None else int(observation_count)

    finite = (
        numpy.isfinite(latitude)
        & numpy.isfinite(longitude)
        & numpy.isfinite(observation_value)
        & numpy.isfinite(model_value)
    )
    latitude, longitude = latitude[finite], longitude[finite]
    observation_value, model_value = observation_value[finite], model_value[finite]
    matched_count = int(latitude.size)

    seed = _selection_seed(dataset_slug, region, variable, depth_bin, start_date, lead_day)
    indices = _selected_indices(matched_count, display_point_cap, seed)
    if indices is not None:
        latitude, longitude = latitude[indices], longitude[indices]
        observation_value, model_value = observation_value[indices], model_value[indices]

    columns = (latitude, longitude, observation_value, model_value)
    quantized = [_quantize(column) for column in columns]
    header = {
        "format": OVERLAY_FORMAT,
        "version": OVERLAY_FORMAT_VERSION,
        "dataset": dataset_slug,
        "region": region,
        "variable": variable,
        "depth_bin": depth_bin,
        "start_date": start_date,
        "lead_day": int(lead_day),
        "observation_count": total,
        "matched_count": matched_count,
        "displayed_count": int(latitude.size),
        "display_point_cap": int(display_point_cap),
        "decimated": bool(indices is not None),
        "columns": list(OVERLAY_VALUE_COLUMNS),
        "quantization": {
            name: _quantization_payload(quantization)
            for name, (_, quantization) in zip(OVERLAY_VALUE_COLUMNS, quantized)
        },
        "oceanbench_version": OCEANBENCH_VERSION,
    }
    payload = _frame_bytes(header, [codes for codes, _ in quantized])
    entry = OverlayEntry(
        variable=variable,
        depth_bin=depth_bin,
        start_date=start_date,
        lead_day=int(lead_day),
        observation_count=total,
        matched_count=matched_count,
        displayed_count=int(latitude.size),
        byte_size=len(payload),
        relative_path=overlay_relative_path(variable, depth_bin, start_date, lead_day),
    )
    return payload, entry


def _frame_bytes(header: dict, columns: list[numpy.ndarray]) -> bytes:
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    prefix_length = len(OVERLAY_MAGIC) + 4 + len(header_bytes)
    padding = (-prefix_length) % _HEADER_ALIGNMENT
    return b"".join(
        (
            OVERLAY_MAGIC,
            numpy.uint32(len(header_bytes) + padding).tobytes(),
            header_bytes,
            b"\x00" * padding,
            *(column.astype("<u2").tobytes() for column in columns),
        )
    )


def read_class4_overlay(path: str) -> tuple[dict, dict]:
    """Read an extract back into its header and decoded float columns (round-trip helper)."""
    return decode_class4_overlay(Path(path).read_bytes())


def decode_class4_overlay(payload: bytes) -> tuple[dict, dict]:
    """Decode ``.obx`` bytes into ``(header, {column: float64 array})``."""
    if payload[: len(OVERLAY_MAGIC)] != OVERLAY_MAGIC:
        raise ValueError("not a Class-4 overlay extract: bad magic")
    header_start = len(OVERLAY_MAGIC) + 4
    header_length = int(numpy.frombuffer(payload, dtype="<u4", count=1, offset=len(OVERLAY_MAGIC))[0])
    header = json.loads(payload[header_start : header_start + header_length].rstrip(b"\x00").decode("utf-8"))
    count = int(header["displayed_count"])
    offset = header_start + header_length
    decoded = {}
    for name in header["columns"]:
        codes = numpy.frombuffer(payload, dtype="<u2", count=count, offset=offset)
        quantization = header["quantization"][name]
        decoded[name] = codes.astype(numpy.float64) * quantization["scale_factor"] + quantization["add_offset"]
        offset += count * 2
    return header, decoded


def write_class4_overlays(
    matchups,
    overlay_directory: str,
    *,
    dataset_slug: str,
    region: str,
    display_point_cap: int = DISPLAY_POINT_CAP,
) -> list[OverlayEntry]:
    """Write one extract per ``(variable, depth_bin, start_date, lead_day)`` of a match-up frame.

    ``matchups`` is a Class-4 match-up dataframe (one forecast start's partition, or a whole
    year); every selection present in it becomes one file under ``overlay_directory``.
    """
    missing = [column for column in (*_OVERLAY_KEY_COLUMNS, *OVERLAY_VALUE_COLUMNS) if column not in matchups.columns]
    if missing:
        raise ValueError(f"match-ups are missing the overlay columns {missing}")
    root = Path(overlay_directory)
    entries = []
    for key, selection in matchups.groupby(list(_OVERLAY_KEY_COLUMNS), sort=True):
        variable, depth_bin, start_date, lead_day = key
        payload, entry = encode_class4_overlay(
            selection["latitude"].to_numpy(),
            selection["longitude"].to_numpy(),
            selection["observation_value"].to_numpy(),
            selection["model_value"].to_numpy(),
            dataset_slug=dataset_slug,
            region=region,
            variable=str(variable),
            depth_bin=str(depth_bin),
            start_date=_start_date_text(start_date),
            lead_day=int(lead_day),
            display_point_cap=display_point_cap,
        )
        path = root / entry.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(entry)
    return entries


def _start_date_text(start_date) -> str:
    return str(numpy.datetime64(start_date, "D")) if not isinstance(start_date, str) else start_date[:10]


def overlay_index_entry(dataset_slug: str, region: str, *, display_point_cap: int = DISPLAY_POINT_CAP) -> dict:
    """The ``class4_overlays`` object a viewer index (``insights.json``) carries for one dataset.

    ``lead_day`` is substituted zero-padded to two digits; every other placeholder is the literal
    value of the selection. A viewer that finds this object paints from the extracts; one that
    does not falls back to range-reading the match-up parquet.
    """
    base = f"./data/insights/{dataset_slug}/{region}/{CLASS4_OVERLAY_DIRECTORY}"
    return {
        "format": f"{OVERLAY_FORMAT}/{OVERLAY_FORMAT_VERSION}",
        "template": f"{base}/{{variable}}/{{depth_bin}}/{{start_date}}-lead{{lead_day}}{OVERLAY_FILE_SUFFIX}",
        "display_point_cap": int(display_point_cap),
        "manifest": f"{base}/{OVERLAY_MANIFEST_FILENAME}",
    }


def write_class4_overlay_manifest(
    entries: list[OverlayEntry],
    overlay_directory: str,
    *,
    dataset_slug: str,
    region: str,
    display_point_cap: int = DISPLAY_POINT_CAP,
) -> str:
    """Write the availability + counts index of a dataset's extracts.

    Nested ``variable -> depth_bin -> start_date -> lead_day -> [observation_count,
    matched_count, displayed_count]`` so the whole year of one variable is one small object. The
    viewer needs it only to enumerate what exists: every count is also inside the extract itself.
    """
    availability: dict = {}
    for entry in sorted(entries, key=lambda item: (item.variable, item.depth_bin, item.start_date, item.lead_day)):
        depth_bins = availability.setdefault(entry.variable, {})
        start_dates = depth_bins.setdefault(entry.depth_bin, {})
        leads = start_dates.setdefault(entry.start_date, {})
        leads[str(entry.lead_day)] = [entry.observation_count, entry.matched_count, entry.displayed_count]
    manifest = {
        **overlay_index_entry(dataset_slug, region, display_point_cap=display_point_cap),
        "dataset": dataset_slug,
        "region": region,
        "counts": ["observation_count", "matched_count", "displayed_count"],
        "extract_count": len(entries),
        "total_byte_size": sum(entry.byte_size for entry in entries),
        "oceanbench_version": OCEANBENCH_VERSION,
        "availability": availability,
    }
    output_path = Path(overlay_directory) / OVERLAY_MANIFEST_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return str(output_path)


def read_class4_overlay_manifest(path: str) -> dict:
    """Read an overlay manifest back (round-trip helper)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def overlay_directory_for(insights_directory: str) -> str:
    """The ``class4-overlays`` directory of one ``insights/<dataset>/<region>`` directory."""
    return os.path.join(insights_directory, CLASS4_OVERLAY_DIRECTORY)
