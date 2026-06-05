# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import unquote, urlparse

import numpy
import pandas
import requests
import xarray

from oceanbench.core.dataset_utils import Dimension, LEAD_DAYS_COUNT, Variable
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.evaluate import evaluate_live_challenger
from oceanbench.core.live_datasets import (
    LIVE_GLONET_FORECAST_ZARR_TEMPLATE,
    live_class4_observation_zarr_template,
)
from oceanbench.core.references.observations import observation_path
from oceanbench.core.regions import RegionLike, resolve_region
from oceanbench.core.runtime_configuration import RuntimeConfiguration, runtime_configuration_from_environment
from oceanbench.core.version import __version__

DEFAULT_NRT_MANIFEST_NAME = "nrt-validation-manifest.json"
DEFAULT_NRT_SYSTEM_ID = "octo-glonet-p1d"
DEFAULT_NRT_SYSTEM_LABEL = "GLONET"
DEFAULT_FORECAST_READY_TIMEOUT_SECONDS = 3600
DEFAULT_FORECAST_READY_POLL_SECONDS = 60
DEFAULT_MANIFEST_WRITE_RETRIES = 5

REQUIRED_CLASS4_OBSERVATION_KEYS = (
    Dimension.TIME.key(),
    Dimension.DEPTH.key(),
    Dimension.LATITUDE.key(),
    Dimension.LONGITUDE.key(),
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    Variable.SEA_WATER_SALINITY.key(),
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
)
REQUIRED_CLASS4_OBSERVATION_VARIABLE_KEYS = (
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    Variable.SEA_WATER_SALINITY.key(),
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
)


@dataclass(frozen=True)
class NrtValidationResult:
    system_id: str
    system_label: str
    region: str
    forecast_init: str
    forecast_lead_days: int
    validated_lead_days: str
    observation_cutoff: str
    observation_zarr_template: str
    forecast_url: str
    forecast_temporary: bool
    forecast_cleanup_status: str | None
    report_notebook: str
    report_url: str
    status: str
    demo: bool
    initial_condition_provenance_validated: bool
    oceanbench_version: str
    note: str | None = None


def _day_string(day: str | datetime | numpy.datetime64 | pandas.Timestamp) -> str:
    return pandas.Timestamp(day).strftime("%Y-%m-%d")


def _format_zarr_template(day: str | datetime | numpy.datetime64 | pandas.Timestamp, zarr_template: str) -> str:
    timestamp = pandas.Timestamp(day)
    day_string = timestamp.strftime("%Y%m%d")
    date_string = timestamp.strftime("%Y-%m-%d")
    path = zarr_template.format(
        compact_date=day_string,
        day=day_string,
        date=date_string,
        yyyymmdd=day_string,
        YYYYMMDD=day_string,
    )
    if path.startswith("file://"):
        parsed_path = urlparse(path)
        return unquote(parsed_path.path)
    return path


def class4_observation_day_is_complete(
    day: str | datetime | numpy.datetime64 | pandas.Timestamp,
    zarr_template: str | None = None,
) -> bool:
    zarr_path = observation_path(
        numpy.datetime64(_day_string(day)),
        zarr_template or live_class4_observation_zarr_template(),
    )
    try:
        dataset = xarray.open_dataset(zarr_path, engine="zarr", decode_cf=False)
    except Exception:
        return False
    try:
        if not all(key in dataset.variables or key in dataset.coords for key in REQUIRED_CLASS4_OBSERVATION_KEYS):
            return False
        time_key = Dimension.TIME.key()
        if dataset[time_key].size == 0:
            return False
        for variable_key in REQUIRED_CLASS4_OBSERVATION_VARIABLE_KEYS:
            if dataset[variable_key].size == 0:
                return False
            if int(dataset[variable_key].count().compute()) == 0:
                return False
        return True
    finally:
        dataset.close()


def forecast_init_for_observation_cutoff(observation_cutoff: str) -> str:
    return _day_string(pandas.Timestamp(observation_cutoff) - pandas.Timedelta(days=LEAD_DAYS_COUNT))


def forecast_zarr_success_exists(forecast_url: str) -> bool:
    if forecast_url.startswith("file://"):
        forecast_path = Path(unquote(urlparse(forecast_url).path))
        return (forecast_path / "_SUCCESS").exists()
    if forecast_url.startswith("http://") or forecast_url.startswith("https://"):
        try:
            response = requests.head(f"{forecast_url.rstrip('/')}/_SUCCESS", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    return (Path(forecast_url) / "_SUCCESS").exists()


def wait_for_forecast_zarr_success(
    forecast_url: str,
    timeout_seconds: int = DEFAULT_FORECAST_READY_TIMEOUT_SECONDS,
    poll_seconds: int = DEFAULT_FORECAST_READY_POLL_SECONDS,
) -> bool:
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be greater than or equal to 0.")
    if poll_seconds < 1:
        raise ValueError("poll_seconds must be greater than or equal to 1.")
    deadline = time.monotonic() + timeout_seconds
    while True:
        if forecast_zarr_success_exists(forecast_url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _s3_forecast_store_from_url(forecast_url: str) -> tuple[str, str, str | None]:
    parsed_url = urlparse(forecast_url)
    if parsed_url.scheme == "s3":
        return parsed_url.netloc, unquote(parsed_url.path).strip("/"), None
    if parsed_url.scheme in {"http", "https"}:
        path_parts = unquote(parsed_url.path).strip("/").split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"Cannot derive S3 bucket and prefix from {forecast_url!r}.")
        endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return path_parts[0], path_parts[1], endpoint_url
    raise ValueError(f"Unsupported remote forecast URL scheme: {forecast_url!r}.")


def _boto3_endpoint_url(endpoint_url: str | None) -> str | None:
    if endpoint_url:
        return endpoint_url
    configured_endpoint = os.environ.get("BOTO3_ENDPOINT_URL")
    if configured_endpoint:
        return configured_endpoint
    if os.environ.get("AWS_S3_ENDPOINT"):
        return f"https://{os.environ['AWS_S3_ENDPOINT']}"
    return None


def _delete_s3_prefix(
    bucket_name: str,
    prefix: str,
    endpoint_url: str | None,
) -> int:
    import boto3

    client = boto3.client("s3", endpoint_url=_boto3_endpoint_url(endpoint_url))
    deleted_objects_count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix.rstrip("/") + "/",
    ):
        for item in page.get("Contents", []):
            client.delete_object(Bucket=bucket_name, Key=item["Key"])
            deleted_objects_count += 1
    return deleted_objects_count


def delete_forecast_zarr_store(forecast_url: str) -> str:
    if forecast_url.startswith("file://"):
        forecast_path = Path(unquote(urlparse(forecast_url).path))
        existed = forecast_path.exists()
        shutil.rmtree(forecast_path, ignore_errors=True)
        return "Deleted local forecast Zarr" if existed else "Forecast Zarr already absent"
    if (
        not forecast_url.startswith("http://")
        and not forecast_url.startswith("https://")
        and not forecast_url.startswith("s3://")
    ):
        forecast_path = Path(forecast_url)
        existed = forecast_path.exists()
        shutil.rmtree(forecast_path, ignore_errors=True)
        return "Deleted local forecast Zarr" if existed else "Forecast Zarr already absent"
    bucket_name, prefix, endpoint_url = _s3_forecast_store_from_url(forecast_url)
    deleted_objects_count = _delete_s3_prefix(bucket_name, prefix, endpoint_url)
    return f"Deleted {deleted_objects_count} forecast Zarr objects"


def _report_file_name(system_label: str, region: RegionLike) -> str:
    resolved_region = resolve_region(region)
    system_slug = re.sub(r"[^a-z0-9]+", "-", system_label.lower()).strip("-")
    return f"{system_slug}.latest.{resolved_region.id}.report.ipynb"


def _report_url(
    report_notebook: str,
    output_bucket: str | None,
    output_prefix: str | None,
) -> str:
    if not output_bucket:
        return report_notebook
    output_name = f"{output_prefix}/{report_notebook}" if output_prefix else report_notebook
    endpoint = os.environ.get("AWS_S3_ENDPOINT", "minio.dive.edito.eu")
    endpoint = endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{endpoint}/{output_bucket}/{output_name}"


def _write_challenger_file(
    directory: Path,
    forecast_url: str,
    forecast_init: str,
) -> Path:
    challenger_path = directory / "nrt_challenger.py"
    challenger_path.write_text(
        "from datetime import datetime\n\n"
        "from oceanbench.core.live_datasets import open_live_forecast_zarr\n\n"
        "challenger_dataset = open_live_forecast_zarr(\n"
        f"    {forecast_url!r},\n"
        f"    datetime.fromisoformat({forecast_init!r}),\n"
        ")\n",
        encoding="utf-8",
    )
    return challenger_path


@contextmanager
def _temporary_environment(updates: dict[str, str | None]):
    previous_values = {name: os.environ.get(name) for name in updates}
    for name, value in updates.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def _manifest_document(result: NrtValidationResult) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "evaluations": [asdict(result)],
    }


def _evaluation_manifest_key(evaluation: dict) -> tuple[str, str] | None:
    system_id = evaluation.get("system_id")
    region = evaluation.get("region")
    if not system_id or not region:
        return None
    return str(system_id), str(region)


def _replace_or_append_evaluation(evaluations: list[dict], new_evaluation: dict) -> list[dict]:
    new_key = _evaluation_manifest_key(new_evaluation)
    if new_key is None:
        return evaluations + [new_evaluation]

    merged_evaluations = []
    replaced = False
    for evaluation in evaluations:
        if _evaluation_manifest_key(evaluation) == new_key:
            if not replaced:
                merged_evaluations.append(new_evaluation)
                replaced = True
            continue
        merged_evaluations.append(evaluation)

    if not replaced:
        merged_evaluations.append(new_evaluation)
    return merged_evaluations


def _merge_nrt_manifest_documents(
    existing_manifest: dict | None,
    new_manifest: dict,
) -> dict:
    existing_evaluations = existing_manifest.get("evaluations", []) if isinstance(existing_manifest, dict) else []
    new_evaluations = new_manifest.get("evaluations", [])
    merged_evaluations = [evaluation for evaluation in existing_evaluations if isinstance(evaluation, dict)]

    for new_evaluation in new_evaluations:
        if isinstance(new_evaluation, dict):
            merged_evaluations = _replace_or_append_evaluation(merged_evaluations, new_evaluation)

    return {
        "schema_version": new_manifest.get(
            "schema_version",
            existing_manifest.get("schema_version", 1) if isinstance(existing_manifest, dict) else 1,
        ),
        "generated_at": new_manifest.get(
            "generated_at",
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        ),
        "evaluations": merged_evaluations,
    }


def _s3_manifest_key(output_prefix: str | None, manifest_name: str) -> str:
    return f"{output_prefix}/{manifest_name}" if output_prefix else manifest_name


def _s3_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", {})
    if not isinstance(response, dict):
        return None
    error_document = response.get("Error", {})
    if not isinstance(error_document, dict):
        return None
    code = error_document.get("Code")
    return str(code) if code is not None else None


def _s3_object_is_missing(error: Exception) -> bool:
    return _s3_error_code(error) in {"NoSuchKey", "NotFound", "404"}


def _s3_put_precondition_failed(error: Exception) -> bool:
    return _s3_error_code(error) in {
        "PreconditionFailed",
        "ConditionalRequestConflict",
        "409",
        "412",
    }


def _read_manifest_from_s3(client, output_bucket: str, key: str) -> tuple[dict | None, str | None]:
    try:
        response = client.get_object(Bucket=output_bucket, Key=key)
    except Exception as error:
        if _s3_object_is_missing(error):
            return None, None
        raise

    body = response["Body"].read().decode("utf-8")
    return json.loads(body), response.get("ETag")


def _write_manifest_to_s3_with_condition(
    client,
    output_bucket: str,
    key: str,
    manifest: dict,
    etag: str | None,
) -> None:
    put_arguments = {
        "Bucket": output_bucket,
        "Key": key,
        "Body": json.dumps(manifest, indent=2).encode("utf-8"),
        "ContentType": "application/json",
    }
    if etag is None:
        put_arguments["IfNoneMatch"] = "*"
    else:
        put_arguments["IfMatch"] = etag
    client.put_object(**put_arguments)


def _write_manifest_to_s3(
    manifest: dict,
    output_bucket: str,
    output_prefix: str | None,
    manifest_name: str,
) -> str:
    import boto3

    key = _s3_manifest_key(output_prefix, manifest_name)
    endpoint = os.environ.get("BOTO3_ENDPOINT_URL")
    if endpoint is None and os.environ.get("AWS_S3_ENDPOINT"):
        endpoint = f"https://{os.environ['AWS_S3_ENDPOINT']}"
    client = boto3.client("s3", endpoint_url=endpoint)
    for _ in range(DEFAULT_MANIFEST_WRITE_RETRIES):
        existing_manifest, etag = _read_manifest_from_s3(client, output_bucket, key)
        merged_manifest = _merge_nrt_manifest_documents(existing_manifest, manifest)
        try:
            _write_manifest_to_s3_with_condition(client, output_bucket, key, merged_manifest, etag)
            break
        except Exception as error:
            if not _s3_put_precondition_failed(error):
                raise
    else:
        raise RuntimeError("Could not update NRT manifest after repeated concurrent writes.")
    return f"s3://{output_bucket}/{key}"


def write_nrt_manifest(
    manifest: dict,
    manifest_path: str | None,
    output_bucket: str | None,
    output_prefix: str | None,
    manifest_name: str = DEFAULT_NRT_MANIFEST_NAME,
) -> str:
    if output_bucket:
        return _write_manifest_to_s3(manifest, output_bucket, output_prefix, manifest_name)
    resolved_path = Path(manifest_path or manifest_name)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    existing_manifest = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else None
    merged_manifest = _merge_nrt_manifest_documents(existing_manifest, manifest)
    resolved_path.write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
    return str(resolved_path)


def validate_nrt_forecast(
    *,
    system_id: str = DEFAULT_NRT_SYSTEM_ID,
    system_label: str = DEFAULT_NRT_SYSTEM_LABEL,
    forecast_zarr_template: str = LIVE_GLONET_FORECAST_ZARR_TEMPLATE,
    observation_zarr_template: str | None = None,
    forecast_init: str | None = None,
    observation_cutoff: str | None = None,
    forecast_temporary: bool = False,
    forecast_ready_timeout_seconds: int = DEFAULT_FORECAST_READY_TIMEOUT_SECONDS,
    forecast_ready_poll_seconds: int = DEFAULT_FORECAST_READY_POLL_SECONDS,
    cleanup_forecast_after_success: bool = True,
    output_bucket: str | None = None,
    output_prefix: str | None = None,
    manifest_path: str | None = None,
    runtime_configuration: RuntimeConfiguration | None = None,
    region: RegionLike = None,
) -> tuple[NrtValidationResult, str]:
    resolved_observation_template = observation_zarr_template or live_class4_observation_zarr_template()
    if forecast_init is None or observation_cutoff is None:
        raise ValueError("--forecast-init and --observation-cutoff are required.")
    forecast_init = _day_string(forecast_init)
    observation_cutoff = _day_string(observation_cutoff)
    expected_forecast_init = forecast_init_for_observation_cutoff(observation_cutoff)
    if forecast_init != expected_forecast_init:
        raise ValueError(
            f"Forecast init {forecast_init} is not {LEAD_DAYS_COUNT} days before "
            f"observation cutoff {observation_cutoff}."
        )
    if not class4_observation_day_is_complete(observation_cutoff, resolved_observation_template):
        raise RuntimeError(f"Class IV observation day {observation_cutoff} is not complete.")
    forecast_url = _format_zarr_template(forecast_init, forecast_zarr_template)
    resolved_forecast_temporary = bool(forecast_temporary)
    forecast_ready = wait_for_forecast_zarr_success(
        forecast_url,
        timeout_seconds=forecast_ready_timeout_seconds,
        poll_seconds=forecast_ready_poll_seconds,
    )
    report_notebook = _report_file_name(system_label, region)
    report_url = _report_url(report_notebook, output_bucket, output_prefix)
    note = (
        "Demonstration only: this forecast was regenerated on demand, and exact operational "
        "initial-condition provenance has not been validated yet."
    )
    status = "Forecast pending"
    forecast_cleanup_status = None
    if forecast_ready:
        with tempfile.TemporaryDirectory(prefix="oceanbench-nrt-") as temporary_directory:
            challenger_path = _write_challenger_file(Path(temporary_directory), forecast_url, forecast_init)
            environment_updates = {
                OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_ZARR_TEMPLATE.value: (
                    resolved_observation_template
                ),
                OceanbenchEnvironmentVariable.OCEANBENCH_LIVE_OBSERVATION_LAST_DAY.value: observation_cutoff,
            }
            with _temporary_environment(environment_updates):
                evaluate_live_challenger(
                    challenger_python_code_uri_or_local_path=str(challenger_path),
                    output_notebook_file_name=report_notebook,
                    output_bucket=output_bucket,
                    output_prefix=output_prefix,
                    runtime_configuration=runtime_configuration or runtime_configuration_from_environment(),
                    region=region,
                )
        status = "Complete"
        if cleanup_forecast_after_success and resolved_forecast_temporary:
            try:
                forecast_cleanup_status = delete_forecast_zarr_store(forecast_url)
            except Exception as error:
                forecast_cleanup_status = f"Cleanup failed: {error}"
        elif resolved_forecast_temporary:
            forecast_cleanup_status = "Kept temporary forecast Zarr"

    result = NrtValidationResult(
        system_id=system_id,
        system_label=system_label,
        region=resolve_region(region).id,
        forecast_init=forecast_init,
        forecast_lead_days=LEAD_DAYS_COUNT,
        validated_lead_days="1-10 days",
        observation_cutoff=observation_cutoff,
        observation_zarr_template=resolved_observation_template,
        forecast_url=forecast_url,
        forecast_temporary=resolved_forecast_temporary,
        forecast_cleanup_status=forecast_cleanup_status,
        report_notebook=report_notebook,
        report_url=report_url,
        status=status,
        demo=True,
        initial_condition_provenance_validated=False,
        oceanbench_version=__version__,
        note=note,
    )
    manifest_path_or_url = write_nrt_manifest(
        _manifest_document(result),
        manifest_path=manifest_path,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
    )
    return result, manifest_path_or_url
