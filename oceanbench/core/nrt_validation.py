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
import subprocess
import sys
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
DEFAULT_OBSERVATION_LOOKBACK_DAYS = 45

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
    octo_process_package_name: str | None = None
    octo_process_package_version: str | None = None
    octo_generation_status: str | None = None
    octo_pending_reason: str | None = None
    octo_gpu_capacity_available: bool | None = None
    octo_running_inference_processes: int | None = None
    note: str | None = None


def _day_string(day: str | datetime | numpy.datetime64 | pandas.Timestamp) -> str:
    return pandas.Timestamp(day).strftime("%Y-%m-%d")


def _format_zarr_template(day: str | datetime | numpy.datetime64 | pandas.Timestamp, zarr_template: str) -> str:
    timestamp = pandas.Timestamp(day)
    day_string = timestamp.strftime("%Y%m%d")
    date_string = timestamp.strftime("%Y-%m-%d")
    path = zarr_template.format(
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


def latest_complete_class4_observation_day(
    zarr_template: str | None = None,
    search_end_day: str | datetime | numpy.datetime64 | pandas.Timestamp | None = None,
    max_lookback_days: int = DEFAULT_OBSERVATION_LOOKBACK_DAYS,
) -> str:
    if max_lookback_days < 1:
        raise ValueError("max_lookback_days must be greater than or equal to 1.")
    end_day = pandas.Timestamp(search_end_day or pandas.Timestamp.utcnow()).normalize()
    for day_offset in range(max_lookback_days):
        candidate_day = end_day - pandas.Timedelta(days=day_offset)
        if class4_observation_day_is_complete(candidate_day, zarr_template):
            return _day_string(candidate_day)
    end_day_string = _day_string(end_day)
    raise RuntimeError(
        f"No complete Class IV observation day was found within {max_lookback_days} days before {end_day_string}."
    )


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


def _json_object_from_output(output: str) -> dict:
    json_start = output.rfind("{")
    if json_start == -1:
        return {}
    try:
        return json.loads(output[json_start:])
    except json.JSONDecodeError:
        return {}


def request_octo_forecast_generation(
    octo_script: str,
    system_id: str,
    forecast_init: str,
    python_executable: str | None = None,
    forecast_output_prefix: str | None = None,
) -> dict:
    command = [
        python_executable or sys.executable,
        octo_script,
        "generate-forecast",
        "--system-id",
        system_id,
        "--forecast-init",
        forecast_init,
    ]
    if forecast_output_prefix:
        command.extend(["--forecast-output-prefix", forecast_output_prefix])
    completed_process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return _json_object_from_output(completed_process.stdout)


def _s3_forecast_store_from_url(forecast_url: str) -> tuple[str, str, str | None]:
    parsed_url = urlparse(forecast_url)
    if parsed_url.scheme == "s3":
        return parsed_url.netloc, unquote(parsed_url.path).strip("/"), None
    if parsed_url.scheme in {"http", "https"}:
        path_parts = unquote(parsed_url.path).strip("/").split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(
                f"Cannot derive S3 bucket and prefix from {forecast_url!r}."
            )
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
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        for chunk_start in range(0, len(objects), 1000):
            objects_chunk = objects[chunk_start : chunk_start + 1000]
            if objects_chunk:
                client.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": objects_chunk},
                )
                deleted_objects_count += len(objects_chunk)
    return deleted_objects_count


def delete_forecast_zarr_store(forecast_url: str) -> str:
    if forecast_url.startswith("file://"):
        forecast_path = Path(unquote(urlparse(forecast_url).path))
        existed = forecast_path.exists()
        shutil.rmtree(forecast_path, ignore_errors=True)
        return (
            "Deleted local forecast Zarr"
            if existed
            else "Forecast Zarr already absent"
        )
    if (
        not forecast_url.startswith("http://")
        and not forecast_url.startswith("https://")
        and not forecast_url.startswith("s3://")
    ):
        forecast_path = Path(forecast_url)
        existed = forecast_path.exists()
        shutil.rmtree(forecast_path, ignore_errors=True)
        return (
            "Deleted local forecast Zarr"
            if existed
            else "Forecast Zarr already absent"
        )
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


def _write_manifest_to_s3(
    manifest: dict,
    output_bucket: str,
    output_prefix: str | None,
    manifest_name: str,
) -> str:
    import boto3

    key = f"{output_prefix}/{manifest_name}" if output_prefix else manifest_name
    endpoint = os.environ.get("BOTO3_ENDPOINT_URL")
    if endpoint is None and os.environ.get("AWS_S3_ENDPOINT"):
        endpoint = f"https://{os.environ['AWS_S3_ENDPOINT']}"
    client = boto3.client("s3", endpoint_url=endpoint)
    client.put_object(
        Bucket=output_bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
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
    resolved_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(resolved_path)


def validate_nrt_forecast(
    *,
    system_id: str = DEFAULT_NRT_SYSTEM_ID,
    system_label: str = DEFAULT_NRT_SYSTEM_LABEL,
    forecast_zarr_template: str = LIVE_GLONET_FORECAST_ZARR_TEMPLATE,
    observation_zarr_template: str | None = None,
    observation_search_end_day: str | None = None,
    max_observation_lookback_days: int = DEFAULT_OBSERVATION_LOOKBACK_DAYS,
    octo_script: str | None = None,
    octo_python: str | None = None,
    octo_forecast_output_prefix: str | None = None,
    skip_forecast_generation: bool = False,
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
    observation_cutoff = latest_complete_class4_observation_day(
        resolved_observation_template,
        search_end_day=observation_search_end_day,
        max_lookback_days=max_observation_lookback_days,
    )
    forecast_init = forecast_init_for_observation_cutoff(observation_cutoff)
    forecast_url = _format_zarr_template(forecast_init, forecast_zarr_template)
    octo_result: dict = {}
    if not skip_forecast_generation:
        if octo_script is None:
            raise ValueError("--octo-script is required unless --skip-forecast-generation is used.")
        octo_result = request_octo_forecast_generation(
            octo_script=octo_script,
            system_id=system_id,
            forecast_init=forecast_init,
            python_executable=octo_python,
            forecast_output_prefix=octo_forecast_output_prefix,
        )
        forecast_url = octo_result.get("forecast_url") or forecast_url

    forecast_temporary = bool(octo_result.get("temporary", bool(octo_result)))
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
        if cleanup_forecast_after_success and forecast_temporary:
            try:
                forecast_cleanup_status = delete_forecast_zarr_store(forecast_url)
            except Exception as error:
                forecast_cleanup_status = f"Cleanup failed: {error}"
        elif forecast_temporary:
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
        forecast_temporary=forecast_temporary,
        forecast_cleanup_status=forecast_cleanup_status,
        report_notebook=report_notebook,
        report_url=report_url,
        status=status,
        demo=True,
        initial_condition_provenance_validated=False,
        oceanbench_version=__version__,
        octo_process_package_name=octo_result.get("process_package_name"),
        octo_process_package_version=octo_result.get("process_package_version"),
        octo_generation_status=octo_result.get("status"),
        octo_pending_reason=octo_result.get("pending_reason"),
        octo_gpu_capacity_available=octo_result.get("gpu_capacity_available"),
        octo_running_inference_processes=octo_result.get(
            "running_inference_processes"
        ),
        note=note,
    )
    manifest_path_or_url = write_nrt_manifest(
        _manifest_document(result),
        manifest_path=manifest_path,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
    )
    return result, manifest_path_or_url
