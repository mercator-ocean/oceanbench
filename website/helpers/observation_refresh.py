# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from __future__ import annotations

from dataclasses import dataclass
import os
import xml.etree.ElementTree as ElementTree

import requests

DAILY_OBSERVATION_PROCESS_CATALOG_ID = "process-playground"
DAILY_OBSERVATION_PROCESS_PACKAGE = "daily-observation-data"
DAILY_OBSERVATION_PROCESS_VERSION = "0.1.6"
DAILY_OBSERVATION_PROCESS_NAME = "daily-observation-data-live-refresh"
DAILY_OBSERVATION_PROCESS_PROJECT = "project-oceanbench"
DAILY_OBSERVATION_OUTPUT_FOLDER = "project-oceanbench/public/live_observations/{compact_date}.zarr"
EDITO_REGION = "waw3-1"
EDITO_TOKEN_URL = "https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token"
EDITO_MY_LAB_APP_URL = "https://datalab.dive.edito.eu/api/my-lab/app"
EDITO_MY_LAB_SERVICES_URL = "https://datalab.dive.edito.eu/api/my-lab/services"
MINIO_ENDPOINT_URL = "https://minio.dive.edito.eu"
MINIO_ENDPOINT_HOST = "minio.dive.edito.eu"


class MissingObservationRefreshCredentials(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessLaunchContext:
    edito_access_token: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    aws_s3_endpoint: str
    aws_default_region: str


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def _enabled_copernicus_marine_config() -> dict[str, str | bool]:
    username = _env_value("COPERNICUSMARINE_SERVICE_USERNAME")
    password = _env_value("COPERNICUSMARINE_SERVICE_PASSWORD")
    return {
        "enabled": bool(username and password),
        "username": username,
        "password": password,
    }


def _refresh_access_token(client_id: str, refresh_token: str) -> str:
    response = requests.post(
        EDITO_TOKEN_URL,
        data={
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": "openid email profile",
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"EDITO token refresh failed with status {response.status_code}.")
    return str(response.json()["access_token"])


def _temporary_s3_credentials(refresh_token: str) -> dict[str, str]:
    access_token = _refresh_access_token("onyxia-minio", refresh_token)
    response = requests.post(
        MINIO_ENDPOINT_URL,
        params={
            "Action": "AssumeRoleWithWebIdentity",
            "WebIdentityToken": access_token,
            "DurationSeconds": "86400",
            "Version": "2011-06-15",
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Temporary S3 credential request failed with status {response.status_code}.")

    namespace = {"sts": "https://sts.amazonaws.com/doc/2011-06-15/"}
    root = ElementTree.fromstring(response.text)
    credentials = root.find(".//sts:Credentials", namespace)
    if credentials is None:
        raise RuntimeError("Temporary S3 credential response is missing Credentials.")

    values = {
        "aws_access_key_id": credentials.findtext("sts:AccessKeyId", namespaces=namespace),
        "aws_secret_access_key": credentials.findtext("sts:SecretAccessKey", namespaces=namespace),
        "aws_session_token": credentials.findtext("sts:SessionToken", namespaces=namespace),
    }
    if not all(values.values()):
        raise RuntimeError("Temporary S3 credential response is incomplete.")
    return {key: str(value) for key, value in values.items()}


def _launch_context_from_environment() -> ProcessLaunchContext:
    edito_access_token = _env_value("EDITO_ACCESS_TOKEN")
    if not edito_access_token:
        edito_refresh_token = _env_value("EDITO_OFFLINE_TOKEN")
        if not edito_refresh_token:
            raise MissingObservationRefreshCredentials("EDITO_ACCESS_TOKEN or EDITO_OFFLINE_TOKEN is required.")
        edito_access_token = _refresh_access_token("edito", edito_refresh_token)

    aws_access_key_id = _env_value("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _env_value("AWS_SECRET_ACCESS_KEY")
    aws_session_token = _env_value("AWS_SESSION_TOKEN")
    if not all((aws_access_key_id, aws_secret_access_key, aws_session_token)):
        minio_refresh_token = _env_value("EDITO_MINIO_OFFLINE_TOKEN")
        if not minio_refresh_token:
            raise MissingObservationRefreshCredentials(
                "AWS temporary credentials or EDITO_MINIO_OFFLINE_TOKEN are required."
            )
        temporary_credentials = _temporary_s3_credentials(minio_refresh_token)
        aws_access_key_id = temporary_credentials["aws_access_key_id"]
        aws_secret_access_key = temporary_credentials["aws_secret_access_key"]
        aws_session_token = temporary_credentials["aws_session_token"]

    return ProcessLaunchContext(
        edito_access_token=edito_access_token,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        aws_s3_endpoint=_env_value("AWS_S3_ENDPOINT") or MINIO_ENDPOINT_HOST,
        aws_default_region=_env_value("AWS_DEFAULT_REGION") or EDITO_REGION,
    )


def _datalab_headers(context: ProcessLaunchContext) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {context.edito_access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "ONYXIA-PROJECT": DAILY_OBSERVATION_PROCESS_PROJECT,
        "ONYXIA-REGION": EDITO_REGION,
    }


def _parse_json_or_text(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text


def _service_entries(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("services", "apps", "items"):
        entries = payload.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _service_matches_process(service: dict[str, object]) -> bool:
    return DAILY_OBSERVATION_PROCESS_NAME in {
        service.get("id"),
        service.get("name"),
        service.get("friendlyName"),
    }


def _process_state_from_details(details: dict[str, object]) -> str:
    tasks = details.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []

    task_statuses = [task.get("status", {}).get("status") for task in tasks if isinstance(task, dict)]
    if any(status in {"Running", "Pending"} for status in task_statuses):
        return "running"
    if any(status == "Succeeded" for status in task_statuses):
        return "succeeded"
    if any(status in {"Failed", "Error"} for status in task_statuses):
        return "failed"
    if details.get("status") == "deployed":
        return "succeeded"
    return "unknown"


def _process_state(context: ProcessLaunchContext) -> str:
    response = requests.get(
        EDITO_MY_LAB_SERVICES_URL,
        headers=_datalab_headers(context),
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Observation refresh process lookup failed with status {response.status_code}.")

    matching_services = [
        service for service in _service_entries(_parse_json_or_text(response)) if _service_matches_process(service)
    ]
    if not matching_services:
        return "missing"
    return _process_state_from_details(matching_services[0])


def _delete_observation_refresh_process(context: ProcessLaunchContext) -> None:
    response = requests.delete(
        EDITO_MY_LAB_APP_URL,
        headers=_datalab_headers(context),
        params={"path": DAILY_OBSERVATION_PROCESS_NAME},
        timeout=60,
    )
    if response.status_code not in {200, 202, 204, 404}:
        raise RuntimeError(f"Observation refresh process delete failed with status {response.status_code}.")


def _launch_payload(context: ProcessLaunchContext) -> dict[str, object]:
    return {
        "catalogId": DAILY_OBSERVATION_PROCESS_CATALOG_ID,
        "packageName": DAILY_OBSERVATION_PROCESS_PACKAGE,
        "packageVersion": DAILY_OBSERVATION_PROCESS_VERSION,
        "name": DAILY_OBSERVATION_PROCESS_NAME,
        "friendlyName": DAILY_OBSERVATION_PROCESS_NAME,
        "share": True,
        "dryRun": False,
        "options": {
            "catalogType": "Process",
            "onyxia": {
                "friendlyName": DAILY_OBSERVATION_PROCESS_NAME,
                "owner": "oceanbench",
                "share": True,
            },
            "resources": {
                "requests": {"cpu": "1000m", "memory": "1Gi"},
                "limits": {"cpu": "1000m", "memory": "1Gi"},
            },
            "s3": {
                "enabled": True,
                "accessKeyId": context.aws_access_key_id,
                "secretAccessKey": context.aws_secret_access_key,
                "sessionToken": context.aws_session_token,
                "endpoint": context.aws_s3_endpoint.removeprefix("https://").removeprefix("http://").rstrip("/"),
                "defaultRegion": context.aws_default_region,
                "bucketName": DAILY_OBSERVATION_PROCESS_PROJECT,
            },
            "vault": {
                "enabled": False,
                "token": "",
                "url": "https://vault.dive.edito.eu",
                "mount": "secret-kv",
                "directory": DAILY_OBSERVATION_PROCESS_PROJECT,
                "secret": "",
            },
            "copernicusMarine": _enabled_copernicus_marine_config(),
            "inputs": {
                "S3_OUTPUT_FOLDER": DAILY_OBSERVATION_OUTPUT_FOLDER,
            },
        },
    }


def _launch_observation_refresh_process(context: ProcessLaunchContext) -> None:
    response = requests.put(
        EDITO_MY_LAB_APP_URL,
        headers=_datalab_headers(context),
        json=_launch_payload(context),
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Observation refresh process launch failed with status {response.status_code}.")


def maybe_launch_daily_observation_refresh() -> None:
    try:
        context = _launch_context_from_environment()
    except MissingObservationRefreshCredentials as error:
        print(f"Skipping daily observation data refresh: {error}")
        return
    except Exception as error:
        print(f"Could not launch daily observation data refresh: {error}")
        return

    try:
        state = _process_state(context)
        if state in {"pending", "running"}:
            print(f"Daily observation data refresh already {state}, skipping launch.")
            return
        if state != "missing":
            _delete_observation_refresh_process(context)
        _launch_observation_refresh_process(context)
        print(f"Launched daily observation data refresh process {DAILY_OBSERVATION_PROCESS_NAME}.")
    except Exception as error:
        print(f"Could not launch daily observation data refresh: {error}")
