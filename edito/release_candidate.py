# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Evaluate every official challenger on EDITO for a release candidate and publish its scores and viewer data.

``launch`` starts one EDITO process per challenger and region. Each process runs ``evaluate_release_candidate.py``,
which writes the scores file under ``public/evaluation-reports/<version>/`` and uploads the viewer data of its
challenger under ``viewer/data-<version>/``, with its index files apart as a fragment. ``assemble`` merges the
fragments into the index files of the viewer data once every process has finished.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import boto3
import requests

REPOSITORY_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_DIRECTORY))

from oceanbench.publish.viewer_index_merge import merge_datasets, merge_insights, merge_scores_summary  # noqa: E402

AUTHENTICATION_URL = "https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token"
AUTHENTICATION_CLIENT = "edito"
PROCESS_API_URL = "https://api.dive.edito.eu/processes/processes"
GENERIC_PROCESS_IDENTIFIER = "core-processes-generic-1.1.1"
EDITO_PROJECT = "oceanbench"
GIT_REPOSITORY = "https://github.com/mercator-ocean/oceanbench.git"
ENTRYPOINT_PATH = "edito/evaluate_release_candidate.py"
ENVIRONMENT_PATH = "edito/release_candidate_environment.yaml"
CODE_SOURCE_SCRIPT = "My code is based on a Python or Shell script"
EXECUTION_TARGET_EDITO = "My process should execute on EDITO Cloud"
CHALLENGER_DIRECTORY = REPOSITORY_DIRECTORY / "challenger_datasets"
OUTPUT_BUCKET = "oceanbench-bucket"
REPORTS_ROOT_PREFIX = "public/evaluation-reports"
VIEWER_ROOT_PREFIX = "viewer"
FRAGMENTS_DIRECTORY = "fragments"
INDEX_FILE_NAMES = ("datasets.json", "insights.json", "scores-summary.json")
FRIENDLY_NAME_MAXIMUM_LENGTH = 53
ONE_DEGREE_SUFFIX = "_1_degree"
HIGH_RESOLUTION_REGIONS = ("global", "ibi")
GLOBAL_ONLY_CHALLENGERS = ("hclimrep",)
NATIVE_TWELFTH_DEGREE_CHALLENGERS = ("glo12", "glo12_persistence")
HTTP_TIMEOUT_SECONDS = 60


class RunMode(Enum):
    FULL = "full"
    DRY = "dry"


DRY_RUN_REQUESTS = {"cpu": "250m", "memory": "1Gi"}
DRY_RUN_LIMITS = {"cpu": "2", "memory": "8Gi"}


class ResourceClass(Enum):
    ONE_DEGREE = {"cpu": "8", "memory": "48Gi"}
    QUARTER_DEGREE = {"cpu": "16", "memory": "128Gi"}
    TWELFTH_DEGREE = {"cpu": "32", "memory": "200Gi"}


@dataclass(frozen=True)
class Job:
    challenger: str
    region: str

    def name(self) -> str:
        return f"{self.challenger}.{self.region}"


@dataclass(frozen=True)
class Release:
    version: str
    git_reference: str

    def reports_prefix(self) -> str:
        return f"{REPORTS_ROOT_PREFIX}/{self.version}"

    def viewer_prefix(self) -> str:
        return f"{VIEWER_ROOT_PREFIX}/data-{self.version}"


def challenger_regions(challenger: str) -> tuple[str, ...]:
    return ("global",) if challenger.endswith(ONE_DEGREE_SUFFIX) or challenger in GLOBAL_ONLY_CHALLENGERS else HIGH_RESOLUTION_REGIONS


def release_jobs(challengers: list[str]) -> list[Job]:
    return [Job(challenger, region) for challenger in sorted(challengers) for region in challenger_regions(challenger)]


def official_challengers() -> list[str]:
    return [path.stem for path in CHALLENGER_DIRECTORY.glob("*.py")]


def resource_class(challenger: str) -> ResourceClass:
    if challenger.endswith(ONE_DEGREE_SUFFIX):
        return ResourceClass.ONE_DEGREE
    if challenger in NATIVE_TWELFTH_DEGREE_CHALLENGERS:
        return ResourceClass.TWELFTH_DEGREE
    return ResourceClass.QUARTER_DEGREE


def friendly_name(release: Release, job: Job) -> str:
    return f"oceanbench-{release.version}-{job.challenger}-{job.region}"[:FRIENDLY_NAME_MAXIMUM_LENGTH]


def runtime_variables(release: Release, job: Job, mode: RunMode) -> list[dict]:
    values = {
        "CHALLENGER": job.challenger,
        "REGION": job.region,
        "OUTPUT_BUCKET": OUTPUT_BUCKET,
        "REPORTS_PREFIX": release.reports_prefix(),
        "VIEWER_PREFIX": release.viewer_prefix(),
        "RELEASE_VERSION": release.version,
        **({"DRY_RUN": "1"} if mode is RunMode.DRY else {}),
    }
    return [{"name": name, "value": value} for name, value in values.items()]


def process_inputs(release: Release, job: Job, credentials: dict, mode: RunMode) -> dict:
    full_resources = resource_class(job.challenger).value
    requests_resources, limits_resources = (
        (DRY_RUN_REQUESTS, DRY_RUN_LIMITS) if mode is RunMode.DRY else (full_resources, full_resources)
    )
    return {
        "configuration": {
            "source": CODE_SOURCE_SCRIPT,
            "git": {"repository": GIT_REPOSITORY, "branch": release.git_reference, "cache": "0"},
            "script": {"entrypointPath": ENTRYPOINT_PATH, "environmentPath": ENVIRONMENT_PATH},
        },
        "target": {
            "type": EXECUTION_TARGET_EDITO,
            "edito": {"requests": requests_resources, "limits": limits_resources},
        },
        "s3": {"enabled": True, "bucketName": OUTPUT_BUCKET, **credentials["s3"]},
        "copernicusMarine": {"enabled": True, **credentials["copernicusMarine"]},
        "processInputs": runtime_variables(release, job, mode),
    }


def process_payload(release: Release, job: Job, credentials: dict, mode: RunMode) -> dict:
    return {
        "metadata": {"project": EDITO_PROJECT},
        "processInputs": {
            **process_inputs(release, job, credentials, mode),
            "onyxia": {"friendlyName": friendly_name(release, job), "share": False},
        },
    }


DISABLED_CREDENTIALS = {"s3": {"enabled": False}, "copernicusMarine": {"enabled": False}}


def credentials_from_environment(environment: dict, mode: RunMode) -> dict:
    if mode is RunMode.DRY:
        return DISABLED_CREDENTIALS
    return {
        "s3": {
            "endpoint": environment["AWS_S3_ENDPOINT"],
            "defaultRegion": environment["AWS_DEFAULT_REGION"],
            "accessKeyId": environment["AWS_ACCESS_KEY_ID"],
            "secretAccessKey": environment["AWS_SECRET_ACCESS_KEY"],
            "sessionToken": environment.get("AWS_SESSION_TOKEN", ""),
        },
        "copernicusMarine": {
            "username": environment["COPERNICUSMARINE_SERVICE_USERNAME"],
            "password": environment["COPERNICUSMARINE_SERVICE_PASSWORD"],
        },
    }


def access_token(environment: dict) -> str:
    if environment.get("EDITO_ACCESS_TOKEN"):
        return environment["EDITO_ACCESS_TOKEN"]
    response = requests.post(
        AUTHENTICATION_URL,
        data={
            "client_id": AUTHENTICATION_CLIENT,
            "grant_type": "refresh_token",
            "refresh_token": environment["EDITO_OFFLINE_TOKEN"],
            "scope": "openid",
        },
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def launch_job(token: str, release: Release, job: Job, credentials: dict, mode: RunMode) -> str:
    response = requests.post(
        f"{PROCESS_API_URL}/{GENERIC_PROCESS_IDENTIFIER}/execution",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=process_payload(release, job, credentials, mode),
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["jobID"]


def selected_jobs(only: list[str]) -> list[Job]:
    jobs = release_jobs(official_challengers())
    return [job for job in jobs if not only or job.name() in only]


def launch(release: Release, only: list[str], mode: RunMode, environment: dict) -> None:
    token = access_token(environment)
    credentials = credentials_from_environment(environment, mode)
    for job in selected_jobs(only):
        print(f"{job.name()}: {launch_job(token, release, job, credentials, mode)}")


def s3_client(environment: dict):
    endpoint = environment.get("BOTO3_ENDPOINT_URL") or f"https://{environment['AWS_S3_ENDPOINT']}"
    return boto3.client("s3", endpoint_url=endpoint)


def viewer_file_destination(viewer_prefix: str, job: Job, relative_path: Path) -> str:
    if relative_path.name in INDEX_FILE_NAMES and len(relative_path.parts) == 1:
        return f"{viewer_prefix}/{FRAGMENTS_DIRECTORY}/{job.name()}/{relative_path.name}"
    return f"{viewer_prefix}/{relative_path.as_posix()}"


def upload_viewer_data(client, directory: Path, bucket: str, viewer_prefix: str, job: Job) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            client.upload_file(
                str(path), bucket, viewer_file_destination(viewer_prefix, job, path.relative_to(directory))
            )


def fragment_keys(client, bucket: str, viewer_prefix: str, file_name: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys = [
        entry["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{viewer_prefix}/{FRAGMENTS_DIRECTORY}/")
        for entry in page.get("Contents", [])
    ]
    return sorted(key for key in keys if key.endswith(f"/{file_name}"))


def read_json(client, bucket: str, key: str):
    return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def assemble(client, bucket: str, viewer_prefix: str) -> dict[str, int]:
    merges = {
        "datasets.json": merge_datasets,
        "insights.json": merge_insights,
        "scores-summary.json": merge_scores_summary,
    }
    counts = {}
    for file_name, merge in merges.items():
        keys = fragment_keys(client, bucket, viewer_prefix, file_name)
        merged = merge([read_json(client, bucket, key) for key in keys])
        client.put_object(
            Bucket=bucket,
            Key=f"{viewer_prefix}/{file_name}",
            Body=json.dumps(merged, sort_keys=True, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        counts[file_name] = len(keys)
    return counts


def release_from_arguments(arguments: argparse.Namespace) -> Release:
    return Release(version=arguments.release_version, git_reference=arguments.git_reference)


def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "assemble"):
        subcommand = subcommands.add_parser(name)
        subcommand.add_argument("--release-version", required=True)
        subcommand.add_argument("--git-reference", required=True)
    launch_parser = subcommands.choices["launch"]
    launch_parser.add_argument("--only", nargs="*", default=[], help="Jobs as <challenger>.<region>")
    launch_parser.add_argument("--run-mode", type=RunMode, choices=list(RunMode), default=RunMode.FULL)
    upload = subcommands.add_parser("upload-viewer-data")
    upload.add_argument("--release-version", required=True)
    upload.add_argument("--git-reference", default="")
    upload.add_argument("--challenger", required=True)
    upload.add_argument("--region", required=True)
    upload.add_argument("--directory", required=True, type=Path)
    return parser


def run_command(arguments: argparse.Namespace) -> None:
    release = release_from_arguments(arguments)
    environment = dict(os.environ)
    match arguments.command:
        case "launch":
            launch(release, arguments.only, arguments.run_mode, environment)
        case "assemble":
            print(assemble(s3_client(environment), OUTPUT_BUCKET, release.viewer_prefix()))
        case "upload-viewer-data":
            job = Job(arguments.challenger, arguments.region)
            upload_viewer_data(s3_client(environment), arguments.directory, OUTPUT_BUCKET, release.viewer_prefix(), job)


def main() -> None:
    run_command(command_line_parser().parse_args())


if __name__ == "__main__":
    main()
