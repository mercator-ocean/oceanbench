# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Evaluate one challenger on one region for a release candidate. Entrypoint of the EDITO process."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).resolve().parent.parent
RELEASE_CANDIDATE_MODULE = REPOSITORY_DIRECTORY / "edito" / "release_candidate.py"
VIEWER_DIRECTORY_NAME = "viewer-data"


def install_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", str(REPOSITORY_DIRECTORY)]


def evaluate_command(environment: dict, viewer_directory: Path) -> list[str]:
    return [
        "oceanbench",
        "evaluate",
        str(REPOSITORY_DIRECTORY / "challenger_datasets" / f"{environment['CHALLENGER']}.py"),
        "--region",
        environment["REGION"],
        "--output-bucket",
        environment["OUTPUT_BUCKET"],
        "--output-prefix",
        environment["REPORTS_PREFIX"],
        "--viewer-artifacts-dir",
        str(viewer_directory),
    ]


def upload_command(environment: dict, viewer_directory: Path) -> list[str]:
    return [
        sys.executable,
        str(RELEASE_CANDIDATE_MODULE),
        "upload-viewer-data",
        "--release-version",
        environment["RELEASE_VERSION"],
        "--challenger",
        environment["CHALLENGER"],
        "--region",
        environment["REGION"],
        "--directory",
        str(viewer_directory),
    ]


def run(command: list[str], working_directory: Path) -> None:
    subprocess.run(command, cwd=working_directory, check=True)


def main() -> None:
    environment = dict(os.environ)
    working_directory = Path(tempfile.mkdtemp())
    viewer_directory = working_directory / VIEWER_DIRECTORY_NAME
    run(install_command(), working_directory)
    if environment.get("DRY_RUN"):
        print(f"Dry run of {environment['CHALLENGER']} {environment['REGION']}: nothing evaluated")
        run([sys.executable, "-c", "from oceanbench.core.version import __version__; print(__version__)"], working_directory)
        return
    run(evaluate_command(environment, viewer_directory), working_directory)
    run(upload_command(environment, viewer_directory), working_directory)


if __name__ == "__main__":
    main()
