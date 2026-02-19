# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import re

import requests

from helpers.challenger_metadata import KNOWN_CHALLENGERS

S3_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench"
REPORTS_PREFIX = "public/evaluation-reports/"


def discover_challengers() -> list[str]:
    try:
        url = f"{S3_BASE_URL}?list-type=2&prefix={REPORTS_PREFIX}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            keys = re.findall(r"<Key>(.*?)</Key>", response.text)
            names = []
            for key in keys:
                if key.endswith(".report.ipynb"):
                    filename = key.split("/")[-1]
                    name = filename.removesuffix(".report.ipynb")
                    if name:
                        names.append(name)
            if names:
                return sorted(set(names))
    except Exception:
        pass
    return list(KNOWN_CHALLENGERS)


def get_notebook_url(challenger_name: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_PREFIX}{challenger_name}.report.ipynb"


def download_notebook(challenger_name: str, destination_directory: str) -> str | None:
    url = get_notebook_url(challenger_name)
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            os.makedirs(destination_directory, exist_ok=True)
            destination_path = os.path.join(destination_directory, f"{challenger_name}.report.ipynb")
            with open(destination_path, "wb") as file:
                file.write(response.content)
            return destination_path
    except Exception as error:
        print(f"Failed to download {challenger_name}: {error}")
    return None
