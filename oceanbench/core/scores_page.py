# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from os import environ
from os.path import abspath
from urllib.parse import urlencode

SCORES_PAGE_URL = "https://oceanbench.edito.eu/"
SCORES_PAGE_URL_PARAMETER = "scores"
ADD_SCORES_BUTTON_LABEL = "Add a scores file"
YOUR_SCORES_ROW_LABEL = "Your scores"


def _s3_endpoint_url() -> str:
    return environ.get("BOTO3_ENDPOINT_URL") or f"https://{environ['AWS_S3_ENDPOINT']}"


def uploaded_scores_file_url(output_bucket: str, output_name: str) -> str:
    """Address of an uploaded scores file, when its bucket is publicly readable."""
    return f"{_s3_endpoint_url().rstrip('/')}/{output_bucket}/{output_name}"


def scores_page_link(scores_file_url: str) -> str:
    """Link to the OceanBench Scores page showing the scores file at the given public URL."""
    return f"{SCORES_PAGE_URL}?{urlencode({SCORES_PAGE_URL_PARAMETER: scores_file_url})}"


def scores_page_instructions(scores_file_name: str, uploaded_url: str | None) -> str:
    """How to see a scores file next to the published scores on the OceanBench website."""
    lines = [
        f"To see these scores next to the published ones, open {SCORES_PAGE_URL}, "
        f'click "{ADD_SCORES_BUTTON_LABEL}" in the "{YOUR_SCORES_ROW_LABEL}" row and choose {abspath(scores_file_name)}.'
    ]
    if uploaded_url:
        lines.append(
            f"Or open {scores_page_link(uploaded_url)} "
            "(the bucket must be publicly readable and allow cross-origin reads)."
        )
    return "\n".join(lines)
