# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from urllib.parse import parse_qs, urlparse

from oceanbench.core.scores_page import (
    SCORES_PAGE_URL,
    scores_page_instructions,
    scores_page_link,
    uploaded_scores_file_url,
)

SCORES_FILE_NAME = "challenger.global.scores.json"


def test_instructions_name_the_local_file_when_it_was_not_uploaded():
    instructions = scores_page_instructions(SCORES_FILE_NAME, None)

    assert SCORES_PAGE_URL in instructions
    assert SCORES_FILE_NAME in instructions
    assert "?scores=" not in instructions


def test_instructions_give_a_link_to_the_uploaded_file(monkeypatch):
    monkeypatch.setenv("BOTO3_ENDPOINT_URL", "https://s3.example.org/")
    uploaded_url = uploaded_scores_file_url("my-bucket", "results/challenger.global.scores.json")

    link = [word for word in scores_page_instructions(SCORES_FILE_NAME, uploaded_url).split() if "?scores=" in word][0]

    assert uploaded_url == "https://s3.example.org/my-bucket/results/challenger.global.scores.json"
    assert parse_qs(urlparse(link).query)["scores"] == [uploaded_url]


def test_page_link_encodes_the_scores_file_url():
    assert scores_page_link("https://host/a b.json") == f"{SCORES_PAGE_URL}?scores=https%3A%2F%2Fhost%2Fa+b.json"
