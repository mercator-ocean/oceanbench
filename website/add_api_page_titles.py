# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re
from functools import partial
from itertools import chain
from pathlib import Path

import yaml

QUARTO_CONFIGURATION_PATH = Path(__file__).parent / "_quarto.yml"
FRONT_MATTER_DELIMITER = "---"
PAGE_KIND = "page"


def _is_page(content: dict) -> bool:
    return content.get("kind") == PAGE_KIND


def _api_pages(quarto_configuration: dict) -> tuple[dict, ...]:
    sections = quarto_configuration["quartodoc"]["sections"]
    return tuple(filter(_is_page, chain.from_iterable(section["contents"] for section in sections)))


def _with_title(page_text: str, title: str) -> str:
    heading_pattern = rf"^#+ {re.escape(title)}[ \t]*\n"
    page_body = re.sub(heading_pattern, "", page_text, count=1)
    return f"{FRONT_MATTER_DELIMITER}\ntitle: {title}\n{FRONT_MATTER_DELIMITER}\n\n{page_body}"


def _add_title(api_directory: Path, page: dict) -> None:
    page_path = api_directory / f"{page['path']}.qmd"
    page_path.write_text(_with_title(page_path.read_text(encoding="utf-8"), page["summary"]["name"]), encoding="utf-8")


def main() -> None:
    quarto_configuration = yaml.safe_load(QUARTO_CONFIGURATION_PATH.read_text(encoding="utf-8"))
    api_directory = Path(__file__).parent / quarto_configuration["quartodoc"]["dir"]
    tuple(map(partial(_add_title, api_directory), _api_pages(quarto_configuration)))


if __name__ == "__main__":
    main()
