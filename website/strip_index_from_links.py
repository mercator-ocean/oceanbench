# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import re

OUTPUT_DIRECTORY = os.path.abspath(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
CURRENT_DIRECTORY_LINK = "./"
INDEX_LINK_PATTERN = re.compile(r'((?:href=|:)")((?:(?!://)[^"#?])*/)?index\.html(?=["#?])')


def _html_file_paths(directory: str) -> list[str]:
    return [
        os.path.join(root, file_name)
        for root, _, file_names in os.walk(directory)
        for file_name in file_names
        if file_name.endswith(".html")
    ]


def _directory_link(match: re.Match) -> str:
    return match.group(1) + (match.group(2) or CURRENT_DIRECTORY_LINK)


def _strip_index_from_links(html: str) -> str:
    return INDEX_LINK_PATTERN.sub(_directory_link, html)


def _rewrite(file_path: str) -> None:
    with open(file_path, encoding="utf-8") as file:
        html = file.read()
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(_strip_index_from_links(html))


def main() -> None:
    html_file_paths = _html_file_paths(OUTPUT_DIRECTORY)
    for file_path in html_file_paths:
        _rewrite(file_path)
    print(f"Stripped index.html from links in {len(html_file_paths)} pages.")


if __name__ == "__main__":
    main()
