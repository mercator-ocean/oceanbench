#!/usr/bin/env sh

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -eu

REPOSITORY_DIRECTORY="$(cd "$(dirname "$0")/../.." && pwd)"

cd "${REPOSITORY_DIRECTORY}"
poetry install --only main --no-interaction
poetry run python edito/catalog/update_catalog.py "$@"
