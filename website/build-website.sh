#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

readonly REPORT_INDEX_URL_OPTION="--report-index-url="
readonly VIEWER_DATA_URL_OPTION="--viewer-data-url="
readonly VIEWER_CONFIG_PATH="viewer/viewer-config.json"

SCRIPT_DIR=$( dirname $(readlink -f "${BASH_SOURCE[0]}") )
pushd "$SCRIPT_DIR" > /dev/null

write_viewer_config() {
    printf '{"dataBaseUrl": "%s"}\n' "$1" > "${VIEWER_CONFIG_PATH}"
}

apply_option() {
    case "$1" in
    "${REPORT_INDEX_URL_OPTION}"*) export OCEANBENCH_REPORT_INDEX_URL="${1#"${REPORT_INDEX_URL_OPTION}"}" ;;
    "${VIEWER_DATA_URL_OPTION}"*) write_viewer_config "${1#"${VIEWER_DATA_URL_OPTION}"}" ;;
    *) echo "Unknown option $1" >&2; exit 1 ;;
    esac
}

rm -f "${VIEWER_CONFIG_PATH}"
for option in "$@"; do
    apply_option "${option}"
done

if ! command -v quarto > /dev/null; then
    curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.23/quarto-1.7.23-linux-amd64.deb --output /tmp/quarto.deb
    dpkg -i /tmp/quarto.deb
fi

rm -rf reports scores api documentation/api _site

pip install -r requirements.txt
quarto render --to html

mkdir -p /app/repository
rm -rf /app/repository/reports
cp -r _site/* /app/repository

popd > /dev/null
