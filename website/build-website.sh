#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

SCRIPT_DIR=$( dirname $(readlink -f "${BASH_SOURCE[0]}") )
pushd "$SCRIPT_DIR" > /dev/null

if ! command -v quarto > /dev/null; then
    curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.23/quarto-1.7.23-linux-amd64.deb --output /tmp/quarto.deb
    dpkg -i /tmp/quarto.deb
fi

rm -rf reports _site

pip install -r requirements.txt
python download_reports.py
quarto render --to html

mkdir -p /app/repository
if command -v rsync > /dev/null; then
    rsync -a --delete _site/ /app/repository/
else
    find /app/repository -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    cp -R _site/. /app/repository/
fi

popd > /dev/null
