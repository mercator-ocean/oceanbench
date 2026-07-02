#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( dirname $(readlink -f "${BASH_SOURCE[0]}") )
pushd "$SCRIPT_DIR" > /dev/null

# install system deps (pandas, matplotlib, etc.)
apt-get update && apt-get install -y build-essential gcc g++ make

# install quarto if missing
if ! command -v quarto > /dev/null; then
  curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.23/quarto-1.7.23-linux-amd64.deb --output /tmp/quarto.deb
  dpkg -i /tmp/quarto.deb
fi

# clean previous build
rm -rf reports _site

# install python deps
pip install -r requirements.txt

# build site
quarto render --to html

# copy to the expected Static-pages source path
mkdir -p /app/repository/$WEBSITE_SOURCE_PATH
cp -r _site/* /app/repository/$WEBSITE_SOURCE_PATH/

# debug (important)
ls -R /app/repository/$WEBSITE_SOURCE_PATH

popd > /dev/null