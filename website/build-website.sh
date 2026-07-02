#!/usr/bin/env bash
set -euo pipefail

# go to repo root
cd /app/repository

# install deps
apt-get update && apt-get install -y build-essential gcc g++ make

# install quarto if missing
if ! command -v quarto > /dev/null; then
  curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.23/quarto-1.7.23-linux-amd64.deb --output /tmp/quarto.deb
  dpkg -i /tmp/quarto.deb
fi

# clean
rm -rf website/_site

# install python deps
pip install -r website/requirements.txt

# build
quarto render website --to html

# copy to source
mkdir -p /app/repository/$WEBSITE_SOURCE_PATH
cp -r website/_site/* /app/repository/$WEBSITE_SOURCE_PATH/

# debug
ls -R /app/repository/$WEBSITE_SOURCE_PATH