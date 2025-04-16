#!/usr/bin/bash
SCRIPT_DIR=$( dirname $(readlink -f "${BASH_SOURCE[0]}") )
pushd $SCRIPT_DIR > /dev/null

which quarto
returnValue=$?
if [ $returnValue -ne 0 ]; then
    curl -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.7.23/quarto-1.7.23-linux-amd64.deb --output /tmp/quarto.deb
    dpkg -i /tmp/quarto.deb
fi

pip install -r requirements.txt
quarto render --to html

cp -r _site/* /app/repository

popd > /dev/null
