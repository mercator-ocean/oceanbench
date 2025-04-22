#!/bin/bash
#
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# This init script download a Jupyter Notebook file
# and opens it in Jupyter Lab at startup with oceanbench pre-installed
# Expected parameters : The URL of the file to download

pip install oceanbench

FILE_URL=$1
FILENAME_WITH_MAYBE_QUERY_PARAMS=${FILE_URL##*/}
FILENAME=${FILENAME_WITH_MAYBE_QUERY_PARAMS%\?*}

wget $FILE_URL --output-document $FILENAME
echo "c.LabApp.default_url = '/lab/tree/$FILENAME'" >> /home/$(whoami)/.jupyter/jupyter_server_config.py
