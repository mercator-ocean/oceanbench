#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Make the served viewer inside the Quarto site (_site/viewer) reflect the source
# (viewer/) live, with no manual copying.
#
# Quarto only copies `resources:` (the viewer app) into _site during a full render,
# so `quarto preview` never picks up viewer edits on its own. This script replaces
# _site/viewer with a symlink to ../viewer, so every edit to a viewer file is served
# immediately — no re-copy, no "is this cached?" guesswork. Run it once per checkout;
# it is idempotent.
#
# _site is a build output (git-ignored); `quarto render` wipes _site and regenerates
# real file copies for production, so this symlink is a dev-only convenience.

set -euo pipefail

website_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )
site_viewer="$website_dir/_site/viewer"

mkdir -p "$website_dir/_site"

if [ -L "$site_viewer" ]; then
  echo "Already live: _site/viewer -> $(readlink "$site_viewer")"
  exit 0
fi

if [ -e "$site_viewer" ]; then
  rm -rf "$site_viewer"
fi

ln -s ../viewer "$site_viewer"
echo "Linked _site/viewer -> ../viewer (viewer edits are now served live)."
