# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import sys
from pathlib import Path
from sphinx.builders.html import StandaloneHTMLBuilder

sys.path.insert(0, str(Path("..").resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OceanBench"
copyright = "2025, Mercator Ocean International"
author = "Mercator Ocean International"
version = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.apidoc", "sphinx_copybutton"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"
pygments_dark_style = "monokai"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "https://minio.dive.edito.eu/project-oceanbench/public/logo/favicon-light.png"
html_css_files = ["css/custom.css"]

apidoc_modules = [
    {"path": "../oceanbench", "destination": "source/", "exclude": ["../oceanbench/cli.py"]},
]
html_logo = "https://minio.dive.edito.eu/project-oceanbench/public/logo/oceanbench-logo-light.png"

html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#002a49"},
}

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/github.html",
    ]
}

# -- Options for different image types -------------------------------------------
# https://stackoverflow.com/questions/45969711/sphinx-doc-how-do-i-render-an-animated-gif-when-building-for-html-but-a-png-wh

StandaloneHTMLBuilder.supported_image_types = [
    "image/svg+xml",
    "image/gif",
    "image/png",
    "image/jpeg",
]
