# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from . import metrics
from .evaluate import generate_notebook_to_evaluate
from .version import __version__

__all__ = [
    "metrics",
    "generate_notebook_to_evaluate",
    "__version__",
]
