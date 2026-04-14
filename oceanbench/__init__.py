# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the package python API to evaluate a challenger.
"""

from . import metrics
from . import datasets
from . import eddies
from . import psd
from . import regions
from . import visualization
from .core.evaluate import evaluate_challenger
from .core.version import __version__

__all__ = [
    "metrics",
    "datasets",
    "eddies",
    "psd",
    "regions",
    "visualization",
    "evaluate_challenger",
    "__version__",
]
