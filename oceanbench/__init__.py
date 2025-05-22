# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the package python API to evaluate a challenger.
"""

from . import metrics
from .core.evaluate import evaluate_challenger
from .core.version import __version__

__all__ = [
    "metrics",
    "evaluate_challenger",
    "__version__",
]
