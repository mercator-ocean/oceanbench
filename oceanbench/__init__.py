# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from . import metrics
from .evaluate import evaluate_challenger
from .version import __version__

__all__ = [
    "metrics",
    "evaluate_challenger",
    "__version__",
]
