# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the package python API to evaluate a challenger.
"""

from . import reference
from . import input
from . import challenger

__all__ = [
    "reference",
    "input",
    "challenger",
]
