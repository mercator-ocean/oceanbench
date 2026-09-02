# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The scoring runner: long-format records, per-start emission, parity harness.

Submodules:

- ``records``: convert legacy per-metric dataframes into long-format records.
- ``run``: compute the metric functions directly and write ``scores.parquet``.
- ``parity``: compare a runner parquet against the published golden.
"""
