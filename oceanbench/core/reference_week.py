# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.remote_http import require_remote_dataset_dimensions


def prepare_reference_week_dataset(
    dataset: xarray.Dataset,
    lead_days_count: int,
    operation_name: str,
) -> xarray.Dataset:
    week_dataset = require_remote_dataset_dimensions(dataset, [Dimension.TIME.key()], operation_name)
    week_dataset = week_dataset.isel({Dimension.TIME.key(): slice(0, lead_days_count)})
    week_lead_days_count = week_dataset.sizes[Dimension.TIME.key()]
    return week_dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
        {Dimension.LEAD_DAY_INDEX.key(): range(week_lead_days_count)}
    )
