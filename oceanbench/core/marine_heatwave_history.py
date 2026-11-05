# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension

MARINE_HEATWAVE_HISTORY_DAYS = 7


def _load_marine_heatwave_analysis_history(
    challenger_dataset: xarray.Dataset,
    analysis_loader: Callable[[xarray.Dataset], xarray.Dataset],
    history_days: int,
) -> xarray.Dataset:
    lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
    first_day_dimension = Dimension.FIRST_DAY_DATETIME.key()

    forecast_first_days = challenger_dataset[first_day_dimension]
    history_request = (
        challenger_dataset.isel({lead_day_dimension: slice(0, 1)})
        .reindex({lead_day_dimension: numpy.arange(history_days)})
        .assign_coords({first_day_dimension: forecast_first_days - numpy.timedelta64(history_days, "D")})
    )
    history_dataset = analysis_loader(history_request)

    return history_dataset.assign_coords(
        {
            first_day_dimension: forecast_first_days.values,
            lead_day_dimension: numpy.arange(-history_days, 0),
        }
    )


def marine_heatwave_history_dataset(
    challenger_dataset: xarray.Dataset,
    analysis_loader: Callable[[xarray.Dataset], xarray.Dataset],
) -> xarray.Dataset | None:
    first_day_dimension = Dimension.FIRST_DAY_DATETIME.key()
    try:
        available_history = _load_marine_heatwave_analysis_history(
            challenger_dataset,
            analysis_loader,
            MARINE_HEATWAVE_HISTORY_DAYS,
        )
    except OSError:
        if challenger_dataset.sizes[first_day_dimension] < 2:
            return None

        available_history = _load_marine_heatwave_analysis_history(
            challenger_dataset.isel({first_day_dimension: slice(1, None)}),
            analysis_loader,
            MARINE_HEATWAVE_HISTORY_DAYS,
        )
    return available_history.reindex({first_day_dimension: challenger_dataset[first_day_dimension]})
