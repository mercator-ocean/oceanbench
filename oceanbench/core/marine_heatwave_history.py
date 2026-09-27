# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import numpy
import xarray
from zarr.errors import GroupNotFoundError

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset


def marine_heatwave_history_is_unavailable(error: Exception) -> bool:
    if not isinstance(error, (FileNotFoundError, GroupNotFoundError)):
        return False

    seen_exceptions = set()
    current_exception: Exception | None = error
    while current_exception is not None and id(current_exception) not in seen_exceptions:
        seen_exceptions.add(id(current_exception))
        if not isinstance(current_exception, (FileNotFoundError, GroupNotFoundError)) and not (
            isinstance(current_exception, KeyError) and current_exception.args == (".zmetadata",)
        ):
            return False
        current_exception = current_exception.__cause__ or current_exception.__context__
    return True


def load_marine_heatwave_analysis_history(
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


def glo12_analysis_history_dataset(
    challenger_dataset: xarray.Dataset,
    history_days: int = 7,
) -> xarray.Dataset:
    return load_marine_heatwave_analysis_history(
        challenger_dataset,
        glo12_analysis_dataset,
        history_days,
    )


def glorys_reanalysis_history_dataset(
    challenger_dataset: xarray.Dataset,
    history_days: int = 7,
) -> xarray.Dataset:
    return load_marine_heatwave_analysis_history(
        challenger_dataset,
        glorys_reanalysis_dataset,
        history_days,
    )
