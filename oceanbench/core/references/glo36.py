# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import logging

import numpy
import xarray
from xarray import Dataset

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.glo36v1 import (
    GLO36V1_FIRST_DAY_DATETIMES,
    GLO36V1_LEAD_DAYS_COUNT,
    Glo36V1ReferenceDataUnavailableError,
    glo36v1_dataset_path,
    is_super_resolution_dataset,
    matching_glo36v1_first_day_datetimes,
    prepare_glo36v1_week_dataset,
)
from oceanbench.core.remote_http import with_remote_http_retries
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset

logger = logging.getLogger(__name__)

_GLO36V1_REFERENCE_DATASET_CACHE: dict[int, Dataset] = {}


def _glo36v1_reference_lead_days_count(challenger_dataset: Dataset) -> int:
    challenger_lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return min(challenger_lead_days_count, GLO36V1_LEAD_DAYS_COUNT)


def _open_glo36v1_week_dataset(first_day_datetime: numpy.datetime64, lead_days_count: int) -> Dataset:
    return prepare_glo36v1_week_dataset(
        xarray.open_dataset(glo36v1_dataset_path(first_day_datetime), engine="zarr"),
        lead_days_count=lead_days_count,
        operation_name="GLO36V1 reference dataset open",
    )


def _remote_glo36v1_reference_dataset(
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
) -> Dataset:
    return xarray.open_mfdataset(
        [glo36v1_dataset_path(first_day_datetime) for first_day_datetime in first_day_datetimes],
        engine="zarr",
        preprocess=lambda dataset: prepare_glo36v1_week_dataset(
            dataset,
            lead_days_count=lead_days_count,
            operation_name="GLO36V1 reference dataset open",
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=False,
    ).assign_coords({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})


def _glo36v1_reference_dataset(
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
) -> Dataset:
    return maybe_stage_weekly_dataset(
        stage_key="references",
        dataset_kind="reference",
        dataset_name="glo36v1",
        first_day_datetimes=first_day_datetimes,
        lead_days_count=lead_days_count,
        open_week_dataset=lambda first_day_datetime: _open_glo36v1_week_dataset(
            first_day_datetime,
            lead_days_count,
        ),
        open_remote_dataset=lambda: _remote_glo36v1_reference_dataset(
            first_day_datetimes,
            lead_days_count,
        ),
        resolution="super_resolution",
    )


def glo36v1_reference() -> Dataset:
    """
    Open the full GLO36V1 reference dataset available on EDITO.
    """
    first_day_datetimes = numpy.array(GLO36V1_FIRST_DAY_DATETIMES)
    return with_remote_http_retries(
        "GLO36V1 full reference dataset open",
        lambda: _glo36v1_reference_dataset(first_day_datetimes, GLO36V1_LEAD_DAYS_COUNT),
    )


def glo36v1_reference_dataset(challenger_dataset: Dataset) -> Dataset:
    if not is_super_resolution_dataset(challenger_dataset):
        raise Glo36V1ReferenceDataUnavailableError(
            "GLO36V1 reference scores are only computed for super-resolution challenger datasets."
        )

    cache_key = id(challenger_dataset)
    cached_dataset = _GLO36V1_REFERENCE_DATASET_CACHE.get(cache_key)
    if cached_dataset is not None:
        return cached_dataset

    first_day_datetimes = matching_glo36v1_first_day_datetimes(challenger_dataset)
    lead_days_count = _glo36v1_reference_lead_days_count(challenger_dataset)
    reference_dataset = with_remote_http_retries(
        "GLO36V1 reference dataset open",
        lambda: _glo36v1_reference_dataset(first_day_datetimes, lead_days_count),
    )
    _GLO36V1_REFERENCE_DATASET_CACHE[cache_key] = reference_dataset

    unmatched_count = challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()] - len(first_day_datetimes)
    if unmatched_count > 0:
        logger.warning(
            "GLO36V1 reference uses %s matching first_day_datetime values and skips %s unavailable values.",
            len(first_day_datetimes),
            unmatched_count,
        )

    return reference_dataset
