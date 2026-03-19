# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
import pandas
from xarray import Dataset, open_dataset, open_mfdataset, merge, concat
import logging
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.resolution import get_dataset_resolution
import copernicusmarine
from oceanbench.core.climate_forecast_standard_names import StandardVariable
from oceanbench.core.instrumentation import instrumented_operation
from oceanbench.core.remote_http import require_remote_dataset_dimensions
from oceanbench.core.references.reference_stage import staged_reference_dataset, should_stage_reference_locally

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _prepare_reference_week_dataset(
    dataset: Dataset,
    lead_days_count: int,
    operation_name: str,
) -> Dataset:
    week_dataset = require_remote_dataset_dimensions(dataset, [Dimension.TIME.key()], operation_name)
    week_dataset = week_dataset.isel({Dimension.TIME.key(): slice(0, lead_days_count)})
    week_lead_days_count = week_dataset.sizes[Dimension.TIME.key()]
    return week_dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
        {Dimension.LEAD_DAY_INDEX.key(): range(week_lead_days_count)}
    )


def _glo12_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo14/{first_day}.zarr"


def _glo12_1_degree_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo12_1degree_2024_V2/{first_day}.zarr"


def _glo12_analysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:

    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    with instrumented_operation("reference_dataset_load", dataset="glo12", resolution="quarter_degree"):
        if should_stage_reference_locally():
            return staged_reference_dataset(
                dataset_name="glo12",
                resolution="quarter_degree",
                first_day_datetimes=first_day_datetimes,
                lead_days_count=lead_days_count,
                open_week_dataset=lambda first_day_datetime: _prepare_reference_week_dataset(
                    open_dataset(_glo12_1_4_path(first_day_datetime), engine="zarr"),
                    lead_days_count=lead_days_count,
                    operation_name="GLO12 quarter-degree dataset open",
                ),
            )
        reference_dataset = open_mfdataset(
            list(map(_glo12_1_4_path, first_day_datetimes)),
            engine="zarr",
            preprocess=lambda dataset: require_remote_dataset_dimensions(
                dataset,
                [Dimension.TIME.key()],
                "GLO12 quarter-degree dataset open",
            )
            .isel(time=slice(0, lead_days_count))
            .rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()})
            .assign({Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=False,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
        return with_dataset_source(
            reference_dataset,
            kind="reference",
            name="glo12",
            resolution="quarter_degree",
        )


def _glo12_1_12_path(first_day_datetime, days_count: int, target_depths: numpy.ndarray) -> Dataset:
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()

    start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
    end_datetime = (first_day + pandas.Timedelta(days=days_count - 1)).strftime("%Y-%m-%dT00:00:00")

    # Load each variable separately as the dataset_ids are different

    dataset_temperature = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=[StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_salinity = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=[StandardVariable.SEA_WATER_SALINITY.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_current = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=[
            StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
        ],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_sea_surface_height = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=[StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_temperature = dataset_temperature.sel(depth=target_depths, method="nearest")
    dataset_salinity = dataset_salinity.sel(depth=target_depths, method="nearest")
    dataset_current = dataset_current.sel(depth=target_depths, method="nearest")

    dataset = merge([dataset_temperature, dataset_salinity, dataset_current, dataset_sea_surface_height])

    return dataset


def _glo12_analysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    target_depths = challenger_dataset[Dimension.DEPTH.key()].values

    with instrumented_operation("reference_dataset_load", dataset="glo12", resolution="twelfth_degree"):
        if should_stage_reference_locally():
            return staged_reference_dataset(
                dataset_name="glo12",
                resolution="twelfth_degree",
                first_day_datetimes=first_day_datetimes,
                lead_days_count=lead_days_count,
                open_week_dataset=lambda first_day_datetime: _prepare_reference_week_dataset(
                    _glo12_1_12_path(first_day_datetime, days_count=lead_days_count, target_depths=target_depths),
                    lead_days_count=lead_days_count,
                    operation_name="GLO12 twelfth-degree dataset open",
                ),
            )
        datasets = []
        for week_index, first_day_datetime in enumerate(first_day_datetimes, start=1):
            with instrumented_operation(
                "reference_dataset_week_load",
                dataset="glo12",
                resolution="twelfth_degree",
                week_index=week_index,
                weeks_count=len(first_day_datetimes),
                first_day_datetime=str(first_day_datetime),
            ):
                dataset = _glo12_1_12_path(first_day_datetime, days_count=lead_days_count, target_depths=target_depths)
                dataset = dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
                    {Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}
                )
                datasets.append(dataset)

        combined_dataset = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
            {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
        )
        return with_dataset_source(
            combined_dataset,
            kind="reference",
            name="glo12",
            resolution="twelfth_degree",
        )


def _glo12_analysis_dataset_1_degree(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    with instrumented_operation("reference_dataset_load", dataset="glo12", resolution="one_degree"):
        if should_stage_reference_locally():
            return staged_reference_dataset(
                dataset_name="glo12",
                resolution="one_degree",
                first_day_datetimes=first_day_datetimes,
                lead_days_count=lead_days_count,
                open_week_dataset=lambda first_day_datetime: _prepare_reference_week_dataset(
                    open_dataset(_glo12_1_degree_path(first_day_datetime), engine="zarr"),
                    lead_days_count=lead_days_count,
                    operation_name="GLO12 one-degree dataset open",
                ),
            )
        reference_dataset = open_mfdataset(
            list(map(_glo12_1_degree_path, first_day_datetimes)),
            engine="zarr",
            preprocess=lambda dataset: require_remote_dataset_dimensions(
                dataset,
                [Dimension.TIME.key()],
                "GLO12 one-degree dataset open",
            )
            .rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()})
            .assign({Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=False,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
        return with_dataset_source(
            reference_dataset,
            kind="reference",
            name="glo12",
            resolution="one_degree",
        )


def glo12_analysis_dataset(challenger_dataset: Dataset) -> Dataset:

    resolution = get_dataset_resolution(challenger_dataset)

    if resolution == "quarter_degree":
        return _glo12_analysis_dataset_1_4(challenger_dataset)
    elif resolution == "twelfth_degree":
        return _glo12_analysis_dataset_1_12(challenger_dataset)
    elif resolution == "one_degree":
        return _glo12_analysis_dataset_1_degree(challenger_dataset)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
