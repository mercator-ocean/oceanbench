# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
SWOT validation : RMSD between the OceanBench challenger models (glonet,
glo12, xihe, wenhai, langya, hclimrep) and the SWOT L3 observations
(oceanbench.core.references.swot_observations_zarr).

Entry point : run_swot_validation(), called by "oceanbench swot-validate".

The RMSD Class IV table is computed for ALL the weekly runs of 2024 (52 runs,
Wednesdays 2024-01-03 -> 2024-12-25) on the shared Class IV population
(ocean_mask, identical for every challenger).

The pointwise outputs (error NetCDF + one error map per calendar day)
are produced ONLY for the requested period : the runs covering the period are
loaded separately, so the map mode never downloads the whole year of SWOT observations.

Logging is left to the caller (the CLI configures it) and every exception
propagates : run_swot_validation raises if a model fails, so the command exits
with a non-zero code. A model whose pointwise step fails still contributes its
RMSD table to the CSV.
"""


# ==============================================================================
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy
import pandas
import xarray

from oceanbench.core.challenger_datasets import (
    glonet,
    _glonet_dataset_path,
    glo12,
    _glo12_dataset_path,
    hclimrep,
    _hclimrep_dataset_path,
    langya,
    _langya_dataset_path,
    wenhai,
    _wenhai_dataset_path,
    xihe,
    _xihe_dataset_path,
)
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.classIV_support import (
    class4_observations_in_shared_population,
    class4_population_layers,
    clear_class4_observations_cache,
    create_class4_observations_dataframe,
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.references.swot_observations_zarr import (
    FIRST_DAY_DATETIME_KEY,
    LATITUDE_KEY,
    LEAD_DAY_INDEX_KEY,
    LONGITUDE_KEY,
    SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY,
    SWOT_STORAGE_OPTIONS,
    TIME_KEY,
    swot_observations,
)
from oceanbench.core.regions import resolve_region, subset_dataset_to_region
from oceanbench.core.remote_http import RetryingHTTPFileSystem, remote_store_exists

# ==============================================================================


logger = logging.getLogger(__name__)

MODELS = {
    "glonet": glonet,
    "glo12": glo12,
    "xihe": xihe,
    "wenhai": wenhai,
    "langya": langya,
    "hclimrep": hclimrep,
}

# Function building the zarr url of one run, by model : the availability check
# probes exactly the url that will be requested at loading time.
MODEL_RUN_PATH_BUILDERS = {
    "glonet": _glonet_dataset_path,
    "glo12": _glo12_dataset_path,
    "xihe": _xihe_dataset_path,
    "wenhai": _wenhai_dataset_path,
    "langya": _langya_dataset_path,
    "hclimrep": _hclimrep_dataset_path,
}

VARIABLES = [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID]

DEFAULT_REGION = "global"
DEFAULT_SUBSAMPLE = 10
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_START_DATE = "2024-01-03"
DEFAULT_RUNS_YEAR = 2024
DEFAULT_RUNS_END_DATE = "2024-12-25"


SWOT_RUN_LOOKBACK_DAYS = 10

_RUN_CHECK_FILE_SYSTEM = RetryingHTTPFileSystem(**SWOT_STORAGE_OPTIONS)


# ==============================================================================
# HELPERS


def _run_zarr_exists(model_name: str, first_day: datetime) -> bool:
    # True if the zarr of the run exists on the bucket. A network error is NOT
    # an absent run : remote_store_exists lets it propagate, after the retries
    # of RetryingHTTPFileSystem, so a connectivity problem is never scored as
    # a missing run.
    run_url = MODEL_RUN_PATH_BUILDERS[model_name](first_day)
    return remote_store_exists(_RUN_CHECK_FILE_SYSTEM, run_url)


def _nearest_available_run_dates(model_name: str, first_day: datetime, maximum_weeks: int = 10) -> list:
    # Closest existing run dates around "first_day", for the error message :
    # probe plus or minus one week, then two, ... (weekly cadence), stop as
    # soon as a week gives results.
    for offset_weeks in range(1, maximum_weeks + 1):
        found = [
            candidate
            for candidate in (
                first_day - pandas.Timedelta(weeks=offset_weeks),
                first_day + pandas.Timedelta(weeks=offset_weeks),
            )
            if _run_zarr_exists(model_name, candidate)
        ]
        if found:
            return sorted(found)
    return []


def _run_dates_for_error_dates(start_error_date: str, end_error_date: str | None) -> list:
    # Weekly run dates whose window intersects the requested error period,
    # within the validated runs of the year. For a day, the most recent run
    # started at or before the day (shortest lead day) is within the last 7
    # days, but OLDER runs can still match the day when the model has more
    # than 7 lead days (the pointwise output keeps every match) : the lookback
    # is therefore SWOT_RUN_LOOKBACK_DAYS, not 7. A run after the end of the
    # period never serves any day of it.
    start = pandas.Timestamp(start_error_date).normalize()
    end = pandas.Timestamp(end_error_date or start_error_date).normalize()
    if end < start:
        raise ValueError(f"end_error_date ({end:%Y-%m-%d}) is BEFORE start_error_date ({start:%Y-%m-%d})")
    weekly_runs = pandas.to_datetime(generate_dates(DEFAULT_START_DATE, DEFAULT_RUNS_END_DATE, 7))
    return [
        run_date for run_date in weekly_runs if start - pandas.Timedelta(days=SWOT_RUN_LOOKBACK_DAYS) < run_date <= end
    ]


def _resolve_output_dir(output_dir: str) -> str:
    return str(Path(output_dir).resolve())


def _find_lead_day_columns(dataframe: pandas.DataFrame) -> list:
    return [column for column in dataframe.columns if isinstance(column, str) and column.startswith("Lead day")]


# ==============================================================================
def _compute_pointwise_error(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    model_name: str = "",
) -> xarray.Dataset:
    # Pointwise error (model - observation) at every observation point, on the
    # same shared Class IV population as the RMSD table (ocean_mask). One line
    # per (observation, run) match : with the weekly cadence an observation
    # can match two consecutive runs, by design, like in OceanBench.
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[LEAD_DAY_INDEX_KEY]
    if lead_days_count < 1:
        raise ValueError(f"{model_name or 'the challenger'} dataset has no lead day (lead_days_count < 1).")

    observations_dataframe = create_class4_observations_dataframe(
        reference_dataset,
        SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY,
        SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY,
        lead_days_count,
    )
    observations_dataframe = observations_dataframe.dropna(subset=["observation_value"]).reset_index(drop=True)
    if observations_dataframe.empty:
        raise ValueError("No usable SWOT observation")
    logger.info(
        f"  Class IV dataframe ({model_name}) : {len(observations_dataframe):,} rows, "
        f"{observations_dataframe.memory_usage(deep=True).sum() / 1e9:.2f} GB"
    )

    # Same population as the RMSD table : keep only the observations over ocean
    # in the OceanBench mask, shared by every challenger whatever its grid.
    population_layers = class4_population_layers(ocean_mask())
    observations_dataframe = class4_observations_in_shared_population(observations_dataframe, population_layers)
    if observations_dataframe.empty:
        raise ValueError("No SWOT observation in the shared Class IV population")

    model_sla = prepare_class4_model_variable(
        challenger[SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY], SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY
    )
    model_values = interpolate_class4_model_to_observations(model_sla, observations_dataframe)

    error = model_values - observations_dataframe["observation_value"].to_numpy(dtype="float64")

    finite_error = error[numpy.isfinite(error)]
    if finite_error.size:
        logger.info(
            f"  Mean signed error ({model_name}) : {numpy.mean(finite_error):+.4f} m "
            f"({finite_error.size:,} obs) — a uniform offset would indicate a reference "
            "mismatch (MDT/MSS) between the model SSH->SLA conversion and the SWOT L3 variable."
        )

    lead_day = observations_dataframe["lead_day"].to_numpy(dtype="int64")
    latitude = observations_dataframe[LATITUDE_KEY].to_numpy(dtype="float64")
    longitude = observations_dataframe[LONGITUDE_KEY].to_numpy(dtype="float64")
    observation_datetimes = pandas.DatetimeIndex(pandas.to_datetime(observations_dataframe[TIME_KEY].to_numpy()))
    first_days_observation = pandas.DatetimeIndex(pandas.to_datetime(observations_dataframe["first_day"].to_numpy()))

    valid = numpy.isfinite(error) & (lead_day >= 0) & (lead_day < lead_days_count)
    total_count, valid_count = error.size, int(valid.sum())
    logger.info(f"  Pointwise errors ({model_name}) : {valid_count:,}/{total_count:,} usable obs")
    if valid_count < 0.9 * total_count:
        logger.warning(
            f"  {total_count - valid_count:,} obs without value in the model (NaN) : "
            "ground / outside the grid, or possible longitude convention mismatch "
            "between the model and the SWOT observations."
        )
    if valid_count == 0:
        raise ValueError("Every interpolated value is NaN : nothing to plot")

    error = error[valid].astype("float32")
    lead_day, latitude, longitude = lead_day[valid], latitude[valid], longitude[valid]
    observation_datetimes, first_days_observation = (
        observation_datetimes[valid],
        first_days_observation[valid],
    )

    first_days = pandas.to_datetime(challenger[FIRST_DAY_DATETIME_KEY].values)

    return xarray.Dataset(
        data_vars={
            "error": (
                ("observations",),
                error,
                {"long_name": "model SLA - SWOT SLA (signed)", "units": "m"},
            ),
        },
        coords={
            TIME_KEY: (("observations",), observation_datetimes.to_numpy()),
            LATITUDE_KEY: (("observations",), latitude),
            LONGITUDE_KEY: (("observations",), longitude),
            FIRST_DAY_DATETIME_KEY: (("observations",), first_days_observation.to_numpy()),
            "lead_day": (("observations",), lead_day),
        },
        attrs={
            "title": "Pointwise error of model SLA against SWOT L3 SLA, " "one line per (observation, run) match",
            "model": model_name,
            "first_day_datetimes": ", ".join(first_days.strftime("%Y-%m-%d")),
            "n_observations": int(valid_count),
            "lead_day_convention": "0-based lead day index, like the model (tables and maps use 1-based)",
            "population": (
                "all (observation, run) matches intersecting the requested period : a day of a "
                "multi-run period carries several lead days, unlike the maps which keep only "
                "the most recent run per day"
            ),
        },
    )


def _save_error_dataset(error_dataset: xarray.Dataset, output_path: str) -> None:
    encoding = {name: {"zlib": True, "complevel": 4} for name in error_dataset.data_vars}
    error_dataset.to_netcdf(output_path, encoding=encoding)


def _pointwise_error_for_period(
    model_name: str,
    model_loader: Callable[..., xarray.Dataset],
    region: str,
    subsample: int,
    start_error_date: str,
    end_error_date: str | None,
    first_day_datetimes: list | None = None,
) -> xarray.Dataset:
    # Pointwise errors for the requested period only : the few weekly runs
    # covering the period are loaded separately, so the map mode never
    # downloads the whole year of SWOT observations. first_day_datetimes :
    # availability-filtered run dates passed by run_swot_validation (a
    # missing run zarr is then skipped, not a crash at the model load) ;
    # None (standalone calls) : every weekly run matching the period.
    start = pandas.Timestamp(start_error_date).normalize()
    end = pandas.Timestamp(end_error_date or start_error_date).normalize()
    if first_day_datetimes is None:
        error_run_dates = _run_dates_for_error_dates(start_error_date, end_error_date)
    else:
        error_run_dates = list(first_day_datetimes)
    if not error_run_dates:
        raise ValueError(
            f"The period [{start:%Y-%m-%d} .. {end:%Y-%m-%d}] is not covered by any "
            f"weekly run of {DEFAULT_RUNS_YEAR} ({DEFAULT_START_DATE} -> {DEFAULT_RUNS_END_DATE})."
        )
    challenger_dataset = model_loader(first_day_datetimes=error_run_dates)
    lead_days_count = challenger_dataset.sizes[LEAD_DAY_INDEX_KEY]
    window_end = error_run_dates[-1] + pandas.Timedelta(days=lead_days_count - 1)
    if end > window_end:
        raise ValueError(
            f"The requested period ends on {end:%Y-%m-%d} but the last {DEFAULT_RUNS_YEAR} run "
            f"({error_run_dates[-1]:%Y-%m-%d}) only covers up to {window_end:%Y-%m-%d}."
        )
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    try:
        # Only the acquisition days of the requested period are downloaded
        # (one day for a single date, instead of the full run windows).
        reference_dataset = swot_observations(
            challenger_dataset,
            subsample=subsample,
            observation_period=(start, end),
        )
        reference_dataset = subset_dataset_to_region(reference_dataset, region)
        # Defense in depth : swot_observations already restricts the days, this
        # guarantees the NetCDF never contains a point outside the request.
        observation_times = pandas.to_datetime(reference_dataset[TIME_KEY].values).normalize()
        in_requested_period = (observation_times >= start) & (observation_times <= end)
        reference_dataset = reference_dataset.isel(observations=numpy.flatnonzero(in_requested_period))
        if reference_dataset.sizes["observations"] == 0:
            raise ValueError(f"No SWOT observation between {start:%Y-%m-%d} and {end:%Y-%m-%d}.")
        return _compute_pointwise_error(challenger_dataset, reference_dataset, model_name=model_name)
    finally:
        clear_class4_observations_cache()


def _plot_error_maps_by_date(
    error_dataset: xarray.Dataset,
    output_dir: str,
    model_name: str,
    start_error_date: str,
    end_error_date: str,
    vmax: float | None = None,
) -> None:
    # One map per calendar day of the requested period. The lead day of the
    # date (base 1, like the OceanBench tables) is in the title. vmax fixes
    # the color interval so maps of different models are comparable; None
    # (default) : 99th percentile of the period (fallback 0.25 m).
    # NOTE : cfeature.LAND downloads the Natural Earth shapefiles on first use
    # (one connection needed, then they are cached locally).

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    start = pandas.Timestamp(start_error_date).normalize()
    end = pandas.Timestamp(end_error_date).normalize()

    if vmax is not None and vmax <= 0:
        raise ValueError(f"vmax must be positive, got {vmax}")

    absolute_error = numpy.abs(error_dataset["error"].values)
    latitude = error_dataset[LATITUDE_KEY].values
    longitude = error_dataset[LONGITUDE_KEY].values
    # Bounded region (e.g. a lat/lon box from --region-file) : zoom on the
    # data instead of showing the whole globe ; a global region stays global.
    latitude_bounds = (float(numpy.nanmin(latitude)), float(numpy.nanmax(latitude)))
    longitude_bounds = (float(numpy.nanmin(longitude)), float(numpy.nanmax(longitude)))
    bounded_region = (latitude_bounds[1] - latitude_bounds[0] < 90.0) or (
        longitude_bounds[1] - longitude_bounds[0] < 180.0
    )
    times = pandas.DatetimeIndex(error_dataset[TIME_KEY].values)
    first_days = pandas.DatetimeIndex(error_dataset[FIRST_DAY_DATETIME_KEY].values)

    # One forecast per day : keep only the observations tagged with the most
    # recent run started at or before the day (shortest lead day). With the
    # weekly cadence, every observation of a day necessarily carries that run
    # tag — robust to identical timestamps, fully vectorized.
    observation_days = times.normalize().to_numpy()
    run_dates = numpy.sort(first_days.unique().to_numpy())
    run_of_day = run_dates[numpy.searchsorted(run_dates, observation_days, side="right") - 1]
    if (run_of_day > observation_days).any():
        raise ValueError("Observation day before the first run : inconsistent dataset.")
    selected_indices = numpy.flatnonzero(first_days.to_numpy() == run_of_day)

    day_of = times[selected_indices].normalize()
    in_period = numpy.asarray((day_of >= start) & (day_of <= end))
    selected_indices = selected_indices[in_period]
    if selected_indices.size == 0:
        logger.warning(f"No observation between {start:%Y-%m-%d} and {end:%Y-%m-%d} : no map.")
        return
    day_of = day_of[in_period]
    selected_absolute_error, selected_latitude, selected_longitude = (
        absolute_error[selected_indices],
        latitude[selected_indices],
        longitude[selected_indices],
    )
    selected_lead_days = (
        (times[selected_indices].normalize().to_numpy() - first_days[selected_indices].to_numpy())
        / numpy.timedelta64(1, "D")
    ).astype("int64")

    # Same color interval on every map, for comparison : fixed vmax or 99th percentile.
    finite = numpy.isfinite(selected_absolute_error)
    if vmax is not None:
        color_max = float(vmax)
    elif finite.any():
        color_max = float(numpy.percentile(selected_absolute_error[finite], 99))
    else:
        color_max = 0.25
    logger.info(f"Error maps by date for {model_name} : {day_of.unique().size} day(s) -> {output_dir}")

    for day in pandas.date_range(start, end, freq="D"):
        day_mask = (day_of == day) & numpy.isfinite(selected_absolute_error)
        points_count = int(day_mask.sum())
        if points_count == 0:
            logger.warning(f"  {day:%Y-%m-%d} : no observation, map ignored")
            continue
        lead_day_values = numpy.unique(selected_lead_days[day_mask])
        lead_text = "Lead day " + ", ".join(str(int(value) + 1) for value in lead_day_values)

        figure, axis = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
        axis.coastlines(color="black", linewidth=0.5)
        axis.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=0)
        if bounded_region:
            axis.set_extent(
                [
                    longitude_bounds[0] - 2.0,
                    longitude_bounds[1] + 2.0,
                    latitude_bounds[0] - 2.0,
                    latitude_bounds[1] + 2.0,
                ],
                crs=ccrs.PlateCarree(),
            )
        else:
            axis.set_global()
        scatter = axis.scatter(
            selected_longitude[day_mask],
            selected_latitude[day_mask],
            c=selected_absolute_error[day_mask],
            s=1,
            linewidths=0,
            cmap="gist_heat_r",
            vmin=0,
            vmax=color_max,
            transform=ccrs.PlateCarree(),
        )
        figure.colorbar(
            scatter,
            ax=axis,
            orientation="vertical",
            pad=0.06,
            shrink=0.5,
            aspect=30,
            label="|error| (m)",
        )
        axis.set_title(
            f"|error| SLA — {model_name} vs SWOT — {day:%Y-%m-%d} ({lead_text}) — "
            f"{points_count:,} observation points",
            fontsize=11,
        )
        output_path = os.path.join(output_dir, f"error_map_{model_name}_{day:%Y%m%d}.png")
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        logger.info(f"  Map saved: {output_path} ({points_count:,} points, {lead_text})")


def _validate_model(
    model_name: str,
    model_loader: Callable[..., xarray.Dataset],
    first_day_datetimes: list,
    region: str = DEFAULT_REGION,
    subsample: int = DEFAULT_SUBSAMPLE,
) -> pandas.DataFrame:
    # RMSD Class IV table of one model on the given runs, on the shared Class
    # IV population. Raises on an empty result (e.g. an all-land region).
    try:
        challenger_dataset = model_loader(first_day_datetimes=first_day_datetimes)
        challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
        if challenger_dataset.sizes[LEAD_DAY_INDEX_KEY] < 1:
            raise ValueError(f"{model_name} dataset has no lead day (lead_days_count < 1).")
        logger.info(f"Model data loaded: {dict(challenger_dataset.sizes)}")

        reference_dataset = swot_observations(challenger_dataset, subsample=subsample)
        reference_dataset = subset_dataset_to_region(reference_dataset, region)
        logger.info(f"SWOT observations loaded: {reference_dataset.sizes['observations']:,} points")

        metrics_dataframe = rmsd_class4_validation(
            challenger_dataset=challenger_dataset,
            reference_dataset=reference_dataset,
            ocean_mask=ocean_mask(),
            variables=VARIABLES,
        )
        if metrics_dataframe.empty:
            raise ValueError("Empty Class IV result : no observation scored for this model.")
        metrics_dataframe.insert(0, "model", model_name)
        metrics_dataframe.insert(1, "region", resolve_region(region).id)
        metrics_dataframe.insert(2, "subsample", subsample)
        metrics_dataframe.insert(3, "runs_used", len(first_day_datetimes))
        metrics_dataframe.insert(
            4,
            "swot_days_used",
            int(pandas.to_datetime(reference_dataset[TIME_KEY].values).normalize().unique().size),
        )
        logger.info(f"RMSD table computed for {model_name} :\n{metrics_dataframe.head()}")
        return metrics_dataframe
    finally:
        clear_class4_observations_cache()


def run_swot_validation(
    models: list[str],
    region: str = DEFAULT_REGION,
    subsample: int = DEFAULT_SUBSAMPLE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output: str = "rmsd_swot_results.csv",
    start_error_date: str | None = None,
    end_error_date: str | None = None,
    plot_maps: bool = False,
    maps_only: bool = False,
    vmax: float | None = None,
) -> pandas.DataFrame:
    """Run the whole SWOT validation (called by "oceanbench swot-validate").

    The RMSD Class IV table is computed for all the weekly runs of 2024
    (52 runs, Wednesdays 2024-01-03 -> 2024-12-25) ; the pointwise outputs
    (error NetCDF of the requested period + one map per calendar day if
    plot_maps) are produced only for the requested period. maps_only skips the
    RMSD table and produces the pointwise outputs alone. All the outputs (CSV,
    error NetCDF, maps) are written in output_dir, resolved against the
    current directory. Returns the concatenated RMSD results (empty DataFrame
    in maps_only mode).

    Raises RuntimeError at the end of the run if at least one model failed
    (the CSV of the successful models is still written) — the CLI turns this
    into a non-zero exit code."""

    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Options : models={models}, region={region}, subsample={subsample}, "
        f"start_error_date={start_error_date}, end_error_date={end_error_date}, "
        f"plot_maps={plot_maps}, maps_only={maps_only}, vmax={vmax}"
    )

    if subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}.")
    if not models:
        raise ValueError("models must contain at least one model name.")
    if plot_maps and not start_error_date:
        raise ValueError("plot_maps requires start_error_date.")
    if maps_only and not start_error_date:
        raise ValueError("maps_only requires start_error_date.")
    if plot_maps:
        try:
            import cartopy  # noqa: F401
        except ImportError as error:
            raise RuntimeError("plot_maps requires cartopy : install oceanbench[maps] (or cartopy).") from error

    first_day_datetimes = pandas.to_datetime(generate_dates(DEFAULT_START_DATE, DEFAULT_RUNS_END_DATE, 7))
    logger.info(
        f"Runs validated ({DEFAULT_RUNS_YEAR}) : {len(first_day_datetimes)} runs, "
        f"{first_day_datetimes[0]:%Y-%m-%d} -> {first_day_datetimes[-1]:%Y-%m-%d}"
    )

    all_results = []
    failed_models = []

    for model_name in models:
        if model_name not in MODELS:
            logger.error(f"Unknown model: {model_name}")
            # An unknown model is a failure, not a silent skip : the command
            # must exit with a non-zero code even if the other models succeed.
            failed_models.append(model_name)
            continue

        if maps_only:
            candidate_run_dates = _run_dates_for_error_dates(start_error_date, end_error_date)
            if not candidate_run_dates:
                raise ValueError(
                    f"The requested period [{start_error_date} .. {end_error_date or start_error_date}] "
                    f"is not covered by any weekly run of {DEFAULT_RUNS_YEAR} "
                    f"({DEFAULT_START_DATE} -> {DEFAULT_RUNS_END_DATE})."
                )
        else:
            # The availability check is unconditional : the results never depend
            # on whether the pointwise outputs are requested.
            candidate_run_dates = first_day_datetimes
        model_run_dates = [run_date for run_date in candidate_run_dates if _run_zarr_exists(model_name, run_date)]
        if not model_run_dates:
            suggestions = _nearest_available_run_dates(model_name, candidate_run_dates[0])
            suggestion_text = ", ".join(f"{run_date:%Y-%m-%d}" for run_date in suggestions) or "none within ±10 weeks"
            logger.error(
                f"  {model_name} : no run available for the requested period "
                f"(closest to {candidate_run_dates[0]:%Y-%m-%d} : {suggestion_text}). Model skipped."
            )
            # A model without run is a failure, not a silent skip : the command
            # must exit with a non-zero code even if the other models succeed.
            failed_models.append(model_name)
            continue
        if not maps_only and len(model_run_dates) < len(first_day_datetimes):
            logger.warning(
                f"  {model_name} : partial coverage of {DEFAULT_RUNS_YEAR} — "
                f"{len(model_run_dates)}/{len(first_day_datetimes)} runs "
                f"({model_run_dates[0]:%Y-%m-%d} -> {model_run_dates[-1]:%Y-%m-%d})"
            )

        logger.info(f"Model: {model_name} | Region: {region} | Subsample: {subsample}")

        if not maps_only:
            try:
                metrics_dataframe = _validate_model(
                    model_name=model_name,
                    model_loader=MODELS[model_name],
                    first_day_datetimes=model_run_dates,
                    region=region,
                    subsample=subsample,
                )
            except Exception:
                logger.exception(f"Failed for {model_name}")
                failed_models.append(model_name)
                continue
            all_results.append(metrics_dataframe)

        if start_error_date:
            # Only the AVAILABLE runs of the period feed the pointwise path
            # (P7) : a missing run zarr is skipped exactly like in the RMSD
            # path, instead of crashing the model load below.
            available_run_dates = set(model_run_dates)
            pointwise_run_dates = [
                run_date
                for run_date in _run_dates_for_error_dates(start_error_date, end_error_date)
                if run_date in available_run_dates
            ]
            if not pointwise_run_dates:
                logger.error(
                    f"  {model_name} : no available run covers "
                    f"[{start_error_date} .. {end_error_date or start_error_date}] — pointwise outputs skipped."
                )
                failed_models.append(model_name)
                continue
            try:
                error_dataset = _pointwise_error_for_period(
                    model_name=model_name,
                    model_loader=MODELS[model_name],
                    region=region,
                    subsample=subsample,
                    start_error_date=start_error_date,
                    end_error_date=end_error_date,
                    first_day_datetimes=pointwise_run_dates,
                )
            except Exception:
                logger.exception(f"Failed for the pointwise outputs of {model_name}")
                failed_models.append(model_name)
            else:
                logger.info(
                    f"Pointwise errors computed for {model_name} "
                    f"[{start_error_date} .. {end_error_date or start_error_date}] : "
                    f"{error_dataset.sizes['observations']:,} points"
                )
                error_output_path = os.path.join(output_dir, f"error_swot_{model_name}.nc")
                _save_error_dataset(error_dataset, error_output_path)
                logger.info(f"  Pointwise error dataset saved: {error_output_path}")

                # One png error map per calendar day of the requested period.
                if plot_maps:
                    try:
                        _plot_error_maps_by_date(
                            error_dataset,
                            output_dir,
                            model_name,
                            start_error_date,
                            end_error_date or start_error_date,
                            vmax=vmax,
                        )
                    except Exception:
                        logger.exception(f"  Failed generating the maps for {model_name}")
                        failed_models.append(model_name)

    if not all_results and not maps_only:
        raise RuntimeError(f"No model could be validated : {models}")

    results = pandas.concat(all_results) if all_results else pandas.DataFrame()
    if not results.empty:
        results.index.name = "variable_label"
        csv_output_path = os.path.join(output_dir, output)
        results.to_csv(csv_output_path)
        logger.info(f"Results saved in: {csv_output_path} ('{output_dir}/')")

        try:
            lead_day_columns = _find_lead_day_columns(results)
            logger.info("Summary of RMSD (meters) by model and lead day:")
            for lead_day_column in lead_day_columns:
                logger.info(f"--- {lead_day_column} ---")
                logger.info(str(results.pivot_table(index="model", values=lead_day_column).round(4)))
        except Exception as error:
            logger.error(f"Impossible to generate the summary table : {error}")

    if failed_models:
        raise RuntimeError(f"swot-validate finished with failures : {failed_models}")
    return results
