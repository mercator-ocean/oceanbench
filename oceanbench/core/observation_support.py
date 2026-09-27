# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from dataclasses import dataclass

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    _CLASS4_OBSERVATIONS_CACHE,
    create_class4_observations_dataframe,
    interpolate_class4_model_to_observations,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import DEPTH_BINS_DEFAULT, Variable
from oceanbench.core.references.observations import _forecast_observation_matches
from oceanbench.core.regions import RegionLike, region_to_dict, resolve_region
from oceanbench.core.remote_http import with_remote_http_retries


SUPPORTED_VARIABLES = {
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): ("temp_raw", "temp_qc"),
    Variable.SEA_WATER_SALINITY.key(): ("psal_raw", "psal_qc"),
}


@dataclass(frozen=True)
class ObservationSupportReport:
    """Tables describing sampled observation support, without an independence claim.

    ``counts`` describes forecast-observation pairs by zero-based lead day.
    Coverage and QC tables deduplicate observations reused across forecasts.
    Profile counts use native ``profile_id`` when present, otherwise estimated
    platform-plus-time groups (including surface measurement groups). Unknown
    platform identities are excluded from platform counts and remain explicit.
    ``provenance`` contains stored source records, including nested attributes.
    """

    counts: pandas.DataFrame
    monthly_coverage: pandas.DataFrame
    spatial_coverage: pandas.DataFrame
    quality_control: pandas.DataFrame
    provenance: pandas.DataFrame
    notes: tuple[str, ...]
    region: dict
    spatial_bin_degrees: float
    minimum_profiles: int
    evaluated_dates: tuple[str, ...]
    independence_status: str = "unknown"


def _lead_days_count(challenger: xarray.Dataset) -> int:
    lead_days = numpy.asarray(challenger["lead_day_index"].values)
    if not len(lead_days) or not numpy.isfinite(lead_days).all():
        raise ValueError("Challenger lead_day_index must contain finite nonnegative integers.")
    if (lead_days < 0).any() or (lead_days != lead_days.astype(int)).any():
        raise ValueError("Challenger lead_day_index must contain finite nonnegative integers.")
    if not challenger.sizes["first_day_datetime"]:
        raise ValueError("Challenger must contain at least one forecast start.")
    if not numpy.array_equal(lead_days, numpy.arange(len(lead_days))):
        raise ValueError("Challenger lead_day_index must be contiguous and ordered from zero (0..N-1).")
    return len(lead_days)


def _paired_observations(
    observations: xarray.Dataset, challenger: xarray.Dataset, lead_days_count: int
) -> xarray.Dataset:
    observations = rename_dataset_with_standard_names(observations)
    observation_dimension = observations["time"].dims[0]
    observation_times = pandas.to_datetime(observations["time"].values)
    first_days = challenger["first_day_datetime"].values.astype("datetime64[ns]")
    observations = observations.assign_coords(time=(observation_dimension, observation_times))
    if "first_day_datetime" not in observations:
        observation_indices, run_indices = _forecast_observation_matches(
            observation_times, pandas.to_datetime(first_days), lead_days_count
        )
        observations = observations.isel({observation_dimension: observation_indices}).assign_coords(
            first_day_datetime=(observation_dimension, first_days[run_indices])
        )
    valid_days = (observations["time"] - observations["first_day_datetime"]) / numpy.timedelta64(1, "D")
    valid = observations["first_day_datetime"].isin(first_days) & (valid_days >= 0) & (valid_days < lead_days_count)
    return observations.isel({observation_dimension: numpy.flatnonzero(valid.values)})


def _domain(region: RegionLike) -> tuple[float, float, float, float]:
    bounds = resolve_region(region).bounds
    if bounds is None:
        return -90.0, 90.0, -180.0, 180.0
    minimum_longitude = (bounds.minimum_longitude + 180) % 360 - 180
    width = (bounds.maximum_longitude - bounds.minimum_longitude) % 360
    if abs(bounds.maximum_longitude - bounds.minimum_longitude) >= 360:
        width = 360.0
    return (
        bounds.minimum_latitude,
        bounds.maximum_latitude,
        minimum_longitude,
        minimum_longitude + width,
    )


def _region_observations(observations: xarray.Dataset, region: RegionLike) -> xarray.Dataset:
    if resolve_region(region).bounds is None:
        return observations
    latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum = _domain(region)
    longitudes = longitude_minimum + (observations["longitude"] - longitude_minimum) % 360
    mask = (
        (observations["latitude"] >= latitude_minimum)
        & (observations["latitude"] <= latitude_maximum)
        & (longitudes <= longitude_maximum)
    )
    return observations.isel({observations["time"].dims[0]: numpy.flatnonzero(mask.values)})


def _identity_metadata(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    platform = dataframe.get("platform_code", pandas.Series("", index=dataframe.index))
    platform = platform.map(_identity_text)
    profile = dataframe.get("profile_id", pandas.Series("", index=dataframe.index)).map(_identity_text)
    time = pandas.to_datetime(dataframe["time"])
    native_profile = profile.ne("")
    estimated_profile = platform.ne("") & time.notna()
    profile_keys = pandas.Series(None, index=dataframe.index, dtype=object)
    profile_keys.loc[native_profile] = list(zip(platform[native_profile], profile[native_profile]))
    profile_keys.loc[~native_profile & estimated_profile] = list(
        zip(platform[~native_profile & estimated_profile], time[~native_profile & estimated_profile])
    )
    observation_ids = dataframe.get("obs_id", pandas.Series("", index=dataframe.index)).map(_identity_text)
    fallback_columns = ["time", "latitude", "longitude", "depth", "platform_code", "profile_id"]
    fallback_columns += [key for key in SUPPORTED_VARIABLES if key in dataframe]
    fallback = dataframe.reindex(columns=fallback_columns).astype(str).agg(tuple, axis=1)
    measurement_keys = pandas.Series(list(zip(observation_ids, fallback)), index=dataframe.index, dtype=object)
    measurement_keys.loc[observation_ids.ne("")] = observation_ids[observation_ids.ne("")]
    return dataframe.assign(_platform=platform.replace("", None), _profile=profile_keys, _measurement=measurement_keys)


def _identity_text(value) -> str:
    if pandas.isna(value):
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf8", errors="replace")
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "none", "unknown", "null") else text


def _support_counts(dataframe: pandas.DataFrame) -> dict[str, int]:
    return {
        "unique_profiles": int(dataframe["_profile"].nunique()),
        "unique_platforms": int(dataframe["_platform"].nunique()),
        "unknown_platform_observations": int(dataframe["_platform"].isna().sum()),
    }


def _coverage_status(profiles: int, measurements: int, minimum_profiles: int) -> str:
    if measurements == 0:
        return "unobserved"
    if profiles == 0:
        return "identity_unknown"
    return "sparse" if profiles < minimum_profiles else "observed"


def _depth_bins(variable: str) -> list[str]:
    return (["surface"] if variable == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key() else []) + list(
        DEPTH_BINS_DEFAULT
    )


def _variable_observations(
    observations: xarray.Dataset, metadata: pandas.DataFrame, variable: str, lead_days_count: int
) -> pandas.DataFrame:
    if variable not in observations:
        return pandas.DataFrame(columns=["depth_bin", "lead_day", "observation_value", *metadata.columns])
    dataframe = create_class4_observations_dataframe(observations, variable, variable, lead_days_count)
    dataframe = dataframe.loc[numpy.isfinite(dataframe["observation_value"])].copy()
    for key in ("_profile", "_platform", "_measurement"):
        dataframe[key] = metadata.loc[dataframe.index, key]
    return dataframe.reset_index(drop=True)


def _pair_counts(
    challenger: xarray.Dataset, dataframe: pandas.DataFrame, variable: str, lead_days_count: int
) -> list[dict]:
    dataframe = dataframe.copy()
    dataframe["model_value"] = numpy.nan
    if variable in challenger and not dataframe.empty:
        available_model_rows = dataframe["lead_day"].isin(challenger["lead_day_index"].values)
        selected = dataframe.loc[available_model_rows].reset_index(drop=True)
        if not selected.empty:
            dataframe.loc[available_model_rows, "model_value"] = interpolate_class4_model_to_observations(
                _model_brackets(challenger[variable], selected), selected
            )
    dataframe["_matched"] = numpy.isfinite(dataframe["model_value"])
    rows = []
    for depth_bin in _depth_bins(variable):
        for lead_day in range(lead_days_count):
            group = dataframe.loc[(dataframe["depth_bin"] == depth_bin) & (dataframe["lead_day"] == lead_day)]
            matched = group.loc[group["_matched"]]
            unique = group.drop_duplicates("_measurement")
            rows.append(
                {
                    "variable": variable,
                    "depth_bin": depth_bin,
                    "lead_day": lead_day,
                    "available_observations": len(group),
                    "model_matched_observations": len(matched),
                    "missing_forecasts": len(group) - len(matched),
                    **_support_counts(unique),
                    "rmsd": numpy.sqrt(((matched["model_value"] - matched["observation_value"]) ** 2).mean()),
                }
            )
    return rows


def _model_brackets(model: xarray.DataArray, observations: pandas.DataFrame) -> xarray.DataArray:
    for coordinate in ("latitude", "longitude"):
        positions = observations.loc[numpy.isfinite(observations[coordinate]), coordinate]
        values = model[coordinate].values
        if positions.empty or len(values) < 2:
            continue
        sorted_indices = numpy.argsort(values)
        sorted_values = values[sorted_indices]
        first = max(0, int(numpy.searchsorted(sorted_values, positions.min(), side="left")) - 1)
        last = min(len(values), int(numpy.searchsorted(sorted_values, positions.max(), side="right")) + 1)
        if last - first < 2:
            first, last = (0, 2) if first == 0 else (len(values) - 2, len(values))
        model = model.isel({coordinate: numpy.sort(sorted_indices[first:last])})
    return model


def _bin_edges(minimum: float, maximum: float, width: float) -> numpy.ndarray:
    return numpy.append(numpy.arange(minimum, maximum, width), maximum)


def _coverage_tables(
    dataframe: pandas.DataFrame,
    variable: str,
    evaluated_days: pandas.DatetimeIndex,
    region: RegionLike,
    width: float,
    minimum_profiles: int,
) -> tuple[list[dict], list[dict]]:
    latitude_minimum, latitude_maximum, longitude_minimum, longitude_maximum = _domain(region)
    latitude_edges = _bin_edges(latitude_minimum, latitude_maximum, width)
    longitude_edges = _bin_edges(longitude_minimum, longitude_maximum, width)
    dataframe = dataframe.drop_duplicates("_measurement").copy()
    dataframe["month"] = pandas.to_datetime(dataframe["time"]).dt.to_period("M").astype(str)
    months = pandas.period_range(evaluated_days.min().to_period("M"), evaluated_days.max().to_period("M"), freq="M")
    month_day_counts = pandas.Series(evaluated_days.to_period("M").astype(str)).value_counts()
    longitudes = longitude_minimum + (dataframe["longitude"] - longitude_minimum) % 360
    located = dataframe["latitude"].between(latitude_minimum, latitude_maximum) & longitudes.between(
        longitude_minimum, longitude_maximum
    )
    dataframe["_latitude_bin"] = numpy.clip(
        numpy.searchsorted(latitude_edges, dataframe["latitude"], side="right") - 1, 0, len(latitude_edges) - 2
    )
    dataframe["_longitude_bin"] = numpy.clip(
        numpy.searchsorted(longitude_edges, longitudes, side="right") - 1, 0, len(longitude_edges) - 2
    )
    monthly_rows, spatial_rows = [], []
    for depth_bin in _depth_bins(variable):
        depth_group = dataframe.loc[dataframe["depth_bin"] == depth_bin]
        for month in months.astype(str):
            group = depth_group.loc[depth_group["month"] == month]
            support = _support_counts(group)
            evaluated_day_count = int(month_day_counts.get(month, 0))
            monthly_rows.append(
                {
                    "variable": variable,
                    "depth_bin": depth_bin,
                    "month": month,
                    **support,
                    "accepted_measurements": len(group),
                    "evaluated_days": evaluated_day_count,
                    "status": (
                        _coverage_status(support["unique_profiles"], len(group), minimum_profiles)
                        if evaluated_day_count
                        else "not_evaluated"
                    ),
                }
            )
        spatial_groups = {
            key: group
            for key, group in depth_group.loc[located].groupby(["_latitude_bin", "_longitude_bin"], sort=False)
        }
        empty = depth_group.iloc[:0]
        for latitude_index in range(len(latitude_edges) - 1):
            for longitude_index in range(len(longitude_edges) - 1):
                group = spatial_groups.get((latitude_index, longitude_index), empty)
                support = _support_counts(group)
                spatial_rows.append(
                    {
                        "variable": variable,
                        "depth_bin": depth_bin,
                        "latitude_min": latitude_edges[latitude_index],
                        "latitude_max": latitude_edges[latitude_index + 1],
                        "longitude_min": longitude_edges[longitude_index],
                        "longitude_max": longitude_edges[longitude_index + 1],
                        **support,
                        "accepted_measurements": len(group),
                        "status": _coverage_status(support["unique_profiles"], len(group), minimum_profiles),
                    }
                )
    return monthly_rows, spatial_rows


def _metadata_availability(
    unique: pandas.DataFrame, keys: tuple[str, ...], provenance: pandas.DataFrame
) -> pandas.Series:
    available = pandas.Series(all(key in unique for key in keys), index=unique.index)
    if "source_day" in unique and {"date", "available_variables"}.issubset(provenance.columns):
        day_availability = {
            str(record["date"]): all(key in record["available_variables"] for key in keys)
            for record in provenance.to_dict("records")
            if isinstance(record["available_variables"], (list, tuple))
        }
        available &= pandas.to_datetime(unique["source_day"]).dt.strftime("%Y-%m-%d").map(day_availability).eq(True)
    return available


def _quality_control(metadata: pandas.DataFrame, variable: str, provenance: pandas.DataFrame) -> dict:
    raw_key, flag_key = SUPPORTED_VARIABLES[variable]
    unique = metadata.drop_duplicates("_measurement")
    applicable_types = (1, 2) if variable == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key() else (1,)
    if "obs_type" in unique:
        unique = unique.loc[unique["obs_type"].isin(applicable_types) | ~unique["obs_type"].isin((1, 2, 3, 4))]
    accepted = numpy.isfinite(unique[variable]) if variable in unique else pandas.Series(False, index=unique.index)
    if "qc_keep" in unique:
        accepted &= unique["qc_keep"].isna() | unique["qc_keep"].eq(1)
    raw_available = _metadata_availability(unique, (raw_key,), provenance)
    qc_available = _metadata_availability(unique, (raw_key, flag_key, "qc_keep", "qc_reason"), provenance)
    if "obs_type" in unique:
        raw_available &= unique["obs_type"].isin(applicable_types)
        qc_available &= unique["obs_type"].isin(applicable_types)
    for key in (flag_key, "qc_keep", "qc_reason"):
        if key in unique:
            qc_available &= unique[key].notna()
    raw_present = bool(raw_available.all()) and raw_key in unique
    raw_finite = numpy.isfinite(unique[raw_key]) if raw_key in unique else pandas.Series(False, index=unique.index)
    rejected = raw_finite & ~accepted & qc_available
    reasons = unique.get("qc_reason", pandas.Series("", index=unique.index)).map(_identity_text)
    rejected_reasons = reasons[rejected].replace("", "variable_qc_or_missing_scored_value")
    flags = unique.loc[raw_finite, flag_key].dropna().value_counts().to_dict() if flag_key in unique else {}
    return {
        "variable": variable,
        "raw_measurements": int(raw_finite.sum()) if raw_present else pandas.NA,
        "accepted_measurements": int(accepted.sum()),
        "rejected_measurements": int(rejected.sum()) if raw_present and qc_available.all() else pandas.NA,
        "raw_missing": int((~raw_finite).sum()) if raw_present else pandas.NA,
        "qc_metadata_available": bool(qc_available.all())
        and all(key in unique for key in (raw_key, flag_key, "qc_keep", "qc_reason")),
        "audited_measurements": int(qc_available.sum()),
        "unknown_metadata_measurements": int((~qc_available).sum()),
        "source_qc_flags": flags,
        "row_first_failure_reasons": rejected_reasons.value_counts().to_dict(),
    }


def _provenance(observations: xarray.Dataset) -> pandas.DataFrame:
    stored = observations.attrs.get("observation_provenance")
    if stored is None:
        return pandas.DataFrame([{"attrs": dict(observations.attrs)}])
    records = json.loads(stored) if isinstance(stored, str) else stored
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("observation_provenance must be a JSON list of source records.")
    return pandas.DataFrame.from_records(records)


def observation_support(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
    *,
    observations_dataset: xarray.Dataset | None = None,
    spatial_bin_degrees: float = 10,
    minimum_profiles: int = 5,
) -> ObservationSupportReport:
    """Describe temperature and salinity observation support for a challenger.

    Supplied observations allow an offline report and may be unpaired point
    observations or the reader's forecast-paired audit dataset. Stored scored
    values define per-variable QC acceptance, additionally requiring qc_keep=1
    when that row has a stored flag; missing audit metadata remains unknown.
    This function does not rebuild the source QC policy.
    Spatial bins describe sampled cells across the selected geographic bounds,
    including land and unobserved cells. They are not an ocean mask or a fraction
    of the ocean supported by observations. Sparse means fewer identified profile
    groups than minimum_profiles, not statistical uncertainty or model skill.
    """
    if not numpy.isfinite(spatial_bin_degrees) or not 0 < spatial_bin_degrees <= 180:
        raise ValueError("spatial_bin_degrees must be finite, greater than zero, and at most 180.")
    if isinstance(minimum_profiles, bool) or not isinstance(minimum_profiles, (int, numpy.integer)):
        raise ValueError("minimum_profiles must be a positive integer.")
    if minimum_profiles < 1:
        raise ValueError("minimum_profiles must be a positive integer.")
    region = resolve_region(region)
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = _lead_days_count(challenger)
    owned_observations = observations_dataset is None
    if owned_observations:
        from oceanbench.core.references.observations import observation_audit

        observations_dataset = observation_audit(challenger)
    try:
        return _build_observation_support(
            challenger, observations_dataset, region, lead_days_count, spatial_bin_degrees, minimum_profiles
        )
    finally:
        if owned_observations:
            observations_dataset.close()


def _build_observation_support(
    challenger: xarray.Dataset,
    observations_dataset: xarray.Dataset,
    region: RegionLike,
    lead_days_count: int,
    spatial_bin_degrees: float,
    minimum_profiles: int,
) -> ObservationSupportReport:
    provenance = _provenance(observations_dataset)
    observations = _region_observations(_paired_observations(observations_dataset, challenger, lead_days_count), region)
    observation_dimension = observations["time"].dims[0]
    observations = with_remote_http_retries("Observation support audit read", observations.compute)
    metadata = observations.to_dataframe()
    metadata = _identity_metadata(metadata.reset_index(drop=True))
    if "qc_keep" in metadata:
        observations = observations.assign(
            {
                variable: observations[variable].where(
                    observations["qc_keep"].isnull() | (observations["qc_keep"] == 1)
                )
                for variable in SUPPORTED_VARIABLES
                if variable in observations
            }
        )
    observations = observations.assign_coords({observation_dimension: numpy.arange(len(metadata))})
    first_days = pandas.to_datetime(challenger["first_day_datetime"].values)
    evaluated_days = (
        pandas.DatetimeIndex(
            numpy.concatenate(
                [
                    pandas.date_range(
                        first_day.normalize(),
                        (
                            first_day + pandas.Timedelta(days=lead_days_count) - pandas.Timedelta(nanoseconds=1)
                        ).normalize(),
                    ).values
                    for first_day in first_days
                ]
            )
        )
        .unique()
        .sort_values()
    )
    counts, monthly, spatial = [], [], []
    _CLASS4_OBSERVATIONS_CACHE.pop((id(observations), lead_days_count), None)
    try:
        for variable in SUPPORTED_VARIABLES:
            dataframe = _variable_observations(observations, metadata, variable, lead_days_count)
            counts.extend(_pair_counts(challenger, dataframe, variable, lead_days_count))
            variable_monthly, variable_spatial = _coverage_tables(
                dataframe, variable, evaluated_days, region, spatial_bin_degrees, minimum_profiles
            )
            monthly.extend(variable_monthly)
            spatial.extend(variable_spatial)
    finally:
        _CLASS4_OBSERVATIONS_CACHE.pop((id(observations), lead_days_count), None)
    notes = (
        "Support method: current Class IV depth bins and model interpolation, with finite values only. "
        "Current score functions drop NaN; infinite values are excluded by this diagnostic.",
        "No shared-population or ocean mask is applied; source selection is the supplied audit population. "
        "Future scoring population changes require explicitly updating this companion method.",
        "Counts are forecast-observation pairs. Coverage and QC deduplicate reused observations by obs_id; "
        "when absent, identical coordinate/value records are an estimated deduplication.",
        "Profiles use native profile_id where available, otherwise estimated platform-plus-time groups, "
        "including surface measurement groups. Unknown identities are not invented.",
        "Coverage samples only evaluated forecast dates within the selected bounds; "
        "zero cells and months are explicit. "
        "Months outside forecast windows are not_evaluated; partial months are not full-month surveys. "
        "Spatial cells include land; missing positions are excluded from the spatial table. "
        "Sparse is a descriptive threshold, not certainty, density extrapolation, or full-ocean coverage.",
        "QC summaries use stored raw/scored measurements and source flags. qc_reason records only the first "
        "row-policy failure; it is not a complete list of causes. Missing raw/QC metadata remains unknown. "
        "When obs_type is stored, temperature QC covers Argo/drifter SST rows and salinity covers Argo rows; "
        "otherwise raw_missing counts rows without a finite raw value, not missing profiles or ocean observations.",
        "Observational independence is unknown. Assessing it requires training-data manifests and dates, "
        "assimilation/initial-condition sources and cycles, and observation/platform identifiers for overlap checks.",
    )
    return ObservationSupportReport(
        counts=pandas.DataFrame(counts),
        monthly_coverage=pandas.DataFrame(monthly),
        spatial_coverage=pandas.DataFrame(spatial),
        quality_control=pandas.DataFrame(
            [_quality_control(metadata, variable, provenance) for variable in SUPPORTED_VARIABLES]
        ),
        provenance=provenance,
        notes=notes,
        region=region_to_dict(region),
        spatial_bin_degrees=float(spatial_bin_degrees),
        minimum_profiles=int(minimum_profiles),
        evaluated_dates=tuple(evaluated_days.strftime("%Y-%m-%d")),
    )
