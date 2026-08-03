# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import timedelta

import numpy
import pandas
from scipy.spatial import cKDTree
import xarray

from oceanbench.core import lagrangian_trajectory
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

DRIFTER_DEPTH_METERS = 15.0
MATCH_CANDIDATE_COUNT = 5
MATCH_DISTANCE_PER_HOUR_KM = 25.0
MAX_TRACK_GAP_HOURS = 6.0
LAST_MATCHED_TIME_COLUMN = "last_matched_time"
MINIMUM_RELIABLE_MATCHED_DRIFTER_COUNT = 50
MINIMUM_RELIABLE_MATCHED_DRIFTER_FRACTION = 0.05
DRIFTER_TRAJECTORY_DEVIATION_ROW_LABEL = "Class-4 drifter trajectory deviation mean (km)"
MATCHED_DRIFTER_COUNT_ROW_LABEL = "Class-4 matched drifter count"

_DRIFTER_COMPARISON_CACHE: dict[tuple[int, int, int], tuple[xarray.Dataset, xarray.Dataset]] = {}


def _drifter_lead_day_labels(lead_day_count: int) -> list[str]:
    labels = lead_day_labels(1, lead_day_count)
    if labels:
        labels[0] = f"{labels[0]} (init)"
    return labels


class Class4DrifterTrajectoryUnavailableError(ValueError):
    pass


def _to_cartesian_coordinates(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> numpy.ndarray:
    latitude_radians = numpy.deg2rad(latitudes)
    longitude_radians = numpy.deg2rad(longitudes)
    cos_latitudes = numpy.cos(latitude_radians)
    return numpy.column_stack(
        [
            cos_latitudes * numpy.cos(longitude_radians),
            cos_latitudes * numpy.sin(longitude_radians),
            numpy.sin(latitude_radians),
        ]
    )


def _haversine_distance_km(
    latitude_start: numpy.ndarray | float,
    longitude_start: numpy.ndarray | float,
    latitude_end: numpy.ndarray | float,
    longitude_end: numpy.ndarray | float,
) -> numpy.ndarray:
    earth_radius_km = 6371.0
    latitude_start_radians = numpy.deg2rad(latitude_start)
    longitude_start_radians = numpy.deg2rad(longitude_start)
    latitude_end_radians = numpy.deg2rad(latitude_end)
    longitude_end_radians = numpy.deg2rad(longitude_end)
    delta_latitude = latitude_end_radians - latitude_start_radians
    delta_longitude = longitude_end_radians - longitude_start_radians
    haversine_term = (
        numpy.sin(delta_latitude / 2.0) ** 2
        + numpy.cos(latitude_start_radians) * numpy.cos(latitude_end_radians) * numpy.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * numpy.arcsin(numpy.sqrt(haversine_term))


def _predict_next_positions(
    active_tracks: pandas.DataFrame,
    time_step_hours: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    time_step_seconds = time_step_hours * 3600.0
    latitudes = active_tracks[Dimension.LATITUDE.key()].to_numpy()
    longitudes = active_tracks[Dimension.LONGITUDE.key()].to_numpy()
    predicted_latitudes = latitudes + (
        active_tracks[Variable.NORTHWARD_SEA_WATER_VELOCITY.key()].to_numpy() * time_step_seconds / 111000.0
    )
    cosine_latitudes = numpy.cos(numpy.deg2rad(numpy.clip(latitudes, -89.9, 89.9)))
    predicted_longitudes = longitudes + (
        active_tracks[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].to_numpy()
        * time_step_seconds
        / (111000.0 * numpy.maximum(cosine_latitudes, 1e-6))
    )
    return predicted_latitudes, predicted_longitudes


def _drifter_observations_dataframe(
    observation_dataset: xarray.Dataset,
    first_day_datetime: numpy.datetime64,
) -> pandas.DataFrame:
    observation_dataset = rename_dataset_with_standard_names(observation_dataset)
    time_key = Dimension.TIME.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    depth_key = Dimension.DEPTH.key()
    eastward_velocity_key = Variable.EASTWARD_SEA_WATER_VELOCITY.key()
    northward_velocity_key = Variable.NORTHWARD_SEA_WATER_VELOCITY.key()
    required_keys = [
        time_key,
        first_day_key,
        latitude_key,
        longitude_key,
        depth_key,
        eastward_velocity_key,
        northward_velocity_key,
    ]
    missing_keys = [key for key in required_keys if key not in observation_dataset.variables]
    if missing_keys:
        raise Class4DrifterTrajectoryUnavailableError(
            "Class-4 drifter trajectories require observation variables: " + ", ".join(missing_keys)
        )

    observation_dimension_key = observation_dataset[time_key].dims[0]
    first_day_observations = observation_dataset.isel(
        {observation_dimension_key: observation_dataset[first_day_key] == numpy.datetime64(first_day_datetime)}
    )
    valid_mask = (
        first_day_observations[eastward_velocity_key].notnull()
        & first_day_observations[northward_velocity_key].notnull()
        & numpy.isclose(first_day_observations[depth_key], DRIFTER_DEPTH_METERS)
    ).compute()
    first_day_observations = first_day_observations.isel({observation_dimension_key: valid_mask})
    if first_day_observations.sizes.get(observation_dimension_key, 0) == 0:
        raise Class4DrifterTrajectoryUnavailableError(
            "No Class-4 drifter observations at 15 m were found for the selected forecast run."
        )

    observations_dataframe = pandas.DataFrame(
        {
            time_key: pandas.to_datetime(first_day_observations[time_key].values),
            latitude_key: first_day_observations[latitude_key].values,
            longitude_key: first_day_observations[longitude_key].values,
            eastward_velocity_key: first_day_observations[eastward_velocity_key].values,
            northward_velocity_key: first_day_observations[northward_velocity_key].values,
        }
    )
    return observations_dataframe.sort_values(time_key).reset_index(drop=True)


def _hourly_drifter_frames(
    observations_dataframe: pandas.DataFrame,
) -> tuple[list[pandas.Timestamp], dict[pandas.Timestamp, pandas.DataFrame]]:
    grouped_frames = {
        timestamp: group.loc[
            :,
            [
                Dimension.LATITUDE.key(),
                Dimension.LONGITUDE.key(),
                Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
                Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
            ],
        ].reset_index(drop=True)
        for timestamp, group in observations_dataframe.groupby(Dimension.TIME.key(), sort=True)
    }
    return sorted(grouped_frames), grouped_frames


def _matched_track_indices(
    active_tracks: pandas.DataFrame,
    next_frame: pandas.DataFrame,
    time_step_hours: numpy.ndarray,
) -> list[tuple[int, int]]:
    predicted_latitudes, predicted_longitudes = _predict_next_positions(active_tracks, time_step_hours=time_step_hours)
    next_track_tree = cKDTree(
        _to_cartesian_coordinates(
            latitudes=next_frame[Dimension.LATITUDE.key()].to_numpy(),
            longitudes=next_frame[Dimension.LONGITUDE.key()].to_numpy(),
        )
    )
    candidate_count = min(MATCH_CANDIDATE_COUNT, len(next_frame))
    _, nearest_indices = next_track_tree.query(
        _to_cartesian_coordinates(predicted_latitudes, predicted_longitudes),
        k=candidate_count,
    )
    if candidate_count == 1:
        nearest_indices = nearest_indices[:, numpy.newaxis]

    candidate_matches: list[tuple[float, int, int]] = []
    match_distance_limits_km = MATCH_DISTANCE_PER_HOUR_KM * numpy.asarray(time_step_hours, dtype=float)
    for active_track_index, candidate_indices in enumerate(nearest_indices):
        match_distance_limit_km = float(match_distance_limits_km[active_track_index])
        for candidate_index in numpy.atleast_1d(candidate_indices):
            next_row = next_frame.iloc[int(candidate_index)]
            distance = float(
                _haversine_distance_km(
                    predicted_latitudes[active_track_index],
                    predicted_longitudes[active_track_index],
                    next_row[Dimension.LATITUDE.key()],
                    next_row[Dimension.LONGITUDE.key()],
                )
            )
            if distance <= match_distance_limit_km:
                candidate_matches.append((distance, active_track_index, int(candidate_index)))

    candidate_matches.sort()
    matched_active_indices: set[int] = set()
    matched_next_indices: set[int] = set()
    selected_matches: list[tuple[int, int]] = []
    for _, active_track_index, next_index in candidate_matches:
        if active_track_index in matched_active_indices or next_index in matched_next_indices:
            continue
        matched_active_indices.add(active_track_index)
        matched_next_indices.add(next_index)
        selected_matches.append((active_track_index, next_index))
    return selected_matches


def _link_hourly_drifter_trajectories(
    observation_dataset: xarray.Dataset,
    first_day_datetime: numpy.datetime64,
) -> pandas.DataFrame:
    observations_dataframe = _drifter_observations_dataframe(observation_dataset, first_day_datetime)
    timestamps, grouped_frames = _hourly_drifter_frames(observations_dataframe)
    if not timestamps:
        raise Class4DrifterTrajectoryUnavailableError(
            "No Class-4 drifter observations were found for the selected forecast run."
        )

    first_timestamp = timestamps[0]
    active_tracks = grouped_frames[first_timestamp].copy()
    active_tracks["track_id"] = numpy.arange(len(active_tracks))
    trajectory_records = [
        {
            "track_id": int(track_id),
            Dimension.TIME.key(): first_timestamp,
            Dimension.LATITUDE.key(): latitude,
            Dimension.LONGITUDE.key(): longitude,
        }
        for track_id, latitude, longitude in zip(
            active_tracks["track_id"].to_numpy(),
            active_tracks[Dimension.LATITUDE.key()].to_numpy(),
            active_tracks[Dimension.LONGITUDE.key()].to_numpy(),
            strict=False,
        )
    ]

    active_tracks[LAST_MATCHED_TIME_COLUMN] = first_timestamp
    for next_timestamp in timestamps[1:]:
        if active_tracks.empty:
            break
        next_frame = grouped_frames[next_timestamp]
        gap_hours = (next_timestamp - active_tracks[LAST_MATCHED_TIME_COLUMN]) / pandas.Timedelta(hours=1)
        matched_indices = _matched_track_indices(active_tracks, next_frame, time_step_hours=gap_hours.to_numpy())

        if matched_indices:
            # Matched positions are unique on both sides, so the per-match writes can be applied
            # as one array assignment per column instead of one scalar assignment per match.
            match_pairs = numpy.asarray(matched_indices, dtype=int)
            active_row_labels = active_tracks.index.to_numpy()[match_pairs[:, 0]]
            next_row_positions = match_pairs[:, 1]
            matched_latitudes = next_frame[Dimension.LATITUDE.key()].to_numpy()[next_row_positions]
            matched_longitudes = next_frame[Dimension.LONGITUDE.key()].to_numpy()[next_row_positions]
            active_tracks.loc[active_row_labels, Dimension.LATITUDE.key()] = matched_latitudes
            active_tracks.loc[active_row_labels, Dimension.LONGITUDE.key()] = matched_longitudes
            active_tracks.loc[active_row_labels, Variable.EASTWARD_SEA_WATER_VELOCITY.key()] = next_frame[
                Variable.EASTWARD_SEA_WATER_VELOCITY.key()
            ].to_numpy()[next_row_positions]
            active_tracks.loc[active_row_labels, Variable.NORTHWARD_SEA_WATER_VELOCITY.key()] = next_frame[
                Variable.NORTHWARD_SEA_WATER_VELOCITY.key()
            ].to_numpy()[next_row_positions]
            active_tracks.loc[active_row_labels, LAST_MATCHED_TIME_COLUMN] = next_timestamp
            matched_track_ids = active_tracks.loc[active_row_labels, "track_id"].to_numpy()
            trajectory_records.extend(
                {
                    "track_id": int(track_id),
                    Dimension.TIME.key(): next_timestamp,
                    Dimension.LATITUDE.key(): latitude,
                    Dimension.LONGITUDE.key(): longitude,
                }
                for track_id, latitude, longitude in zip(
                    matched_track_ids,
                    matched_latitudes,
                    matched_longitudes,
                    strict=False,
                )
            )

        # Unmatched tracks stay alive, dead reckoned from their last observed position and velocity,
        # until their gap grows past the bounded tolerance.
        elapsed_hours = (next_timestamp - active_tracks[LAST_MATCHED_TIME_COLUMN]) / pandas.Timedelta(hours=1)
        active_tracks = active_tracks[elapsed_hours <= MAX_TRACK_GAP_HOURS].copy()

    return pandas.DataFrame(trajectory_records)


def class4_drifter_reference_trajectories(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    challenger_dataset = rename_dataset_with_standard_names(challenger_dataset)
    first_day_datetime = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values[first_day_index]
    linked_trajectories = _link_hourly_drifter_trajectories(
        observation_dataset=observation_dataset,
        first_day_datetime=first_day_datetime,
    )
    if linked_trajectories.empty:
        raise Class4DrifterTrajectoryUnavailableError("No linked Class-4 drifter trajectories were reconstructed.")

    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    initial_timestamp = linked_trajectories[Dimension.TIME.key()].min()
    output_timestamps = [initial_timestamp + timedelta(days=offset) for offset in range(lead_days_count)]
    track_ids = numpy.arange(linked_trajectories["track_id"].max() + 1)
    latitude_values = numpy.full((len(track_ids), len(output_timestamps)), numpy.nan)
    longitude_values = numpy.full((len(track_ids), len(output_timestamps)), numpy.nan)

    trajectory_lookup = linked_trajectories.set_index(["track_id", Dimension.TIME.key()]).sort_index()
    for time_index, output_timestamp in enumerate(output_timestamps):
        if output_timestamp not in linked_trajectories[Dimension.TIME.key()].values:
            continue
        time_slice = trajectory_lookup.xs(output_timestamp, level=Dimension.TIME.key(), drop_level=False)
        for track_id, row in time_slice.groupby(level="track_id"):
            latitude_values[int(track_id), time_index] = float(row[Dimension.LATITUDE.key()].iloc[0])
            longitude_values[int(track_id), time_index] = float(row[Dimension.LONGITUDE.key()].iloc[0])

    initial_latitudes = latitude_values[:, 0]
    initial_longitudes = longitude_values[:, 0]
    valid_initial_positions = numpy.isfinite(initial_latitudes) & numpy.isfinite(initial_longitudes)
    if not numpy.any(valid_initial_positions):
        raise Class4DrifterTrajectoryUnavailableError("No Class-4 drifter trajectories have finite initial positions.")

    return xarray.Dataset(
        {
            "lat": (["particle", "time"], latitude_values[valid_initial_positions]),
            "lon": (["particle", "time"], longitude_values[valid_initial_positions]),
        },
        coords={
            "particle": numpy.arange(int(numpy.count_nonzero(valid_initial_positions))),
            "time": pandas.to_datetime(output_timestamps),
            "lat0": ("particle", initial_latitudes[valid_initial_positions]),
            "lon0": ("particle", initial_longitudes[valid_initial_positions]),
        },
    )


def _bracketing_depth_indices(depth_values: numpy.ndarray, target_depth: float) -> list[int]:
    """Positions of the two depth levels that linear interpolation to the target depth reads.

    They are the levels bracketing the target depth, or the two closest levels of the axis when
    the target depth falls outside the axis range, so that interpolating on that pair keeps the
    out-of-range result of the full axis.
    """
    ascending_order = numpy.argsort(depth_values)
    ascending_values = depth_values[ascending_order]
    insertion_position = int(numpy.searchsorted(ascending_values, target_depth, side="right"))
    lower_position = min(max(insertion_position - 1, 0), ascending_values.size - 2)
    return sorted(int(position) for position in ascending_order[lower_position : lower_position + 2])


def _drifter_depth_current_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    """Currents used to advect the model particles, taken at the drifter drogue depth.

    Class-4 drifters drift at 15 m, so the model must be sampled there rather than at its first
    depth level. Models published without a depth axis, or with a single surface level, are
    surface only by design and are used as they are.

    Only the two levels that linear interpolation reads are kept before interpolating, because
    interpolating over the full depth axis copies every level of both current components.
    """
    depth_key = Dimension.DEPTH.key()
    current_dataset = rename_dataset_with_standard_names(dataset)[
        [
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        ]
    ]
    if current_dataset.sizes.get(depth_key, 0) < 2:
        return lagrangian_trajectory.surface_current_dataset(dataset)
    bracketing_indices = _bracketing_depth_indices(
        current_dataset[depth_key].values,
        DRIFTER_DEPTH_METERS,
    )
    bracketing_dataset = current_dataset.isel({depth_key: bracketing_indices})
    return bracketing_dataset.interp({depth_key: DRIFTER_DEPTH_METERS}).drop_vars(depth_key)


def class4_drifter_challenger_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_trajectories: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    challenger_standard_dataset = lagrangian_trajectory._harmonise_dataset(challenger_dataset)
    challenger_run_dataset = lagrangian_trajectory._split_dataset(challenger_standard_dataset)[first_day_index]
    challenger_particles = lagrangian_trajectory._get_particle_dataset(
        dataset=_drifter_depth_current_dataset(challenger_run_dataset),
        latitudes=reference_trajectories["lat0"].values,
        longitudes=reference_trajectories["lon0"].values,
    )
    time_count = min(challenger_particles.sizes["time"], reference_trajectories.sizes["time"])
    challenger_particles = challenger_particles.isel(time=slice(0, time_count))
    reference_times = reference_trajectories["time"].isel(time=slice(0, time_count)).values
    return challenger_particles.assign_coords(time=reference_times)


def class4_drifter_trajectory_distance_km(
    challenger_particles: xarray.Dataset,
    reference_particles: xarray.Dataset,
) -> numpy.ndarray:
    challenger_particles, reference_particles = xarray.align(challenger_particles, reference_particles, join="inner")
    return _haversine_distance_km(
        reference_particles["lat"].values,
        reference_particles["lon"].values,
        challenger_particles["lat"].values,
        challenger_particles["lon"].values,
    )


def reported_class4_drifter_time_count(
    challenger_dataset: xarray.Dataset,
    available_time_count: int,
) -> int:
    challenger_dataset = rename_dataset_with_standard_names(challenger_dataset)
    forecast_lead_day_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return min(available_time_count, max(1, forecast_lead_day_count - 1))


def _minimum_reliable_matched_drifter_count(matched_count_row: numpy.ndarray) -> int:
    if matched_count_row.size == 0:
        return 1
    initial_matched_count = int(matched_count_row[0])
    return max(
        MINIMUM_RELIABLE_MATCHED_DRIFTER_COUNT,
        int(numpy.ceil(initial_matched_count * MINIMUM_RELIABLE_MATCHED_DRIFTER_FRACTION)),
    )


def class4_drifter_trajectory_comparison(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> tuple[xarray.Dataset, xarray.Dataset]:
    cache_key = (id(challenger_dataset), id(observation_dataset), first_day_index)
    cached_comparison = _DRIFTER_COMPARISON_CACHE.get(cache_key)
    if cached_comparison is not None:
        return cached_comparison
    reference_trajectories = class4_drifter_reference_trajectories(
        challenger_dataset=challenger_dataset,
        observation_dataset=observation_dataset,
        first_day_index=first_day_index,
    )
    challenger_trajectories = class4_drifter_challenger_trajectories(
        challenger_dataset=challenger_dataset,
        reference_trajectories=reference_trajectories,
        first_day_index=first_day_index,
    )
    comparison = (challenger_trajectories, reference_trajectories)
    _DRIFTER_COMPARISON_CACHE[cache_key] = comparison
    return comparison


def deviation_of_lagrangian_trajectories_compared_to_class4_observations(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    challenger_dataset = rename_dataset_with_standard_names(challenger_dataset)
    mean_distances = []
    matched_counts = []

    for first_day_index in range(challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]):
        try:
            challenger_trajectories, reference_trajectories = class4_drifter_trajectory_comparison(
                challenger_dataset=challenger_dataset,
                observation_dataset=observation_dataset,
                first_day_index=first_day_index,
            )
        except Class4DrifterTrajectoryUnavailableError:
            continue
        particle_distances = class4_drifter_trajectory_distance_km(
            challenger_particles=challenger_trajectories,
            reference_particles=reference_trajectories,
        )
        mean_distances.append(numpy.nanmean(particle_distances, axis=0))
        matched_counts.append(numpy.sum(numpy.isfinite(particle_distances), axis=0))

    if not mean_distances:
        raise Class4DrifterTrajectoryUnavailableError("No linked Class-4 drifter trajectories were reconstructed.")

    lead_day_count = reported_class4_drifter_time_count(
        challenger_dataset,
        max(len(distances) for distances in mean_distances),
    )
    if lead_day_count < 1:
        raise Class4DrifterTrajectoryUnavailableError("No linked Class-4 drifter trajectories were reconstructed.")

    distance_rows = numpy.full((len(mean_distances), lead_day_count), numpy.nan)
    count_rows = numpy.zeros((len(matched_counts), lead_day_count))
    for run_index, (distances, counts) in enumerate(zip(mean_distances, matched_counts, strict=True)):
        run_lead_day_count = min(lead_day_count, len(distances))
        distance_rows[run_index, :run_lead_day_count] = distances[:run_lead_day_count]
        count_rows[run_index, :run_lead_day_count] = counts[:run_lead_day_count]

    mean_distance_row = numpy.nanmean(distance_rows, axis=0)
    matched_count_row = numpy.sum(count_rows, axis=0)
    mean_distance_row[matched_count_row < _minimum_reliable_matched_drifter_count(matched_count_row)] = numpy.nan

    return pandas.DataFrame(
        data=numpy.vstack(
            [
                mean_distance_row,
                matched_count_row,
            ]
        ),
        index=[
            DRIFTER_TRAJECTORY_DEVIATION_ROW_LABEL,
            MATCHED_DRIFTER_COUNT_ROW_LABEL,
        ],
        columns=_drifter_lead_day_labels(lead_day_count),
    )
