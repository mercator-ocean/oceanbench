# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import timedelta

import numpy
import pandas
from scipy.spatial import cKDTree
import xarray

from oceanbench.core import lagrangian_trajectory
from oceanbench.core import regions
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

DRIFTER_DEPTH_METERS = 15.0
MATCH_CANDIDATE_COUNT = 5
MATCH_DISTANCE_PER_HOUR_KM = 25.0
_DRIFTER_COMPARISON_CACHE: dict[tuple[int, int], tuple[xarray.Dataset, xarray.Dataset]] = {}


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
    latitude_start: numpy.ndarray,
    longitude_start: numpy.ndarray,
    latitude_end: numpy.ndarray,
    longitude_end: numpy.ndarray,
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
        + numpy.cos(latitude_start_radians)
        * numpy.cos(latitude_end_radians)
        * numpy.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * numpy.arcsin(numpy.sqrt(haversine_term))


def _predict_next_positions(
    active_tracks: pandas.DataFrame,
    time_step_hours: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    time_step_seconds = time_step_hours * 3600.0
    predicted_latitudes = active_tracks["latitude"].to_numpy() + (
        active_tracks["northward_sea_water_velocity"].to_numpy() * time_step_seconds / 111000.0
    )
    cosine_latitudes = numpy.cos(numpy.deg2rad(numpy.clip(active_tracks["latitude"].to_numpy(), -89.9, 89.9)))
    predicted_longitudes = active_tracks["longitude"].to_numpy() + (
        active_tracks["eastward_sea_water_velocity"].to_numpy()
        * time_step_seconds
        / (111000.0 * numpy.maximum(cosine_latitudes, 1e-6))
    )
    return predicted_latitudes, predicted_longitudes


def _drifter_observations_dataframe(
    observation_dataset: xarray.Dataset,
    first_day_datetime: numpy.datetime64,
) -> pandas.DataFrame:
    time_key = Dimension.TIME.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    observation_dimension_key = next(iter(observation_dataset[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].dims))
    first_day_mask = observation_dataset[first_day_key] == numpy.datetime64(first_day_datetime)
    first_day_observations = observation_dataset.isel({observation_dimension_key: first_day_mask})
    valid_mask = (
        first_day_observations[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].notnull()
        & first_day_observations[Variable.NORTHWARD_SEA_WATER_VELOCITY.key()].notnull()
        & numpy.isclose(first_day_observations[Dimension.DEPTH.key()], DRIFTER_DEPTH_METERS)
    ).compute()
    first_day_observations = first_day_observations.isel({observation_dimension_key: valid_mask})
    observations_dataframe = pandas.DataFrame(
        {
            time_key: pandas.to_datetime(first_day_observations[time_key].values),
            Dimension.LATITUDE.key(): first_day_observations[Dimension.LATITUDE.key()].values,
            Dimension.LONGITUDE.key(): first_day_observations[Dimension.LONGITUDE.key()].values,
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): first_day_observations[
                Variable.EASTWARD_SEA_WATER_VELOCITY.key()
            ].values,
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): first_day_observations[
                Variable.NORTHWARD_SEA_WATER_VELOCITY.key()
            ].values,
        }
    )
    return observations_dataframe.sort_values(time_key).reset_index(drop=True)


def _hourly_drifter_frames(observations_dataframe: pandas.DataFrame) -> tuple[list[pandas.Timestamp], dict[pandas.Timestamp, pandas.DataFrame]]:
    grouped_frames = {
        timestamp: group.rename(
            columns={
                Dimension.LATITUDE.key(): "latitude",
                Dimension.LONGITUDE.key(): "longitude",
                Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "eastward_sea_water_velocity",
                Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "northward_sea_water_velocity",
            }
        )
        .loc[
            :,
            [
                "latitude",
                "longitude",
                "eastward_sea_water_velocity",
                "northward_sea_water_velocity",
            ],
        ]
        .reset_index(drop=True)
        for timestamp, group in observations_dataframe.groupby(Dimension.TIME.key(), sort=True)
    }
    timestamps = sorted(grouped_frames)
    return timestamps, grouped_frames


def _matched_track_indices(
    active_tracks: pandas.DataFrame,
    next_frame: pandas.DataFrame,
    time_step_hours: float,
) -> list[tuple[int, int]]:
    predicted_latitudes, predicted_longitudes = _predict_next_positions(active_tracks, time_step_hours=time_step_hours)
    next_track_tree = cKDTree(
        _to_cartesian_coordinates(
            latitudes=next_frame["latitude"].to_numpy(),
            longitudes=next_frame["longitude"].to_numpy(),
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
    match_distance_limit_km = MATCH_DISTANCE_PER_HOUR_KM * time_step_hours
    for active_track_index, candidate_indices in enumerate(nearest_indices):
        for candidate_index in numpy.atleast_1d(candidate_indices):
            distance = float(
                _haversine_distance_km(
                    predicted_latitudes[active_track_index],
                    predicted_longitudes[active_track_index],
                    next_frame.iloc[int(candidate_index)]["latitude"],
                    next_frame.iloc[int(candidate_index)]["longitude"],
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
        raise ValueError("No Class-4 drifter observations were found for the selected forecast run.")

    first_timestamp = timestamps[0]
    active_tracks = grouped_frames[first_timestamp].copy()
    active_tracks["track_id"] = numpy.arange(len(active_tracks))
    trajectory_records = [
        {
            "track_id": int(track_id),
            "time": first_timestamp,
            "latitude": latitude,
            "longitude": longitude,
        }
        for track_id, latitude, longitude in zip(
            active_tracks["track_id"].to_numpy(),
            active_tracks["latitude"].to_numpy(),
            active_tracks["longitude"].to_numpy(),
            strict=False,
        )
    ]

    previous_timestamp = first_timestamp
    for next_timestamp in timestamps[1:]:
        next_frame = grouped_frames[next_timestamp]
        time_step_hours = (next_timestamp - previous_timestamp) / pandas.Timedelta(hours=1)
        matched_indices = _matched_track_indices(active_tracks, next_frame, time_step_hours=float(time_step_hours))
        if not matched_indices:
            break

        matched_rows = []
        for active_track_index, next_index in matched_indices:
            next_row = next_frame.iloc[next_index]
            track_id = int(active_tracks.iloc[active_track_index]["track_id"])
            matched_rows.append(
                {
                    "track_id": track_id,
                    "latitude": next_row["latitude"],
                    "longitude": next_row["longitude"],
                    "eastward_sea_water_velocity": next_row["eastward_sea_water_velocity"],
                    "northward_sea_water_velocity": next_row["northward_sea_water_velocity"],
                }
            )
            trajectory_records.append(
                {
                    "track_id": track_id,
                    "time": next_timestamp,
                    "latitude": next_row["latitude"],
                    "longitude": next_row["longitude"],
                }
            )
        active_tracks = pandas.DataFrame(matched_rows)
        previous_timestamp = next_timestamp

    return pandas.DataFrame(trajectory_records)


def class4_drifter_reference_trajectories(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    first_day_datetime = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values[first_day_index]
    linked_trajectories = _link_hourly_drifter_trajectories(
        observation_dataset=observation_dataset,
        first_day_datetime=first_day_datetime,
    )
    linked_trajectories = regions.filter_trajectory_dataframe_from_challenger_region(
        linked_trajectories,
        challenger_dataset=challenger_dataset,
        latitude_column="latitude",
        longitude_column="longitude",
    )
    if linked_trajectories.empty:
        raise ValueError("No linked Class-4 drifter trajectories were reconstructed.")

    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    initial_timestamp = linked_trajectories["time"].min()
    output_timestamps = [initial_timestamp + timedelta(days=offset) for offset in range(lead_days_count)]
    track_ids = numpy.arange(linked_trajectories["track_id"].max() + 1)
    latitude_values = numpy.full((len(track_ids), len(output_timestamps)), numpy.nan)
    longitude_values = numpy.full((len(track_ids), len(output_timestamps)), numpy.nan)

    trajectory_lookup = linked_trajectories.set_index(["track_id", "time"]).sort_index()
    for time_index, output_timestamp in enumerate(output_timestamps):
        if output_timestamp not in linked_trajectories["time"].values:
            continue
        time_slice = trajectory_lookup.xs(output_timestamp, level="time", drop_level=False)
        for track_id, row in time_slice.groupby(level="track_id"):
            latitude_values[int(track_id), time_index] = float(row["latitude"].iloc[0])
            longitude_values[int(track_id), time_index] = float(row["longitude"].iloc[0])

    initial_latitudes = latitude_values[:, 0]
    initial_longitudes = longitude_values[:, 0]
    return xarray.Dataset(
        {
            "lat": (["particle", "time"], latitude_values),
            "lon": (["particle", "time"], longitude_values),
        },
        coords={
            "particle": track_ids,
            "time": pandas.to_datetime(output_timestamps),
            "lat0": ("particle", initial_latitudes),
            "lon0": ("particle", initial_longitudes),
        },
    )


def class4_drifter_challenger_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_trajectories: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    challenger_standard_dataset = lagrangian_trajectory._harmonise_dataset(challenger_dataset)
    challenger_run_dataset = lagrangian_trajectory._split_dataset(challenger_standard_dataset)[first_day_index]
    challenger_particles = lagrangian_trajectory._get_particle_dataset(
        dataset=challenger_run_dataset.isel({Dimension.DEPTH.key(): 0}),
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
    latitude_reference_set_radians = numpy.deg2rad(reference_particles["lat"].values)
    delta_latitude = (challenger_particles["lat"].values - reference_particles["lat"].values) * 111.0
    delta_longitude = (
        (challenger_particles["lon"].values - reference_particles["lon"].values)
        * 111.0
        * numpy.cos(latitude_reference_set_radians)
    )
    return numpy.sqrt(delta_latitude**2 + delta_longitude**2)


def class4_drifter_trajectory_comparison(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> tuple[xarray.Dataset, xarray.Dataset]:
    cache_key = (id(challenger_dataset), first_day_index)
    if cache_key in _DRIFTER_COMPARISON_CACHE:
        return _DRIFTER_COMPARISON_CACHE[cache_key]
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
    _DRIFTER_COMPARISON_CACHE[cache_key] = (challenger_trajectories, reference_trajectories)
    return _DRIFTER_COMPARISON_CACHE[cache_key]


def deviation_of_lagrangian_trajectories_compared_to_class4_observations(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    mean_distances = []
    matched_counts = []

    for first_day_index in range(challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]):
        challenger_trajectories, reference_trajectories = class4_drifter_trajectory_comparison(
            challenger_dataset=challenger_dataset,
            observation_dataset=observation_dataset,
            first_day_index=first_day_index,
        )
        particle_distances = class4_drifter_trajectory_distance_km(
            challenger_particles=challenger_trajectories,
            reference_particles=reference_trajectories,
        )
        mean_distances.append(numpy.nanmean(particle_distances, axis=0))
        matched_counts.append(numpy.sum(numpy.isfinite(particle_distances), axis=0))

    lead_day_count = len(mean_distances[0])
    return pandas.DataFrame(
        data=numpy.vstack(
            [
                numpy.nanmean(mean_distances, axis=0),
                numpy.sum(matched_counts, axis=0),
            ]
        ),
        index=[
            "Class-4 drifter trajectory deviation mean (km)",
            "Class-4 matched drifter count",
        ],
        columns=lead_day_labels(1, lead_day_count),
    )
