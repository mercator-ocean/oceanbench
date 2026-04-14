# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension

REGION_NAME_ATTR = "oceanbench_region_name"
REGION_MINIMUM_LATITUDE_ATTR = "oceanbench_region_minimum_latitude"
REGION_MAXIMUM_LATITUDE_ATTR = "oceanbench_region_maximum_latitude"
REGION_MINIMUM_LONGITUDE_ATTR = "oceanbench_region_minimum_longitude"
REGION_MAXIMUM_LONGITUDE_ATTR = "oceanbench_region_maximum_longitude"


@dataclass(frozen=True)
class RegionDefinition:
    name: str
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float

    def to_attrs(self) -> dict[str, float | str]:
        return {
            REGION_NAME_ATTR: self.name,
            REGION_MINIMUM_LATITUDE_ATTR: float(self.minimum_latitude),
            REGION_MAXIMUM_LATITUDE_ATTR: float(self.maximum_latitude),
            REGION_MINIMUM_LONGITUDE_ATTR: float(self.minimum_longitude),
            REGION_MAXIMUM_LONGITUDE_ATTR: float(self.maximum_longitude),
        }


NORTH_ATLANTIC = RegionDefinition(
    name="North Atlantic",
    minimum_latitude=0.0,
    maximum_latitude=75.0,
    minimum_longitude=-100.0,
    maximum_longitude=20.0,
)


def _wrap_longitudes(longitudes: numpy.ndarray) -> numpy.ndarray:
    wrapped_longitudes = ((numpy.asarray(longitudes, dtype=float) + 180.0) % 360.0) - 180.0
    positive_dateline_mask = (wrapped_longitudes == -180.0) & (numpy.asarray(longitudes, dtype=float) > 0)
    wrapped_longitudes[positive_dateline_mask] = 180.0
    return wrapped_longitudes


def region_from_dataset(dataset: xarray.Dataset) -> RegionDefinition | None:
    if REGION_NAME_ATTR not in dataset.attrs:
        return None
    return RegionDefinition(
        name=str(dataset.attrs[REGION_NAME_ATTR]),
        minimum_latitude=float(dataset.attrs[REGION_MINIMUM_LATITUDE_ATTR]),
        maximum_latitude=float(dataset.attrs[REGION_MAXIMUM_LATITUDE_ATTR]),
        minimum_longitude=float(dataset.attrs[REGION_MINIMUM_LONGITUDE_ATTR]),
        maximum_longitude=float(dataset.attrs[REGION_MAXIMUM_LONGITUDE_ATTR]),
    )


def _point_mask(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    region: RegionDefinition,
) -> numpy.ndarray:
    wrapped_longitudes = _wrap_longitudes(longitudes)
    latitude_mask = (latitudes >= region.minimum_latitude) & (latitudes <= region.maximum_latitude)
    if region.minimum_longitude <= region.maximum_longitude:
        longitude_mask = (wrapped_longitudes >= region.minimum_longitude) & (
            wrapped_longitudes <= region.maximum_longitude
        )
    else:
        longitude_mask = (wrapped_longitudes >= region.minimum_longitude) | (
            wrapped_longitudes <= region.maximum_longitude
        )
    return latitude_mask & longitude_mask


def subset_dataset_to_region(
    dataset: xarray.Dataset,
    region: RegionDefinition,
) -> xarray.Dataset:
    standard_dataset = rename_dataset_with_standard_names(dataset)
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    latitude_values = standard_dataset[latitude_key].values
    longitude_values = standard_dataset[longitude_key].values
    latitude_mask = (latitude_values >= region.minimum_latitude) & (latitude_values <= region.maximum_latitude)
    longitude_mask = _point_mask(
        latitudes=numpy.zeros_like(longitude_values, dtype=float),
        longitudes=longitude_values,
        region=RegionDefinition(
            name=region.name,
            minimum_latitude=-90.0,
            maximum_latitude=90.0,
            minimum_longitude=region.minimum_longitude,
            maximum_longitude=region.maximum_longitude,
        ),
    )
    subset = standard_dataset.isel(
        {
            latitude_key: latitude_mask,
            longitude_key: longitude_mask,
        }
    )
    return subset.assign_attrs({**subset.attrs, **region.to_attrs()})


def subset_dataset_from_challenger_region(
    dataset: xarray.Dataset,
    challenger_dataset: xarray.Dataset,
) -> xarray.Dataset:
    region = region_from_dataset(challenger_dataset)
    if region is None:
        return dataset
    return subset_dataset_to_region(dataset, region)


def filter_observation_dataset_to_region(
    observation_dataset: xarray.Dataset,
    region: RegionDefinition,
) -> xarray.Dataset:
    standard_dataset = rename_dataset_with_standard_names(observation_dataset)
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    observation_dimension = standard_dataset[latitude_key].dims[0]
    point_mask = _point_mask(
        latitudes=standard_dataset[latitude_key].values,
        longitudes=standard_dataset[longitude_key].values,
        region=region,
    )
    filtered = standard_dataset.isel({observation_dimension: point_mask})
    return filtered.assign_attrs({**filtered.attrs, **region.to_attrs()})


def filter_observation_dataset_from_challenger_region(
    observation_dataset: xarray.Dataset,
    challenger_dataset: xarray.Dataset,
) -> xarray.Dataset:
    region = region_from_dataset(challenger_dataset)
    if region is None:
        return observation_dataset
    return filter_observation_dataset_to_region(observation_dataset, region)


def filter_dataframe_to_region(
    dataframe: pandas.DataFrame,
    region: RegionDefinition,
    latitude_column: str = Dimension.LATITUDE.key(),
    longitude_column: str = Dimension.LONGITUDE.key(),
) -> pandas.DataFrame:
    if dataframe.empty:
        return dataframe
    point_mask = _point_mask(
        latitudes=dataframe[latitude_column].to_numpy(dtype=float),
        longitudes=dataframe[longitude_column].to_numpy(dtype=float),
        region=region,
    )
    return dataframe.loc[point_mask].reset_index(drop=True)


def filter_dataframe_from_challenger_region(
    dataframe: pandas.DataFrame,
    challenger_dataset: xarray.Dataset,
    latitude_column: str = Dimension.LATITUDE.key(),
    longitude_column: str = Dimension.LONGITUDE.key(),
) -> pandas.DataFrame:
    region = region_from_dataset(challenger_dataset)
    if region is None:
        return dataframe
    return filter_dataframe_to_region(
        dataframe,
        region=region,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
    )


def filter_trajectory_dataframe_by_initial_region(
    dataframe: pandas.DataFrame,
    region: RegionDefinition,
    track_id_column: str = "track_id",
    latitude_column: str = Dimension.LATITUDE.key(),
    longitude_column: str = Dimension.LONGITUDE.key(),
    time_column: str = Dimension.TIME.key(),
) -> pandas.DataFrame:
    if dataframe.empty:
        return dataframe
    sorted_dataframe = dataframe.sort_values([track_id_column, time_column]).reset_index(drop=True)
    initial_rows = sorted_dataframe.groupby(track_id_column, sort=False).first().reset_index()
    initial_mask = _point_mask(
        latitudes=initial_rows[latitude_column].to_numpy(dtype=float),
        longitudes=initial_rows[longitude_column].to_numpy(dtype=float),
        region=region,
    )
    selected_track_ids = set(initial_rows.loc[initial_mask, track_id_column].tolist())
    return sorted_dataframe.loc[sorted_dataframe[track_id_column].isin(selected_track_ids)].reset_index(drop=True)


def filter_trajectory_dataframe_from_challenger_region(
    dataframe: pandas.DataFrame,
    challenger_dataset: xarray.Dataset,
    track_id_column: str = "track_id",
    latitude_column: str = Dimension.LATITUDE.key(),
    longitude_column: str = Dimension.LONGITUDE.key(),
    time_column: str = Dimension.TIME.key(),
) -> pandas.DataFrame:
    region = region_from_dataset(challenger_dataset)
    if region is None:
        return dataframe
    return filter_trajectory_dataframe_by_initial_region(
        dataframe,
        region=region,
        track_id_column=track_id_column,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        time_column=time_column,
    )
