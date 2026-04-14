# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.measure import find_contours, regionprops
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

CYCLONE = "cyclone"
ANTICYCLONE = "anticyclone"
POLARITY_ORDER = [CYCLONE, ANTICYCLONE]
POLARITY_LABELS = {
    CYCLONE: "Cyclones",
    ANTICYCLONE: "Anticyclones",
}

DEFAULT_BACKGROUND_SIGMA_GRID = 12.0
DEFAULT_DETECTION_SIGMA_GRID = 1.5
DEFAULT_MIN_DISTANCE_GRID = 8
DEFAULT_AMPLITUDE_THRESHOLD_METERS = 0.04
DEFAULT_MAX_ABS_LATITUDE_DEGREES = 70.0
DEFAULT_MATCH_DISTANCE_KM = 200.0
DEFAULT_CONTOUR_LEVEL_STEP_METERS = 0.01
DEFAULT_MIN_CONTOUR_PIXEL_COUNT = 16
DEFAULT_MAX_CONTOUR_PIXEL_COUNT = 6000
DEFAULT_MIN_CONTOUR_CONVEXITY = 0.75

LEAD_DAY_COLUMN = "lead_day"
LATITUDE_COLUMN = Dimension.LATITUDE.key()
LONGITUDE_COLUMN = Dimension.LONGITUDE.key()
POLARITY_COLUMN = "polarity"
AMPLITUDE_COLUMN = "sea_surface_height_anomaly"
SEA_SURFACE_HEIGHT_COLUMN = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
DISTANCE_COLUMN = "distance_km"
CONTOUR_LEVEL_COLUMN = "contour_level"
CONTOUR_PIXEL_COUNT_COLUMN = "contour_pixel_count"
CONTOUR_CONVEXITY_COLUMN = "contour_convexity"
CONTOUR_LATITUDES_COLUMN = "contour_latitudes"
CONTOUR_LONGITUDES_COLUMN = "contour_longitudes"


def _standard_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _surface_ssh_field(
    dataset: xarray.Dataset,
    first_day_index: int,
    lead_day_index: int,
) -> xarray.DataArray:
    standard_dataset = _standard_dataset(dataset)
    field = standard_dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()]
    if Dimension.FIRST_DAY_DATETIME.key() in field.dims:
        field = field.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index})
    if Dimension.LEAD_DAY_INDEX.key() in field.dims:
        field = field.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_index})
    elif Dimension.TIME.key() in field.dims:
        field = field.isel({Dimension.TIME.key(): lead_day_index})
    return field.compute()


def _lead_day_indices(dataset: xarray.Dataset, lead_day_indices: list[int] | None = None) -> list[int]:
    standard_dataset = _standard_dataset(dataset)
    if Dimension.LEAD_DAY_INDEX.key() in standard_dataset.dims:
        available_count = standard_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    elif Dimension.TIME.key() in standard_dataset.dims:
        available_count = standard_dataset.sizes[Dimension.TIME.key()]
    else:
        return [0]
    if lead_day_indices is None:
        return list(range(available_count))
    return [lead_day_index for lead_day_index in lead_day_indices if 0 <= lead_day_index < available_count]


def _gaussian_filter_with_mask(values: numpy.ndarray, sigma: float) -> numpy.ndarray:
    valid_mask = numpy.isfinite(values)
    if not numpy.any(valid_mask):
        return numpy.full_like(values, numpy.nan, dtype=float)
    filled_values = numpy.where(valid_mask, values, 0.0)
    weights = valid_mask.astype(float)
    filtered_values = gaussian_filter(filled_values, sigma=sigma, mode=("nearest", "wrap"))
    filtered_weights = gaussian_filter(weights, sigma=sigma, mode=("nearest", "wrap"))
    with numpy.errstate(invalid="ignore", divide="ignore"):
        smoothed_values = filtered_values / filtered_weights
    smoothed_values[filtered_weights <= 0] = numpy.nan
    return smoothed_values


def _ssh_anomaly(
    field: xarray.DataArray,
    background_sigma_grid: float,
    detection_sigma_grid: float,
) -> numpy.ndarray:
    field_values = numpy.asarray(field.values, dtype=float)
    background_values = _gaussian_filter_with_mask(field_values, sigma=background_sigma_grid)
    anomaly_values = field_values - background_values
    return _gaussian_filter_with_mask(anomaly_values, sigma=detection_sigma_grid)


def _valid_detection_mask(field: xarray.DataArray, max_abs_latitude_degrees: float) -> numpy.ndarray:
    latitude_values = field[LATITUDE_COLUMN].values
    longitude_count = field.sizes[LONGITUDE_COLUMN]
    finite_mask = numpy.isfinite(field.values)
    latitude_mask = numpy.abs(latitude_values) <= max_abs_latitude_degrees
    return finite_mask & latitude_mask[:, None] & numpy.ones((1, longitude_count), dtype=bool)


def _detect_polarity_peaks(
    anomaly_values: numpy.ndarray,
    valid_mask: numpy.ndarray,
    polarity: str,
    min_distance_grid: int,
    amplitude_threshold_meters: float,
) -> numpy.ndarray:
    masked_values = numpy.where(valid_mask, anomaly_values, numpy.nan)
    if polarity == ANTICYCLONE:
        image_values = numpy.where(numpy.isfinite(masked_values), masked_values, -numpy.inf)
    else:
        image_values = numpy.where(numpy.isfinite(masked_values), -masked_values, -numpy.inf)
    coordinates = peak_local_max(
        image_values,
        min_distance=min_distance_grid,
        threshold_abs=amplitude_threshold_meters,
        exclude_border=False,
    )
    if coordinates.size == 0:
        return coordinates
    amplitude_values = masked_values[coordinates[:, 0], coordinates[:, 1]]
    if polarity == ANTICYCLONE:
        polarity_mask = amplitude_values >= amplitude_threshold_meters
    else:
        polarity_mask = amplitude_values <= -amplitude_threshold_meters
    return coordinates[polarity_mask]


def detect_mesoscale_eddies(
    dataset: xarray.Dataset,
    first_day_index: int = 0,
    lead_day_indices: list[int] | None = None,
    background_sigma_grid: float = DEFAULT_BACKGROUND_SIGMA_GRID,
    detection_sigma_grid: float = DEFAULT_DETECTION_SIGMA_GRID,
    min_distance_grid: int = DEFAULT_MIN_DISTANCE_GRID,
    amplitude_threshold_meters: float = DEFAULT_AMPLITUDE_THRESHOLD_METERS,
    max_abs_latitude_degrees: float = DEFAULT_MAX_ABS_LATITUDE_DEGREES,
) -> pandas.DataFrame:
    detection_rows: list[dict[str, float | int | str]] = []
    for lead_day_index in _lead_day_indices(dataset, lead_day_indices):
        field = _surface_ssh_field(dataset, first_day_index=first_day_index, lead_day_index=lead_day_index)
        anomaly_values = _ssh_anomaly(
            field,
            background_sigma_grid=background_sigma_grid,
            detection_sigma_grid=detection_sigma_grid,
        )
        valid_mask = _valid_detection_mask(field, max_abs_latitude_degrees=max_abs_latitude_degrees)
        latitude_values = field[LATITUDE_COLUMN].values
        longitude_values = field[LONGITUDE_COLUMN].values
        for polarity in POLARITY_ORDER:
            peak_coordinates = _detect_polarity_peaks(
                anomaly_values=anomaly_values,
                valid_mask=valid_mask,
                polarity=polarity,
                min_distance_grid=min_distance_grid,
                amplitude_threshold_meters=amplitude_threshold_meters,
            )
            for latitude_index, longitude_index in peak_coordinates:
                detection_rows.append(
                    {
                        LEAD_DAY_COLUMN: lead_day_index,
                        LATITUDE_COLUMN: float(latitude_values[latitude_index]),
                        LONGITUDE_COLUMN: float(longitude_values[longitude_index]),
                        POLARITY_COLUMN: polarity,
                        AMPLITUDE_COLUMN: float(anomaly_values[latitude_index, longitude_index]),
                        SEA_SURFACE_HEIGHT_COLUMN: float(field.values[latitude_index, longitude_index]),
                    }
                )
    return pandas.DataFrame(
        detection_rows,
        columns=[
            LEAD_DAY_COLUMN,
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN,
            POLARITY_COLUMN,
            AMPLITUDE_COLUMN,
            SEA_SURFACE_HEIGHT_COLUMN,
        ],
    )


def _haversine_distance_km(
    latitude_a: numpy.ndarray,
    longitude_a: numpy.ndarray,
    latitude_b: numpy.ndarray,
    longitude_b: numpy.ndarray,
) -> numpy.ndarray:
    earth_radius_km = 6371.0
    latitude_a_rad = numpy.deg2rad(latitude_a)[:, None]
    longitude_a_rad = numpy.deg2rad(longitude_a)[:, None]
    latitude_b_rad = numpy.deg2rad(latitude_b)[None, :]
    longitude_b_rad = numpy.deg2rad(longitude_b)[None, :]

    dlatitude = latitude_b_rad - latitude_a_rad
    dlongitude = longitude_b_rad - longitude_a_rad
    dlongitude = (dlongitude + numpy.pi) % (2.0 * numpy.pi) - numpy.pi

    haversine = (
        numpy.sin(dlatitude / 2.0) ** 2
        + numpy.cos(latitude_a_rad) * numpy.cos(latitude_b_rad) * numpy.sin(dlongitude / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * numpy.arcsin(numpy.sqrt(haversine))


def match_mesoscale_eddies(
    challenger_detections: pandas.DataFrame,
    reference_detections: pandas.DataFrame,
    max_match_distance_km: float = DEFAULT_MATCH_DISTANCE_KM,
) -> pandas.DataFrame:
    match_rows: list[dict[str, float | int | str]] = []
    for lead_day_index in sorted(set(challenger_detections[LEAD_DAY_COLUMN]).union(reference_detections[LEAD_DAY_COLUMN])):
        for polarity in POLARITY_ORDER:
            challenger_subset = challenger_detections.loc[
                (challenger_detections[LEAD_DAY_COLUMN] == lead_day_index)
                & (challenger_detections[POLARITY_COLUMN] == polarity)
            ]
            reference_subset = reference_detections.loc[
                (reference_detections[LEAD_DAY_COLUMN] == lead_day_index)
                & (reference_detections[POLARITY_COLUMN] == polarity)
            ]
            if challenger_subset.empty or reference_subset.empty:
                continue
            distance_matrix_km = _haversine_distance_km(
                challenger_subset[LATITUDE_COLUMN].to_numpy(),
                challenger_subset[LONGITUDE_COLUMN].to_numpy(),
                reference_subset[LATITUDE_COLUMN].to_numpy(),
                reference_subset[LONGITUDE_COLUMN].to_numpy(),
            )
            cost_matrix = distance_matrix_km.copy()
            cost_matrix[cost_matrix > max_match_distance_km] = max_match_distance_km + 1.0e6
            challenger_indices, reference_indices = linear_sum_assignment(cost_matrix)
            for challenger_position, reference_position in zip(challenger_indices, reference_indices, strict=False):
                distance_km = distance_matrix_km[challenger_position, reference_position]
                if distance_km > max_match_distance_km:
                    continue
                challenger_row = challenger_subset.iloc[challenger_position]
                reference_row = reference_subset.iloc[reference_position]
                match_rows.append(
                    {
                        LEAD_DAY_COLUMN: lead_day_index,
                        POLARITY_COLUMN: polarity,
                        "challenger_detection_index": int(challenger_subset.index[challenger_position]),
                        "reference_detection_index": int(reference_subset.index[reference_position]),
                        DISTANCE_COLUMN: float(distance_km),
                        "challenger_latitude": float(challenger_row[LATITUDE_COLUMN]),
                        "challenger_longitude": float(challenger_row[LONGITUDE_COLUMN]),
                        "reference_latitude": float(reference_row[LATITUDE_COLUMN]),
                        "reference_longitude": float(reference_row[LONGITUDE_COLUMN]),
                    }
                )
    return pandas.DataFrame(match_rows)


def mesoscale_eddy_summary_from_detections(
    challenger_detections: pandas.DataFrame,
    reference_detections: pandas.DataFrame,
    matches: pandas.DataFrame,
    lead_day_count: int,
    challenger_name: str = "GLONET",
    reference_name: str = "GLORYS",
) -> pandas.DataFrame:
    summary_rows: dict[str, numpy.ndarray] = {}
    for polarity in POLARITY_ORDER:
        polarity_label = POLARITY_LABELS[polarity]
        challenger_counts = []
        reference_counts = []
        hit_counts = []
        miss_counts = []
        for lead_day_index in range(lead_day_count):
            challenger_count = int(
                challenger_detections.loc[
                    (challenger_detections[LEAD_DAY_COLUMN] == lead_day_index)
                    & (challenger_detections[POLARITY_COLUMN] == polarity)
                ].shape[0]
            )
            reference_count = int(
                reference_detections.loc[
                    (reference_detections[LEAD_DAY_COLUMN] == lead_day_index)
                    & (reference_detections[POLARITY_COLUMN] == polarity)
                ].shape[0]
            )
            hit_count = int(
                matches.loc[
                    (matches[LEAD_DAY_COLUMN] == lead_day_index) & (matches[POLARITY_COLUMN] == polarity)
                ].shape[0]
            )
            challenger_counts.append(challenger_count)
            reference_counts.append(reference_count)
            hit_counts.append(hit_count)
            miss_counts.append(reference_count - hit_count)

        summary_rows[f"{challenger_name} {polarity_label.lower()}"] = numpy.asarray(challenger_counts, dtype=int)
        summary_rows[f"{reference_name} {polarity_label.lower()}"] = numpy.asarray(reference_counts, dtype=int)
        summary_rows[f"{polarity_label[:-1]} hits"] = numpy.asarray(hit_counts, dtype=int)
        summary_rows[f"{polarity_label[:-1]} misses"] = numpy.asarray(miss_counts, dtype=int)

    return pandas.DataFrame(summary_rows, index=lead_day_labels(1, lead_day_count)).T


def mesoscale_eddy_summary(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    first_day_index: int = 0,
    lead_day_indices: list[int] | None = None,
    background_sigma_grid: float = DEFAULT_BACKGROUND_SIGMA_GRID,
    detection_sigma_grid: float = DEFAULT_DETECTION_SIGMA_GRID,
    min_distance_grid: int = DEFAULT_MIN_DISTANCE_GRID,
    amplitude_threshold_meters: float = DEFAULT_AMPLITUDE_THRESHOLD_METERS,
    max_abs_latitude_degrees: float = DEFAULT_MAX_ABS_LATITUDE_DEGREES,
    max_match_distance_km: float = DEFAULT_MATCH_DISTANCE_KM,
    contour_level_step_meters: float = DEFAULT_CONTOUR_LEVEL_STEP_METERS,
    min_contour_pixel_count: int = DEFAULT_MIN_CONTOUR_PIXEL_COUNT,
    max_contour_pixel_count: int = DEFAULT_MAX_CONTOUR_PIXEL_COUNT,
    min_contour_convexity: float = DEFAULT_MIN_CONTOUR_CONVEXITY,
    challenger_name: str = "GLONET",
    reference_name: str = "GLORYS",
) -> pandas.DataFrame:
    challenger_detections = detect_mesoscale_eddies(
        challenger_dataset,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        background_sigma_grid=background_sigma_grid,
        detection_sigma_grid=detection_sigma_grid,
        min_distance_grid=min_distance_grid,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
    )
    challenger_contours = mesoscale_eddy_contours_from_detections(
        challenger_detections,
        challenger_dataset,
        first_day_index=first_day_index,
        background_sigma_grid=background_sigma_grid,
        detection_sigma_grid=detection_sigma_grid,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
        contour_level_step_meters=contour_level_step_meters,
        min_contour_pixel_count=min_contour_pixel_count,
        max_contour_pixel_count=max_contour_pixel_count,
        min_contour_convexity=min_contour_convexity,
    )
    challenger_detections = filter_mesoscale_eddy_detections_by_contours(
        challenger_detections,
        challenger_contours,
    )
    reference_detections = detect_mesoscale_eddies(
        reference_dataset,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        background_sigma_grid=background_sigma_grid,
        detection_sigma_grid=detection_sigma_grid,
        min_distance_grid=min_distance_grid,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
    )
    reference_contours = mesoscale_eddy_contours_from_detections(
        reference_detections,
        reference_dataset,
        first_day_index=first_day_index,
        background_sigma_grid=background_sigma_grid,
        detection_sigma_grid=detection_sigma_grid,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
        contour_level_step_meters=contour_level_step_meters,
        min_contour_pixel_count=min_contour_pixel_count,
        max_contour_pixel_count=max_contour_pixel_count,
        min_contour_convexity=min_contour_convexity,
    )
    reference_detections = filter_mesoscale_eddy_detections_by_contours(
        reference_detections,
        reference_contours,
    )
    matches = match_mesoscale_eddies(
        challenger_detections,
        reference_detections,
        max_match_distance_km=max_match_distance_km,
    )
    lead_day_count = len(_lead_day_indices(challenger_dataset, lead_day_indices))
    return mesoscale_eddy_summary_from_detections(
        challenger_detections=challenger_detections,
        reference_detections=reference_detections,
        matches=matches,
        lead_day_count=lead_day_count,
        challenger_name=challenger_name,
        reference_name=reference_name,
    )


def _nearest_coordinate_indices(coordinates: numpy.ndarray, values: numpy.ndarray) -> numpy.ndarray:
    insertion_indices = numpy.searchsorted(coordinates, values)
    insertion_indices = numpy.clip(insertion_indices, 1, len(coordinates) - 1)
    left_values = coordinates[insertion_indices - 1]
    right_values = coordinates[insertion_indices]
    choose_left = numpy.abs(values - left_values) <= numpy.abs(right_values - values)
    return insertion_indices - choose_left.astype(int)


def _component_contour_info(
    labels: numpy.ndarray,
    component_label: int,
    latitude_values: numpy.ndarray,
    longitude_values: numpy.ndarray,
) -> dict[str, object] | None:
    component_positions = numpy.argwhere(labels == component_label)
    if component_positions.size == 0:
        return None
    latitude_min = max(int(component_positions[:, 0].min()) - 1, 0)
    latitude_max = min(int(component_positions[:, 0].max()) + 2, labels.shape[0])
    longitude_min = max(int(component_positions[:, 1].min()) - 1, 0)
    longitude_max = min(int(component_positions[:, 1].max()) + 2, labels.shape[1])

    component_mask = labels[latitude_min:latitude_max, longitude_min:longitude_max] == component_label
    properties = regionprops(component_mask.astype(numpy.uint8))
    if not properties:
        return None
    contours = find_contours(component_mask.astype(float), level=0.5)
    if not contours:
        return None

    contour = max(contours, key=len)
    contour_latitude_indices = contour[:, 0] + latitude_min
    contour_longitude_indices = contour[:, 1] + longitude_min
    contour_latitudes = numpy.interp(contour_latitude_indices, numpy.arange(len(latitude_values)), latitude_values)
    contour_longitudes = numpy.interp(contour_longitude_indices, numpy.arange(len(longitude_values)), longitude_values)

    region = properties[0]
    return {
        CONTOUR_LATITUDES_COLUMN: contour_latitudes,
        CONTOUR_LONGITUDES_COLUMN: contour_longitudes,
        CONTOUR_PIXEL_COUNT_COLUMN: int(region.area),
        CONTOUR_CONVEXITY_COLUMN: float(region.solidity),
    }


def mesoscale_eddy_contours_from_detections(
    detections: pandas.DataFrame,
    dataset: xarray.Dataset,
    first_day_index: int = 0,
    background_sigma_grid: float = DEFAULT_BACKGROUND_SIGMA_GRID,
    detection_sigma_grid: float = DEFAULT_DETECTION_SIGMA_GRID,
    amplitude_threshold_meters: float = DEFAULT_AMPLITUDE_THRESHOLD_METERS,
    max_abs_latitude_degrees: float = DEFAULT_MAX_ABS_LATITUDE_DEGREES,
    contour_level_step_meters: float = DEFAULT_CONTOUR_LEVEL_STEP_METERS,
    min_contour_pixel_count: int = DEFAULT_MIN_CONTOUR_PIXEL_COUNT,
    max_contour_pixel_count: int = DEFAULT_MAX_CONTOUR_PIXEL_COUNT,
    min_contour_convexity: float = DEFAULT_MIN_CONTOUR_CONVEXITY,
) -> pandas.DataFrame:
    if contour_level_step_meters <= 0:
        raise ValueError("contour_level_step_meters must be positive")

    contour_rows: list[dict[str, object]] = []
    lead_day_indices = sorted(detections[LEAD_DAY_COLUMN].unique()) if not detections.empty else []
    connectivity = numpy.ones((3, 3), dtype=int)

    for lead_day_index in lead_day_indices:
        anomaly_field = surface_ssh_anomaly_field(
            dataset,
            first_day_index=first_day_index,
            lead_day_index=int(lead_day_index),
            background_sigma_grid=background_sigma_grid,
            detection_sigma_grid=detection_sigma_grid,
            max_abs_latitude_degrees=max_abs_latitude_degrees,
        )
        anomaly_values = numpy.asarray(anomaly_field.values, dtype=float)
        latitude_values = anomaly_field[LATITUDE_COLUMN].values
        longitude_values = anomaly_field[LONGITUDE_COLUMN].values

        for polarity in POLARITY_ORDER:
            subset = detections.loc[
                (detections[LEAD_DAY_COLUMN] == lead_day_index) & (detections[POLARITY_COLUMN] == polarity)
            ].copy()
            if subset.empty:
                continue

            subset = subset.reset_index().rename(columns={"index": "detection_index"})
            center_latitudes = subset[LATITUDE_COLUMN].to_numpy(dtype=float)
            center_longitudes = subset[LONGITUDE_COLUMN].to_numpy(dtype=float)
            center_latitude_indices = _nearest_coordinate_indices(latitude_values, center_latitudes)
            center_longitude_indices = _nearest_coordinate_indices(longitude_values, center_longitudes)
            center_magnitudes = numpy.abs(subset[AMPLITUDE_COLUMN].to_numpy(dtype=float))
            max_center_magnitude = float(numpy.nanmax(center_magnitudes))
            if not numpy.isfinite(max_center_magnitude) or max_center_magnitude < amplitude_threshold_meters:
                continue

            level_values = numpy.arange(
                amplitude_threshold_meters,
                max_center_magnitude + contour_level_step_meters,
                contour_level_step_meters,
            )
            unresolved_mask = numpy.ones(len(subset), dtype=bool)

            for level_value in level_values:
                candidate_mask = unresolved_mask & (center_magnitudes >= level_value)
                if not numpy.any(candidate_mask):
                    continue

                if polarity == ANTICYCLONE:
                    component_mask = anomaly_values >= level_value
                else:
                    component_mask = anomaly_values <= -level_value
                component_mask = component_mask & numpy.isfinite(anomaly_values)
                labels, component_count = label(component_mask, structure=connectivity)
                if component_count == 0:
                    continue

                center_component_labels = labels[center_latitude_indices, center_longitude_indices]
                active_label_mask = (center_component_labels > 0) & (center_magnitudes >= level_value)
                if not numpy.any(active_label_mask):
                    continue
                active_label_counts = numpy.bincount(center_component_labels[active_label_mask])
                component_cache: dict[int, dict[str, object] | None] = {}

                for subset_index in numpy.where(candidate_mask)[0]:
                    component_label = int(center_component_labels[subset_index])
                    if component_label <= 0:
                        continue
                    if component_label >= len(active_label_counts) or active_label_counts[component_label] != 1:
                        continue

                    contour_info = component_cache.get(component_label)
                    if contour_info is None:
                        contour_info = _component_contour_info(
                            labels,
                            component_label=component_label,
                            latitude_values=latitude_values,
                            longitude_values=longitude_values,
                        )
                        component_cache[component_label] = contour_info
                    if contour_info is None:
                        continue
                    if contour_info[CONTOUR_PIXEL_COUNT_COLUMN] < min_contour_pixel_count:
                        continue
                    if contour_info[CONTOUR_PIXEL_COUNT_COLUMN] > max_contour_pixel_count:
                        continue
                    if contour_info[CONTOUR_CONVEXITY_COLUMN] < min_contour_convexity:
                        continue

                    contour_rows.append(
                        {
                            "detection_index": int(subset.loc[subset_index, "detection_index"]),
                            LEAD_DAY_COLUMN: int(lead_day_index),
                            POLARITY_COLUMN: polarity,
                            LATITUDE_COLUMN: float(subset.loc[subset_index, LATITUDE_COLUMN]),
                            LONGITUDE_COLUMN: float(subset.loc[subset_index, LONGITUDE_COLUMN]),
                            CONTOUR_LEVEL_COLUMN: float(level_value),
                            CONTOUR_PIXEL_COUNT_COLUMN: contour_info[CONTOUR_PIXEL_COUNT_COLUMN],
                            CONTOUR_CONVEXITY_COLUMN: contour_info[CONTOUR_CONVEXITY_COLUMN],
                            CONTOUR_LATITUDES_COLUMN: contour_info[CONTOUR_LATITUDES_COLUMN],
                            CONTOUR_LONGITUDES_COLUMN: contour_info[CONTOUR_LONGITUDES_COLUMN],
                        }
                    )
                    unresolved_mask[subset_index] = False

                if not numpy.any(unresolved_mask):
                    break

    return pandas.DataFrame(
        contour_rows,
        columns=[
            "detection_index",
            LEAD_DAY_COLUMN,
            POLARITY_COLUMN,
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN,
            CONTOUR_LEVEL_COLUMN,
            CONTOUR_PIXEL_COUNT_COLUMN,
            CONTOUR_CONVEXITY_COLUMN,
            CONTOUR_LATITUDES_COLUMN,
            CONTOUR_LONGITUDES_COLUMN,
        ],
    )


def filter_mesoscale_eddy_detections_by_contours(
    detections: pandas.DataFrame,
    contours: pandas.DataFrame,
) -> pandas.DataFrame:
    if detections.empty or contours.empty:
        return detections.iloc[0:0].copy()
    accepted_detection_indices = pandas.Index(contours["detection_index"]).unique()
    return detections.loc[detections.index.isin(accepted_detection_indices)].copy()


def mesoscale_eddy_concentration_from_detections(
    detections: pandas.DataFrame,
    template_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    template_field = _surface_ssh_field(template_dataset, first_day_index=first_day_index, lead_day_index=0)
    latitude_values = template_field[LATITUDE_COLUMN].values
    longitude_values = template_field[LONGITUDE_COLUMN].values
    concentration_arrays = {
        polarity: numpy.zeros(template_field.shape, dtype=float)
        for polarity in POLARITY_ORDER
    }
    for polarity in POLARITY_ORDER:
        polarity_subset = detections.loc[detections[POLARITY_COLUMN] == polarity]
        if polarity_subset.empty:
            continue
        latitude_indices = _nearest_coordinate_indices(latitude_values, polarity_subset[LATITUDE_COLUMN].to_numpy())
        longitude_indices = _nearest_coordinate_indices(longitude_values, polarity_subset[LONGITUDE_COLUMN].to_numpy())
        for latitude_index, longitude_index in zip(latitude_indices, longitude_indices, strict=False):
            concentration_arrays[polarity][latitude_index, longitude_index] += 1.0
    return xarray.Dataset(
        {
            f"{polarity}_concentration": (
                (LATITUDE_COLUMN, LONGITUDE_COLUMN),
                concentration_arrays[polarity],
            )
            for polarity in POLARITY_ORDER
        },
        coords={
            LATITUDE_COLUMN: latitude_values,
            LONGITUDE_COLUMN: longitude_values,
        },
    )


def _wrapped_longitude_values(longitude_values: numpy.ndarray) -> numpy.ndarray:
    wrapped_values = ((numpy.asarray(longitude_values, dtype=float) + 180.0) % 360.0) - 180.0
    positive_dateline_mask = (wrapped_values == -180.0) & (numpy.asarray(longitude_values, dtype=float) > 0)
    wrapped_values[positive_dateline_mask] = 180.0
    return wrapped_values


def _longitude_values_for_polygon(template_longitudes: numpy.ndarray, polygon_longitudes: numpy.ndarray) -> numpy.ndarray:
    wrapped_template_longitudes = _wrapped_longitude_values(template_longitudes)
    wrapped_polygon_longitudes = _wrapped_longitude_values(polygon_longitudes)
    if numpy.nanmax(wrapped_polygon_longitudes) - numpy.nanmin(wrapped_polygon_longitudes) <= 180.0:
        return wrapped_template_longitudes

    shifted_template_longitudes = numpy.where(wrapped_template_longitudes < 0.0, wrapped_template_longitudes + 360.0, wrapped_template_longitudes)
    return shifted_template_longitudes


def _polygon_longitude_values(polygon_longitudes: numpy.ndarray) -> numpy.ndarray:
    wrapped_polygon_longitudes = _wrapped_longitude_values(polygon_longitudes)
    if numpy.nanmax(wrapped_polygon_longitudes) - numpy.nanmin(wrapped_polygon_longitudes) <= 180.0:
        return wrapped_polygon_longitudes
    return numpy.where(wrapped_polygon_longitudes < 0.0, wrapped_polygon_longitudes + 360.0, wrapped_polygon_longitudes)


def mesoscale_eddy_concentration_from_contours(
    contours: pandas.DataFrame,
    template_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    template_field = _surface_ssh_field(template_dataset, first_day_index=first_day_index, lead_day_index=0)
    latitude_values = template_field[LATITUDE_COLUMN].values
    longitude_values = template_field[LONGITUDE_COLUMN].values
    latitude_grid, longitude_grid = numpy.meshgrid(latitude_values, longitude_values, indexing="ij")
    concentration_arrays = {
        polarity: numpy.zeros(template_field.shape, dtype=float)
        for polarity in POLARITY_ORDER
    }
    finite_mask = numpy.isfinite(template_field.values)

    for polarity in POLARITY_ORDER:
        polarity_subset = contours.loc[contours[POLARITY_COLUMN] == polarity]
        if polarity_subset.empty:
            continue
        for _, contour_row in polarity_subset.iterrows():
            contour_latitudes = numpy.asarray(contour_row[CONTOUR_LATITUDES_COLUMN], dtype=float)
            contour_longitudes = numpy.asarray(contour_row[CONTOUR_LONGITUDES_COLUMN], dtype=float)
            if contour_latitudes.size < 3 or contour_longitudes.size < 3:
                continue
            polygon_longitudes = _polygon_longitude_values(contour_longitudes)
            grid_longitudes = _longitude_values_for_polygon(longitude_values, contour_longitudes)
            polygon_vertices = numpy.column_stack([polygon_longitudes, contour_latitudes])
            polygon_path = Path(polygon_vertices, closed=True)
            candidate_mask = (
                (latitude_grid >= numpy.nanmin(contour_latitudes))
                & (latitude_grid <= numpy.nanmax(contour_latitudes))
            )
            if not numpy.any(candidate_mask):
                continue
            candidate_mask &= (
                (grid_longitudes[None, :] >= numpy.nanmin(polygon_longitudes))
                & (grid_longitudes[None, :] <= numpy.nanmax(polygon_longitudes))
            )
            if not numpy.any(candidate_mask):
                continue
            candidate_points = numpy.column_stack(
                [
                    numpy.broadcast_to(grid_longitudes[None, :], latitude_grid.shape)[candidate_mask],
                    latitude_grid[candidate_mask],
                ]
            )
            inside_mask = polygon_path.contains_points(candidate_points)
            if not numpy.any(inside_mask):
                continue
            filled_mask = numpy.zeros(template_field.shape, dtype=bool)
            filled_mask[candidate_mask] = inside_mask
            filled_mask &= finite_mask
            concentration_arrays[polarity][filled_mask] += 1.0

    return xarray.Dataset(
        {
            f"{polarity}_concentration": (
                (LATITUDE_COLUMN, LONGITUDE_COLUMN),
                concentration_arrays[polarity],
            )
            for polarity in POLARITY_ORDER
        },
        coords={
            LATITUDE_COLUMN: latitude_values,
            LONGITUDE_COLUMN: longitude_values,
        },
    )


def default_eddy_detection_parameters() -> dict[str, float | int]:
    return {
        "background_sigma_grid": DEFAULT_BACKGROUND_SIGMA_GRID,
        "detection_sigma_grid": DEFAULT_DETECTION_SIGMA_GRID,
        "min_distance_grid": DEFAULT_MIN_DISTANCE_GRID,
        "amplitude_threshold_meters": DEFAULT_AMPLITUDE_THRESHOLD_METERS,
        "max_abs_latitude_degrees": DEFAULT_MAX_ABS_LATITUDE_DEGREES,
        "max_match_distance_km": DEFAULT_MATCH_DISTANCE_KM,
        "contour_level_step_meters": DEFAULT_CONTOUR_LEVEL_STEP_METERS,
        "min_contour_pixel_count": DEFAULT_MIN_CONTOUR_PIXEL_COUNT,
        "max_contour_pixel_count": DEFAULT_MAX_CONTOUR_PIXEL_COUNT,
        "min_contour_convexity": DEFAULT_MIN_CONTOUR_CONVEXITY,
    }


def surface_ssh_field(
    dataset: xarray.Dataset,
    first_day_index: int = 0,
    lead_day_index: int = 0,
) -> xarray.DataArray:
    return _surface_ssh_field(dataset=dataset, first_day_index=first_day_index, lead_day_index=lead_day_index)


def surface_ssh_anomaly_field(
    dataset: xarray.Dataset,
    first_day_index: int = 0,
    lead_day_index: int = 0,
    background_sigma_grid: float = DEFAULT_BACKGROUND_SIGMA_GRID,
    detection_sigma_grid: float = DEFAULT_DETECTION_SIGMA_GRID,
    max_abs_latitude_degrees: float | None = None,
) -> xarray.DataArray:
    field = _surface_ssh_field(dataset=dataset, first_day_index=first_day_index, lead_day_index=lead_day_index)
    anomaly_values = _ssh_anomaly(
        field,
        background_sigma_grid=background_sigma_grid,
        detection_sigma_grid=detection_sigma_grid,
    )
    if max_abs_latitude_degrees is not None:
        valid_mask = _valid_detection_mask(field, max_abs_latitude_degrees=max_abs_latitude_degrees)
        anomaly_values = numpy.where(valid_mask, anomaly_values, numpy.nan)
    return xarray.DataArray(
        anomaly_values,
        dims=field.dims,
        coords=field.coords,
        attrs={"standard_name": "sea_surface_height_anomaly"},
    )
