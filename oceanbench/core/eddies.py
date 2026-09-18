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

EARTH_RADIUS_KM = 6371.0
ONE_DEGREE_LATITUDE_KM = numpy.pi * EARTH_RADIUS_KM / 180.0

# Detection parameters are true physical scales in kilometres (converted to grid
# cells per dataset by `_kilometres_to_grid_sigma` and the haversine helpers), so
# the same defaults apply at 1 degree, 1/4 degree and 1/12 degree resolution.
#
# History: through commit c1f1099 these constants encoded *1-degree-grid-cell
# counts* rather than physical scales (e.g. min_peak_separation "8" meant 8 cells,
# which only equals ~890 km on a 1-degree grid and shrinks on finer grids). At
# native resolution that made detection wildly over-restrictive -- an adversarial
# audit (2026-07-07) measured only ~70 eddies globally at 1 degree, versus the
# ~1500-2500 expected from the literature. The values below are literature-derived
# (Chelton et al. 2011, DOI:10.1016/j.pocean.2011.01.002; the META / py-eddy-tracker
# product line, Mason et al. 2014) and resolution-independent.
#
# Background high-pass, as a (latitude, longitude) Gaussian sigma in kilometres.
# These are SIGMAS, not cutoffs: a Gaussian low-pass exp(-k^2 sigma^2 / 2) reaches half
# power at a wavelength of roughly 7.5 sigma, so (265, 130) km corresponds to half-power
# wavelengths of about 1000 km meridional x 2000 km zonal, the scale of Chelton et al.
# (2011)'s 20-degree zonal x 10-degree meridional half-power block. The earlier default of
# 12 degrees of latitude (1334 km) was applied directly as the sigma, i.e. a ~10000 km
# half-power filter, which left gyre and front-scale sea surface height in the "anomaly".
DEFAULT_BACKGROUND_SIGMA_KM = (130.0, 265.0)
# Second smoothing pass of the anomaly is OFF by default: Chelton/META and
# py-eddy-tracker do not blur the mesoscale field before peak detection. The
# parameter is kept so old artifacts remain reproducible by passing an explicit value.
DEFAULT_DETECTION_SIGMA_KM = None
DEFAULT_MIN_PEAK_SEPARATION_KM = 100.0  # ~ one mesoscale eddy diameter
# Chelton/META 1 cm amplitude. This is the Chelton amplitude: the peak anomaly measured
# ABOVE the level of the outermost closed contour that passes the area and solidity tests,
# not the raw peak value. It doubles as the cheap `peak_local_max` prefilter threshold and
# as the level ladder's first rung, but a centre is only accepted if peak minus contour
# level clears it, so a broad plateau with a 1 cm bump on it no longer counts as an eddy.
DEFAULT_AMPLITUDE_THRESHOLD_METERS = 0.01
DEFAULT_MAX_ABS_LATITUDE_DEGREES = 70.0
DEFAULT_MATCH_DISTANCE_KM = 200.0
DEFAULT_CONTOUR_LEVEL_STEP_METERS = 0.01
DEFAULT_MIN_EDDY_AREA_KM2 = 2000.0  # radius ~ 25 km (small mesoscale floor)
DEFAULT_MAX_EDDY_AREA_KM2 = 300_000.0  # radius ~ 300 km (Chelton-order upper bound on eddy size)
DEFAULT_MIN_CONTOUR_CONVEXITY = 0.75
DEFAULT_APPLY_CONTOUR_FILTERING = True
GLOBAL_LONGITUDE_SPAN_THRESHOLD_DEGREES = 300.0
GLOBAL_LONGITUDE_PERIOD_DEGREES = 360.0

LEAD_DAY_COLUMN = "lead_day"
LATITUDE_COLUMN = Dimension.LATITUDE.key()
LONGITUDE_COLUMN = Dimension.LONGITUDE.key()
POLARITY_COLUMN = "polarity"
AMPLITUDE_COLUMN = "sea_surface_height_anomaly"
SEA_SURFACE_HEIGHT_COLUMN = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
DISTANCE_COLUMN = "distance_km"
CONTOUR_LEVEL_COLUMN = "contour_level"
CONTOUR_PIXEL_COUNT_COLUMN = "contour_pixel_count"
CONTOUR_AREA_KM2_COLUMN = "contour_area_km2"
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


def _gaussian_filter_with_mask(
    values: numpy.ndarray,
    sigma: float | tuple[float, float],
) -> numpy.ndarray:
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
    background_sigma_km: float | tuple[float, float],
    detection_sigma_km: float | None,
) -> numpy.ndarray:
    field_values = numpy.asarray(field.values, dtype=float)
    background_values = _gaussian_filter_with_mask(
        field_values, sigma=_kilometres_to_grid_sigma(field, background_sigma_km)
    )
    anomaly_values = field_values - background_values
    if detection_sigma_km is None or detection_sigma_km <= 0:
        return anomaly_values
    return _gaussian_filter_with_mask(anomaly_values, sigma=_kilometres_to_grid_sigma(field, detection_sigma_km))


def _median_positive_spacing(values: numpy.ndarray) -> float:
    differences = numpy.abs(numpy.diff(numpy.asarray(values, dtype=float)))
    positive = differences[numpy.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        raise ValueError("eddy detection requires at least two distinct coordinate values")
    return float(numpy.median(positive))


def _kilometres_to_grid_sigma(
    field: xarray.DataArray,
    sigma_km: float | tuple[float, float],
) -> tuple[float, float]:
    # `sigma_km` is either one isotropic physical sigma or a (latitude, longitude) pair;
    # the result is always the pair of grid sigmas for the (latitude, longitude) axes.
    latitude_sigma_km, longitude_sigma_km = sigma_km if isinstance(sigma_km, tuple) else (sigma_km, sigma_km)
    latitude_values = numpy.asarray(field[LATITUDE_COLUMN].values, dtype=float)
    latitude_spacing_km = _median_positive_spacing(latitude_values) * ONE_DEGREE_LATITUDE_KM
    characteristic_latitude = float(numpy.nanmean(latitude_values))
    longitude_spacing_km = (
        _median_positive_spacing(field[LONGITUDE_COLUMN].values)
        * ONE_DEGREE_LATITUDE_KM
        * numpy.cos(numpy.deg2rad(characteristic_latitude))
    )
    return latitude_sigma_km / latitude_spacing_km, longitude_sigma_km / longitude_spacing_km


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
    min_peak_separation_km: float,
    amplitude_threshold_meters: float,
    latitude_values: numpy.ndarray,
    longitude_values: numpy.ndarray,
) -> numpy.ndarray:
    masked_values = numpy.where(valid_mask, anomaly_values, numpy.nan)
    if polarity == ANTICYCLONE:
        image_values = numpy.where(numpy.isfinite(masked_values), masked_values, -numpy.inf)
    else:
        image_values = numpy.where(numpy.isfinite(masked_values), -masked_values, -numpy.inf)
    coordinates = peak_local_max(
        image_values,
        min_distance=1,
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
    coordinates = coordinates[polarity_mask]
    if coordinates.size == 0:
        return coordinates
    ranked = coordinates[numpy.argsort(-numpy.abs(masked_values[coordinates[:, 0], coordinates[:, 1]]))]
    # A great-circle distance is never shorter than the latitude difference alone, so an
    # already accepted peak further than `min_peak_separation_km` in latitude cannot be the
    # one that rejects this candidate. Only the peaks inside that latitude band are measured,
    # which leaves the accept/reject decision exactly as the all-pairs sweep had it.
    latitude_window_degrees = min_peak_separation_km / ONE_DEGREE_LATITUDE_KM
    accepted_by_latitude_index: dict[int, list[int]] = {}
    accepted: list[numpy.ndarray] = []
    for coordinate in ranked:
        latitude_index = int(coordinate[0])
        if accepted:
            candidate_latitude = latitude_values[latitude_index]
            neighbour_latitude_indices = numpy.searchsorted(
                latitude_values,
                [candidate_latitude - latitude_window_degrees, candidate_latitude + latitude_window_degrees],
                side="left",
            )
            neighbour_rows: list[int] = []
            neighbours: list[int] = []
            for row in range(int(neighbour_latitude_indices[0]) - 1, int(neighbour_latitude_indices[1]) + 1):
                row_longitude_indices = accepted_by_latitude_index.get(row)
                if row_longitude_indices:
                    neighbour_rows.extend([row] * len(row_longitude_indices))
                    neighbours.extend(row_longitude_indices)
            if neighbours:
                distances = _haversine_distance_km(
                    numpy.asarray([candidate_latitude]),
                    numpy.asarray([longitude_values[coordinate[1]]]),
                    latitude_values[numpy.asarray(neighbour_rows)],
                    longitude_values[numpy.asarray(neighbours)],
                )
                if not numpy.all(distances >= min_peak_separation_km):
                    continue
        accepted.append(coordinate)
        accepted_by_latitude_index.setdefault(latitude_index, []).append(int(coordinate[1]))
    return numpy.asarray(sorted(accepted, key=lambda item: (item[0], item[1])), dtype=int)


def detect_mesoscale_eddies(
    dataset: xarray.Dataset,
    first_day_index: int = 0,
    lead_day_indices: list[int] | None = None,
    background_sigma_km: float | tuple[float, float] = DEFAULT_BACKGROUND_SIGMA_KM,
    detection_sigma_km: float | None = DEFAULT_DETECTION_SIGMA_KM,
    min_peak_separation_km: float = DEFAULT_MIN_PEAK_SEPARATION_KM,
    amplitude_threshold_meters: float = DEFAULT_AMPLITUDE_THRESHOLD_METERS,
    max_abs_latitude_degrees: float = DEFAULT_MAX_ABS_LATITUDE_DEGREES,
) -> pandas.DataFrame:
    detection_rows: list[dict[str, float | int | str]] = []
    for lead_day_index in _lead_day_indices(dataset, lead_day_indices):
        field = _surface_ssh_field(dataset, first_day_index=first_day_index, lead_day_index=lead_day_index)
        anomaly_values = _ssh_anomaly(
            field,
            background_sigma_km=background_sigma_km,
            detection_sigma_km=detection_sigma_km,
        )
        valid_mask = _valid_detection_mask(field, max_abs_latitude_degrees=max_abs_latitude_degrees)
        latitude_values = field[LATITUDE_COLUMN].values
        longitude_values = field[LONGITUDE_COLUMN].values
        for polarity in POLARITY_ORDER:
            peak_coordinates = _detect_polarity_peaks(
                anomaly_values=anomaly_values,
                valid_mask=valid_mask,
                polarity=polarity,
                min_peak_separation_km=min_peak_separation_km,
                amplitude_threshold_meters=amplitude_threshold_meters,
                latitude_values=latitude_values,
                longitude_values=longitude_values,
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
    return 2.0 * EARTH_RADIUS_KM * numpy.arcsin(numpy.sqrt(haversine))


def match_mesoscale_eddies(
    challenger_detections: pandas.DataFrame,
    reference_detections: pandas.DataFrame,
    max_match_distance_km: float = DEFAULT_MATCH_DISTANCE_KM,
) -> pandas.DataFrame:
    match_rows: list[dict[str, float | int | str]] = []
    for lead_day_index in sorted(
        set(challenger_detections[LEAD_DAY_COLUMN]).union(reference_detections[LEAD_DAY_COLUMN])
    ):
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
    background_sigma_km: float | tuple[float, float] = DEFAULT_BACKGROUND_SIGMA_KM,
    detection_sigma_km: float | None = DEFAULT_DETECTION_SIGMA_KM,
    min_peak_separation_km: float = DEFAULT_MIN_PEAK_SEPARATION_KM,
    amplitude_threshold_meters: float = DEFAULT_AMPLITUDE_THRESHOLD_METERS,
    max_abs_latitude_degrees: float = DEFAULT_MAX_ABS_LATITUDE_DEGREES,
    max_match_distance_km: float = DEFAULT_MATCH_DISTANCE_KM,
    contour_level_step_meters: float = DEFAULT_CONTOUR_LEVEL_STEP_METERS,
    min_eddy_area_km2: float = DEFAULT_MIN_EDDY_AREA_KM2,
    max_eddy_area_km2: float = DEFAULT_MAX_EDDY_AREA_KM2,
    min_contour_convexity: float = DEFAULT_MIN_CONTOUR_CONVEXITY,
    challenger_name: str = "GLONET",
    reference_name: str = "GLORYS",
) -> pandas.DataFrame:
    challenger_detections = detect_mesoscale_eddies(
        challenger_dataset,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        background_sigma_km=background_sigma_km,
        detection_sigma_km=detection_sigma_km,
        min_peak_separation_km=min_peak_separation_km,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
    )
    challenger_contours = mesoscale_eddy_contours_from_detections(
        challenger_detections,
        challenger_dataset,
        first_day_index=first_day_index,
        background_sigma_km=background_sigma_km,
        detection_sigma_km=detection_sigma_km,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
        contour_level_step_meters=contour_level_step_meters,
        min_eddy_area_km2=min_eddy_area_km2,
        max_eddy_area_km2=max_eddy_area_km2,
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
        background_sigma_km=background_sigma_km,
        detection_sigma_km=detection_sigma_km,
        min_peak_separation_km=min_peak_separation_km,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
    )
    reference_contours = mesoscale_eddy_contours_from_detections(
        reference_detections,
        reference_dataset,
        first_day_index=first_day_index,
        background_sigma_km=background_sigma_km,
        detection_sigma_km=detection_sigma_km,
        amplitude_threshold_meters=amplitude_threshold_meters,
        max_abs_latitude_degrees=max_abs_latitude_degrees,
        contour_level_step_meters=contour_level_step_meters,
        min_eddy_area_km2=min_eddy_area_km2,
        max_eddy_area_km2=max_eddy_area_km2,
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


def _is_periodic_longitude_domain(longitude_values: numpy.ndarray) -> bool:
    finite_longitudes = numpy.asarray(longitude_values, dtype=float)
    finite_longitudes = finite_longitudes[numpy.isfinite(finite_longitudes)]
    if finite_longitudes.size < 2:
        return False
    longitude_span = float(numpy.nanmax(finite_longitudes) - numpy.nanmin(finite_longitudes))
    return longitude_span >= GLOBAL_LONGITUDE_SPAN_THRESHOLD_DEGREES


def _periodic_longitude_delta(start: float, end: float) -> float:
    return (
        (end - start + GLOBAL_LONGITUDE_PERIOD_DEGREES / 2.0) % GLOBAL_LONGITUDE_PERIOD_DEGREES
    ) - GLOBAL_LONGITUDE_PERIOD_DEGREES / 2.0


def _unwrapped_periodic_longitudes(longitude_values: numpy.ndarray) -> numpy.ndarray:
    longitudes = numpy.asarray(longitude_values, dtype=float)
    if longitudes.size == 0:
        return longitudes
    # `cumsum` accumulates left to right over [first longitude, delta, delta, ...], which is
    # the same association, and so the same rounding, as the running sum this replaced.
    steps = numpy.empty(longitudes.size, dtype=float)
    steps[0] = longitudes[0]
    steps[1:] = (
        (numpy.diff(longitudes) + GLOBAL_LONGITUDE_PERIOD_DEGREES / 2.0) % GLOBAL_LONGITUDE_PERIOD_DEGREES
    ) - GLOBAL_LONGITUDE_PERIOD_DEGREES / 2.0
    return numpy.cumsum(steps)


def _cell_area_by_latitude_row(
    latitude_values: numpy.ndarray,
    longitude_values: numpy.ndarray,
) -> numpy.ndarray:
    latitude_spacing_radians = numpy.deg2rad(_median_positive_spacing(latitude_values))
    longitude_spacing_radians = numpy.deg2rad(_median_positive_spacing(longitude_values))
    return (
        EARTH_RADIUS_KM**2
        * latitude_spacing_radians
        * longitude_spacing_radians
        * numpy.cos(numpy.deg2rad(latitude_values))
    )


def _anchor_periodic_longitudes(longitudes: numpy.ndarray, anchor_longitude: float) -> numpy.ndarray:
    if longitudes.size == 0:
        return longitudes
    mean_longitude = float(numpy.nanmean(longitudes))
    if not numpy.isfinite(mean_longitude):
        return longitudes
    longitude_shift = round((anchor_longitude - mean_longitude) / GLOBAL_LONGITUDE_PERIOD_DEGREES)
    return longitudes + longitude_shift * GLOBAL_LONGITUDE_PERIOD_DEGREES


def _merge_periodic_longitude_labels(
    labels: numpy.ndarray,
    component_count: int,
) -> tuple[numpy.ndarray, int]:
    if component_count <= 1 or labels.shape[1] < 2:
        return labels, component_count

    parent_labels = numpy.arange(component_count + 1, dtype=int)

    def find(label_id: int) -> int:
        while parent_labels[label_id] != label_id:
            parent_labels[label_id] = parent_labels[parent_labels[label_id]]
            label_id = int(parent_labels[label_id])
        return label_id

    def union(first_label: int, second_label: int) -> None:
        first_root = find(first_label)
        second_root = find(second_label)
        if first_root != second_root:
            parent_labels[second_root] = first_root

    latitude_count = labels.shape[0]
    for latitude_index in range(latitude_count):
        left_label = int(labels[latitude_index, 0])
        if left_label <= 0:
            continue
        for neighbor_latitude_index in range(
            max(0, latitude_index - 1),
            min(latitude_count, latitude_index + 2),
        ):
            right_label = int(labels[neighbor_latitude_index, -1])
            if right_label > 0:
                union(left_label, right_label)

    root_to_periodic_label: dict[int, int] = {}
    label_mapping = numpy.zeros(component_count + 1, dtype=int)
    for label_id in range(1, component_count + 1):
        root_label = find(label_id)
        if root_label not in root_to_periodic_label:
            root_to_periodic_label[root_label] = len(root_to_periodic_label) + 1
        label_mapping[label_id] = root_to_periodic_label[root_label]

    return label_mapping[labels], len(root_to_periodic_label)


def _connected_component_labels(
    component_mask: numpy.ndarray,
    structure: numpy.ndarray,
    periodic_longitude: bool,
) -> tuple[numpy.ndarray, int]:
    labels, component_count = label(component_mask, structure=structure)
    if not periodic_longitude:
        return labels, component_count
    return _merge_periodic_longitude_labels(labels, component_count)


def _component_positions_by_label(
    labels: numpy.ndarray,
    component_count: int,
    wanted_labels: numpy.ndarray,
) -> dict[int, tuple[numpy.ndarray, numpy.ndarray]]:
    # One pass over the label field for every component a level needs, instead of one
    # `labels == component_label` scan per eddy centre. Rows come out of `nonzero` in
    # row-major order and the sort is stable, so each component's positions stay in the
    # order `numpy.argwhere` produced them.
    unique_labels = numpy.unique(numpy.asarray(wanted_labels, dtype=int))
    unique_labels = unique_labels[unique_labels > 0]
    if unique_labels.size == 0:
        return {}
    selector = numpy.zeros(component_count + 1, dtype=bool)
    selector[unique_labels] = True
    rows, columns = numpy.nonzero(selector[labels])
    if rows.size == 0:
        return {}
    values = labels[rows, columns]
    order = numpy.argsort(values, kind="stable")
    sorted_values = values[order]
    boundaries = numpy.searchsorted(sorted_values, unique_labels, side="left")
    ends = numpy.searchsorted(sorted_values, unique_labels, side="right")
    positions: dict[int, tuple[numpy.ndarray, numpy.ndarray]] = {}
    for position, component_label in enumerate(unique_labels):
        span = order[boundaries[position] : ends[position]]
        positions[int(component_label)] = (rows[span], columns[span])
    return positions


def _component_contour_info(
    component_rows: numpy.ndarray,
    component_columns: numpy.ndarray,
    grid_shape: tuple[int, int],
    latitude_values: numpy.ndarray,
    longitude_values: numpy.ndarray,
    center_longitude_index: int | None = None,
    periodic_longitude: bool = False,
    unwrapped_longitude_cache: dict[int, numpy.ndarray] | None = None,
    cell_area_by_latitude_row: numpy.ndarray | None = None,
) -> dict[str, object] | None:
    # `component_rows` / `component_columns` are the component's positions in the unrolled
    # label field. The periodic case used to roll the whole label array per centre; rolling
    # the positions arithmetically gives the same working frame for a few dozen values.
    working_columns = component_columns
    working_longitude_values = longitude_values
    if periodic_longitude and center_longitude_index is not None and grid_shape[1] > 1:
        longitude_roll = grid_shape[1] // 2 - int(center_longitude_index)
        working_columns = (component_columns + longitude_roll) % grid_shape[1]
        cache = unwrapped_longitude_cache if unwrapped_longitude_cache is not None else {}
        cached_longitudes = cache.get(longitude_roll)
        if cached_longitudes is None:
            cached_longitudes = _unwrapped_periodic_longitudes(numpy.roll(longitude_values, longitude_roll))
            cache[longitude_roll] = cached_longitudes
        working_longitude_values = cached_longitudes

    if component_rows.size == 0:
        return None
    latitude_min = max(int(component_rows.min()) - 1, 0)
    latitude_max = min(int(component_rows.max()) + 2, grid_shape[0])
    longitude_min = max(int(working_columns.min()) - 1, 0)
    longitude_max = min(int(working_columns.max()) + 2, grid_shape[1])

    component_mask = numpy.zeros((latitude_max - latitude_min, longitude_max - longitude_min), dtype=bool)
    component_mask[component_rows - latitude_min, working_columns - longitude_min] = True
    component_positions = numpy.argwhere(component_mask)
    component_positions[:, 0] += latitude_min
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
    contour_longitudes = numpy.interp(
        contour_longitude_indices,
        numpy.arange(len(working_longitude_values)),
        working_longitude_values,
    )
    if periodic_longitude and center_longitude_index is not None:
        contour_longitudes = _anchor_periodic_longitudes(
            contour_longitudes,
            float(longitude_values[center_longitude_index]),
        )

    region = properties[0]
    component_latitude_indices = component_positions[:, 0]
    if cell_area_by_latitude_row is None:
        cell_area_by_latitude_row = _cell_area_by_latitude_row(latitude_values, longitude_values)
    cell_areas_km2 = cell_area_by_latitude_row[component_latitude_indices]
    return {
        CONTOUR_LATITUDES_COLUMN: contour_latitudes,
        CONTOUR_LONGITUDES_COLUMN: contour_longitudes,
        CONTOUR_PIXEL_COUNT_COLUMN: int(region.area),
        CONTOUR_AREA_KM2_COLUMN: float(numpy.sum(cell_areas_km2)),
        CONTOUR_CONVEXITY_COLUMN: float(region.solidity),
    }


def mesoscale_eddy_contours_from_detections(
    detections: pandas.DataFrame,
    dataset: xarray.Dataset,
    first_day_index: int = 0,
    background_sigma_km: float | tuple[float, float] = DEFAULT_BACKGROUND_SIGMA_KM,
    detection_sigma_km: float | None = DEFAULT_DETECTION_SIGMA_KM,
    amplitude_threshold_meters: float = DEFAULT_AMPLITUDE_THRESHOLD_METERS,
    max_abs_latitude_degrees: float = DEFAULT_MAX_ABS_LATITUDE_DEGREES,
    contour_level_step_meters: float = DEFAULT_CONTOUR_LEVEL_STEP_METERS,
    min_eddy_area_km2: float = DEFAULT_MIN_EDDY_AREA_KM2,
    max_eddy_area_km2: float = DEFAULT_MAX_EDDY_AREA_KM2,
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
            background_sigma_km=background_sigma_km,
            detection_sigma_km=detection_sigma_km,
            max_abs_latitude_degrees=max_abs_latitude_degrees,
        )
        anomaly_values = numpy.asarray(anomaly_field.values, dtype=float)
        latitude_values = anomaly_field[LATITUDE_COLUMN].values
        longitude_values = anomaly_field[LONGITUDE_COLUMN].values
        periodic_longitude = _is_periodic_longitude_domain(longitude_values)
        # Per-field quantities the level ladder used to rebuild for every level and every
        # centre: the finite mask, the per-row cell area and the unwrapped longitudes of a
        # given roll all depend on the grid alone.
        finite_mask = numpy.isfinite(anomaly_values)
        cell_area_by_latitude_row = _cell_area_by_latitude_row(latitude_values, longitude_values)
        unwrapped_longitude_cache: dict[int, numpy.ndarray] = {}
        grid_shape = (latitude_values.size, longitude_values.size)

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
                component_mask &= finite_mask
                labels, component_count = _connected_component_labels(
                    component_mask,
                    structure=connectivity,
                    periodic_longitude=periodic_longitude,
                )
                if component_count == 0:
                    continue

                center_component_labels = labels[center_latitude_indices, center_longitude_indices]
                active_label_mask = (center_component_labels > 0) & (center_magnitudes >= level_value)
                if not numpy.any(active_label_mask):
                    continue
                active_label_counts = numpy.bincount(center_component_labels[active_label_mask])
                component_cache: dict[int, dict[str, object] | None] = {}

                candidate_indices = numpy.where(candidate_mask)[0]
                candidate_labels = center_component_labels[candidate_indices]
                resolvable = (candidate_labels > 0) & (candidate_labels < len(active_label_counts))
                resolvable[resolvable] = active_label_counts[candidate_labels[resolvable]] == 1
                # Every component this level needs is located in one sweep of the label
                # field, rather than one `labels == component_label` sweep per centre.
                wanted_labels = candidate_labels[resolvable]
                component_positions = _component_positions_by_label(labels, component_count, wanted_labels)

                for subset_index, component_label, is_resolvable in zip(
                    candidate_indices, candidate_labels, resolvable, strict=True
                ):
                    if not is_resolvable:
                        continue
                    component_label = int(component_label)

                    contour_info = component_cache.get(component_label)
                    if contour_info is None:
                        rows, columns = component_positions[component_label]
                        contour_info = _component_contour_info(
                            rows,
                            columns,
                            grid_shape=grid_shape,
                            latitude_values=latitude_values,
                            longitude_values=longitude_values,
                            center_longitude_index=int(center_longitude_indices[subset_index]),
                            periodic_longitude=periodic_longitude,
                            unwrapped_longitude_cache=unwrapped_longitude_cache,
                            cell_area_by_latitude_row=cell_area_by_latitude_row,
                        )
                        component_cache[component_label] = contour_info
                    if contour_info is None:
                        continue
                    if contour_info[CONTOUR_AREA_KM2_COLUMN] < min_eddy_area_km2:
                        continue
                    if contour_info[CONTOUR_AREA_KM2_COLUMN] > max_eddy_area_km2:
                        continue
                    if contour_info[CONTOUR_CONVEXITY_COLUMN] < min_contour_convexity:
                        continue

                    # Chelton amplitude: height of the peak above the outermost closed
                    # contour, not the raw peak anomaly. Levels are walked from low to
                    # high, so this first accepted level IS the outermost valid contour
                    # and gives the largest amplitude this centre can ever have; if it
                    # falls short of the threshold the centre is rejected outright rather
                    # than retried at a higher level.
                    contour_amplitude = float(center_magnitudes[subset_index]) - float(level_value)
                    if contour_amplitude < amplitude_threshold_meters:
                        unresolved_mask[subset_index] = False
                        continue

                    contour_rows.append(
                        {
                            "detection_index": int(subset.loc[subset_index, "detection_index"]),
                            AMPLITUDE_COLUMN: contour_amplitude if polarity == ANTICYCLONE else -contour_amplitude,
                            LEAD_DAY_COLUMN: int(lead_day_index),
                            POLARITY_COLUMN: polarity,
                            LATITUDE_COLUMN: float(subset.loc[subset_index, LATITUDE_COLUMN]),
                            LONGITUDE_COLUMN: float(subset.loc[subset_index, LONGITUDE_COLUMN]),
                            CONTOUR_LEVEL_COLUMN: float(level_value),
                            CONTOUR_PIXEL_COUNT_COLUMN: contour_info[CONTOUR_PIXEL_COUNT_COLUMN],
                            CONTOUR_AREA_KM2_COLUMN: contour_info[CONTOUR_AREA_KM2_COLUMN],
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
            AMPLITUDE_COLUMN,
            LEAD_DAY_COLUMN,
            POLARITY_COLUMN,
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN,
            CONTOUR_LEVEL_COLUMN,
            CONTOUR_PIXEL_COUNT_COLUMN,
            CONTOUR_AREA_KM2_COLUMN,
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
    accepted = detections.loc[detections.index.isin(accepted_detection_indices)].copy()
    # Republish the Chelton amplitude carried by the accepted contour over the raw peak
    # anomaly the detector recorded, so every consumer of AMPLITUDE_COLUMN sees the height
    # above the outermost closed contour.
    contour_amplitudes = contours.drop_duplicates("detection_index").set_index("detection_index")[AMPLITUDE_COLUMN]
    accepted[AMPLITUDE_COLUMN] = accepted.index.map(contour_amplitudes)
    return accepted


def mesoscale_eddy_concentration_from_detections(
    detections: pandas.DataFrame,
    template_dataset: xarray.Dataset,
    first_day_index: int = 0,
) -> xarray.Dataset:
    template_field = _surface_ssh_field(template_dataset, first_day_index=first_day_index, lead_day_index=0)
    latitude_values = template_field[LATITUDE_COLUMN].values
    longitude_values = template_field[LONGITUDE_COLUMN].values
    concentration_arrays = {polarity: numpy.zeros(template_field.shape, dtype=float) for polarity in POLARITY_ORDER}
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


def _longitude_values_for_polygon(
    template_longitudes: numpy.ndarray, polygon_longitudes: numpy.ndarray
) -> numpy.ndarray:
    wrapped_template_longitudes = _wrapped_longitude_values(template_longitudes)
    wrapped_polygon_longitudes = _wrapped_longitude_values(polygon_longitudes)
    if numpy.nanmax(wrapped_polygon_longitudes) - numpy.nanmin(wrapped_polygon_longitudes) <= 180.0:
        return wrapped_template_longitudes

    shifted_template_longitudes = numpy.where(
        wrapped_template_longitudes < 0.0, wrapped_template_longitudes + 360.0, wrapped_template_longitudes
    )
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
    concentration_arrays = {polarity: numpy.zeros(template_field.shape, dtype=float) for polarity in POLARITY_ORDER}
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
            candidate_mask = (latitude_grid >= numpy.nanmin(contour_latitudes)) & (
                latitude_grid <= numpy.nanmax(contour_latitudes)
            )
            if not numpy.any(candidate_mask):
                continue
            candidate_mask &= (grid_longitudes[None, :] >= numpy.nanmin(polygon_longitudes)) & (
                grid_longitudes[None, :] <= numpy.nanmax(polygon_longitudes)
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
        "background_sigma_km": DEFAULT_BACKGROUND_SIGMA_KM,
        "detection_sigma_km": DEFAULT_DETECTION_SIGMA_KM,
        "min_peak_separation_km": DEFAULT_MIN_PEAK_SEPARATION_KM,
        "amplitude_threshold_meters": DEFAULT_AMPLITUDE_THRESHOLD_METERS,
        "max_abs_latitude_degrees": DEFAULT_MAX_ABS_LATITUDE_DEGREES,
        "max_match_distance_km": DEFAULT_MATCH_DISTANCE_KM,
        "contour_level_step_meters": DEFAULT_CONTOUR_LEVEL_STEP_METERS,
        "min_eddy_area_km2": DEFAULT_MIN_EDDY_AREA_KM2,
        "max_eddy_area_km2": DEFAULT_MAX_EDDY_AREA_KM2,
        "min_contour_convexity": DEFAULT_MIN_CONTOUR_CONVEXITY,
        "apply_contour_filtering": DEFAULT_APPLY_CONTOUR_FILTERING,
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
    background_sigma_km: float | tuple[float, float] = DEFAULT_BACKGROUND_SIGMA_KM,
    detection_sigma_km: float | None = DEFAULT_DETECTION_SIGMA_KM,
    max_abs_latitude_degrees: float | None = None,
) -> xarray.DataArray:
    field = _surface_ssh_field(dataset=dataset, first_day_index=first_day_index, lead_day_index=lead_day_index)
    anomaly_values = _ssh_anomaly(
        field,
        background_sigma_km=background_sigma_km,
        detection_sigma_km=detection_sigma_km,
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
