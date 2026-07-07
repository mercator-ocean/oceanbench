# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from oceanbench.core import eddies
from oceanbench.core.dataset_utils import Dimension, Variable


def _global_boundary_eddy_dataset() -> xarray.Dataset:
    latitudes = numpy.linspace(-10.0, 10.0, 41)
    longitudes = numpy.arange(0.0, 360.0, 1.0)
    longitude_distance = ((longitudes[None, :] - 359.0 + 180.0) % 360.0) - 180.0
    latitude_distance = latitudes[:, None]
    sea_surface_height = -numpy.exp(-((longitude_distance / 3.0) ** 2 + (latitude_distance / 2.5) ** 2))

    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ),
                sea_surface_height[None, None, :, :],
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-01"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def test_mesoscale_eddy_contours_are_periodic_at_longitude_boundary() -> None:
    dataset = _global_boundary_eddy_dataset()
    detections = pandas.DataFrame(
        [
            {
                eddies.LEAD_DAY_COLUMN: 0,
                eddies.LATITUDE_COLUMN: 0.0,
                eddies.LONGITUDE_COLUMN: 359.0,
                eddies.POLARITY_COLUMN: eddies.CYCLONE,
                eddies.AMPLITUDE_COLUMN: -1.0,
                eddies.SEA_SURFACE_HEIGHT_COLUMN: -1.0,
            }
        ]
    )

    contours = eddies.mesoscale_eddy_contours_from_detections(
        detections,
        dataset,
        background_sigma_km=12.0 * eddies.ONE_DEGREE_LATITUDE_KM,
        detection_sigma_km=0.0,
        amplitude_threshold_meters=0.2,
        min_eddy_area_km2=0.0,
        max_eddy_area_km2=1.0e9,
        min_contour_convexity=0.0,
    )

    assert len(contours) == 1
    contour_longitudes = numpy.asarray(contours.iloc[0][eddies.CONTOUR_LONGITUDES_COLUMN], dtype=float)
    assert numpy.nanmin(contour_longitudes) < 358.0
    assert numpy.nanmax(contour_longitudes) > 360.0
    assert numpy.nanmax(contour_longitudes) - numpy.nanmin(contour_longitudes) < 20.0

    accepted_detections = eddies.filter_mesoscale_eddy_detections_by_contours(detections, contours)
    assert len(accepted_detections) == 1


def test_literature_km_defaults_detect_planted_eddies_without_extra_smoothing() -> None:
    # The literature-derived defaults (Chelton et al. 2011; META/py-eddy-tracker):
    # 1 cm amplitude threshold, 100 km peak separation, and NO second anomaly-smoothing
    # pass (detection_sigma_km defaults to None). A single-degree field carrying one
    # strong anticyclone and one strong cyclone must therefore yield exactly those two
    # detections with the correct polarity and near-exact centres. This replaces the
    # old test that pinned the pre-c1f1099 grid-cell-count behaviour (extra 1.5-cell
    # blur, 8-cell separation, 4 cm threshold), which is no longer the default.
    assert eddies.DEFAULT_DETECTION_SIGMA_KM is None
    assert eddies.DEFAULT_AMPLITUDE_THRESHOLD_METERS == 0.01
    assert eddies.DEFAULT_MIN_PEAK_SEPARATION_KM == 100.0

    latitudes = numpy.arange(-20.0, 21.0)
    longitudes = numpy.arange(0.0, 80.0)
    latitude_grid = latitudes[:, None]
    longitude_grid = longitudes[None, :]
    values = 0.5 * numpy.exp(
        -(((latitude_grid + 5.0) / 2.0) ** 2 + ((longitude_grid - 20.0) / 2.0) ** 2)
    ) - 0.6 * numpy.exp(-(((latitude_grid - 7.0) / 2.5) ** 2 + ((longitude_grid - 55.0) / 2.5) ** 2))
    dataset = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ),
                values[None, None],
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-01"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )

    detections = eddies.detect_mesoscale_eddies(dataset)

    anticyclones = detections[detections[eddies.POLARITY_COLUMN] == eddies.ANTICYCLONE]
    cyclones = detections[detections[eddies.POLARITY_COLUMN] == eddies.CYCLONE]
    assert not anticyclones.empty
    assert not cyclones.empty
    # The strongest detection of each polarity must sit on the planted eddy centre.
    anticyclone = anticyclones.loc[anticyclones[eddies.AMPLITUDE_COLUMN].idxmax()]
    cyclone = cyclones.loc[cyclones[eddies.AMPLITUDE_COLUMN].idxmin()]
    assert abs(anticyclone[eddies.LATITUDE_COLUMN] - (-5.0)) <= 2.0
    assert abs(anticyclone[eddies.LONGITUDE_COLUMN] - 20.0) <= 2.0
    assert abs(cyclone[eddies.LATITUDE_COLUMN] - 7.0) <= 2.0
    assert abs(cyclone[eddies.LONGITUDE_COLUMN] - 55.0) <= 2.0


def test_detection_sigma_parameter_reproduces_pre_change_smoothing() -> None:
    # The detection_sigma_km parameter is retained for reproducing pre-change artifacts:
    # passing the old 1.5-cell (at 1 degree) value must re-enable the second smoothing pass.
    latitudes = numpy.arange(-20.0, 21.0)
    longitudes = numpy.arange(0.0, 80.0)
    latitude_grid = latitudes[:, None]
    longitude_grid = longitudes[None, :]
    values = 0.5 * numpy.exp(-(((latitude_grid + 5.0) / 2.0) ** 2 + ((longitude_grid - 20.0) / 2.0) ** 2))
    dataset = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ),
                values[None, None],
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-01"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )

    legacy_background = gaussian_filter(values, sigma=12.0, mode=("nearest", "wrap"))
    legacy_anomaly = gaussian_filter(values - legacy_background, sigma=1.5, mode=("nearest", "wrap"))
    legacy_rows = sorted(
        (eddies.ANTICYCLONE, float(latitudes[latitude_index]), float(longitudes[longitude_index]))
        for latitude_index, longitude_index in peak_local_max(
            legacy_anomaly, min_distance=8, threshold_abs=0.04, exclude_border=False
        )
    )

    detections = eddies.detect_mesoscale_eddies(
        dataset,
        detection_sigma_km=1.5 * eddies.ONE_DEGREE_LATITUDE_KM,
        min_peak_separation_km=8.0 * eddies.ONE_DEGREE_LATITUDE_KM,
        amplitude_threshold_meters=0.04,
    )
    new_rows = sorted(
        (row.polarity, row.latitude, row.longitude) for row in detections.itertuples(index=False)
    )
    assert new_rows == legacy_rows
