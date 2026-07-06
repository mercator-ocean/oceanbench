# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
from scipy.ndimage import gaussian_filter1d

from oceanbench.core import eddies
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.runner import realism, records

SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()


def _context() -> records.RunContext:
    return records.RunContext(
        challenger="synthetic",
        challenger_version="0.0.0",
        year=2024,
        region="global",
        oceanbench_version="test",
    )


def _sea_surface_height_dataset(
    values: numpy.ndarray,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    start_dates: numpy.ndarray,
    lead_days: numpy.ndarray,
) -> xarray.Dataset:
    return xarray.Dataset(
        {
            SEA_SURFACE_HEIGHT_KEY: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
                {"standard_name": SEA_SURFACE_HEIGHT_KEY},
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): start_dates,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _metric_frame(result: realism.RealismResult) -> "object":
    return records.records_to_dataframe(result.records)


def test_single_wavelength_field_attributes_energy_to_the_expected_band() -> None:
    latitudes = numpy.linspace(-5.0, 5.0, 11)
    longitudes = numpy.arange(0.0, 360.0, 1.0)
    start_dates = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
    lead_days = numpy.arange(2)
    large_scale_pattern = numpy.sin(numpy.deg2rad(longitudes) * 5.0)
    values = numpy.broadcast_to(
        large_scale_pattern[None, None, None, :], (2, 2, len(latitudes), len(longitudes))
    ).copy()
    dataset = _sea_surface_height_dataset(values, latitudes, longitudes, start_dates, lead_days)

    result = realism.compute_realism_battery(
        dataset, {"glorys": dataset}, region="global", context=_context(), lead_days=(1,)
    )
    frame = _metric_frame(result)
    band_fractions = frame[
        (frame["metric"] == records.METRIC_PSD_BAND_ENERGY_FRACTION) & (frame["lead_day"] == 1)
    ].set_index("band")["value"]

    dominant_band = band_fractions.idxmax()
    assert dominant_band == realism.BAND_LARGE
    assert band_fractions[realism.BAND_LARGE] > 0.5


def test_smoothed_challenger_has_a_finite_effective_resolution_cutoff() -> None:
    latitudes = numpy.linspace(-5.0, 5.0, 11)
    longitudes = numpy.arange(0.0, 360.0, 1.0)
    start_dates = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
    lead_days = numpy.arange(1)
    generator = numpy.random.default_rng(0)
    reference_values = generator.standard_normal((2, 1, len(latitudes), len(longitudes)))
    challenger_values = gaussian_filter1d(reference_values, sigma=6.0, axis=-1, mode="wrap")

    reference = _sea_surface_height_dataset(reference_values, latitudes, longitudes, start_dates, lead_days)
    challenger = _sea_surface_height_dataset(challenger_values, latitudes, longitudes, start_dates, lead_days)

    result = realism.compute_realism_battery(
        challenger, {"glorys": reference}, region="global", context=_context(), lead_days=(1,)
    )
    frame = _metric_frame(result)
    effective_resolution = frame[
        (frame["metric"] == records.METRIC_EFFECTIVE_RESOLUTION_KILOMETRES) & (frame["lead_day"] == 1)
    ]["value"].iloc[0]

    assert numpy.isfinite(effective_resolution)
    assert effective_resolution > 0.0


def test_damped_challenger_activity_ratio_is_below_one() -> None:
    latitudes = numpy.linspace(-5.0, 5.0, 11)
    longitudes = numpy.arange(0.0, 360.0, 1.0)
    start_dates = numpy.array(["2024-01-03", "2024-01-10", "2024-01-17"], dtype="datetime64[ns]")
    lead_days = numpy.arange(1)
    base_pattern = numpy.sin(numpy.deg2rad(longitudes) * 8.0)[None, None, None, :]
    start_amplitudes = numpy.array([1.0, 1.5, 2.0])[:, None, None, None]
    reference_values = numpy.broadcast_to(base_pattern, (3, 1, len(latitudes), len(longitudes))) * start_amplitudes
    challenger_values = 0.4 * reference_values

    reference = _sea_surface_height_dataset(reference_values, latitudes, longitudes, start_dates, lead_days)
    challenger = _sea_surface_height_dataset(challenger_values, latitudes, longitudes, start_dates, lead_days)

    result = realism.compute_realism_battery(
        challenger, {"glorys": reference}, region="global", context=_context(), lead_days=(1,)
    )
    frame = _metric_frame(result)
    activity_ratio = frame[(frame["metric"] == records.METRIC_ACTIVITY_RATIO) & (frame["lead_day"] == 1)]["value"].iloc[
        0
    ]

    assert activity_ratio < 1.0
    assert numpy.isclose(activity_ratio, 0.4, atol=1.0e-6)


def _gaussian_eddy_field(latitudes: numpy.ndarray, longitudes: numpy.ndarray, centre_longitude: float) -> numpy.ndarray:
    latitude_grid = latitudes[:, None]
    longitude_grid = longitudes[None, :]
    return 0.5 * numpy.exp(-(((longitude_grid - centre_longitude) / 2.0) ** 2 + (latitude_grid / 2.0) ** 2))


def test_two_shifted_gaussian_eddies_produce_one_match_with_expected_displacement() -> None:
    latitudes = numpy.linspace(-15.0, 15.0, 31)
    longitudes = numpy.arange(0.0, 60.0, 1.0)
    start_dates = numpy.array(["2024-01-03"], dtype="datetime64[ns]")
    lead_days = numpy.array([0])
    challenger_values = _gaussian_eddy_field(latitudes, longitudes, 15.0)[None, None]
    reference_values = _gaussian_eddy_field(latitudes, longitudes, 16.0)[None, None]

    challenger = _sea_surface_height_dataset(challenger_values, latitudes, longitudes, start_dates, lead_days)
    reference = _sea_surface_height_dataset(reference_values, latitudes, longitudes, start_dates, lead_days)

    result = realism.compute_realism_battery(
        challenger,
        {"glorys": reference},
        region="global",
        context=_context(),
        lead_days=(1,),
        start_indices=[0],
        eddy_start_indices=[0],
    )
    frame = _metric_frame(result)
    anticyclone = frame[(frame["polarity"] == "anticyclone") & (frame["lead_day"] == 1)].set_index("metric")["value"]

    assert anticyclone[records.METRIC_EDDY_COUNT] == 1.0
    assert anticyclone[records.METRIC_EDDY_HIT_RATE] == 1.0
    assert anticyclone[records.METRIC_EDDY_MISS_RATE] == 0.0
    assert numpy.isclose(anticyclone[records.METRIC_EDDY_MEAN_DISPLACEMENT_KILOMETRES], 111.2, atol=2.0)

    census_frame = result.eddy_census[0]["frames"][0]
    parameters = result.eddy_census[0]["parameters"]
    assert parameters["background_sigma_km"] == eddies.DEFAULT_BACKGROUND_SIGMA_KM
    assert parameters["apply_contour_filtering"] is False
    assert parameters["oceanbench_version"]
    assert len(census_frame["matches"]) == 1
    assert census_frame["matches"][0]["challenger"]["polarity"] == "anticyclone"
    assert numpy.isclose(census_frame["matches"][0]["displacement_km"], 111.2, atol=2.0)
