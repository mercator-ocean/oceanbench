# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray
from scipy.ndimage import gaussian_filter

from oceanbench.core.dataset_utils import DepthLevel, Dimension, Variable
from oceanbench.core.ocean_mask import OCEAN_MASK_DEPTHS
from oceanbench.core.rmsd import MISSING_FRACTION_COLUMN, rmsd
from oceanbench.core.smoothing_controlled_gridded_rmsd import smoothing_controlled_rmsd_gridded
from oceanbench.core.smoothing_controlled_rmsd import (
    KILOMETERS_PER_DEGREE,
    ROW_DIAGNOSTICS_ATTRIBUTE,
    SMOOTHING_WIDTH_ROW_FIELD,
    SUFFICIENT_STATISTICS_ATTRIBUTE,
    grid_smoothed_fields,
)

TEMPERATURE = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
FIRST_DAYS = pandas.date_range("2023-01-04", periods=6, freq="7D").values
LEAD_DAYS_COUNT = 2
MODEL_DEPTHS = numpy.array([level.value for level in DepthLevel])
STEP_DEGREES = 1 / 12
FIRST_LATITUDE = 20.0
FIRST_LONGITUDE = -50.0
DOMAIN_DEGREES = 6.0
EDDY_RADIUS_KILOMETERS = 40.0
EDDY_COUNT = 12
DISPLACEMENT_KILOMETERS = 45.0
SPATIAL_DIMENSIONS = [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
FORECAST_DIMENSIONS = [Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key(), Dimension.DEPTH.key()]


def _grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    count = int(round(DOMAIN_DEGREES / STEP_DEGREES)) + 1
    return FIRST_LATITUDE + STEP_DEGREES * numpy.arange(count), FIRST_LONGITUDE + STEP_DEGREES * numpy.arange(count)


def _climatology_field() -> numpy.ndarray:
    latitudes, longitudes = _grid()
    return (
        20.0
        - 0.3 * (latitudes[:, None] - FIRST_LATITUDE)
        + 1.5
        * numpy.sin(numpy.deg2rad(120.0 * (latitudes[:, None] - FIRST_LATITUDE)))
        * numpy.cos(numpy.deg2rad(90.0 * (longitudes[None, :] - FIRST_LONGITUDE)))
    )


def _eddy_field(seed: int, displacement_kilometers: float) -> numpy.ndarray:
    generator = numpy.random.default_rng(seed)
    eddy_latitudes = generator.uniform(FIRST_LATITUDE, FIRST_LATITUDE + DOMAIN_DEGREES, EDDY_COUNT)
    eddy_longitudes = generator.uniform(FIRST_LONGITUDE, FIRST_LONGITUDE + DOMAIN_DEGREES, EDDY_COUNT)
    amplitudes = generator.choice([-1.0, 1.0], EDDY_COUNT)
    directions = generator.uniform(0, 2 * numpy.pi, EDDY_COUNT)
    eddy_latitudes = eddy_latitudes + displacement_kilometers * numpy.sin(directions) / KILOMETERS_PER_DEGREE
    eddy_longitudes = eddy_longitudes + displacement_kilometers * numpy.cos(directions) / (
        KILOMETERS_PER_DEGREE * numpy.cos(numpy.deg2rad(eddy_latitudes))
    )
    latitudes, longitudes = _grid()
    northward = (latitudes[:, None, None] - eddy_latitudes) * KILOMETERS_PER_DEGREE
    eastward = (
        (longitudes[None, :, None] - eddy_longitudes) * KILOMETERS_PER_DEGREE * numpy.cos(numpy.deg2rad(eddy_latitudes))
    )
    return numpy.sum(amplitudes * numpy.exp(-(northward**2 + eastward**2) / (2 * EDDY_RADIUS_KILOMETERS**2)), axis=-1)


def _anomalies(displacement_kilometers: float, seed_offset: int = 0) -> numpy.ndarray:
    """Anomalies of shape (first days, lead days, depths, latitudes, longitudes), weaker with depth."""
    fields = numpy.stack(
        [
            numpy.stack(
                [
                    _eddy_field(10 * start + lead + seed_offset, displacement_kilometers)
                    for lead in range(LEAD_DAYS_COUNT)
                ]
            )
            for start in range(len(FIRST_DAYS))
        ]
    )
    depth_factors = numpy.linspace(1.0, 0.4, len(MODEL_DEPTHS))
    return fields[:, :, None] * depth_factors[None, None, :, None, None]


def _dataset(anomalies: numpy.ndarray) -> xarray.Dataset:
    latitudes, longitudes = _grid()
    return xarray.Dataset(
        {TEMPERATURE: (FORECAST_DIMENSIONS + SPATIAL_DIMENSIONS, anomalies + _climatology_field())},
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(LEAD_DAYS_COUNT),
            Dimension.DEPTH.key(): MODEL_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _climatology() -> xarray.Dataset:
    latitudes, longitudes = _grid()
    field = _climatology_field()
    return xarray.Dataset(
        {
            TEMPERATURE: (
                [Dimension.TIME.key(), Dimension.DEPTH.key()] + SPATIAL_DIMENSIONS,
                numpy.broadcast_to(field, (12, len(MODEL_DEPTHS)) + field.shape).copy(),
            )
        },
        coords={
            Dimension.TIME.key(): pandas.date_range("2004-01-01", periods=12, freq="MS").values,
            Dimension.DEPTH.key(): MODEL_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _ocean_mask(land_east_of: float | None = None) -> xarray.DataArray:
    latitudes, longitudes = _grid()
    is_ocean = numpy.full((len(OCEAN_MASK_DEPTHS), len(latitudes), len(longitudes)), True)
    if land_east_of is not None:
        is_ocean[:, :, longitudes > land_east_of] = False
    return xarray.DataArray(
        is_ocean,
        dims=[Dimension.DEPTH.key()] + SPATIAL_DIMENSIONS,
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


@pytest.fixture(scope="module")
def reference() -> xarray.Dataset:
    return _dataset(_anomalies(0.0))


@pytest.fixture(scope="module")
def displaced_anomalies() -> numpy.ndarray:
    return _anomalies(DISPLACEMENT_KILOMETERS)


def _controlled(challenger: xarray.Dataset, reference: xarray.Dataset, mask=None) -> pandas.DataFrame:
    return smoothing_controlled_rmsd_gridded(
        challenger_dataset=challenger,
        reference_dataset=reference,
        variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
        ocean_mask=_ocean_mask() if mask is None else mask,
        climatology_dataset=_climatology(),
    )


def _lead_values(table: pandas.DataFrame) -> numpy.ndarray:
    return table.drop(columns=[MISSING_FRACTION_COLUMN]).values.astype(float)


def test_forcing_no_smoothing_and_unit_slope_reproduces_the_gridded_rmsd(reference, displaced_anomalies) -> None:
    challenger = _dataset(displaced_anomalies)
    land_mask = _ocean_mask(land_east_of=FIRST_LONGITUDE + 4.0)
    table = _controlled(challenger, reference, land_mask)
    statistics = table.attrs[SUFFICIENT_STATISTICS_ATTRIBUTE][TEMPERATURE]
    unsmoothed = statistics.loc[statistics["width_km"] == 0.0].copy()
    unsmoothed["rmsd"] = numpy.sqrt(
        (unsmoothed["sum_yy"] - 2 * unsmoothed["sum_ya"] + unsmoothed["sum_aa"]) / unsmoothed["count"]
    )
    forced = unsmoothed.groupby(["depth_bin", "lead_day"])["rmsd"].mean().unstack("lead_day")
    plain = rmsd(challenger, reference, [Variable.SEA_WATER_POTENTIAL_TEMPERATURE], land_mask)
    depth_labels = [label.split("{")[1].rstrip("}") for label in plain.index]
    numpy.testing.assert_allclose(forced.loc[depth_labels].values, _lead_values(plain), rtol=1e-6)
    assert list(table.index) == list(plain.index)
    assert list(table.columns) == list(plain.columns)
    numpy.testing.assert_allclose(table[MISSING_FRACTION_COLUMN], plain[MISSING_FRACTION_COLUMN])


@pytest.mark.parametrize("factor", [0.3, 2.5])
def test_damping_or_inflating_the_anomaly_leaves_the_score_unchanged(reference, displaced_anomalies, factor) -> None:
    original = _lead_values(_controlled(_dataset(displaced_anomalies), reference))
    scaled = _lead_values(_controlled(_dataset(factor * displaced_anomalies), reference))
    numpy.testing.assert_allclose(scaled, original, rtol=1e-9)


def _blurred(anomalies: numpy.ndarray, sigma_kilometers: float) -> numpy.ndarray:
    sigma_cells = sigma_kilometers / (STEP_DEGREES * KILOMETERS_PER_DEGREE)
    zonal_cells = sigma_cells / numpy.cos(numpy.deg2rad(FIRST_LATITUDE + DOMAIN_DEGREES / 2))
    return gaussian_filter(anomalies, sigma=(0, 0, 0, sigma_cells, zonal_cells), mode="nearest")


def test_a_blurred_forecast_does_not_beat_the_original_by_more_than_two_percent(reference, displaced_anomalies) -> None:
    """Without observation noise the gain of a 30 km blur reaches 1.6 percent here, between two fitted widths."""
    original = _lead_values(_controlled(_dataset(displaced_anomalies), reference))
    for sigma_kilometers in (15.0, 30.0):
        blurred = _lead_values(_controlled(_dataset(_blurred(displaced_anomalies, sigma_kilometers)), reference))
        assert numpy.all(blurred >= 0.98 * original)


def test_blur_with_noise_or_fake_eddies_is_worse_than_the_original(reference, displaced_anomalies) -> None:
    original = _lead_values(_controlled(_dataset(displaced_anomalies), reference))
    blurred = _blurred(displaced_anomalies, 30.0)
    noise = 1.0 * numpy.random.default_rng(7).standard_normal(displaced_anomalies.shape)
    fake_eddies = 0.5 * _anomalies(0.0, seed_offset=1000)
    for degraded in (blurred + noise, blurred + fake_eddies):
        assert numpy.all(_lead_values(_controlled(_dataset(degraded), reference)) > original)


def test_a_forecast_equal_to_the_reference_scores_zero(reference) -> None:
    table = _controlled(reference, reference)
    numpy.testing.assert_allclose(_lead_values(table), 0.0, atol=1e-6)
    widths = table.attrs[ROW_DIAGNOSTICS_ATTRIBUTE][SMOOTHING_WIDTH_ROW_FIELD]
    assert numpy.all(widths.values.astype(float) == 0.0)


def test_land_values_do_not_reach_the_score(reference, displaced_anomalies) -> None:
    land_mask = _ocean_mask(land_east_of=FIRST_LONGITUDE + 3.0)
    longitudes = _grid()[1]
    polluted = displaced_anomalies.copy()
    polluted[..., longitudes > FIRST_LONGITUDE + 3.0] = 1e6
    clean = _lead_values(_controlled(_dataset(displaced_anomalies), reference, land_mask))
    dirty = _lead_values(_controlled(_dataset(polluted), reference, land_mask))
    numpy.testing.assert_allclose(dirty, clean, rtol=1e-12)
    statistics = _controlled(_dataset(displaced_anomalies), reference, land_mask).attrs[
        SUFFICIENT_STATISTICS_ATTRIBUTE
    ][TEMPERATURE]
    full = _controlled(_dataset(displaced_anomalies), reference).attrs[SUFFICIENT_STATISTICS_ATTRIBUTE][TEMPERATURE]
    assert statistics["count"].sum() < 0.6 * full["count"].sum()


def _second_moment_kilometers(field: numpy.ndarray, coordinates_kilometers: numpy.ndarray) -> float:
    profile = field / field.sum()
    centre = numpy.sum(profile * coordinates_kilometers)
    return float(numpy.sqrt(numpy.sum(profile * (coordinates_kilometers - centre) ** 2)))


@pytest.mark.parametrize("centre_latitude", [0.0, 60.0])
def test_the_kernel_is_in_km_at_every_latitude(centre_latitude) -> None:
    step = 1 / 12
    latitudes = centre_latitude + step * numpy.arange(-60, 61)
    longitudes = step * numpy.arange(-120, 121)
    bump_sigma_kilometers = 20.0
    northward = latitudes[:, None] * 0 + (latitudes[:, None] - centre_latitude) * KILOMETERS_PER_DEGREE
    eastward = longitudes[None, :] * KILOMETERS_PER_DEGREE * numpy.cos(numpy.deg2rad(centre_latitude))
    bump = numpy.exp(-(northward**2 + eastward**2) / (2 * bump_sigma_kilometers**2))
    field = xarray.DataArray(
        bump, dims=SPATIAL_DIMENSIONS, coords={SPATIAL_DIMENSIONS[0]: latitudes, SPATIAL_DIMENSIONS[1]: longitudes}
    )
    smoothed = grid_smoothed_fields(field)
    for sigma_kilometers, smoothed_field in zip((15.0, 30.0, 45.0), smoothed[:3]):
        expected = numpy.hypot(bump_sigma_kilometers, sigma_kilometers)
        values = smoothed_field.values
        centre_row = values[60]
        centre_column = values[:, 120]
        zonal = _second_moment_kilometers(centre_row, eastward[0])
        meridional = _second_moment_kilometers(centre_column, northward[:, 0])
        assert zonal == pytest.approx(expected, rel=0.02)
        assert meridional == pytest.approx(expected, rel=0.02)
