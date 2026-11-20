# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray
from scipy.ndimage import gaussian_filter

import oceanbench.core.smoothing_controlled_rmsd as smoothing_controlled_rmsd
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ocean_mask import OCEAN_MASK_DEPTHS
from oceanbench.core.scores import score_rows, scores_document
from oceanbench.core.regions import resolve_region
from oceanbench.core.smoothing_controlled_rmsd import (
    GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    KILOMETERS_PER_DEGREE,
    ROW_DIAGNOSTICS_ATTRIBUTE,
    SIGMAS_KILOMETERS,
    SMOOTHING_METHODS,
    gaussian_means,
    prepare_climatology_variable,
    score_smoothing_controlled_rmsd,
    smoothing_controlled_rmsd_class4_validation,
    stencil_offsets_kilometers,
    stencil_response,
    stencil_weights,
    sum_of_squared_errors,
)

TEMPERATURE = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
FIRST_DAYS = pandas.date_range("2024-01-03", periods=6, freq="7D").values
LEAD_DAYS_COUNT = 2
MODEL_DEPTHS = numpy.array([0.494025, 47.37369])
FIRST_LATITUDE = 20.0
FIRST_LONGITUDE = -50.0
DOMAIN_DEGREES = 10.0
EDDY_RADIUS_KILOMETERS = 40.0
EDDY_COUNT = 15
DISPLACEMENT_KILOMETERS = 45.0
OBSERVATIONS_PER_DAY = 300
OBSERVATION_NOISE = 0.1


def _grid(step: float) -> tuple[numpy.ndarray, numpy.ndarray]:
    count = int(round(DOMAIN_DEGREES / step)) + 1
    return FIRST_LATITUDE + step * numpy.arange(count), FIRST_LONGITUDE + step * numpy.arange(count)


def _climatology_values(latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> numpy.ndarray:
    """A climatology with fronts of a few degrees, so its Gaussian means differ from its point values."""
    return (
        20.0
        - 0.3 * (latitudes - FIRST_LATITUDE)
        + 1.5
        * numpy.sin(numpy.deg2rad(120.0 * (latitudes - FIRST_LATITUDE)))
        * numpy.cos(numpy.deg2rad(90.0 * (longitudes - FIRST_LONGITUDE)))
    )


def _climatology_field(latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> numpy.ndarray:
    return _climatology_values(latitudes[:, None], longitudes[None, :])


def _eddies(seed: int) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    generator = numpy.random.default_rng(seed)
    return (
        generator.uniform(FIRST_LATITUDE, FIRST_LATITUDE + DOMAIN_DEGREES, EDDY_COUNT),
        generator.uniform(FIRST_LONGITUDE, FIRST_LONGITUDE + DOMAIN_DEGREES, EDDY_COUNT),
        generator.choice([-1.0, 1.0], EDDY_COUNT),
    )


def _eddy_field(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    eddies: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
    displacement_kilometers: float = 0.0,
) -> numpy.ndarray:
    eddy_latitudes, eddy_longitudes, amplitudes = eddies
    directions = numpy.random.default_rng(len(eddy_latitudes)).uniform(0, 2 * numpy.pi, len(eddy_latitudes))
    shifted_latitudes = eddy_latitudes + displacement_kilometers * numpy.sin(directions) / KILOMETERS_PER_DEGREE
    shifted_longitudes = eddy_longitudes + displacement_kilometers * numpy.cos(directions) / (
        KILOMETERS_PER_DEGREE * numpy.cos(numpy.deg2rad(eddy_latitudes))
    )
    northward = (latitudes[..., None] - shifted_latitudes) * KILOMETERS_PER_DEGREE
    eastward = (
        (longitudes[..., None] - shifted_longitudes)
        * KILOMETERS_PER_DEGREE
        * numpy.cos(numpy.deg2rad(shifted_latitudes))
    )
    return numpy.sum(amplitudes * numpy.exp(-(northward**2 + eastward**2) / (2 * EDDY_RADIUS_KILOMETERS**2)), axis=-1)


def _forecast_anomalies(step: float, displacement_kilometers: float) -> numpy.ndarray:
    latitudes, longitudes = _grid(step)
    latitude_grid, longitude_grid = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    return numpy.stack(
        [
            numpy.stack(
                [
                    _eddy_field(latitude_grid, longitude_grid, _eddies(10 * start + lead), displacement_kilometers)
                    for lead in range(LEAD_DAYS_COUNT)
                ]
            )
            for start in range(len(FIRST_DAYS))
        ]
    )


def _challenger(anomalies: numpy.ndarray, step: float = 1 / 12) -> xarray.Dataset:
    latitudes, longitudes = _grid(step)
    values = anomalies[:, :, None, :, :] + _climatology_field(latitudes, longitudes)[None, None, None]
    return xarray.Dataset(
        {
            TEMPERATURE: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                numpy.repeat(values, len(MODEL_DEPTHS), axis=2),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(LEAD_DAYS_COUNT),
            Dimension.DEPTH.key(): MODEL_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _climatology() -> xarray.Dataset:
    latitudes, longitudes = _grid(1 / 12)
    field = _climatology_field(latitudes, longitudes)
    return xarray.Dataset(
        {
            TEMPERATURE: (
                [Dimension.TIME.key(), Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
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


def _observations() -> xarray.Dataset:
    generator = numpy.random.default_rng(1988)
    records = []
    for start, first_day in enumerate(FIRST_DAYS):
        for lead in range(LEAD_DAYS_COUNT):
            latitudes = generator.uniform(FIRST_LATITUDE + 2, FIRST_LATITUDE + DOMAIN_DEGREES - 2, OBSERVATIONS_PER_DAY)
            longitudes = generator.uniform(
                FIRST_LONGITUDE + 2, FIRST_LONGITUDE + DOMAIN_DEGREES - 2, OBSERVATIONS_PER_DAY
            )
            truth = _eddy_field(latitudes, longitudes, _eddies(10 * start + lead)) + _climatology_values(
                latitudes, longitudes
            )
            records.append(
                pandas.DataFrame(
                    {
                        Dimension.TIME.key(): first_day + numpy.timedelta64(lead, "D"),
                        Dimension.LATITUDE.key(): latitudes,
                        Dimension.LONGITUDE.key(): longitudes,
                        Dimension.FIRST_DAY_DATETIME.key(): first_day,
                        Dimension.DEPTH.key(): 0.0,
                        TEMPERATURE: truth + OBSERVATION_NOISE * generator.standard_normal(OBSERVATIONS_PER_DAY),
                    }
                )
            )
    dataframe = pandas.concat(records, ignore_index=True)
    return xarray.Dataset({column: ("observation", dataframe[column].values) for column in dataframe.columns})


def _ocean_mask() -> xarray.DataArray:
    latitudes, longitudes = _grid(1 / 12)
    return xarray.DataArray(
        numpy.full((len(OCEAN_MASK_DEPTHS), len(latitudes), len(longitudes)), True),
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


@pytest.fixture(scope="module")
def observations() -> xarray.Dataset:
    return _observations()


@pytest.fixture(scope="module")
def displaced_anomalies() -> numpy.ndarray:
    return _forecast_anomalies(1 / 12, DISPLACEMENT_KILOMETERS)


def _statistics(challenger: xarray.Dataset, observations: xarray.Dataset) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    captured = []
    original = smoothing_controlled_rmsd.score_smoothing_controlled_rmsd

    def capture(statistics: pandas.DataFrame) -> pandas.DataFrame:
        captured.append(statistics)
        return original(statistics)

    smoothing_controlled_rmsd.score_smoothing_controlled_rmsd = capture
    try:
        table = _controlled(challenger, observations)
    finally:
        smoothing_controlled_rmsd.score_smoothing_controlled_rmsd = original
    return captured[0], table


def _controlled(challenger: xarray.Dataset, observations: xarray.Dataset) -> pandas.DataFrame:
    return smoothing_controlled_rmsd_class4_validation(
        challenger_dataset=challenger,
        reference_dataset=observations,
        ocean_mask=_ocean_mask(),
        variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
        climatology_dataset=_climatology(),
    )


def _plain(challenger: xarray.Dataset, observations: xarray.Dataset) -> pandas.DataFrame:
    return rmsd_class4_validation(
        challenger_dataset=challenger,
        reference_dataset=observations,
        ocean_mask=_ocean_mask(),
        variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
    )


def _in_sample_rmsd(statistics: pandas.DataFrame) -> pandas.Series:
    def best(group: pandas.DataFrame) -> float:
        sums = group.groupby("width_km")[["count", "sum_yy", "sum_aa", "sum_ya"]].sum()
        errors = sums["sum_yy"] - sums["sum_ya"] ** 2 / sums["sum_aa"]
        return float(numpy.sqrt(errors.min() / sums["count"].iloc[0]))

    return statistics.groupby("lead_day").apply(best)


def _lead_columns(table: pandas.DataFrame) -> list[str]:
    return [column for column in table.columns if column.startswith("Lead day")]


def test_stencil_response_is_within_five_hundredths_of_the_gaussian_for_wavelengths_from_20_km() -> None:
    wavelengths = numpy.concatenate([numpy.geomspace(20.0, 20000.0, 600), [1e7]])
    directions = numpy.linspace(0.0, numpy.pi, 721)
    response = stencil_response(wavelengths, directions)
    target = numpy.exp(
        -2 * numpy.pi**2 * numpy.array(SIGMAS_KILOMETERS)[:, None, None] ** 2 / wavelengths[None, :, None] ** 2
    )

    assert numpy.abs(response - target).max() < 0.05
    numpy.testing.assert_allclose(stencil_weights().sum(axis=1), 1.0)
    assert (stencil_weights() >= 0).all()
    assert len(stencil_offsets_kilometers()) == 331


def test_forcing_no_smoothing_and_unit_slope_reproduces_the_rmsd(observations, displaced_anomalies) -> None:
    challenger = _challenger(displaced_anomalies)
    statistics, _ = _statistics(challenger, observations)
    plain = _plain(challenger, observations)

    forced = [
        numpy.sqrt(
            sum_of_squared_errors(statistics[statistics["lead_day"] == lead], 0.0, 1.0)
            / statistics.loc[(statistics["lead_day"] == lead) & (statistics["width_km"] == 0.0), "count"].sum()
        )
        for lead in range(LEAD_DAYS_COUNT)
    ]

    numpy.testing.assert_allclose(forced, plain[_lead_columns(plain)].iloc[0].values, rtol=1e-12)


def test_the_table_has_the_shape_and_counts_of_the_rmsd_table(observations, displaced_anomalies) -> None:
    challenger = _challenger(displaced_anomalies)
    plain = _plain(challenger, observations)
    controlled = _controlled(challenger, observations)

    assert controlled.index.tolist() == plain.index.tolist()
    assert controlled.columns.tolist() == plain.columns.tolist()
    pandas.testing.assert_frame_equal(controlled[["Observations", "Missing"]], plain[["Observations", "Missing"]])
    diagnostics = controlled.attrs[ROW_DIAGNOSTICS_ATTRIBUTE]
    assert set(diagnostics["smoothing_width_km"].iloc[0]) <= {0.0, *SIGMAS_KILOMETERS}


def test_the_rmsd_table_is_unchanged_by_the_controlled_metric(observations, displaced_anomalies) -> None:
    challenger = _challenger(displaced_anomalies)
    before = _plain(challenger, observations)
    _controlled(challenger, observations)
    after = _plain(challenger, observations)

    pandas.testing.assert_frame_equal(before, after, check_exact=True)


@pytest.mark.parametrize("factor", [0.3, 2.5])
def test_damping_or_inflating_the_anomaly_leaves_the_score_unchanged(observations, displaced_anomalies, factor) -> None:
    reference_statistics, reference_table = _statistics(_challenger(displaced_anomalies), observations)
    scaled_statistics, scaled_table = _statistics(_challenger(factor * displaced_anomalies), observations)

    numpy.testing.assert_allclose(
        _in_sample_rmsd(scaled_statistics), _in_sample_rmsd(reference_statistics), rtol=0, atol=1e-10
    )
    lead_columns = _lead_columns(reference_table)
    numpy.testing.assert_allclose(
        scaled_table[lead_columns].values, reference_table[lead_columns].values, rtol=0, atol=1e-10
    )


def test_the_in_sample_controlled_rmsd_is_not_above_the_rmsd(observations, displaced_anomalies) -> None:
    challenger = _challenger(displaced_anomalies)
    statistics, _ = _statistics(challenger, observations)
    plain = _plain(challenger, observations)

    assert (_in_sample_rmsd(statistics).values <= plain[_lead_columns(plain)].iloc[0].values + 1e-12).all()


def _blurred(anomalies: numpy.ndarray, sigma_kilometers: float) -> numpy.ndarray:
    grid_kilometers = KILOMETERS_PER_DEGREE / 12
    return gaussian_filter(
        anomalies,
        sigma=(0, 0, sigma_kilometers / grid_kilometers, sigma_kilometers / grid_kilometers / numpy.cos(0.43)),
        mode="nearest",
    )


def _controlled_values(anomalies: numpy.ndarray, observations: xarray.Dataset) -> numpy.ndarray:
    table = _controlled(_challenger(anomalies), observations)
    return table[_lead_columns(table)].iloc[0].values


def test_a_blurred_forecast_does_not_beat_the_original_by_more_than_one_percent(
    observations, displaced_anomalies
) -> None:
    original = _controlled_values(displaced_anomalies, observations)
    plain_original = _plain(_challenger(displaced_anomalies), observations)
    plain_blurred = _plain(_challenger(_blurred(displaced_anomalies, 30.0)), observations)
    blurred = _controlled_values(_blurred(displaced_anomalies, 30.0), observations)

    assert (
        plain_blurred[_lead_columns(plain_blurred)].values < plain_original[_lead_columns(plain_original)].values
    ).all()
    assert (blurred >= 0.99 * original).all()


def test_blur_with_white_noise_or_fake_eddies_is_worse_than_the_original(observations, displaced_anomalies) -> None:
    original = _controlled_values(displaced_anomalies, observations)
    blurred = _blurred(displaced_anomalies, 30.0)
    noise = 0.3 * numpy.random.default_rng(7).standard_normal(blurred.shape)
    fake_eddies = numpy.stack(
        [_forecast_anomalies(1 / 12, 0.0)[(start + 1) % len(FIRST_DAYS)] * 0.5 for start in range(len(FIRST_DAYS))]
    )

    assert (_controlled_values(blurred + noise, observations) > original).all()
    assert (_controlled_values(blurred + fake_eddies, observations) > original).all()


_INTERIOR_OBSERVATIONS = pandas.DataFrame(
    {
        Dimension.LATITUDE.key(): numpy.linspace(23.0, 27.0, 40),
        Dimension.LONGITUDE.key(): numpy.linspace(-47.0, -43.0, 40),
        Dimension.DEPTH.key(): 0.0,
    }
)


def _eddy_data_array(step: float, land_east_of: float | None = None) -> xarray.DataArray:
    latitudes, longitudes = _grid(step)
    latitude_grid, longitude_grid = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    values = _eddy_field(latitude_grid, longitude_grid, _eddies(3))[None]
    if land_east_of is not None:
        values[:, :, longitudes > land_east_of] = numpy.nan
    return xarray.DataArray(
        values,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): [0.0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


@pytest.mark.parametrize("smoothing_method", SMOOTHING_METHODS)
def test_gaussian_means_of_one_field_on_quarter_and_twelfth_degree_grids_agree_from_30_km(smoothing_method) -> None:
    def means(step: float) -> numpy.ndarray:
        return gaussian_means(_eddy_data_array(step), _INTERIOR_OBSERVATIONS, numpy.array([0.0]), smoothing_method)

    quarter, twelfth = means(1 / 4), means(1 / 12)
    from_30_km = numpy.array(SIGMAS_KILOMETERS) >= 30

    assert numpy.abs(quarter[from_30_km] - twelfth[from_30_km]).max() < 0.03 * numpy.abs(twelfth).max()


@pytest.mark.parametrize("step", [1 / 4, 1 / 12])
def test_grid_and_stencil_gaussian_means_agree_within_five_hundredths(step) -> None:
    field = _eddy_data_array(step)
    grid = gaussian_means(field, _INTERIOR_OBSERVATIONS, numpy.array([0.0]), "grid")
    stencil = gaussian_means(field, _INTERIOR_OBSERVATIONS, numpy.array([0.0]), "stencil")

    assert numpy.abs(grid - stencil).max() < 0.05 * numpy.abs(field.values).max()


def test_grid_and_stencil_controlled_rmsd_agree_within_one_percent(observations, displaced_anomalies) -> None:
    challenger = _challenger(displaced_anomalies)
    tables = {
        smoothing_method: smoothing_controlled_rmsd_class4_validation(
            challenger_dataset=challenger,
            reference_dataset=observations,
            ocean_mask=_ocean_mask(),
            variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
            climatology_dataset=_climatology(),
            smoothing_method=smoothing_method,
        )
        for smoothing_method in SMOOTHING_METHODS
    }
    lead_columns = _lead_columns(tables["grid"])

    numpy.testing.assert_allclose(
        tables["grid"][lead_columns].values, tables["stencil"][lead_columns].values, rtol=0.01
    )


@pytest.mark.parametrize("smoothing_method", SMOOTHING_METHODS)
def test_gaussian_means_skip_land_and_renormalise(smoothing_method) -> None:
    latitudes, longitudes = _grid(1 / 12)
    values = numpy.full((1, len(latitudes), len(longitudes)), 2.0)
    values[:, :, longitudes > -45.0] = numpy.nan
    field = xarray.DataArray(
        values,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): [0.0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )
    observation_group = pandas.DataFrame(
        {Dimension.LATITUDE.key(): [25.0], Dimension.LONGITUDE.key(): [-45.5], Dimension.DEPTH.key(): [0.0]}
    )

    numpy.testing.assert_allclose(
        gaussian_means(field, observation_group, numpy.array([0.0]), smoothing_method), 2.0, rtol=1e-9
    )


def _statistics_rows(first_days: list[str], sum_ya: list[float]) -> pandas.DataFrame:
    return pandas.DataFrame(
        {
            "first_day": pandas.to_datetime(numpy.repeat(first_days, 2)),
            "depth_bin": "surface",
            "lead_day": 0,
            "width_km": [0.0, 15.0] * len(first_days),
            "count": 10,
            "sum_yy": 10.0,
            "sum_aa": 10.0,
            "sum_ya": sum_ya,
        }
    )


def test_scoring_fits_on_one_fold_and_scores_the_other() -> None:
    statistics = _statistics_rows(["2024-01-03", "2024-01-10"], [8.0, 2.0, 2.0, 8.0])

    score = score_smoothing_controlled_rmsd(statistics).iloc[0]

    assert score["cross_fitted"]
    expected_error = 2 * (10.0 - 2 * 0.8 * 2.0 + 0.8**2 * 10.0)
    assert score["rmsd"] == pytest.approx(numpy.sqrt(expected_error / 20))


def test_scoring_a_single_start_is_in_sample() -> None:
    statistics = _statistics_rows(["2024-01-03"], [5.0, 8.0])

    score = score_smoothing_controlled_rmsd(statistics).iloc[0]

    assert not score["cross_fitted"]
    assert score["width_km"] == 15.0
    assert score["slope"] == pytest.approx(0.8)
    assert score["rmsd"] == pytest.approx(numpy.sqrt((10.0 - 6.4) / 10))


def test_scores_file_rows_carry_the_fitted_width_and_slope(observations, displaced_anomalies) -> None:
    challenger = _challenger(displaced_anomalies)
    table = _controlled(challenger, observations)
    rows = score_rows("smoothing_controlled_rmsd_of_variables_compared_to_observations", table)

    assert {row["metric"] for row in rows} == {"smoothing_controlled_rmsd_of_variables"}
    assert all(row["smoothing_width_km"] in {0.0, *SIGMAS_KILOMETERS} for row in rows)
    scores_document(
        "synthetic",
        "",
        challenger,
        resolve_region("global"),
        {"smoothing_controlled_rmsd_of_variables_compared_to_observations": table},
        {},
    )


def test_climatology_sea_level_uses_the_glorys_shift(monkeypatch) -> None:
    latitudes, longitudes = _grid(1 / 12)
    coordinates = {Dimension.LATITUDE.key(): latitudes, Dimension.LONGITUDE.key(): longitudes}
    dimensions = [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    sea_surface_height = xarray.DataArray(
        numpy.full((len(latitudes), len(longitudes)), 0.5), dims=dimensions, coords=coordinates
    )
    mean_dynamic_topography = xarray.DataArray(
        numpy.full((len(latitudes), len(longitudes)), 0.2), dims=dimensions, coords=coordinates
    )
    monkeypatch.setattr(
        smoothing_controlled_rmsd, "load_mean_dynamic_topography", lambda resolution: mean_dynamic_topography
    )

    sea_level_anomaly = prepare_climatology_variable(sea_surface_height, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())

    numpy.testing.assert_allclose(sea_level_anomaly.values, 0.5 - 0.2 - GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT)
