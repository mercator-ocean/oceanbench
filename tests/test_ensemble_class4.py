# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.classIV_support import interpolate_class4_model_to_observations
from oceanbench.core.curvilinear_staging import CurvilinearChallenger
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ensemble_class4 import (
    ENSEMBLE_DIMENSION,
    METRIC_CRPS_FAIR,
    METRIC_ENSEMBLE_MEAN_RMSD,
    METRIC_ENSEMBLE_SPREAD,
    METRIC_MEMBER_RMSD,
    METRIC_SIGMA_TOTAL_RMS,
    METRIC_SSR_ADD,
    METRIC_SSR_UNCORRECTED,
    Class4EnsembleMatchup,
    SigmaLookup,
    class4_group_statistics,
    class4_observation_type,
    crps_fair,
    dressed_rank_histogram,
    ensemble_class4_matchup,
    ensemble_class4_rank_histograms,
    ensemble_class4_records,
    group_metric_values,
    interpolate_class4_ensemble_to_observations,
    ranks_with_random_ties,
)
from oceanbench.core.score_records import RunContext, records_to_dataframe
from test_classiv_interpolation import _model_data, _observations_dataframe

DETERMINISTIC_MODEL_VALUES = [0.0, 11.5, 23.0, 100.75, 122.25]
MEMBER_OFFSETS = numpy.array([-1.0, 0.0, 2.0])


# ---------------------------------------------------------------------------
# Fair CRPS
# ---------------------------------------------------------------------------


def test_fair_crps_matches_a_hand_computed_case():
    """Members 1, 2, 4 against an observation of 3.

    The absolute error term is (2 + 1 + 1) / 3 = 4 / 3. The double sum counts every ordered
    pair, so it is 2 * (1 + 3 + 2) = 12, and the fair estimator divides it by
    2 * M * (M - 1) = 12. The fair CRPS is therefore 4 / 3 - 1 = 1 / 3.
    """
    members = numpy.array([[1.0, 2.0, 4.0]])
    observations = numpy.array([3.0])

    numpy.testing.assert_allclose(crps_fair(members, observations), [1.0 / 3.0], rtol=1e-12)


def test_fair_crps_matches_the_explicit_double_sum():
    generator = numpy.random.default_rng(3)
    members = generator.normal(size=(50, 7))
    observations = generator.normal(size=50)
    member_count = members.shape[1]
    absolute_error_term = numpy.abs(members - observations[:, None]).mean(axis=1)
    pairwise = numpy.abs(members[:, :, None] - members[:, None, :]).sum(axis=(1, 2))
    expected = absolute_error_term - pairwise / (2 * member_count * (member_count - 1))

    numpy.testing.assert_allclose(crps_fair(members, observations), expected, rtol=1e-12)


def test_fair_crps_at_one_member_is_the_mean_absolute_error():
    """The fair estimator divides by M - 1 and has no value at a single member.

    This module follows the gridded axis and returns the mean absolute error there, which is
    what the CRPS reduces to when the forecast is a point mass.
    """
    generator = numpy.random.default_rng(11)
    members = generator.normal(size=(40, 1))
    observations = generator.normal(size=40)

    numpy.testing.assert_allclose(crps_fair(members, observations), numpy.abs(members[:, 0] - observations))


# ---------------------------------------------------------------------------
# Spread-skill ratio
# ---------------------------------------------------------------------------


def _known_spread_case() -> tuple[numpy.ndarray, numpy.ndarray]:
    """Four members per observation whose sample variance and mean error are exact.

    Every row holds the members -3, -1, 1, 3 around its own centre, so the sample variance
    with one degree of freedom is 20 / 3 in every row. The centres sit two units below the
    observations, so the ensemble-mean squared error is 4 in every row.
    """
    members = numpy.array([[-3.0, -1.0, 1.0, 3.0]]) + numpy.array([[0.0], [10.0], [-4.0]])
    observations = numpy.array([2.0, 12.0, -2.0])
    return members, observations


def test_ssr_uncorrected_is_the_plain_spread_over_error():
    members, observations = _known_spread_case()
    statistics = class4_group_statistics(members, observations)
    values = group_metric_values(statistics)

    expected_spread = numpy.sqrt(5 / 4 * (20 / 3))
    assert values[METRIC_ENSEMBLE_SPREAD] == pytest.approx(expected_spread)
    assert values[METRIC_ENSEMBLE_MEAN_RMSD] == pytest.approx(2.0)
    assert values[METRIC_SSR_UNCORRECTED] == pytest.approx(expected_spread / 2.0)
    assert METRIC_SSR_ADD not in values
    assert METRIC_SIGMA_TOTAL_RMS not in values


def test_ssr_add_puts_the_observation_error_on_the_spread_side():
    members, observations = _known_spread_case()
    sigma_total = numpy.full(len(observations), 0.5)
    values = group_metric_values(class4_group_statistics(members, observations, sigma_total))

    expected_spread_variance = 5 / 4 * (20 / 3)
    assert values[METRIC_SSR_ADD] == pytest.approx(numpy.sqrt(expected_spread_variance + 0.25) / 2.0)
    assert values[METRIC_SSR_ADD] > values[METRIC_SSR_UNCORRECTED]
    assert values[METRIC_SIGMA_TOTAL_RMS] == pytest.approx(0.5)


def test_ssr_add_returns_to_one_for_a_calibrated_ensemble_with_observation_error():
    """A perfect-model ensemble whose observations carry a known extra error.

    The uncorrected ratio must fall below one, because the realised error contains an
    observation term the spread does not. The additive form adds that term to the spread and
    must come back to one, which is the whole reason the campaign switched conventions.
    """
    generator = numpy.random.default_rng(7)
    observation_count = 40000
    member_count = 25
    centre = generator.normal(size=observation_count)
    members = centre[:, None] + generator.normal(size=(observation_count, member_count))
    truth = centre + generator.normal(size=observation_count)
    sigma = 0.5
    observations = truth + sigma * generator.normal(size=observation_count)

    values = group_metric_values(class4_group_statistics(members, observations, numpy.full(observation_count, sigma)))

    assert values[METRIC_SSR_UNCORRECTED] < 0.98
    assert values[METRIC_SSR_ADD] == pytest.approx(1.0, abs=0.02)


def test_the_ensemble_mean_beats_the_typical_member():
    generator = numpy.random.default_rng(5)
    truth = generator.normal(size=500)
    members = truth[:, None] + generator.normal(scale=0.8, size=(500, 20))
    values = group_metric_values(class4_group_statistics(members, truth))

    assert values[METRIC_ENSEMBLE_MEAN_RMSD] < values[METRIC_MEMBER_RMSD]


# ---------------------------------------------------------------------------
# Rank histograms
# ---------------------------------------------------------------------------


def test_rank_histogram_counts_a_constructed_case_exactly():
    """Three members at 0, 1, 2 and four observations, one falling in each of the four bins.

    With no observation error the dressing draw has zero width, so the histogram is the
    plain rank histogram and every count is known in advance.
    """
    members = numpy.tile(numpy.array([0.0, 1.0, 2.0]), (4, 1))
    observations = numpy.array([-0.5, 0.5, 1.5, 2.5])
    generator = numpy.random.default_rng(0)

    histogram = dressed_rank_histogram(members, observations, numpy.zeros(4), generator, mode="member", draws=1)

    assert histogram.size == 4
    numpy.testing.assert_array_equal(histogram, [1.0, 1.0, 1.0, 1.0])


def test_rank_histogram_puts_every_observation_in_the_end_bins_when_the_ensemble_is_offset():
    members = numpy.tile(numpy.array([0.0, 1.0, 2.0]), (6, 1))
    observations = numpy.array([-1.0, -2.0, -3.0, 5.0, 6.0, 7.0])
    generator = numpy.random.default_rng(0)

    histogram = dressed_rank_histogram(members, observations, numpy.zeros(6), generator, mode="member", draws=1)

    numpy.testing.assert_array_equal(histogram, [3.0, 0.0, 0.0, 3.0])


def test_rank_histogram_averages_the_dressing_draws_without_changing_the_sample_size():
    generator = numpy.random.default_rng(1)
    members = generator.normal(size=(300, 8))
    observations = generator.normal(size=300)

    histogram = dressed_rank_histogram(members, observations, numpy.full(300, 0.3), generator, mode="member", draws=4)

    assert histogram.size == 9
    assert histogram.sum() == pytest.approx(300.0)


def test_obs_dressing_widens_the_observation_and_member_dressing_widens_the_members():
    generator = numpy.random.default_rng(2)
    members = numpy.tile(numpy.array([0.0, 1.0, 2.0]), (2000, 1))
    observations = numpy.full(2000, 1.0)

    member_dressed = dressed_rank_histogram(
        members, observations, numpy.full(2000, 5.0), generator, mode="member", draws=1
    )
    obs_dressed = dressed_rank_histogram(members, observations, numpy.full(2000, 5.0), generator, mode="obs", draws=1)

    # Dressing the observation alone throws it far outside a needle-thin ensemble, so almost
    # every rank lands in an end bin. Dressing the members spreads them instead.
    assert obs_dressed[[0, -1]].sum() > member_dressed[[0, -1]].sum()


def test_ranks_with_random_ties_stay_inside_the_bin_range():
    generator = numpy.random.default_rng(4)
    members = numpy.zeros((100, 5))
    observations = numpy.zeros(100)

    ranks = ranks_with_random_ties(members, observations, generator)

    assert ranks.min() >= 0
    assert ranks.max() <= 5


# ---------------------------------------------------------------------------
# Sigma lookup
# ---------------------------------------------------------------------------

SIGMA_OBSERVATION_TYPES = [
    "drifter_sst",
    "argo_temperature",
    "argo_salinity",
    "currents_u",
    "currents_v",
    "sla",
]
SIGMA_DEPTH_OBSERVATION_TYPES = ["argo_temperature", "argo_salinity"]
SIGMA_REGIONS = ["gulf_stream", "kuroshio", "gyre_interior", "ibi", "southern_ocean", "GLOBAL"]
SIGMA_BASES = ["rms_over_cells", "median_cell"]
SIGMA_INSTRUMENT = {"drifter_sst": 0.051, "argo_temperature": 0.002, "argo_salinity": 0.01, "sla": 0.02}
SIGMA_LEVEL_DEPTHS = numpy.array([0.494025, 30.0], dtype="float32")


def _synthetic_sigma_dataset() -> xarray.Dataset:
    """A tiny lookup with the schema of sigma-lookup-v3.0.0.

    Only the first few cells of the artifact grid are shipped, so every observation used in
    these tests sits in the far south-west corner. The cell arithmetic the loader verifies is
    anchored at (-90, -180) with a row offset of 40, which puts row 0 at latitude -80.
    """
    latitudes = -79.875 + 0.25 * numpy.arange(4, dtype="float32")
    longitudes = -179.875 + 0.25 * numpy.arange(4, dtype="float32")
    months = numpy.arange(1, 13, dtype="int8")
    shape = (len(SIGMA_OBSERVATION_TYPES), len(months), len(latitudes), len(longitudes))

    sigma_r = numpy.full(shape, 0.4, dtype="float32")
    days = numpy.full(shape, 10, dtype="int16")
    # One dry cell for drifter_sst in January, so the regional fallback is exercised.
    days[SIGMA_OBSERVATION_TYPES.index("drifter_sst"), 0, 1, 1] = 0

    depth_shape = (
        len(SIGMA_DEPTH_OBSERVATION_TYPES),
        len(months),
        len(SIGMA_LEVEL_DEPTHS),
        len(latitudes),
        len(longitudes),
    )
    sigma_r_z = numpy.empty(depth_shape, dtype="float32")
    sigma_r_z[:, :, 0] = 1.0
    sigma_r_z[:, :, 1] = 3.0

    return xarray.Dataset(
        {
            "sigma_r": (("obs_type", "month", "lat", "lon"), sigma_r),
            "n_days": (("obs_type", "month", "lat", "lon"), days),
            "sigma_i": (
                ("obs_type",),
                numpy.array([SIGMA_INSTRUMENT.get(name, 0.004) for name in SIGMA_OBSERVATION_TYPES], dtype="float32"),
            ),
            "sigma_r_fallback": (
                ("obs_type", "month", "region", "basis"),
                numpy.full(
                    (len(SIGMA_OBSERVATION_TYPES), len(months), len(SIGMA_REGIONS), len(SIGMA_BASES)),
                    0.7,
                    dtype="float32",
                ),
            ),
            "sigma_r_z": (("obs_type_z", "month", "level", "lat", "lon"), sigma_r_z),
            "n_days_z": (("obs_type_z", "month", "level", "lat", "lon"), numpy.full(depth_shape, 5, dtype="int16")),
            "sigma_r_fallback_z": (
                ("obs_type_z", "month", "level", "region", "basis"),
                numpy.full(
                    (
                        len(SIGMA_DEPTH_OBSERVATION_TYPES),
                        len(months),
                        len(SIGMA_LEVEL_DEPTHS),
                        len(SIGMA_REGIONS),
                        len(SIGMA_BASES),
                    ),
                    0.9,
                    dtype="float32",
                ),
            ),
        },
        coords={
            "obs_type": SIGMA_OBSERVATION_TYPES,
            "obs_type_z": SIGMA_DEPTH_OBSERVATION_TYPES,
            "region": SIGMA_REGIONS,
            "basis": SIGMA_BASES,
            "month": months,
            "lat": latitudes,
            "lon": longitudes,
            "level": numpy.arange(len(SIGMA_LEVEL_DEPTHS), dtype="int8"),
            "depth": ("level", SIGMA_LEVEL_DEPTHS),
        },
    )


@pytest.fixture
def sigma_lookup(tmp_path) -> SigmaLookup:
    store = tmp_path / "sigma-lookup-test.zarr"
    _synthetic_sigma_dataset().to_zarr(store, consolidated=True)
    return SigmaLookup(str(store))


def test_sigma_total_combines_the_instrument_and_representativity_terms(sigma_lookup):
    sigma_total, sigma_instrument, diagnostics = sigma_lookup.total(
        "sla", numpy.array([3]), numpy.array([-79.9]), numpy.array([-179.9])
    )

    assert sigma_instrument == pytest.approx(0.02, abs=1e-6)
    assert sigma_total[0] == pytest.approx(numpy.sqrt(0.02**2 + 0.4**2), abs=1e-6)
    assert diagnostics["sigma_r_fallback_rows"] == 0
    assert diagnostics["sigma_depth_resolved"] is False


def test_sigma_falls_back_to_the_region_value_where_the_cell_has_no_days(sigma_lookup):
    sigma_total, _instrument, diagnostics = sigma_lookup.total(
        "drifter_sst", numpy.array([1, 1]), numpy.array([-79.6, -79.9]), numpy.array([-179.6, -179.9])
    )

    assert diagnostics["sigma_r_fallback_rows"] == 1
    assert sigma_total[0] == pytest.approx(numpy.sqrt(0.051**2 + 0.7**2), abs=1e-6)
    assert sigma_total[1] == pytest.approx(numpy.sqrt(0.051**2 + 0.4**2), abs=1e-6)


def test_sigma_interpolates_linearly_in_depth_for_the_argo_streams(sigma_lookup):
    midpoint = float(SIGMA_LEVEL_DEPTHS[0] + SIGMA_LEVEL_DEPTHS[1]) / 2.0
    sigma_total, _instrument, diagnostics = sigma_lookup.total(
        "argo_temperature",
        numpy.array([6]),
        numpy.array([-79.9]),
        numpy.array([-179.9]),
        depth=numpy.array([midpoint]),
    )

    assert diagnostics["sigma_depth_resolved"] is True
    assert sigma_total[0] == pytest.approx(numpy.sqrt(0.002**2 + 2.0**2), abs=1e-5)


def test_sigma_clamps_outside_the_level_range_rather_than_extrapolating(sigma_lookup):
    shallow, _instrument, _diagnostics = sigma_lookup.total(
        "argo_temperature", numpy.array([6]), numpy.array([-79.9]), numpy.array([-179.9]), numpy.array([0.0])
    )
    deep, _instrument, _diagnostics = sigma_lookup.total(
        "argo_temperature", numpy.array([6]), numpy.array([-79.9]), numpy.array([-179.9]), numpy.array([5000.0])
    )

    assert shallow[0] == pytest.approx(numpy.sqrt(0.002**2 + 1.0**2), abs=1e-5)
    assert deep[0] == pytest.approx(numpy.sqrt(0.002**2 + 3.0**2), abs=1e-5)


def test_sigma_ignores_depth_for_a_stream_the_artifact_does_not_resolve(sigma_lookup):
    sigma_total, _instrument, diagnostics = sigma_lookup.total(
        "currents_u", numpy.array([2]), numpy.array([-79.9]), numpy.array([-179.9]), numpy.array([15.0])
    )

    assert diagnostics["sigma_depth_resolved"] is False
    assert sigma_total[0] == pytest.approx(numpy.sqrt(0.004**2 + 0.4**2), abs=1e-6)


def test_sigma_lookup_rejects_an_unknown_basis_or_region(tmp_path):
    store = tmp_path / "sigma-lookup-reject.zarr"
    _synthetic_sigma_dataset().to_zarr(store, consolidated=True)

    with pytest.raises(ValueError, match="basis"):
        SigmaLookup(str(store), basis="not_a_basis")
    with pytest.raises(ValueError, match="region"):
        SigmaLookup(str(store), fallback_region="atlantis")


def test_sigma_lookup_refuses_a_transposed_array():
    transposed = _synthetic_sigma_dataset()
    transposed["sigma_r_z"] = transposed["sigma_r_z"].transpose("obs_type_z", "level", "month", "lat", "lon")

    with pytest.raises(ValueError, match="sigma_r_z"):
        SigmaLookup(transposed)


def test_the_closing_edge_of_the_sigma_grid_resolves_to_the_last_cell(sigma_lookup):
    # The fixture ships the south-west corner only, so the coordinates of the full artifact
    # grid are put on the loader to reach the cells at latitude 90 and longitude 180.
    sigma_lookup.latitude = -79.875 + 0.25 * numpy.arange(680)
    sigma_lookup.longitude = -179.875 + 0.25 * numpy.arange(1440)

    row, column, inside = sigma_lookup.cell_index(numpy.array([90.0, 0.0]), numpy.array([180.0, 0.0]))

    assert row[0] == sigma_lookup.latitude.size - 1
    assert column[0] == sigma_lookup.longitude.size - 1
    assert inside.all()


def test_an_observation_south_of_the_sigma_grid_stays_outside_it(sigma_lookup):
    row, column, inside = sigma_lookup.cell_index(numpy.array([-85.0]), numpy.array([-179.9]))

    assert not inside[0]
    assert 0 <= row[0] < sigma_lookup.latitude.size
    assert 0 <= column[0] < sigma_lookup.longitude.size


def test_surface_temperature_is_scored_against_the_drifter_stream():
    temperature = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()

    assert class4_observation_type(temperature, "surface") == "drifter_sst"
    assert class4_observation_type(temperature, "100-300m") == "argo_temperature"
    assert class4_observation_type(Variable.SEA_WATER_SALINITY.key(), "0-5m") == "argo_salinity"


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------


def _ensemble_model_data() -> xarray.DataArray:
    """The deterministic test field, offset by a constant per member."""
    deterministic = _model_data()
    members = xarray.concat(
        [deterministic + offset for offset in MEMBER_OFFSETS],
        dim=ENSEMBLE_DIMENSION,
    ).assign_coords({ENSEMBLE_DIMENSION: numpy.arange(len(MEMBER_OFFSETS))})
    return members.rename(deterministic.name)


def test_the_deterministic_class4_matchup_is_unchanged():
    """Main's Class IV interpolation, called exactly as main calls it, still agrees.

    The ensemble module never reaches inside classIV_support, so this is the guard that the
    deterministic path kept its values while the member loop was added around it.
    """
    model_values = interpolate_class4_model_to_observations(_model_data(), _observations_dataframe())

    numpy.testing.assert_allclose(model_values, DETERMINISTIC_MODEL_VALUES)


def test_every_member_column_is_the_deterministic_matchup_of_that_member():
    observations = _observations_dataframe()

    member_values = interpolate_class4_ensemble_to_observations(_ensemble_model_data(), observations)

    assert member_values.shape == (len(observations), len(MEMBER_OFFSETS))
    for member_index, offset in enumerate(MEMBER_OFFSETS):
        expected = interpolate_class4_model_to_observations(_model_data() + offset, observations)
        numpy.testing.assert_allclose(member_values[:, member_index], expected)
        numpy.testing.assert_allclose(member_values[:, member_index], numpy.array(DETERMINISTIC_MODEL_VALUES) + offset)


def test_a_single_member_ensemble_reproduces_the_deterministic_values():
    observations = _observations_dataframe()
    single = _model_data().expand_dims({ENSEMBLE_DIMENSION: [0]})

    member_values = interpolate_class4_ensemble_to_observations(single, observations)

    numpy.testing.assert_allclose(member_values[:, 0], DETERMINISTIC_MODEL_VALUES)


def test_the_member_dimension_is_required():
    with pytest.raises(ValueError, match=ENSEMBLE_DIMENSION):
        interpolate_class4_ensemble_to_observations(_model_data(), _observations_dataframe())


def _observations_dataset() -> xarray.Dataset:
    """A Class IV observation store shaped like the reference one, for two forecast starts."""
    frame = _observations_dataframe()
    observation_count = len(frame)
    return xarray.Dataset(
        {
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                ("observation",),
                numpy.array(DETERMINISTIC_MODEL_VALUES) + 0.5,
            ),
            Dimension.TIME.key(): (("observation",), frame[Dimension.TIME.key()].to_numpy()),
            Dimension.LATITUDE.key(): (("observation",), frame[Dimension.LATITUDE.key()].to_numpy()),
            Dimension.LONGITUDE.key(): (("observation",), frame[Dimension.LONGITUDE.key()].to_numpy()),
            Dimension.FIRST_DAY_DATETIME.key(): (("observation",), frame["first_day"].to_numpy()),
            Dimension.DEPTH.key(): (("observation",), numpy.zeros(observation_count)),
        }
    )


def test_ensemble_matchup_runs_mains_pipeline_once_per_member():
    challenger = _ensemble_model_data().to_dataset()

    matchups = ensemble_class4_matchup(
        challenger,
        _observations_dataset(),
        [Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
    )

    assert len(matchups) == 1
    matchup = matchups[0]
    assert matchup.variable == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    assert matchup.member_count == len(MEMBER_OFFSETS)
    assert len(matchup.observations) == len(matchup.member_values)
    assert set(matchup.observations["depth_bin"]) == {"surface"}
    expected = numpy.array(DETERMINISTIC_MODEL_VALUES)[:, None] + MEMBER_OFFSETS[None, :]
    numpy.testing.assert_allclose(matchup.member_values, expected)


# ---------------------------------------------------------------------------
# Matchup of a challenger left on its native curvilinear grid
# ---------------------------------------------------------------------------

NATIVE_FIRST_DAY = numpy.datetime64("2024-01-04")
NATIVE_OBSERVATION_LATITUDES = numpy.array([40.1, 41.1, 42.1])
NATIVE_OBSERVATION_LONGITUDES = numpy.array([10.1, 11.1, 12.1])


def _native_tracer_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    row = numpy.arange(4, dtype="float64")
    column = numpy.arange(4, dtype="float64")
    return (
        numpy.broadcast_to(40.0 + row[:, numpy.newaxis], (4, 4)).copy(),
        numpy.broadcast_to(10.0 + column[numpy.newaxis, :], (4, 4)).copy(),
    )


def _native_challenger_dataset() -> xarray.Dataset:
    """A three-dimensional NEMO store, describing its cells twice as the real ones do."""
    latitude, longitude = _native_tracer_grid()
    dimensions = (
        ENSEMBLE_DIMENSION,
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        "y",
        "x",
    )
    field = numpy.arange(16.0).reshape(1, 1, 1, 4, 4)
    return xarray.Dataset(
        {
            "thetao": (dimensions, field, {"standard_name": "sea_water_potential_temperature"}),
            "zos": (dimensions, numpy.zeros((1, 1, 1, 4, 4)), {"standard_name": "sea_surface_height_above_geoid"}),
            "uo": (dimensions, numpy.ones((1, 1, 1, 4, 4)), {"standard_name": "sea_water_x_velocity"}),
            "vo": (dimensions, numpy.zeros((1, 1, 1, 4, 4)), {"standard_name": "sea_water_y_velocity"}),
        },
        coords={
            ENSEMBLE_DIMENSION: [0],
            Dimension.FIRST_DAY_DATETIME.key(): [NATIVE_FIRST_DAY],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            "nav_lat": (("y", "x"), latitude, {"standard_name": "latitude"}),
            "nav_lon": (("y", "x"), longitude, {"standard_name": "longitude"}),
            Dimension.LATITUDE.key(): (("y", "x"), latitude, {"standard_name": "latitude"}),
            Dimension.LONGITUDE.key(): (("y", "x"), longitude, {"standard_name": "longitude"}),
        },
    )


def _native_observations_dataset() -> xarray.Dataset:
    observation_count = len(NATIVE_OBSERVATION_LATITUDES)
    observation_values = numpy.zeros(observation_count)
    return xarray.Dataset(
        {
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (("observation",), observation_values),
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (("observation",), observation_values),
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (("observation",), observation_values),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (("observation",), observation_values),
            Dimension.TIME.key(): (
                ("observation",),
                numpy.full(observation_count, NATIVE_FIRST_DAY),
            ),
            Dimension.LATITUDE.key(): (("observation",), NATIVE_OBSERVATION_LATITUDES),
            Dimension.LONGITUDE.key(): (("observation",), NATIVE_OBSERVATION_LONGITUDES),
            Dimension.FIRST_DAY_DATETIME.key(): (("observation",), numpy.full(observation_count, NATIVE_FIRST_DAY)),
            Dimension.DEPTH.key(): (("observation",), numpy.zeros(observation_count)),
        }
    )


def _declare_native_challenger(monkeypatch) -> xarray.Dataset:
    latitude, longitude = _native_tracer_grid()
    monkeypatch.setattr(
        "oceanbench.core.curvilinear_staging.CURVILINEAR_CHALLENGERS",
        {
            "curvy": CurvilinearChallenger(
                tracer_grid=lambda dataset: (latitude, longitude),
                tracer_ocean_mask=lambda dataset: numpy.ones(latitude.shape, dtype=bool),
            )
        },
    )
    return with_dataset_source(_native_challenger_dataset(), kind="challenger", name="curvy")


def test_a_store_that_describes_its_cells_twice_reaches_the_native_matchup(monkeypatch):
    challenger = _declare_native_challenger(monkeypatch)

    matchups = ensemble_class4_matchup(
        challenger,
        _native_observations_dataset(),
        [Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
    )

    assert [matchup.variable for matchup in matchups] == [Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    numpy.testing.assert_array_equal(matchups[0].member_values[:, 0], [0.0, 5.0, 10.0])


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _synthetic_matchup(observation_count: int = 60, member_count: int = 8) -> Class4EnsembleMatchup:
    generator = numpy.random.default_rng(19)
    first_days = numpy.array(["2024-01-04", "2024-01-11"], dtype="datetime64[ns]")
    truth = generator.normal(size=observation_count)
    members = truth[:, None] + generator.normal(scale=0.6, size=(observation_count, member_count))
    observations = pandas.DataFrame(
        {
            "observation_value": truth,
            Dimension.TIME.key(): pandas.to_datetime(
                numpy.repeat(["2024-01-05", "2024-01-12"], observation_count // 2)
            ),
            Dimension.LATITUDE.key(): numpy.full(observation_count, -79.9),
            Dimension.LONGITUDE.key(): numpy.full(observation_count, -179.9),
            "first_day": numpy.repeat(first_days, observation_count // 2),
            Dimension.DEPTH.key(): numpy.zeros(observation_count),
            "lead_day": numpy.tile([1, 2], observation_count // 2),
            "depth_bin": "surface",
        }
    )
    return Class4EnsembleMatchup(Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(), observations, members)


def _run_context() -> RunContext:
    return RunContext(
        challenger="gloens",
        challenger_version="test",
        year=2024,
        region="global",
        oceanbench_version="0.0.0",
    )


def test_records_carry_every_metric_per_start_and_an_aggregate():
    records = ensemble_class4_records([_synthetic_matchup()], context=_run_context(), reference="class4")
    dataframe = records_to_dataframe(records)

    expected_metrics = {
        METRIC_CRPS_FAIR,
        METRIC_ENSEMBLE_MEAN_RMSD,
        METRIC_MEMBER_RMSD,
        METRIC_ENSEMBLE_SPREAD,
        METRIC_SSR_UNCORRECTED,
    }
    assert set(dataframe["metric"]) == expected_metrics
    # Two lead days, each with one aggregate record and two per-start records.
    assert len(dataframe) == len(expected_metrics) * 2 * 3
    assert dataframe["start_date"].isna().sum() == len(expected_metrics) * 2
    assert set(dataframe["depth"]) == {"surface"}
    assert set(dataframe[dataframe["metric"] == METRIC_SSR_UNCORRECTED]["unit"]) == {"1"}


def test_records_gain_the_sigma_aware_ratio_only_with_a_sigma_lookup(sigma_lookup):
    without_sigma = records_to_dataframe(
        ensemble_class4_records([_synthetic_matchup()], context=_run_context(), reference="class4")
    )
    with_sigma = records_to_dataframe(
        ensemble_class4_records(
            [_synthetic_matchup()], context=_run_context(), reference="class4", sigma_lookup=sigma_lookup
        )
    )

    assert METRIC_SSR_ADD not in set(without_sigma["metric"])
    assert {METRIC_SSR_ADD, METRIC_SIGMA_TOTAL_RMS} <= set(with_sigma["metric"])
    # Compared group by group: the ratio only rises where the same observations are scored.
    sigma_free_ratio = without_sigma[without_sigma["metric"] == METRIC_SSR_UNCORRECTED]["value"].to_numpy()
    sigma_aware_ratio = with_sigma[with_sigma["metric"] == METRIC_SSR_ADD]["value"].to_numpy()
    assert (sigma_aware_ratio > sigma_free_ratio).all()


def test_records_drop_rows_where_a_member_is_missing():
    matchup = _synthetic_matchup()
    matchup.member_values[0, 3] = numpy.nan
    lead_day = int(matchup.observations.loc[0, "lead_day"])
    first_day = matchup.observations.loc[0, "first_day"]

    dataframe = records_to_dataframe(ensemble_class4_records([matchup], context=_run_context(), reference="class4"))
    counts = dataframe[
        (dataframe["metric"] == METRIC_CRPS_FAIR)
        & (dataframe["lead_day"] == lead_day)
        & (dataframe["start_date"] == pandas.Timestamp(first_day))
    ]["n"]

    assert counts.to_numpy().tolist() == [len(matchup.observations) // 4 - 1]


def test_rank_histograms_are_keyed_on_the_group_and_hold_every_observation():
    matchup = _synthetic_matchup()

    histograms = ensemble_class4_rank_histograms([matchup])

    assert set(histograms) == {
        (matchup.variable, "surface", lead_day, mode) for lead_day in (1, 2) for mode in ("member", "obs")
    }
    for counts in histograms.values():
        assert counts.size == matchup.member_count + 1
        assert counts.sum() == pytest.approx(len(matchup.observations) / 2)
