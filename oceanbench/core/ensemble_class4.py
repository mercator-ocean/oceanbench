# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Observation-space (Class IV) probabilistic scores for an ensemble challenger.

The deterministic Class IV axis interpolates one model field to every observation position,
depth and time and reports the RMSD of the residual. An ensemble carries M fields, so the
same matchup runs once per member and every observation ends up with M model values instead
of one. Nothing about the matchup itself changes: the observation dataframe, the horizontal
and vertical interpolation and the SSH to SLA conversion are
:mod:`oceanbench.core.classIV_support` untouched, called member by member. There is no
superobbing and no distance guard, exactly as on the deterministic path.

The metric math is ported from the frozen campaign scorer
``/scratch/jseillade/probax/campaign/src/score_matchup.py`` (scorer-v2.0.0) on lir, which is
the code that produced the published GloEns and glonet2-ens class-4 numbers. Ported here:

``crps_fair``
    The fair (finite-ensemble unbiased) CRPS of Ferro (2014),

        CRPS_fair = mean_i |x_i - y| - sum_i sum_j |x_i - x_j| / (2 M (M - 1))

    evaluated through :func:`oceanbench.core.ensemble_gridded.continuous_ranked_probability_score`
    so the observation-space and gridded axes cannot drift apart. That function uses the
    same sorted-ensemble identity ``sum_i sum_j |x_i - x_j| = 2 sum_k (2k - M - 1) x_(k)``
    the campaign scorer uses, so the values are the same to floating point.

``ssr_add``
    The additive spread-skill ratio, the campaign's primary convention:

        SSR_add = sqrt( (M + 1) / M * mean(s^2) + mean(sigma_total^2) ) / sqrt(mean((y - xbar)^2))

    Observation error sits on the spread side, where the predicted innovation variance
    belongs: the ensemble predicts the ocean and the instrument adds its own error on top,
    and the residual ``y - xbar`` realises both. The ratio is therefore always defined and
    needs no clipping, unlike the subtractive convention it replaced.

``ssr_uncorrected``
    The same ratio with no observation error at all, which is the campaign scorer's name for
    ``spread / rmse``. It is the only spread-skill number available when no sigma lookup is
    passed, and it is emitted under its own metric name so a sigma-free run can never be
    read as a sigma-aware one.

``rank histograms``
    M + 1 bins, ties in the ranking broken uniformly at random, averaged over several
    independent dressing draws. The member-dressed variant adds an independent N(0, sigma)
    to every member (Saetra 2004) and is the primary diagnostic; the obs-dressed variant
    adds one draw to the observation instead and double counts the observation error, so it
    is kept only as a diagnostic.

``ensemble_mean_rmsd`` and ``member_rmsd``
    The RMSD of the ensemble mean against the observations, which is what puts an ensemble
    challenger in the same table as a deterministic one, and the root mean squared error
    averaged over the members as the single-realisation comparator.

Deliberately not ported, because they are campaign-analysis rather than library concerns:
the dressed deterministic null and its Gaussian CRPS, the bootstrap confidence intervals,
the biased CRPS, the subtractive SSR, and every ``crps_*_dressed`` quantity.

Observation error is optional. When a sigma lookup is supplied it is applied per row as
``sigma_total^2 = sigma_i^2 + sigma_r^2(obs_type, month, 0.25 degree cell[, depth])`` from
the sigma-v3 artifact; when it is not, ``sigma_total`` is zero throughout and only
``ssr_uncorrected`` is published.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    create_class4_observations_dataframe,
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import VARIABLE_METADATA, Dimension, Variable
from oceanbench.core.ensemble_gridded import (
    ENSEMBLE_DIMENSION,
    METRIC_CRPS_FAIR,
    METRIC_ENSEMBLE_MEAN_RMSD,
    METRIC_ENSEMBLE_SPREAD,
    METRIC_MEMBER_RMSD,
    continuous_ranked_probability_score,
    finite_ensemble_correction,
)
from oceanbench.core.score_records import RunContext, score_record

METRIC_SSR_ADD = "ssr_add"
METRIC_SSR_UNCORRECTED = "ssr_uncorrected"
METRIC_SIGMA_TOTAL_RMS = "sigma_total_rms"

RATIO_UNIT = "1"

#: Dressing draws averaged into each rank histogram. One draw makes the histogram itself a
#: random variable at the same order as the signal in the end bins; four cuts that noise in
#: half at negligible cost. The campaign scorer uses the same number.
RANK_DRESSING_DRAWS = 4

RANK_DRESSING_MODES = ("member", "obs")

OBSERVATION_DIMENSION = "observation"

#: Class IV variable to sigma artifact ``obs_type``. The artifact splits temperature between
#: the drifting buoys that measure the surface and the Argo floats that measure the profile,
#: so surface temperature is resolved separately by :func:`class4_observation_type`.
CLASS4_OBSERVATION_TYPES = {
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "argo_temperature",
    Variable.SEA_WATER_SALINITY.key(): "argo_salinity",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "currents_u",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "currents_v",
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "sla",
}

SURFACE_TEMPERATURE_OBSERVATION_TYPE = "drifter_sst"
SURFACE_DEPTH_BIN = "surface"


def class4_observation_type(variable_key: str, depth_bin: str) -> str:
    """The sigma artifact ``obs_type`` a Class IV group is scored against.

    Only temperature is ambiguous: the ``surface`` bin is drifting buoy SST, whose
    instrument sigma is an order of magnitude above the Argo one, and every other bin is
    Argo profile temperature.
    """
    if variable_key == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key() and depth_bin == SURFACE_DEPTH_BIN:
        return SURFACE_TEMPERATURE_OBSERVATION_TYPE
    return CLASS4_OBSERVATION_TYPES[variable_key]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def crps_fair(member_values: numpy.ndarray, observation_values: numpy.ndarray) -> numpy.ndarray:
    """Per-observation fair CRPS of ``member_values`` ``(n, M)`` against ``observation_values`` ``(n,)``.

    The campaign scorer refuses at ``M = 1`` because the fair estimator divides by ``M - 1``.
    This module follows :mod:`oceanbench.core.ensemble_gridded` instead and returns the mean
    absolute error there, which is the value the CRPS takes at a single member, so that the
    two ensemble axes answer a degenerate ensemble the same way.
    """
    members = numpy.asarray(member_values, dtype="float64")
    observations = numpy.asarray(observation_values, dtype="float64")
    values = continuous_ranked_probability_score(
        xarray.DataArray(members, dims=(OBSERVATION_DIMENSION, ENSEMBLE_DIMENSION)),
        xarray.DataArray(observations, dims=(OBSERVATION_DIMENSION,)),
        fair=True,
    )
    return numpy.asarray(values.values, dtype="float64")


def ranks_with_random_ties(
    member_values: numpy.ndarray,
    observation_values: numpy.ndarray,
    generator: numpy.random.Generator,
) -> numpy.ndarray:
    """Rank of each observation among its M members, ``0..M``, ties broken uniformly at random."""
    members = numpy.asarray(member_values, dtype="float64")
    column = numpy.asarray(observation_values, dtype="float64")[:, None]
    below = (members < column).sum(axis=1)
    equal = (members == column).sum(axis=1)
    extra = numpy.where(equal > 0, generator.integers(0, numpy.maximum(equal, 1) + 1), 0)
    return (below + extra).astype("int64")


def dressed_ranks(
    member_values: numpy.ndarray,
    observation_values: numpy.ndarray,
    sigma_total: numpy.ndarray,
    generator: numpy.random.Generator,
    mode: str,
) -> numpy.ndarray:
    """Ranks after one dressing draw.

    ``member`` adds an independent N(0, sigma) to every member, which is the Saetra (2004)
    treatment and the primary one. ``obs`` adds a single draw to the observation instead,
    which double counts the observation error and is kept as a diagnostic only.
    """
    members = numpy.asarray(member_values, dtype="float64")
    observations = numpy.asarray(observation_values, dtype="float64")
    sigma = numpy.asarray(sigma_total, dtype="float64")
    if mode == "member":
        noise = generator.normal(0.0, 1.0, size=members.shape) * sigma[:, None]
        return ranks_with_random_ties(members + noise, observations, generator)
    if mode == "obs":
        noise = generator.normal(0.0, 1.0, size=observations.shape) * sigma
        return ranks_with_random_ties(members, observations + noise, generator)
    raise ValueError(f"unknown dressing mode {mode}")


def dressed_rank_histogram(
    member_values: numpy.ndarray,
    observation_values: numpy.ndarray,
    sigma_total: numpy.ndarray,
    generator: numpy.random.Generator,
    *,
    mode: str = "member",
    draws: int = RANK_DRESSING_DRAWS,
    bins: int | None = None,
) -> numpy.ndarray:
    """Rank histogram over ``M + 1`` bins, averaged over several independent dressing draws.

    Counts are divided by the number of draws, so the histogram still sums to the number of
    observations and is directly comparable with a single-draw one. The bin count follows the
    ensemble width rather than a fixed member count, so an eight-member ensemble is not
    padded with empty bins that no flatness test would survive.
    """
    members = numpy.asarray(member_values, dtype="float64")
    if bins is None:
        bins = members.shape[1] + 1
    counts = numpy.zeros(bins, dtype="float64")
    for _draw in range(draws):
        rank = dressed_ranks(members, observation_values, sigma_total, generator, mode)
        counts += numpy.bincount(rank, minlength=bins).astype("float64")
    return counts / float(draws)


def spread_error_ratio_additive(
    mean_ensemble_variance: float,
    mean_sigma_variance: float,
    mean_squared_error: float,
    member_count: int,
) -> float:
    """``sqrt((M + 1) / M * mean(s^2) + mean(sigma_total^2)) / sqrt(mean((y - xbar)^2))``.

    Pass ``mean_sigma_variance`` of zero for the sigma-free ratio the campaign scorer calls
    ``ssr_uncorrected``.
    """
    if member_count < 2 or not math.isfinite(mean_ensemble_variance) or mean_squared_error <= 0:
        return float("nan")
    spread_variance = finite_ensemble_correction(member_count) * mean_ensemble_variance
    return math.sqrt(spread_variance + mean_sigma_variance) / math.sqrt(mean_squared_error)


@dataclass(frozen=True)
class Class4GroupStatistics:
    """Probabilistic statistics of one pooled group of Class IV matchups.

    A group is every observation sharing a variable, a depth bin and a lead day, optionally
    also a forecast start. Observations are pooled unweighted: the Class IV sample is already
    a sample of where the ocean is observed, and the deterministic axis pools it the same way.
    """

    observation_count: int
    member_count: int
    crps_fair: float
    ensemble_mean_squared_error: float
    member_squared_error: float
    ensemble_variance: float
    sigma_variance: float
    has_sigma: bool


def class4_group_statistics(
    member_values: numpy.ndarray,
    observation_values: numpy.ndarray,
    sigma_total: numpy.ndarray | None = None,
) -> Class4GroupStatistics:
    """Reduce one pooled group of ``(n, M)`` member values and ``(n,)`` observations."""
    members = numpy.asarray(member_values, dtype="float64")
    observations = numpy.asarray(observation_values, dtype="float64")
    member_count = members.shape[1]
    ensemble_mean = members.mean(axis=1)
    variance = members.var(axis=1, ddof=1) if member_count > 1 else numpy.full(members.shape[0], numpy.nan)
    sigma = numpy.zeros(members.shape[0]) if sigma_total is None else numpy.asarray(sigma_total, dtype="float64")
    return Class4GroupStatistics(
        observation_count=int(members.shape[0]),
        member_count=int(member_count),
        crps_fair=float(crps_fair(members, observations).mean()),
        ensemble_mean_squared_error=float(((observations - ensemble_mean) ** 2).mean()),
        member_squared_error=float(((members - observations[:, None]) ** 2).mean()),
        ensemble_variance=float("nan") if member_count < 2 else float(variance.mean()),
        sigma_variance=float((sigma**2).mean()),
        has_sigma=sigma_total is not None,
    )


def group_metric_values(statistics: Class4GroupStatistics) -> dict[str, float]:
    """The published metric values of one group, keyed by metric name.

    ``ssr_add`` appears only when the group carried an observation error, so a sigma-free run
    publishes ``ssr_uncorrected`` alone and can never be misread as a sigma-aware one.
    """
    spread = math.sqrt(finite_ensemble_correction(statistics.member_count) * statistics.ensemble_variance)
    values = {
        METRIC_CRPS_FAIR: statistics.crps_fair,
        METRIC_ENSEMBLE_MEAN_RMSD: math.sqrt(statistics.ensemble_mean_squared_error),
        METRIC_MEMBER_RMSD: math.sqrt(statistics.member_squared_error),
        METRIC_ENSEMBLE_SPREAD: spread,
        METRIC_SSR_UNCORRECTED: spread_error_ratio_additive(
            statistics.ensemble_variance,
            0.0,
            statistics.ensemble_mean_squared_error,
            statistics.member_count,
        ),
    }
    if statistics.has_sigma:
        values[METRIC_SSR_ADD] = spread_error_ratio_additive(
            statistics.ensemble_variance,
            statistics.sigma_variance,
            statistics.ensemble_mean_squared_error,
            statistics.member_count,
        )
        values[METRIC_SIGMA_TOTAL_RMS] = math.sqrt(statistics.sigma_variance)
    return values


# ---------------------------------------------------------------------------
# Sigma lookup
# ---------------------------------------------------------------------------


class SigmaLookup:
    """Per-observation ``sigma_total`` read from a sigma-v3 lookup artifact.

    ``sigma_total^2 = sigma_i^2 + sigma_r^2(obs_type, month, cell[, depth])``. ``sigma_i`` is
    a scalar instrument term per stream; ``sigma_r`` is a monthly representativity map on
    0.25 degree cells anchored at (-90, -180), with a regional fallback wherever the map has
    no contributing days.

    Cell keying uses the arithmetic the artifact documents,
    ``cell_i = floor((lat + 90) / 0.25) - 40`` and ``cell_j = floor((lon + 180) / 0.25)``,
    and then verifies the result against the shipped coordinate arrays rather than trusting
    it. An earlier campaign aggregator used a row offset of 160 and produced nonsense that no
    aggregate would have shown.

    Depth resolution applies only to the ``obs_type_z`` streams the artifact wires in, which
    is Argo temperature and salinity. The v3 artifact also ships ``sigma_r_z_extra`` for the
    current components, and deliberately keeps them off ``obs_type_z``: the current matchups
    carry a depth column, so listing them there would silently switch those streams from the
    flat 15 m map to depth interpolation and move the published scores. This loader honours
    that and reads ``sigma_r_z_extra`` never.

    ``store`` is a path or URL handed straight to :func:`xarray.open_zarr`, so a local
    directory, an ``s3://`` prefix or an ``https://`` store all work. An already opened
    dataset is accepted too, which is what the tests use.
    """

    CELL_DEGREES = 0.25
    ROW_OFFSET = 40
    FIRST_ROW_LATITUDE = -80.0
    FIRST_COLUMN_LONGITUDE = -180.0

    def __init__(
        self,
        store: str | xarray.Dataset,
        *,
        basis: str = "rms_over_cells",
        fallback_region: str = "GLOBAL",
        storage_options: Mapping[str, object] | None = None,
    ) -> None:
        self.store = (
            store
            if isinstance(store, xarray.Dataset)
            else xarray.open_zarr(store, consolidated=True, storage_options=dict(storage_options or {}) or None)
        )
        self.basis = basis
        self.fallback_region = fallback_region
        self.latitude = self.store["lat"].values.astype("float64")
        self.longitude = self.store["lon"].values.astype("float64")
        self.observation_types = [str(value) for value in self.store["obs_type"].values]
        self.regions = [str(value) for value in self.store["region"].values]
        self.bases = [str(value) for value in self.store["basis"].values]
        if basis not in self.bases:
            raise ValueError(f"basis {basis} is not one of the artifact bases {self.bases}")
        if fallback_region not in self.regions:
            raise ValueError(f"region {fallback_region} is not one of the artifact regions {self.regions}")
        self.has_depth = "sigma_r_z" in self.store
        self.depth_observation_types = (
            [str(value) for value in self.store["obs_type_z"].values] if self.has_depth else []
        )
        self.depths = self.store["depth"].values.astype("float64") if self.has_depth else numpy.array([])
        self._surface_cache: dict[str, tuple[numpy.ndarray, numpy.ndarray, float]] = {}
        self._depth_cache: dict[tuple[str, int], tuple[numpy.ndarray, numpy.ndarray]] = {}

    def _surface_fields(self, observation_type: str) -> tuple[numpy.ndarray, numpy.ndarray, float]:
        if observation_type not in self._surface_cache:
            if observation_type not in self.observation_types:
                raise ValueError(f"obs_type {observation_type} is not in the artifact {self.observation_types}")
            index = self.observation_types.index(observation_type)
            # n_days carries an obs_type dimension. Indexing it as (month, lat, lon) would
            # silently read the first stream's coverage for every other stream.
            self._surface_cache[observation_type] = (
                self.store["sigma_r"].isel(obs_type=index).values,
                self.store["n_days"].isel(obs_type=index).values,
                float(self.store["sigma_i"].isel(obs_type=index).values),
            )
        return self._surface_cache[observation_type]

    def _depth_fields(self, observation_type: str, month_index: int) -> tuple[numpy.ndarray, numpy.ndarray]:
        """``(sigma_r_z, n_days_z)`` for one stream and one month, shape ``(level, lat, lon)``.

        ``month_index`` is zero based here, as it is everywhere else in this class. Taking a
        one-based month and subtracting inside would put two conventions on one call path,
        which reads December for January and is invisible in every aggregate.

        Loaded a month at a time rather than whole: the full depth array runs to hundreds of
        megabytes and one scoring group spans at most two calendar months.
        """
        key = (observation_type, int(month_index))
        if key not in self._depth_cache:
            selection = {
                "obs_type_z": self.depth_observation_types.index(observation_type),
                "month": int(month_index),
            }
            self._depth_cache[key] = (
                self.store["sigma_r_z"].isel(**selection).values,
                self.store["n_days_z"].isel(**selection).values,
            )
        return self._depth_cache[key]

    def cell_index(self, latitude: numpy.ndarray, longitude: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
        """Documented floor arithmetic, verified against the shipped cell midpoints."""
        latitude = numpy.asarray(latitude, dtype="float64")
        longitude = numpy.asarray(longitude, dtype="float64")
        adjusted_longitude = numpy.where(longitude > 180.0, longitude - 360.0, longitude)
        row = numpy.floor((latitude + 90.0) / self.CELL_DEGREES).astype("int64") - self.ROW_OFFSET
        column = numpy.floor((adjusted_longitude + 180.0) / self.CELL_DEGREES).astype("int64")
        inside = (row >= 0) & (row < self.latitude.size) & (column >= 0) & (column < self.longitude.size)
        row = numpy.clip(row, 0, self.latitude.size - 1)
        column = numpy.clip(column, 0, self.longitude.size - 1)
        half_cell = self.CELL_DEGREES / 2.0
        latitude_gap = numpy.abs(self.latitude[row] - latitude)
        longitude_gap = numpy.abs(self.longitude[column] - adjusted_longitude)
        wrong = inside & ((latitude_gap > half_cell + 1e-6) | (longitude_gap > half_cell + 1e-6))
        if wrong.any():
            raise RuntimeError(
                f"sigma cell keying is wrong for {int(wrong.sum())} rows: "
                f"worst latitude gap {float(latitude_gap[wrong].max()):.4f} degrees, "
                f"worst longitude gap {float(longitude_gap[wrong].max()):.4f} degrees"
            )
        return row, column, inside

    def depth_bracket(self, depth: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Bracketing level indices and the weight on the shallower one.

        Linear in sigma, not in variance, and clamped at both ends: an observation above the
        first level takes the first level and one below the last takes the last, rather than
        extrapolating a quantity bounded below by zero.
        """
        clamped = numpy.clip(numpy.asarray(depth, dtype="float64"), self.depths[0], self.depths[-1])
        upper = numpy.clip(numpy.searchsorted(self.depths, clamped, side="left"), 1, self.depths.size - 1)
        lower = upper - 1
        span = self.depths[upper] - self.depths[lower]
        weight_lower = numpy.where(clamped <= self.depths[0], 1.0, (self.depths[upper] - clamped) / span)
        return lower, upper, numpy.clip(weight_lower, 0.0, 1.0)

    def _depth_resolved_sigma_r(
        self,
        observation_type: str,
        month_index: numpy.ndarray,
        row: numpy.ndarray,
        column: numpy.ndarray,
        inside: numpy.ndarray,
        depth: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        lower, upper, weight = self.depth_bracket(depth)
        sigma_r = numpy.full(row.shape, numpy.nan)
        fallback_level = lower.copy()
        for month_value in numpy.unique(month_index):
            sigma_z, days_z = self._depth_fields(observation_type, month_value)
            here = month_index == month_value
            rows = row[here]
            columns = column[here]
            low = lower[here]
            high = upper[here]

            def sample(level: numpy.ndarray) -> numpy.ndarray:
                value = sigma_z[level, rows, columns].astype("float64")
                usable = (days_z[level, rows, columns] > 0) & numpy.isfinite(value) & inside[here]
                return numpy.where(usable, value, numpy.nan)

            shallow = sample(low)
            deep = sample(high)
            both = numpy.isfinite(shallow) & numpy.isfinite(deep)
            sigma_r[here] = numpy.where(
                both,
                weight[here] * numpy.nan_to_num(shallow) + (1.0 - weight[here]) * numpy.nan_to_num(deep),
                numpy.where(numpy.isfinite(shallow), shallow, deep),
            )
            # Where the deeper level is dry the shallower one carried the value, so the
            # fallback level must follow whichever side actually had data.
            fallback_level[here] = numpy.where(numpy.isfinite(shallow), low, high)
        return sigma_r, fallback_level

    def total(
        self,
        observation_type: str,
        month: numpy.ndarray,
        latitude: numpy.ndarray,
        longitude: numpy.ndarray,
        depth: numpy.ndarray | None = None,
    ) -> tuple[numpy.ndarray, float, dict]:
        """``sqrt(sigma_i^2 + sigma_r^2)`` per row, plus ``sigma_i`` and a diagnostics dict.

        ``month`` is one based, as calendar months are. ``depth`` is read only for the
        streams the artifact resolves in depth; passing it for any other stream uses the
        surface map, which is what an artifact without depth support did at every depth.
        """
        sigma_r_map, days_map, sigma_i = self._surface_fields(observation_type)
        row, column, inside = self.cell_index(latitude, longitude)
        month_index = numpy.asarray(month, dtype="int64") - 1
        use_depth = depth is not None and self.has_depth and observation_type in self.depth_observation_types

        if use_depth:
            sigma_r, fallback_level = self._depth_resolved_sigma_r(
                observation_type, month_index, row, column, inside, depth
            )
        else:
            sigma_r = sigma_r_map[month_index, row, column].astype("float64")
            available = (days_map[month_index, row, column] > 0) & inside
            sigma_r = numpy.where(available & numpy.isfinite(sigma_r), sigma_r, numpy.nan)
            fallback_level = None

        missing = ~numpy.isfinite(sigma_r)
        fallback_count = int(missing.sum())
        if fallback_count:
            if use_depth:
                table = (
                    self.store["sigma_r_fallback_z"]
                    .isel(obs_type_z=self.depth_observation_types.index(observation_type))
                    .sel(region=self.fallback_region, basis=self.basis)
                    .values
                )
                replacement = table[month_index, fallback_level]
            else:
                table = (
                    self.store["sigma_r_fallback"]
                    .isel(obs_type=self.observation_types.index(observation_type))
                    .sel(region=self.fallback_region, basis=self.basis)
                    .values
                )
                replacement = table[month_index]
            sigma_r = numpy.where(missing, replacement, sigma_r)
        sigma_r = numpy.where(numpy.isfinite(sigma_r), sigma_r, 0.0)
        diagnostics = {
            "sigma_r_fallback_rows": fallback_count,
            "sigma_r_fallback_fraction": float(fallback_count / max(len(sigma_r), 1)),
            "rows_outside_sigma_grid": int((~inside).sum()),
            "sigma_basis": self.basis,
            "sigma_fallback_region": self.fallback_region,
            "sigma_depth_resolved": bool(use_depth),
        }
        return numpy.sqrt(sigma_i * sigma_i + sigma_r * sigma_r), sigma_i, diagnostics


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------


def interpolate_class4_ensemble_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> numpy.ndarray:
    """Model values at every observation for every member, shape ``(n, M)``.

    The member loop lives here and not inside
    :func:`oceanbench.core.classIV_support.interpolate_class4_model_to_observations`, which
    is called once per member on a slice that has no member dimension left. That function
    therefore sees exactly the array a deterministic challenger hands it, and the
    deterministic path is untouched.
    """
    if ensemble_dimension not in model_data.dims:
        raise ValueError(f"model data has no {ensemble_dimension} dimension, found {list(model_data.dims)}")
    member_count = model_data.sizes[ensemble_dimension]
    columns = [
        interpolate_class4_model_to_observations(
            model_data.isel({ensemble_dimension: member_index}),
            observations_dataframe,
        )
        for member_index in range(member_count)
    ]
    return numpy.stack(columns, axis=1)


@dataclass(frozen=True)
class Class4EnsembleMatchup:
    """One variable's Class IV matchup, with M model values per observation.

    ``observations`` is main's Class IV observation dataframe unchanged, indexed 0..n-1, and
    ``member_values`` is aligned with it row for row.
    """

    variable: str
    observations: pandas.DataFrame
    member_values: numpy.ndarray

    @property
    def member_count(self) -> int:
        return int(self.member_values.shape[1])


def ensemble_class4_matchup(
    challenger_dataset: xarray.Dataset,
    observations_dataset: xarray.Dataset,
    variables: Sequence[Variable],
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> list[Class4EnsembleMatchup]:
    """Run main's Class IV matchup once per member, for every requested variable.

    Step for step this is :func:`oceanbench.core.classIV.rmsd_class4_validation` up to the
    point where it computes an RMSD: the same standard-name rename, the same observation
    dataframe, the same SSH to SLA conversion and the same interpolation. The only extension
    is the member loop.
    """
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]

    matchups = []
    for variable in variables:
        variable_key = variable.key()
        observations_dataframe = create_class4_observations_dataframe(
            observations_dataset,
            variable_key,
            variable_key,
            lead_days_count,
        )
        if observations_dataframe.empty:
            continue
        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"]).reset_index(drop=True)
        if observations_dataframe.empty:
            continue
        model_variable = prepare_class4_model_variable(challenger[variable_key], variable_key)
        member_values = interpolate_class4_ensemble_to_observations(
            model_variable,
            observations_dataframe,
            ensemble_dimension=ensemble_dimension,
        )
        matchups.append(Class4EnsembleMatchup(variable_key, observations_dataframe, member_values))
    return matchups


def _finite_matchup_rows(matchup: Class4EnsembleMatchup) -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Rows where the observation and every member value are finite.

    The deterministic table drops a row whose single model value is missing; an ensemble row
    is only usable when every member reached it, otherwise the ensemble statistics of that
    row would be taken over a different member set than its neighbours.
    """
    usable = numpy.isfinite(matchup.member_values).all(axis=1) & numpy.isfinite(
        matchup.observations["observation_value"].to_numpy("float64")
    )
    return matchup.observations.loc[usable].reset_index(drop=True), matchup.member_values[usable]


def _group_sigma(
    sigma_lookup: SigmaLookup | None,
    variable_key: str,
    depth_bin: str,
    group: pandas.DataFrame,
) -> numpy.ndarray | None:
    if sigma_lookup is None:
        return None
    times = pandas.to_datetime(group[Dimension.TIME.key()])
    depth_values = group[Dimension.DEPTH.key()].to_numpy("float64") if Dimension.DEPTH.key() in group.columns else None
    sigma_total, _sigma_i, _diagnostics = sigma_lookup.total(
        class4_observation_type(variable_key, depth_bin),
        times.dt.month.to_numpy(),
        group[Dimension.LATITUDE.key()].to_numpy("float64"),
        group[Dimension.LONGITUDE.key()].to_numpy("float64"),
        depth=depth_values,
    )
    return sigma_total


def _metric_records(
    metric_values: Mapping[str, float],
    *,
    context: RunContext,
    reference: str,
    variable: str,
    depth_bin: str,
    lead_day: int,
    start_date: object,
    observation_count: int,
) -> list[dict]:
    variable_unit = VARIABLE_METADATA[variable][1]
    ratio_metrics = {METRIC_SSR_ADD, METRIC_SSR_UNCORRECTED}
    return [
        score_record(
            context=context,
            metric=metric,
            value=value,
            unit=RATIO_UNIT if metric in ratio_metrics else variable_unit,
            reference=reference,
            variable=variable,
            depth=depth_bin,
            lead_day=lead_day,
            start_date=start_date,
            sample_count=observation_count,
        )
        for metric, value in metric_values.items()
    ]


def ensemble_class4_records(
    matchups: Iterable[Class4EnsembleMatchup],
    *,
    context: RunContext,
    reference: str,
    sigma_lookup: SigmaLookup | None = None,
) -> list[dict]:
    """Long-format ``scores.parquet`` records for a whole ensemble Class IV run.

    One record per metric is emitted for every ``(forecast start, lead day, variable, depth
    bin)`` group, plus one record per ``(lead day, variable, depth bin)`` carrying a null
    start date, which is the value the published tables read. That aggregate pools the
    observations of every start rather than averaging the per-start numbers: Class IV groups
    are ragged in sample count, sometimes by an order of magnitude between starts, and
    pooling weights each observation once. This is the one place the observation-space axis
    aggregates differently from the gridded one, where every start covers the same grid and
    the per-start roots are averaged.
    """
    records = []
    for matchup in matchups:
        observations, member_values = _finite_matchup_rows(matchup)
        if observations.empty:
            continue
        for (depth_bin, lead_day), group in observations.groupby(["depth_bin", "lead_day"], sort=True):
            sigma_total = _group_sigma(sigma_lookup, matchup.variable, depth_bin, group)
            group_members = member_values[group.index.to_numpy()]
            observation_values = group["observation_value"].to_numpy("float64")
            records += _metric_records(
                group_metric_values(class4_group_statistics(group_members, observation_values, sigma_total)),
                context=context,
                reference=reference,
                variable=matchup.variable,
                depth_bin=depth_bin,
                lead_day=int(lead_day),
                start_date=None,
                observation_count=len(group),
            )
            for start_date, start_group in group.groupby("first_day", sort=True):
                start_sigma = None if sigma_total is None else sigma_total[group.index.get_indexer(start_group.index)]
                records += _metric_records(
                    group_metric_values(
                        class4_group_statistics(
                            member_values[start_group.index.to_numpy()],
                            start_group["observation_value"].to_numpy("float64"),
                            start_sigma,
                        )
                    ),
                    context=context,
                    reference=reference,
                    variable=matchup.variable,
                    depth_bin=depth_bin,
                    lead_day=int(lead_day),
                    start_date=start_date,
                    observation_count=len(start_group),
                )
    return records


def ensemble_class4_rank_histograms(
    matchups: Iterable[Class4EnsembleMatchup],
    *,
    sigma_lookup: SigmaLookup | None = None,
    seed: int = 0,
    modes: Sequence[str] = RANK_DRESSING_MODES,
    draws: int = RANK_DRESSING_DRAWS,
) -> dict[tuple[str, str, int, str], numpy.ndarray]:
    """Rank histograms keyed on ``(variable, depth bin, lead day, dressing mode)``.

    Histograms are not scalar scores and do not belong in the score records, so they are
    returned on their own. With no sigma lookup the dressing is a zero-width draw, which
    leaves the plain rank histogram.
    """
    generator = numpy.random.default_rng(seed)
    histograms: dict[tuple[str, str, int, str], numpy.ndarray] = {}
    for matchup in matchups:
        observations, member_values = _finite_matchup_rows(matchup)
        if observations.empty:
            continue
        for (depth_bin, lead_day), group in observations.groupby(["depth_bin", "lead_day"], sort=True):
            sigma_total = _group_sigma(sigma_lookup, matchup.variable, depth_bin, group)
            group_members = member_values[group.index.to_numpy()]
            if sigma_total is None:
                sigma_total = numpy.zeros(len(group))
            observation_values = group["observation_value"].to_numpy("float64")
            for mode in modes:
                histograms[(matchup.variable, depth_bin, int(lead_day), mode)] = dressed_rank_histogram(
                    group_members,
                    observation_values,
                    sigma_total,
                    generator,
                    mode=mode,
                    draws=draws,
                )
    return histograms
