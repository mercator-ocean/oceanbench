# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Evaluation entry point: score a forecast and emit the standard artifacts.

``evaluate`` scores either a **registered challenger** (by slug) or a **user's own forecast
zarr(s)**, and emits the long-format per-start records parquet plus the aggregated summary.
For a user forecast it also writes a self-contained overlay scorecard laying that model over
the published challengers. It reuses the scoring runner
(:func:`oceanbench.runner.run.run_challenger_scores`) and the aggregation library
(:mod:`oceanbench.publish.aggregate`); no new score code lives here.

References and observations are read **live from the public EDITO objects by default**,
through the resilient chunk-fetch engine and its persistent cache (contracts.md §1). An
**offline reference bundle** may be supplied instead: a self-describing directory whose
``pack-manifest.json`` resolves every reference, the observation store, the
mean-dynamic-topography and the baseline forecasts without touching the network. The bundle
is an optimisation for offline or repeated runs, never a prerequisite.

Skill-vs-baseline works the same way offline as it does online: the baselines are scored as
challengers in their own right, their per-start records land in the same ``scores.parquet``,
and :func:`oceanbench.publish.aggregate.aggregate_scores` derives the paired skill from that
one frame. The only difference is where the baseline forecasts come from, the pack's
``contents.baselines`` instead of the published objects. A pack that bundles no baselines
(every pack built before they existed) scores without skill and says so in a flag.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy
import pandas
import xarray

from oceanbench.core import runtime_configuration as runtime_configuration_module
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.regions import GLOBAL_REGION_NAME, normalize_region_name, subset_dataset_to_region
from oceanbench.core.remote_json import read_json_url
from oceanbench.core.runtime_configuration import (
    RuntimeConfiguration,
    current_runtime_configuration,
    set_runtime_configuration,
)
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.packs.manifest import PACK_MANIFEST_FILENAME
from oceanbench.packs.scorecard import write_overlay_scorecard
from oceanbench.publish.aggregate import aggregate_scores, summary_to_json_records
from oceanbench.publish.insights import write_realism_insights
from oceanbench.runner import records
from oceanbench.runner.run import (
    LIVE_REFERENCE_OPENERS,
    is_registered_challenger,
    live_observation_opener,
    open_registered_challenger,
    registered_challengers,
    run_challenger_scores,
)

YOUR_MODEL_SLUG = "your_model"
CHALLENGER_DATASET_VARIABLE_NAME = "challenger_dataset"
YOUR_MODEL_DISPLAY_NAME = "Your model"

SCORES_FILENAME = "scores.parquet"
SCORES_SUMMARY_FILENAME = "scores-summary.json"
SCORECARD_DIRECTORY = "scorecard"

DEFAULT_EVALUATION_YEAR = 2024

_PER_START_KEY_COLUMNS = ["metric", "reference", "variable", "depth", "lead_day", "start_date"]
METRIC_NAMES = ("rmsd", "mld", "geostrophic", "class4", "lagrangian", "realism")

# Skill is quoted against one baseline. Climatology is the conventional reference for a
# forecast, so it wins when a pack bundles several; the resolution variants are the same
# baseline on a different grid.
_PREFERRED_SKILL_BASELINES = ("climatology", "climatology_1_degree", "persistence", "persistence_1_degree")
NO_BUNDLED_BASELINES_FLAG = (
    "the offline reference directory bundles no baselines, so skill-vs-baseline was not computed; "
    "score live instead, or add a baseline store to the directory and its manifest"
)


@dataclass(frozen=True)
class EvaluateLocalResult:
    """What an evaluation produced.

    ``scores`` is the evaluated target's own long-format records. The parquet at
    ``scores_path`` additionally carries ``baseline_scores``, the records of the pack's
    bundled baselines, so the ``skill_vs_<baseline>`` columns of the summary stay
    recomputable from the artifact alone. ``skill_baseline`` names the baseline skill is
    quoted against, or is ``None`` when none was available.
    """

    scores_path: str | None = None
    summary_path: str | None = None
    scorecard_path: str | None = None
    scores: pandas.DataFrame = field(default_factory=pandas.DataFrame)
    baseline_scores: pandas.DataFrame = field(default_factory=pandas.DataFrame)
    skill_baseline: str | None = None
    viewer_directory: str | None = None
    viewer_datasets_path: str | None = None
    viewer_zarr_path: str | None = None
    viewer_manifest_path: str | None = None
    matchup_parquet_path: str | None = None
    eddy_census_path: str | None = None
    year_error_geography_path: str | None = None
    year_rmsd_by_start_path: str | None = None
    published_prefix: str | None = None
    flags: list[str] = field(default_factory=list)


def load_pack_manifest(pack_directory: str) -> dict:
    """Load and schema-validate a pack's ``pack-manifest.json`` (contracts.md §7)."""
    manifest = json.loads((Path(pack_directory) / PACK_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    validate_against_schema(manifest, "pack-manifest")
    missing = [field for field in ("year", "region") if field not in manifest]
    if missing:
        raise ValueError(f"pack manifest is missing required field(s): {', '.join(missing)}")
    return manifest


def _load_json(path_or_url: str) -> dict:
    if "://" in path_or_url:
        return read_json_url(path_or_url)
    return json.loads(Path(path_or_url).read_text(encoding="utf-8"))


def _looks_like_weekly_store_directory(path: Path) -> bool:
    return (
        path.is_dir()
        and not (path / "zarr.json").exists()
        and not (path / ".zgroup").exists()
        and any(child.name.endswith(".zarr") for child in path.iterdir())
    )


def _open_weekly_forecast_directory(path: Path) -> xarray.Dataset:
    from datetime import datetime as _datetime

    from oceanbench.core.challenger_datasets import _prepared_challenger_week_dataset

    weekly_paths = sorted(child for child in path.iterdir() if child.name.endswith(".zarr"))
    first_day_datetimes = [_datetime.strptime(weekly_path.stem, "%Y%m%d") for weekly_path in weekly_paths]
    weeks = [
        _prepared_challenger_week_dataset(
            xarray.open_dataset(str(weekly_path), engine="zarr"),
            "local forecast open",
        )
        for weekly_path in weekly_paths
    ]
    return xarray.concat(weeks, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )


def open_forecast_dataset(forecasts_path: str) -> xarray.Dataset:
    """Open a user forecast in the weekly-store conventions of the challenger datasets.

    Two accepted layouts (documented in ``docs/local-evaluation.md``):

    - a **single combined zarr** with dims ``(first_day_datetime, lead_day_index, depth,
      latitude, longitude)`` and the CF-named forecast variables;
    - a **directory of weekly zarr stores** named ``YYYYMMDD.zarr`` (one per forecast start),
      each with a ``time`` lead-day dimension — the same shape a challenger publishes.
    """
    path = Path(forecasts_path)
    if _looks_like_weekly_store_directory(path):
        return _open_weekly_forecast_directory(path)
    return xarray.open_dataset(str(path), engine="zarr")


def _pack_reference_opener(pack_directory: Path, reference_relative_path: str):
    reference_dataset = xarray.open_dataset(str(pack_directory / reference_relative_path), engine="zarr")

    def opener(regional_challenger: xarray.Dataset) -> xarray.Dataset:
        return reference_dataset

    return opener


def _pack_observation_opener(pack_directory: Path, observation_relative_path: str):
    observation_dataset = xarray.open_dataset(str(pack_directory / observation_relative_path), engine="zarr")
    first_day_values = observation_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    def opener(regional_challenger: xarray.Dataset) -> xarray.Dataset:
        challenger_starts = regional_challenger[Dimension.FIRST_DAY_DATETIME.key()].values
        selected = numpy.flatnonzero(numpy.isin(first_day_values, challenger_starts))
        return observation_dataset.isel(observations=selected)

    return opener


@contextmanager
def _pack_runtime_configuration(pack_directory: Path):
    """Point the stage at the pack (so the ported Class-4 SSH->SLA code finds the bundled MDT offline)
    for the duration of the scoring, then restore the previous global runtime configuration exactly.
    """
    previous = runtime_configuration_module._runtime_configuration
    existing = current_runtime_configuration()
    set_runtime_configuration(
        RuntimeConfiguration(
            staged_components=("observations",),
            stage_directory=str(pack_directory),
            remote_retries=existing.remote_retries,
            local_cache_directory_path=existing.local_cache_directory_path,
        )
    )
    try:
        yield
    finally:
        runtime_configuration_module._runtime_configuration = previous


@dataclass(frozen=True)
class EvaluationSources:
    """Where the references, the observations and the evaluation context come from.

    ``kind`` is ``"quick"`` when the sources only carry surface fields, which restricts the
    emitted scores to the surface. ``start_dates`` limits scoring to the starts the sources
    can serve; ``None`` means every start the forecast carries.
    """

    reference_openers: dict
    observation_opener: object
    year: int
    region: str
    kind: str
    start_dates: numpy.ndarray | None
    offline_directory: Path | None
    baseline_store_paths: dict[str, str] = field(default_factory=dict)


def _live_sources(region: str | None, year: int | None) -> EvaluationSources:
    return EvaluationSources(
        reference_openers=dict(LIVE_REFERENCE_OPENERS),
        observation_opener=live_observation_opener,
        year=DEFAULT_EVALUATION_YEAR if year is None else year,
        region=GLOBAL_REGION_NAME if region is None else normalize_region_name(region),
        kind="full",
        start_dates=None,
        offline_directory=None,
        baseline_store_paths={},
    )


def _offline_sources(directory: str, region: str | None, year: int | None) -> EvaluationSources:
    """Resolve every source from an offline reference bundle's ``pack-manifest.json``.

    The bundle's manifest defines the evaluation context, so an explicit region or year that
    contradicts it is rejected rather than silently ignored (contracts.md §7).
    """
    bundle_path = Path(directory)
    manifest = load_pack_manifest(directory)
    if region is not None and normalize_region_name(region) != manifest["region"]:
        raise ValueError(
            f"the offline reference bundle covers region {manifest['region']!r}, not {region!r}; "
            "drop the region argument or point at a bundle for that region"
        )
    if year is not None and year != manifest["year"]:
        raise ValueError(
            f"the offline reference bundle covers year {manifest['year']}, not {year}; "
            "drop the year argument or point at a bundle for that year"
        )
    return EvaluationSources(
        reference_openers={
            name: _pack_reference_opener(bundle_path, entry["path"])
            for name, entry in manifest["contents"]["references"].items()
        },
        observation_opener=_pack_observation_opener(bundle_path, manifest["contents"]["observations"]["path"]),
        year=manifest["year"],
        region=manifest["region"],
        kind=manifest["kind"],
        start_dates=numpy.asarray(manifest["start_dates"], dtype="datetime64[D]"),
        offline_directory=bundle_path,
        baseline_store_paths={
            slug: str(bundle_path / entry["path"]) for slug, entry in manifest["contents"].get("baselines", {}).items()
        },
    )


def _resolve_sources(
    offline_references_directory: str | None,
    region: str | None,
    year: int | None,
) -> EvaluationSources:
    if offline_references_directory is None:
        return _live_sources(region, year)
    return _offline_sources(offline_references_directory, region, year)


@contextmanager
def _sources_runtime_configuration(sources: EvaluationSources):
    """Point the stage at an offline bundle for the duration of the scoring, if there is one."""
    if sources.offline_directory is None:
        yield
        return
    with _pack_runtime_configuration(sources.offline_directory):
        yield


def skill_baseline_slug(baseline_slugs) -> str | None:
    """Which bundled baseline skill is quoted against, or ``None`` when a pack bundles none."""
    available = sorted(baseline_slugs)
    if not available:
        return None
    for preferred in _PREFERRED_SKILL_BASELINES:
        if preferred in available:
            return preferred
    return available[0]


def score_pack_baselines(
    sources: EvaluationSources,
    *,
    forecast_start_values: numpy.ndarray,
    selected_metrics: set,
    output_root: str,
) -> tuple[pandas.DataFrame, list[str]]:
    """Score the pack's bundled baselines as challengers, on the forecast's own starts.

    Same scoring call, same pack references and observations, same metric selection as the
    model being evaluated, so the per-start records share metric keys with it and
    :func:`aggregate_scores` can pair them into skill. A baseline that fails to score is
    reported as a flag and dropped: it must never abort the user's own evaluation.
    """
    frames: list[pandas.DataFrame] = []
    flags: list[str] = []
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    for slug, store_path in sorted(sources.baseline_store_paths.items()):
        try:
            baseline_dataset = xarray.open_dataset(store_path, engine="zarr")
            shared_starts = numpy.flatnonzero(numpy.isin(baseline_dataset[first_day_key].values, forecast_start_values))
            if not len(shared_starts):
                flags.append(f"baseline {slug} shares no start date with the forecast; skill excludes it")
                continue
            baseline_run = run_challenger_scores(
                slug,
                sources.region,
                sources.year,
                references=tuple(sources.reference_openers.keys()),
                include_gridded="rmsd" in selected_metrics,
                include_mixed_layer_depth=(sources.kind == "full" and "mld" in selected_metrics),
                include_geostrophic="geostrophic" in selected_metrics,
                include_class4="class4" in selected_metrics,
                include_lagrangian="lagrangian" in selected_metrics,
                area_weighted=True,
                challenger_version="pack",
                output_root=str(Path(output_root) / f"_baseline_{slug}"),
                dataset=baseline_dataset.isel({first_day_key: shared_starts}),
                reference_openers=sources.reference_openers,
                observation_opener=sources.observation_opener,
            )
        except Exception as error:  # noqa: BLE001 - a baseline must not abort the user's run
            flags.append(f"baseline {slug} skipped: {error}")
            continue
        frames.append(baseline_run.scores)
        flags.extend(f"baseline {slug}: {flag}" for flag in baseline_run.flags)
    if not frames:
        return pandas.DataFrame(), flags
    return pandas.concat(frames, ignore_index=True), flags


def _write_scores_and_summary(
    scores: pandas.DataFrame,
    baseline_scores: pandas.DataFrame,
    *,
    skill_baseline: str | None,
    scores_path: Path,
    summary_path: Path,
) -> pandas.DataFrame:
    """Write the canonical scores parquet and the aggregated summary derived from it.

    The parquet carries the evaluated model **and** any bundled baseline that was scored, the
    same multi-challenger shape the published ``scores.parquet`` has, so the skill columns in
    the summary stay recomputable from the artifact alone.

    Only per-start metrics (gridded / Class-4) carry a start distribution to aggregate into a
    mean and bootstrap CI. Realism records are already aggregates over the starts (start_date
    is null, contracts.md §3.2); they stay in the long-format parquet but are not re-aggregated.
    """
    combined = scores if baseline_scores.empty else pandas.concat([scores, baseline_scores], ignore_index=True)
    combined.to_parquet(str(scores_path), index=False)
    per_start_scores = combined[combined["start_date"].notna()].reset_index(drop=True)
    summary = aggregate_scores(per_start_scores, baseline_challenger=skill_baseline)
    summary_path.write_text(
        json.dumps(summary_to_json_records(summary), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def is_python_challenger_file(target: str) -> bool:
    """Whether ``target`` is a challenger ``.py`` file, the input the notebook route consumed."""
    return target.split("?", 1)[0].endswith(".py")


def _read_python_challenger_source(path_or_url: str) -> str:
    if "://" in path_or_url:
        with urlopen(path_or_url, timeout=30) as response:  # noqa: S310
            return response.read().decode("utf-8")
    try:
        return Path(path_or_url).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read the challenger file {path_or_url!r}: {error}") from error


def open_python_challenger_file(path_or_url: str) -> xarray.Dataset:
    """Open the forecast a challenger ``.py`` file describes.

    Same contract the notebook route used: the file is executed as a script and must bind a
    ``challenger_dataset`` xarray dataset, exactly as the files under ``challenger_datasets/``
    do. The file is trusted code, run in this process with no sandbox, just as the generated
    notebook ran it.
    """
    source = _read_python_challenger_source(path_or_url)
    namespace: dict = {"__name__": "__oceanbench_challenger__", "__file__": path_or_url}
    exec(compile(source, path_or_url, "exec"), namespace)  # noqa: S102
    if CHALLENGER_DATASET_VARIABLE_NAME not in namespace:
        raise ValueError(
            f"{path_or_url} does not define {CHALLENGER_DATASET_VARIABLE_NAME!r}; a challenger file must "
            f"assign the forecast to {CHALLENGER_DATASET_VARIABLE_NAME} (see challenger_datasets/)"
        )
    dataset = namespace[CHALLENGER_DATASET_VARIABLE_NAME]
    if not isinstance(dataset, xarray.Dataset):
        raise ValueError(
            f"{path_or_url} assigned {CHALLENGER_DATASET_VARIABLE_NAME} of type "
            f"{type(dataset).__name__}, expected an xarray.Dataset"
        )
    return dataset


def _unknown_target_message(target: str) -> str:
    accepted = (
        f"unknown evaluation target {target!r}. Accepted targets are: "
        "a registered challenger slug (" + ", ".join(registered_challengers()) + "); "
        "a path or URL to a forecast zarr, either a combined store or a directory of weekly "
        "YYYYMMDD.zarr stores; "
        f"or a path or URL to a challenger .py file assigning {CHALLENGER_DATASET_VARIABLE_NAME}"
    )
    closest = difflib.get_close_matches(target, registered_challengers(), n=1)
    if closest:
        return f"{accepted}. Did you mean {closest[0]!r}?"
    return accepted


def _open_evaluation_target(target: str) -> tuple[xarray.Dataset, str, str]:
    """Open what is being scored, and return it with its slug and version.

    ``target`` is a registered challenger slug (opened from its published objects), a path to
    the user's own forecast in the weekly-store conventions, or a challenger ``.py`` file.
    """
    if is_registered_challenger(target):
        return open_registered_challenger(target), target, "published"
    if is_python_challenger_file(target):
        return open_python_challenger_file(target), YOUR_MODEL_SLUG, "local"
    if "://" not in target and not Path(target).exists():
        raise ValueError(_unknown_target_message(target))
    return open_forecast_dataset(target), YOUR_MODEL_SLUG, "local"


def _realism_result(
    regional_challenger: xarray.Dataset,
    reference_openers: dict,
    region: str,
    context: records.RunContext,
    start_limit: int | None,
):
    from oceanbench.runner.realism import compute_realism_battery

    reference_datasets = {
        name: subset_dataset_to_region(opener(regional_challenger), region)
        for name, opener in reference_openers.items()
    }
    start_count = regional_challenger.sizes.get(Dimension.FIRST_DAY_DATETIME.key(), 1)
    start_indices = list(range(start_count if start_limit is None else min(start_limit, start_count)))
    result = compute_realism_battery(
        regional_challenger,
        reference_datasets,
        region=region,
        context=context,
        variable=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        start_indices=start_indices,
        eddy_start_indices=[start_indices[0]] if start_indices else [0],
    )
    return result


def per_start_agreement(local_scores: pandas.DataFrame, published_scores: pandas.DataFrame) -> pandas.DataFrame:
    """Join local and published per-start records on the metric key + start and report the differences.

    Returns one row per shared (metric, reference, variable, depth, lead_day, start_date) key with
    ``local_value``, ``published_value`` and ``absolute_difference``. The overlay's claim of agreement
    is proven here: for the same forecast starts the per-start values must be numerically identical.
    """
    key = _PER_START_KEY_COLUMNS

    def _normalise(frame: pandas.DataFrame) -> pandas.DataFrame:
        normalised = frame.copy()
        normalised["start_date"] = pandas.to_datetime(normalised["start_date"])
        for column in ["reference", "variable", "depth"]:
            normalised[column] = normalised[column].astype(object).where(normalised[column].notna(), None)
        return normalised

    local = _normalise(local_scores)[key + ["value"]].rename(columns={"value": "local_value"})
    published = _normalise(published_scores)[key + ["value"]].rename(columns={"value": "published_value"})
    merged = local.merge(published, on=key, how="inner")
    merged["absolute_difference"] = (merged["local_value"] - merged["published_value"]).abs()
    return merged


def evaluate(
    target: str,
    *,
    output_directory: str,
    offline_references_directory: str | None = None,
    region: str | None = None,
    year: int | None = None,
    published_scores_path: str | None = None,
    published_challengers_path: str | None = None,
    metrics: tuple[str, ...] | list[str] | None = None,
    viewer_artifacts: bool = False,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_endpoint: str | None = None,
    s3_env_file: str | None = None,
) -> EvaluateLocalResult:
    """Score ``target`` and emit the long-format records and the aggregated summary.

    ``target`` is either a registered challenger slug or a path to the user's own forecast;
    a user forecast additionally gets the overlay scorecard laying it over the published
    challengers. References and observations are read live from the public EDITO objects
    unless ``offline_references_directory`` points at an offline reference bundle, in which
    case the bundle's manifest also fixes the year and region, and its bundled baselines are
    scored alongside the target so the summary carries skill-vs-baseline. A bundle with no
    baselines scores without skill and reports that as a flag.

    Scores are the only default output. ``viewer_artifacts`` additionally builds everything the map
    needs (Class-4 match-up parquet, eddy census, field pyramid, year-mode JSON) plus a local
    viewer site that opens over a plain file server. When ``s3_bucket`` is set the whole
    ``output_directory`` tree is uploaded to ``s3://<s3_bucket>/<s3_prefix>/`` using the existing
    publish machinery.
    """
    sources = _resolve_sources(offline_references_directory, region, year)
    kind = sources.kind
    year = sources.year
    region = sources.region
    selected_metrics = set(METRIC_NAMES if metrics is None else metrics)
    unknown_metrics = selected_metrics.difference(METRIC_NAMES)
    if unknown_metrics:
        raise ValueError(f"unknown metrics: {', '.join(sorted(unknown_metrics))}")

    forecast_dataset, challenger_slug, challenger_version = _open_evaluation_target(target)

    # Created only once the target opened, so a failed run leaves no empty output directory behind.
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    if sources.start_dates is not None:
        forecast_start_values = forecast_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
        selected_start_indices = numpy.flatnonzero(
            numpy.isin(forecast_start_values.astype("datetime64[D]"), sources.start_dates)
        )
        if not len(selected_start_indices):
            raise ValueError("the forecast and the offline reference bundle have no start dates in common")
        forecast_dataset = forecast_dataset.isel({Dimension.FIRST_DAY_DATETIME.key(): selected_start_indices})

    reference_openers = sources.reference_openers
    observation_opener = sources.observation_opener
    references = tuple(reference_openers.keys())

    with _sources_runtime_configuration(sources):
        run_result = run_challenger_scores(
            challenger_slug,
            region,
            year,
            references=references,
            include_gridded="rmsd" in selected_metrics,
            include_mixed_layer_depth=(kind == "full" and "mld" in selected_metrics),
            include_geostrophic="geostrophic" in selected_metrics,
            include_class4="class4" in selected_metrics,
            include_lagrangian="lagrangian" in selected_metrics,
            area_weighted=True,
            challenger_version=challenger_version,
            output_root=str(output_path / "_run"),
            dataset=forecast_dataset,
            reference_openers=reference_openers,
            observation_opener=observation_opener,
        )
        flags = list(run_result.flags)
        scores = run_result.scores

        if kind == "quick":
            scores = scores[scores["depth"].isna() | (scores["depth"] == "surface")].reset_index(drop=True)

        # Baselines come from the pack, never from a remote source, and are scored right here
        # so their per-start records share this run's references, observations and metric keys.
        baseline_scores, baseline_flags = score_pack_baselines(
            sources,
            forecast_start_values=forecast_dataset[Dimension.FIRST_DAY_DATETIME.key()].values,
            selected_metrics=selected_metrics,
            output_root=str(output_path / "_run"),
        )
        flags.extend(baseline_flags)
        if kind == "quick" and not baseline_scores.empty:
            baseline_scores = baseline_scores[
                baseline_scores["depth"].isna() | (baseline_scores["depth"] == "surface")
            ].reset_index(drop=True)
        if sources.offline_directory is not None and not sources.baseline_store_paths:
            flags.append(NO_BUNDLED_BASELINES_FLAG)
        skill_baseline = skill_baseline_slug(
            set(baseline_scores["challenger"].unique()) if not baseline_scores.empty else set()
        )

        if "realism" in selected_metrics:
            regional_challenger = subset_dataset_to_region(
                forecast_dataset,
                region,
            )
            context = records.RunContext(
                challenger=challenger_slug,
                challenger_version=challenger_version,
                year=year,
                region=region,
                oceanbench_version=OCEANBENCH_VERSION,
            )
            try:
                realism_result = _realism_result(regional_challenger, reference_openers, region, context, None)
                scores = pandas.concat(
                    [scores, records.records_to_dataframe(realism_result.records)], ignore_index=True
                )
                flags.extend(realism_result.flags)
            except Exception as error:  # noqa: BLE001 - realism must not abort the local run
                flags.append(f"realism battery skipped: {error}")
                realism_result = None

            # The spectra and eddies payloads live next to the other per-(challenger, region)
            # insights the viewer reads (viewer README: insights/<slug>/<region>/spectra.json).
            if realism_result is not None:
                try:
                    write_realism_insights(
                        realism_result.spectra_entries,
                        realism_result.eddy_census,
                        str(output_path / "insights" / challenger_slug / region),
                        variable=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
                    )
                except Exception as error:  # noqa: BLE001 - one artifact must not abort the others
                    flags.append(f"realism insights skipped: {error}")

    scores_path = output_path / SCORES_FILENAME
    summary_path = output_path / SCORES_SUMMARY_FILENAME
    summary = _write_scores_and_summary(
        scores,
        baseline_scores,
        skill_baseline=skill_baseline,
        scores_path=scores_path,
        summary_path=summary_path,
    )

    # The scorecard lays *your* model over the published ones. Scoring a challenger that is
    # already published has nothing to overlay, so only a user forecast gets one.
    scorecard_directory = output_path / SCORECARD_DIRECTORY if challenger_slug == YOUR_MODEL_SLUG else None
    if scorecard_directory is not None:
        published_scores = pandas.read_parquet(published_scores_path) if published_scores_path is not None else None
        published_challengers = (
            _load_json(published_challengers_path) if published_challengers_path is not None else None
        )
        write_overlay_scorecard(
            scorecard_directory,
            your_model_scores=scores,
            published_scores=published_scores,
            published_challengers=published_challengers,
            region=region,
            year=year,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    viewer_result = None
    viewer_artifacts_result = None
    if viewer_artifacts:
        from oceanbench.packs.local_viewer import build_local_viewer
        from oceanbench.publish.viewer_artifacts import write_viewer_artifacts

        matchups_context = records.RunContext(
            challenger=challenger_slug,
            challenger_version=challenger_version,
            year=year,
            region=region,
            oceanbench_version=OCEANBENCH_VERSION,
        )
        regional_forecast = subset_dataset_to_region(forecast_dataset, region)
        try:
            with _sources_runtime_configuration(sources):
                observation_dataset = subset_dataset_to_region(observation_opener(regional_forecast), region)
                viewer_artifacts_result = write_viewer_artifacts(
                    forecast_dataset=regional_forecast,
                    observation_dataset=observation_dataset,
                    region=region,
                    dataset_slug=challenger_slug,
                    output_directory=output_directory,
                    year=year,
                    matchups_context=matchups_context,
                )
            flags.extend(viewer_artifacts_result.flags)
        except Exception as error:  # noqa: BLE001 - the map itself must still be buildable
            flags.append(f"viewer serving artifacts skipped: {error}")

        # The serving artifacts already wrote the field pyramid under viewer/data/; the local site
        # adopts it rather than rebuilding the same tiles, and builds its own if they were skipped.
        viewer_result = build_local_viewer(
            forecast_dataset,
            output_directory=output_directory,
            year=year,
            dataset_slug=challenger_slug,
            label=YOUR_MODEL_DISPLAY_NAME + " (local)" if challenger_slug == YOUR_MODEL_SLUG else challenger_slug,
            pyramid_zarr_path=viewer_artifacts_result.pyramid_zarr_path if viewer_artifacts_result else None,
            pyramid_manifest_path=viewer_artifacts_result.pyramid_manifest_path if viewer_artifacts_result else None,
        )

        if viewer_artifacts_result and viewer_artifacts_result.class4_bias_records:
            bias_frame = records.records_to_dataframe(viewer_artifacts_result.class4_bias_records)
            scores = pandas.concat([scores, bias_frame], ignore_index=True)
            summary = _write_scores_and_summary(
                scores,
                baseline_scores,
                skill_baseline=skill_baseline,
                scores_path=scores_path,
                summary_path=summary_path,
            )

        # Per-challenger scores file: the viewer loads one challenger at a time, so a per-challenger
        # file avoids the monolithic-summary cold-load cost. The merged scores-summary.json above
        # stays for the scores site's compatibility.
        per_challenger_path = output_path / f"scores-{challenger_slug}.json"
        per_challenger_path.write_text(
            json.dumps(summary_to_json_records(summary), sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )

    published_prefix = None
    if s3_bucket is not None:
        from oceanbench.publish.s3 import EDITO_MINIO_ENDPOINT, upload_tree

        if s3_prefix is None:
            raise ValueError("s3_prefix is required when s3_bucket is given")
        upload_tree(
            output_directory,
            bucket=s3_bucket,
            prefix=s3_prefix,
            endpoint=s3_endpoint if s3_endpoint is not None else EDITO_MINIO_ENDPOINT,
            env_file=s3_env_file,
            compress_json=True,
        )
        published_prefix = f"s3://{s3_bucket}/{s3_prefix.strip('/')}/"

    return EvaluateLocalResult(
        scores_path=str(scores_path),
        summary_path=str(summary_path),
        scorecard_path=str(scorecard_directory / "index.html") if scorecard_directory is not None else None,
        scores=scores,
        baseline_scores=baseline_scores,
        skill_baseline=skill_baseline,
        flags=flags,
        matchup_parquet_path=viewer_artifacts_result.matchup_parquet_path if viewer_artifacts_result else None,
        eddy_census_path=viewer_artifacts_result.eddy_census_path if viewer_artifacts_result else None,
        year_error_geography_path=(
            viewer_artifacts_result.year_error_geography_path if viewer_artifacts_result else None
        ),
        year_rmsd_by_start_path=viewer_artifacts_result.year_rmsd_by_start_path if viewer_artifacts_result else None,
        published_prefix=published_prefix,
        viewer_directory=viewer_result.viewer_directory if viewer_result else None,
        viewer_datasets_path=viewer_result.datasets_path if viewer_result else None,
        viewer_zarr_path=viewer_result.zarr_path if viewer_result else None,
        viewer_manifest_path=viewer_result.manifest_path if viewer_result else None,
    )
