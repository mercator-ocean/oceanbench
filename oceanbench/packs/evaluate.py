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
``pack-manifest.json`` resolves every reference, the observation store and the
mean-dynamic-topography without touching the network. The bundle is an optimisation for
offline or repeated runs, never a prerequisite.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import urlopen

import numpy
import pandas
import xarray

from oceanbench.core import runtime_configuration as runtime_configuration_module
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.regions import GLOBAL_REGION_NAME, normalize_region_name, subset_dataset_to_region
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
from oceanbench.runner import records
from oceanbench.runner.run import (
    LIVE_REFERENCE_OPENERS,
    is_registered_challenger,
    live_observation_opener,
    open_registered_challenger,
    run_challenger_scores,
)

YOUR_MODEL_SLUG = "your_model"
YOUR_MODEL_DISPLAY_NAME = "Your model"

SCORES_FILENAME = "scores.parquet"
SCORES_SUMMARY_FILENAME = "scores-summary.json"
SCORECARD_DIRECTORY = "scorecard"

DEFAULT_EVALUATION_YEAR = 2024

_PER_START_KEY_COLUMNS = ["metric", "reference", "variable", "depth", "lead_day", "start_date"]
METRIC_NAMES = ("rmsd", "mld", "geostrophic", "class4", "lagrangian", "realism")


@dataclass(frozen=True)
class EvaluateLocalResult:
    scores_path: str | None = None
    summary_path: str | None = None
    scorecard_path: str | None = None
    scores: pandas.DataFrame = field(default_factory=pandas.DataFrame)
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
        with urlopen(path_or_url, timeout=30) as response:  # noqa: S310
            return json.load(response)
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


def _live_sources(region: str | None, year: int | None) -> EvaluationSources:
    return EvaluationSources(
        reference_openers=dict(LIVE_REFERENCE_OPENERS),
        observation_opener=live_observation_opener,
        year=DEFAULT_EVALUATION_YEAR if year is None else year,
        region=GLOBAL_REGION_NAME if region is None else normalize_region_name(region),
        kind="full",
        start_dates=None,
        offline_directory=None,
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


def _open_evaluation_target(target: str) -> tuple[xarray.Dataset, str, str]:
    """Open what is being scored, and return it with its slug and version.

    ``target`` is either a registered challenger slug (opened from its published objects) or
    a path to the user's own forecast in the weekly-store conventions.
    """
    if is_registered_challenger(target):
        return open_registered_challenger(target), target, "published"
    return open_forecast_dataset(target), YOUR_MODEL_SLUG, "local"


def _realism_records(
    regional_challenger: xarray.Dataset,
    reference_openers: dict,
    region: str,
    context: records.RunContext,
    start_limit: int | None,
) -> tuple[list[dict], list[str]]:
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
    return result.records, result.flags


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
    case the bundle's manifest also fixes the year and region.

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

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    forecast_dataset, challenger_slug, challenger_version = _open_evaluation_target(target)
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
                realism_records, realism_flags = _realism_records(
                    regional_challenger, reference_openers, region, context, None
                )
                scores = pandas.concat([scores, records.records_to_dataframe(realism_records)], ignore_index=True)
                flags.extend(realism_flags)
            except Exception as error:  # noqa: BLE001 - realism must not abort the local run
                flags.append(f"realism battery skipped: {error}")

    scores_path = output_path / SCORES_FILENAME
    scores.to_parquet(str(scores_path), index=False)

    # Only per-start metrics (gridded / Class-4) carry a start distribution to aggregate into a
    # mean and bootstrap CI. Realism records are already aggregates over the starts (start_date
    # is null, contracts.md §3.2); they stay in the long-format parquet but are not re-aggregated.
    per_start_scores = scores[scores["start_date"].notna()].reset_index(drop=True)
    summary = aggregate_scores(per_start_scores)
    summary_path = output_path / SCORES_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary_to_json_records(summary), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
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
            scores.to_parquet(str(scores_path), index=False)
            per_start_scores = scores[scores["start_date"].notna()].reset_index(drop=True)
            summary = aggregate_scores(per_start_scores)
            summary_path.write_text(
                json.dumps(summary_to_json_records(summary), sort_keys=True, indent=2, default=str),
                encoding="utf-8",
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
