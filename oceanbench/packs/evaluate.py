# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Local evaluation against an evaluation pack (contracts.md §7).

``evaluate_local`` scores a user's forecast zarr(s) against the bundled references of an
evaluation pack and emits the same artifacts as the hosted run — the long-format per-start
records parquet and the aggregated summary — plus a self-contained overlay scorecard that
lays the user's model over the published challengers. It reuses the scoring runner
(:func:`oceanbench.runner.run.run_challenger_scores`) and the aggregation library
(:mod:`oceanbench.publish.aggregate`); no new score code lives here.

The pack is self-describing: every reference, the observation store and the
mean-dynamic-topography are resolved from ``pack-manifest.json`` alone.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy
import pandas
import xarray

from oceanbench.core import runtime_configuration as runtime_configuration_module
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.regions import subset_dataset_to_region
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
from oceanbench.runner.run import run_challenger_scores

YOUR_MODEL_SLUG = "your_model"
YOUR_MODEL_DISPLAY_NAME = "Your model"

SCORES_FILENAME = "scores.parquet"
SCORES_SUMMARY_FILENAME = "scores-summary.json"
SCORECARD_DIRECTORY = "scorecard"

_PER_START_KEY_COLUMNS = ["metric", "reference", "variable", "depth", "lead_day", "start_date"]


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
    flags: list[str] = field(default_factory=list)


def load_pack_manifest(pack_directory: str) -> dict:
    """Load and schema-validate a pack's ``pack-manifest.json`` (contracts.md §7)."""
    manifest = json.loads((Path(pack_directory) / PACK_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    validate_against_schema(manifest, "pack-manifest")
    return manifest


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


def evaluate_local(
    forecasts_path: str,
    *,
    pack_directory: str,
    output_directory: str,
    year: int = 2024,
    region: str = "global",
    published_scores_path: str | None = None,
    published_challengers_path: str | None = None,
    starts_limit: int | None = None,
    with_lagrangian: bool = False,
    include_class4: bool = True,
    include_realism: bool = True,
    artifacts: str = "scores",
) -> EvaluateLocalResult:
    """Score ``forecasts_path`` against ``pack_directory`` and emit records, summary and overlay scorecard."""
    if artifacts not in {"scores", "viewer", "all"}:
        raise ValueError("artifacts must be one of: scores, viewer, all")
    pack_path = Path(pack_directory)
    manifest = load_pack_manifest(pack_directory)
    kind = manifest["kind"]

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    forecast_dataset = open_forecast_dataset(forecasts_path)

    viewer_result = None
    if artifacts in {"viewer", "all"}:
        from oceanbench.packs.local_viewer import build_local_viewer

        viewer_result = build_local_viewer(
            forecast_dataset, output_directory=output_directory, year=year, starts_limit=starts_limit
        )

    if artifacts == "viewer":
        return EvaluateLocalResult(
            viewer_directory=viewer_result.viewer_directory,
            viewer_datasets_path=viewer_result.datasets_path,
            viewer_zarr_path=viewer_result.zarr_path,
            viewer_manifest_path=viewer_result.manifest_path,
        )

    reference_openers = {
        name: _pack_reference_opener(pack_path, entry["path"])
        for name, entry in manifest["contents"]["references"].items()
    }
    observation_opener = _pack_observation_opener(pack_path, manifest["contents"]["observations"]["path"])
    references = tuple(reference_openers.keys())

    with _pack_runtime_configuration(pack_path):
        run_result = run_challenger_scores(
            YOUR_MODEL_SLUG,
            region,
            year,
            references=references,
            include_gridded=True,
            include_mixed_layer_depth=(kind == "full"),
            include_geostrophic=True,
            include_class4=include_class4,
            include_lagrangian=with_lagrangian,
            area_weighted=True,
            challenger_version="local",
            output_root=str(output_path / "_run"),
            dataset=forecast_dataset,
            reference_openers=reference_openers,
            observation_opener=observation_opener,
            start_limit=starts_limit,
        )
        flags = list(run_result.flags)
        scores = run_result.scores

        if kind == "quick":
            scores = scores[scores["depth"].isna() | (scores["depth"] == "surface")].reset_index(drop=True)

        if include_realism:
            regional_challenger = subset_dataset_to_region(
                (
                    forecast_dataset
                    if starts_limit is None
                    else forecast_dataset.isel({Dimension.FIRST_DAY_DATETIME.key(): slice(0, starts_limit)})
                ),
                region,
            )
            context = records.RunContext(
                challenger=YOUR_MODEL_SLUG,
                challenger_version="local",
                year=year,
                region=region,
                oceanbench_version=OCEANBENCH_VERSION,
            )
            try:
                realism_records, realism_flags = _realism_records(
                    regional_challenger, reference_openers, region, context, starts_limit
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

    published_scores = pandas.read_parquet(published_scores_path) if published_scores_path is not None else None
    published_challengers = (
        json.loads(Path(published_challengers_path).read_text(encoding="utf-8"))
        if published_challengers_path is not None
        else None
    )
    scorecard_directory = output_path / SCORECARD_DIRECTORY
    write_overlay_scorecard(
        scorecard_directory,
        your_model_scores=scores,
        published_scores=published_scores,
        published_challengers=published_challengers,
        region=region,
        year=year,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    return EvaluateLocalResult(
        scores_path=str(scores_path),
        summary_path=str(summary_path),
        scorecard_path=str(scorecard_directory / "index.html"),
        scores=scores,
        flags=flags,
        viewer_directory=viewer_result.viewer_directory if viewer_result else None,
        viewer_datasets_path=viewer_result.datasets_path if viewer_result else None,
        viewer_zarr_path=viewer_result.zarr_path if viewer_result else None,
        viewer_manifest_path=viewer_result.manifest_path if viewer_result else None,
    )
