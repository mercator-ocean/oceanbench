# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Evaluation-pack builder (contracts.md §7).

``build_pack`` produces a self-describing pack directory from ingested / staged data:
the gridded references (surface subset for a ``quick`` pack, all depths for ``full``), the
Class-4 observation match-up store, the mean-dynamic-topography for the SSH->SLA conversion,
a validated ``pack-manifest.json`` (stamping the upstream products and retrieval dates,
contracts.md §1) and a ``README.md`` carrying the Copernicus Marine credit and disclaimer
(contracts.md §11) verbatim. Baselines (climatology / persistence) are not available yet;
the manifest supports optional baseline entries and flags their absence.

Only 1-degree data is cached locally, so the demo ``quick`` pack is built from the
glonet-compatible 1-degree GLORYS + GLO12 surface subsets plus the observation store.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import shutil
from pathlib import Path

import numpy
import pandas
import xarray

from oceanbench.core import challenger_datasets
from oceanbench.core.attribution import (
    COPERNICUS_MARINE_CREDIT,
    COPERNICUS_MARINE_DISCLAIMER,
    COPERNICUS_MARINE_SOURCE_PRODUCTS,
    copernicus_marine_attribution_attrs,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import DepthLevel, Dimension, Variable
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import (
    _mean_dynamic_topography_stage_path,
    load_mean_dynamic_topography,
    observations,
)
from oceanbench.core.regions import GLOBAL_REGION_NAME, subset_dataset_to_region
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.packs.manifest import PACK_MANIFEST_SCHEMA_VERSION, PackManifestResult, write_pack_manifest

_CORE_REFERENCE_VARIABLES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
]

_REFERENCE_OPENERS = {
    "glorys": glorys_reanalysis_dataset,
    "glo12": glo12_analysis_dataset,
}

_REFERENCE_PRODUCT_IDS = {
    "glorys": COPERNICUS_MARINE_SOURCE_PRODUCTS["GLORYS12"],
    "glo12": COPERNICUS_MARINE_SOURCE_PRODUCTS["GLO12"],
}

_OBSERVATION_PRODUCT_ID = "OceanBench-observations-2024-v3 (INSITU/SLA/SST Copernicus Marine match-ups)"

_ALL_DEPTH_LABELS = ["surface", "50m", "100m", "200m", "300m", "500m"]

_MEAN_DYNAMIC_TOPOGRAPHY_RESOLUTION = "one_degree"


@dataclass(frozen=True)
class PackSources:
    """Where a pack's data comes from.

    ``template_challenger`` is the challenger slug whose native grid and forecast starts
    define the pack (references are aligned to it); it is a template, not a scored model.
    ``start_limit`` bounds the number of forecast starts bundled (a demo pack keeps a handful).
    ``baselines`` names any climatology / persistence baseline slugs to bundle (none yet).
    """

    template_challenger: str
    references: tuple[str, ...] = ("glorys", "glo12")
    region: str = GLOBAL_REGION_NAME
    start_limit: int | None = None
    baselines: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackBuildResult:
    pack_directory: str
    manifest: dict
    manifest_path: str
    flags: list[str] = field(default_factory=list)


def _iso_date(value: numpy.datetime64) -> str:
    return pandas.Timestamp(value).strftime("%Y-%m-%d")


def _write_zarr(dataset: xarray.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = dataset.load()
    prepared.attrs.update(copernicus_marine_attribution_attrs())
    for variable_name in prepared.variables:
        prepared[variable_name].encoding.pop("chunks", None)
    prepared.to_zarr(str(path), mode="w", consolidated=True)


def _surface_reference(reference: xarray.Dataset) -> xarray.Dataset:
    depth_key = Dimension.DEPTH.key()
    if depth_key not in reference.dims:
        return reference
    return reference.sel({depth_key: [DepthLevel.SURFACE.value]}, method="nearest")


def _reference_subset(reference: xarray.Dataset, region: str, kind: str) -> xarray.Dataset:
    standard = rename_dataset_with_standard_names(reference)
    regional = subset_dataset_to_region(standard, region)
    present_variables = [variable.key() for variable in _CORE_REFERENCE_VARIABLES if variable.key() in regional]
    selected = regional[present_variables]
    return _surface_reference(selected) if kind == "quick" else selected


def _build_reference_store(
    template: xarray.Dataset,
    reference_name: str,
    region: str,
    kind: str,
    references_directory: Path,
) -> dict:
    reference = _REFERENCE_OPENERS[reference_name](template)
    subset = _reference_subset(reference, region, kind)
    store_path = references_directory / f"{reference_name}.zarr"
    _write_zarr(subset, store_path)
    depth_labels = ["surface"] if kind == "quick" else _ALL_DEPTH_LABELS
    return {
        "path": f"references/{reference_name}.zarr",
        "variables": sorted(str(name) for name in subset.data_vars),
        "depths": depth_labels,
    }


def _build_observation_store(
    full_template: xarray.Dataset,
    start_dates: numpy.ndarray,
    region: str,
    observations_directory: Path,
) -> dict:
    observation_dataset = subset_dataset_to_region(observations(full_template), region)
    first_day_values = observation_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    selected = numpy.flatnonzero(numpy.isin(first_day_values, start_dates))
    observation_subset = observation_dataset.isel(observations=selected)
    store_path = observations_directory / "observations.zarr"
    _write_zarr(observation_subset, store_path)
    return {"path": "observations/observations.zarr"}


def _bundle_mean_dynamic_topography(pack_directory: Path) -> dict:
    load_mean_dynamic_topography(_MEAN_DYNAMIC_TOPOGRAPHY_RESOLUTION)
    staged_path = _mean_dynamic_topography_stage_path(_MEAN_DYNAMIC_TOPOGRAPHY_RESOLUTION)
    destination = pack_directory / staged_path.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(staged_path, destination)
    return {"path": staged_path.name, "resolution": _MEAN_DYNAMIC_TOPOGRAPHY_RESOLUTION}


def _pack_readme(manifest: dict) -> str:
    reference_lines = "\n".join(
        f"- `{name}` — {entry['path']} (variables: {', '.join(entry['variables'])}; "
        f"depths: {', '.join(entry['depths'])})"
        for name, entry in sorted(manifest["contents"]["references"].items())
    )
    baseline_note = (
        "Climatology / persistence baselines are bundled (skill-vs-baseline available)."
        if manifest["baselines_available"]
        else "No climatology / persistence baselines are bundled yet, so skill-vs-baseline "
        "cannot be computed locally from this pack."
    )
    notes = "\n".join(f"- {note}" for note in manifest.get("notes", []))
    return f"""<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench evaluation pack — {manifest["kind"]} ({manifest["year"]})

Self-describing bundle for scoring an ocean-forecast model locally with
`oceanbench evaluate-local`. The `pack-manifest.json` next to this file locates
every reference, the observation match-up store and the mean-dynamic-topography.

- kind: {manifest["kind"]}
- year: {manifest["year"]}
- region: {manifest["region"]}
- resolution: {manifest["resolution"]}
- forecast starts: {len(manifest["start_dates"])} ({manifest["start_dates"][0]} … {manifest["start_dates"][-1]})
- oceanbench version: {manifest["oceanbench_version"]}
- generated at: {manifest["generated_at"]}

## Gridded references

{reference_lines}

## Baselines

{baseline_note}

## Notes

{notes or "- (none)"}

## Attribution (Copernicus Marine)

{COPERNICUS_MARINE_CREDIT}

## Disclaimer

{COPERNICUS_MARINE_DISCLAIMER}
"""


def build_pack(kind: str, year: int, sources: PackSources, output_dir: str) -> PackBuildResult:
    """Build an evaluation pack directory (contracts.md §7) and return its manifest.

    ``kind`` is ``quick`` (surface reference fields, minutes to score) or ``full`` (all
    depths). ``sources`` selects the template challenger, references, region and start count.
    The produced directory carries the reference / observation / MDT stores, a validated
    ``pack-manifest.json`` and a ``README.md`` with the Copernicus Marine attribution.
    """
    if kind not in ("quick", "full"):
        raise ValueError(f"Unsupported pack kind: {kind!r} (expected 'quick' or 'full').")

    pack_directory = Path(output_dir)
    pack_directory.mkdir(parents=True, exist_ok=True)

    full_template = challenger_datasets.__dict__[sources.template_challenger]()
    regional_full_template = subset_dataset_to_region(full_template, sources.region)
    template = (
        regional_full_template
        if sources.start_limit is None
        else regional_full_template.isel({Dimension.FIRST_DAY_DATETIME.key(): slice(0, sources.start_limit)})
    )
    start_date_values = template[Dimension.FIRST_DAY_DATETIME.key()].values
    start_dates = [_iso_date(value) for value in start_date_values]
    resolution = "one_degree"

    references = {
        reference_name: _build_reference_store(
            template, reference_name, sources.region, kind, pack_directory / "references"
        )
        for reference_name in sources.references
    }
    observation_entry = _build_observation_store(
        regional_full_template, start_date_values, sources.region, pack_directory / "observations"
    )
    mean_dynamic_topography_entry = _bundle_mean_dynamic_topography(pack_directory)

    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    upstream = [
        {"name": reference_name, "product_id": _REFERENCE_PRODUCT_IDS[reference_name], "retrieved": retrieved}
        for reference_name in sources.references
    ] + [{"name": "observations", "product_id": _OBSERVATION_PRODUCT_ID, "retrieved": retrieved}]

    flags: list[str] = []
    baselines_available = bool(sources.baselines)
    notes: list[str] = []
    if not baselines_available:
        note = "Baselines (climatology / persistence) are not available yet; skill-vs-baseline is unavailable locally."
        notes.append(note)
        flags.append(note)
    if kind == "quick":
        notes.append(
            "Quick pack carries surface reference fields only; subsurface gridded RMSD and "
            "mixed-layer-depth need the full pack."
        )

    manifest = {
        "schema_version": PACK_MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "year": year,
        "region": sources.region,
        "resolution": resolution,
        "oceanbench_version": OCEANBENCH_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_dates": start_dates,
        "attribution": COPERNICUS_MARINE_CREDIT,
        "disclaimer": COPERNICUS_MARINE_DISCLAIMER,
        "source_products": dict(COPERNICUS_MARINE_SOURCE_PRODUCTS),
        "upstream": upstream,
        "contents": {
            "references": references,
            "observations": observation_entry,
            "mean_dynamic_topography": mean_dynamic_topography_entry,
            "baselines": {},
        },
        "baselines_available": baselines_available,
        "notes": notes,
    }

    result: PackManifestResult = write_pack_manifest(manifest, str(pack_directory))
    (pack_directory / "README.md").write_text(_pack_readme(manifest), encoding="utf-8")

    return PackBuildResult(
        pack_directory=str(pack_directory),
        manifest=result.manifest,
        manifest_path=result.manifest_path,
        flags=flags,
    )
