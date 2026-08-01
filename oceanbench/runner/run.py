# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The scoring runner: compute the metric functions directly (no papermill /
no notebook) and write per-start long-format records to
``runs/<challenger>/<year>/<region>/scores.parquet``.

Every metric is emitted per forecast start date. Gridded RMSD (variables, mixed
layer depth, geostrophic) and the Lagrangian deviation are per-start means over
the region. Class-4 RMSD is emitted per start too: each row is the RMSD over the
observations of one forecast start (per variable x depth_bin x lead_day) with
``n`` that observation count. The published value is a single RMSD pooled over
every observation at a given lead day across the whole year; it is recovered
exactly from the per-start rows via ``sqrt(sum(value ** 2 * n) / sum(n))`` (see
``oceanbench.core.classIV_support.recombine_class4_pooled_from_per_start`` and the
class4 branch of ``oceanbench.runner.parity.aggregate_runner_scores``).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
import os

import pandas
import xarray

import oceanbench.datasets.challenger as challenger_datasets
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.grid_alignment import GridAlignment, align_reference_to_challenger_grid
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.regions import GLOBAL_REGION_NAME, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd_per_start_date
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.runner import records

_GRIDDED_VARIABLES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
]
_MIXED_LAYER_DEPTH_VARIABLES = [Variable.MIXED_LAYER_DEPTH]
_GEOSTROPHIC_VARIABLES = [
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
]
_CLASS4_VARIABLES = _GRIDDED_VARIABLES

# The default sources: read straight from the public EDITO objects through the resilient
# chunk-fetch engine and its persistent cache. An offline reference bundle substitutes its
# own openers for these; nothing else changes.
LIVE_REFERENCE_OPENERS = {
    "glorys": glorys_reanalysis_dataset,
    "glo12": glo12_analysis_dataset,
}


@dataclass
class RunResult:
    parquet_path: str
    scores: pandas.DataFrame
    flags: list[str] = field(default_factory=list)


def _open_challenger(challenger: str) -> xarray.Dataset:
    opener = getattr(challenger_datasets, challenger, None)
    if opener is None:
        raise ValueError(f"Unknown challenger slug: {challenger}")
    return opener()


ReferenceOpener = Callable[[xarray.Dataset], xarray.Dataset]
ObservationOpener = Callable[[xarray.Dataset], xarray.Dataset]


def _aligned_reference(
    regional_challenger: xarray.Dataset,
    *,
    reference_name: str,
    reference_openers: dict[str, ReferenceOpener],
    region: str,
) -> tuple[xarray.Dataset, GridAlignment]:
    """Open a reference onto the challenger's own grid, once for every gridded metric family.

    Aligning here rather than inside each metric keeps the coverage reportable instead of
    absorbed (issue #305). A genuine mismatch raises and aborts the run.

    Both sides go through the CF standard-name rename first, exactly as every metric does
    before it computes: the quarter-degree GLORYS store names its axes ``lat``/``lon``, and
    the alignment reads ``latitude``/``longitude``.
    """
    reference = subset_dataset_to_region(reference_openers[reference_name](regional_challenger), region)
    return align_reference_to_challenger_grid(
        rename_dataset_with_standard_names(regional_challenger),
        rename_dataset_with_standard_names(reference),
    )


def _gridded_records(
    regional_challenger: xarray.Dataset,
    *,
    reference_name: str,
    aligned_reference: xarray.Dataset,
    variables: list[Variable],
    context: records.RunContext,
    area_weighted: bool,
    depth_applicable: bool,
    transform: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
) -> list[dict]:
    challenger_input = transform(regional_challenger) if transform is not None else regional_challenger
    reference_input = transform(aligned_reference) if transform is not None else aligned_reference
    per_start_frames = rmsd_per_start_date(challenger_input, reference_input, variables, area_weighted=area_weighted)
    return [
        record
        for start_date, frame in per_start_frames.items()
        for record in records.gridded_rmsd_records(
            frame,
            reference=reference_name,
            context=context,
            start_date=start_date,
            depth_applicable=depth_applicable,
        )
    ]


def _class4_records(
    regional_challenger: xarray.Dataset,
    region: str,
    context: records.RunContext,
    observation_opener: ObservationOpener,
) -> tuple[list[dict], str | None]:
    from oceanbench.core.classIV import rmsd_class4_validation_per_start
    from oceanbench.core.references.observations import ObservationDataUnavailableError

    try:
        observation_dataset = subset_dataset_to_region(observation_opener(regional_challenger), region)
        per_start_table = rmsd_class4_validation_per_start(
            regional_challenger,
            observation_dataset,
            variables=_CLASS4_VARIABLES,
            challenger_slug=context.challenger,
        )
    except (ObservationDataUnavailableError, KeyError, ValueError) as error:
        return [], f"class4_rmsd unavailable: {error}"
    if per_start_table.empty:
        return [], "class4_rmsd produced no rows"
    return records.class4_per_start_records(per_start_table, context=context), None


def _lagrangian_records(
    regional_challenger: xarray.Dataset,
    *,
    reference_name: str,
    reference_openers: dict[str, ReferenceOpener],
    region: str,
    context: records.RunContext,
) -> list[dict]:
    from oceanbench.core.dataset_utils import Dimension
    from oceanbench.core.lagrangian_support import LAGRANGIAN_ROW_LABEL
    from oceanbench.core.lead_day_utils import lead_day_labels
    from oceanbench.core.lagrangian_trajectory import (
        LEAD_DAY_START,
        _all_deviation_of_lagrangian_trajectories,
        _get_random_ocean_points_from_file,
        _harmonise_dataset,
        lagrangian_particle_count_for_region,
    )

    reference = subset_dataset_to_region(reference_openers[reference_name](regional_challenger), region)
    particle_count = lagrangian_particle_count_for_region(regional_challenger, regional_challenger)
    harmonised_challenger = _harmonise_dataset(regional_challenger)
    harmonised_reference = _harmonise_dataset(reference)
    lead_day_stop = harmonised_challenger.sizes[Dimension.LEAD_DAY_INDEX.key()] - 1
    latitudes, longitudes = _get_random_ocean_points_from_file(
        harmonised_challenger,
        variable_name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        n=particle_count,
        seed=123,
    )
    weekly_deviations = _all_deviation_of_lagrangian_trajectories(
        harmonised_challenger, harmonised_reference, latitudes, longitudes
    )
    start_dates = harmonised_challenger[Dimension.FIRST_DAY_DATETIME.key()].values
    index_labels = lead_day_labels(LEAD_DAY_START, lead_day_stop)
    emitted = []
    for start_date, weekly_deviation in zip(start_dates, weekly_deviations):
        frame = pandas.DataFrame({LAGRANGIAN_ROW_LABEL: weekly_deviation[LEAD_DAY_START - 1 : lead_day_stop]})
        frame.index = index_labels
        emitted.extend(
            records.lagrangian_records(
                frame.T,
                reference=reference_name,
                context=context,
                start_date=start_date,
            )
        )
    return emitted


def live_observation_opener(regional_challenger: xarray.Dataset) -> xarray.Dataset:
    """Open the Class-4 observation store from its public EDITO objects."""
    from oceanbench.core.references.observations import observations

    return observations(regional_challenger)


def registered_challengers() -> tuple[str, ...]:
    """Every challenger slug this installation can open by itself, in alphabetical order.

    Read off the public openers of :mod:`oceanbench.datasets.challenger` rather than restated
    here, so adding a challenger there is enough to make it evaluable by slug. Restricted to
    functions that module defines so its imports (``xarray`` and friends) never look like slugs.
    """
    return tuple(
        sorted(
            name
            for name, member in vars(challenger_datasets).items()
            if not name.startswith("_")
            and inspect.isfunction(member)
            and member.__module__ == challenger_datasets.__name__
        )
    )


def is_registered_challenger(name: str) -> bool:
    """Whether ``name`` is a challenger slug this installation can open by itself."""
    return name in registered_challengers()


def open_registered_challenger(challenger: str) -> xarray.Dataset:
    """Open a registered challenger's forecast dataset by slug."""
    return _open_challenger(challenger)


def run_challenger_scores(
    challenger: str,
    region: str = GLOBAL_REGION_NAME,
    year: int = 2024,
    *,
    references: tuple[str, ...] = ("glorys", "glo12"),
    include_gridded: bool = True,
    include_mixed_layer_depth: bool = True,
    include_geostrophic: bool = True,
    include_class4: bool = True,
    include_lagrangian: bool = True,
    area_weighted: bool = True,
    challenger_version: str = "0.2.1",
    output_root: str = "runs",
    dataset: xarray.Dataset | None = None,
    reference_openers: dict[str, ReferenceOpener] | None = None,
    observation_opener: ObservationOpener | None = None,
    start_limit: int | None = None,
) -> RunResult:
    """Score ``challenger`` on ``region``/``year`` and write per-start records to parquet.

    By default the challenger is opened by slug and references/observations are fetched
    live (through the resilient engine + persistent cache). The data sources are injectable
    so the same scoring code can run against a pre-opened forecast dataset and the bundled
    references of an evaluation pack (contracts.md §7): pass ``dataset`` for an already-open
    challenger, ``reference_openers`` mapping a reference name to a callable returning that
    reference aligned to the challenger, ``observation_opener`` for the Class-4 observation
    store, and ``start_limit`` to score only the first N forecast starts (quick-look mode).
    """
    opened_dataset = dataset if dataset is not None else _open_challenger(challenger)
    if start_limit is not None:
        opened_dataset = opened_dataset.isel({Dimension.FIRST_DAY_DATETIME.key(): slice(0, start_limit)})
    resolved_reference_openers = reference_openers if reference_openers is not None else LIVE_REFERENCE_OPENERS
    resolved_observation_opener = observation_opener if observation_opener is not None else live_observation_opener
    regional_challenger = subset_dataset_to_region(opened_dataset, region)
    context = records.RunContext(
        challenger=challenger,
        challenger_version=challenger_version,
        year=year,
        region=region,
        oceanbench_version=OCEANBENCH_VERSION,
    )

    all_records: list[dict] = []
    flags: list[str] = []

    for reference_name in references:
        if not (include_gridded or include_mixed_layer_depth or include_geostrophic):
            continue
        aligned_reference, alignment = _aligned_reference(
            regional_challenger,
            reference_name=reference_name,
            reference_openers=resolved_reference_openers,
            region=region,
        )
        all_records.append(
            records.grid_coverage_record(
                context=context,
                reference=reference_name,
                coverage=alignment.coverage,
                matched_cell_count=alignment.matched_cell_count,
            )
        )
        if alignment.snapped:
            flags.append(f"grid alignment vs {reference_name}: {alignment.describe()}")
        if include_gridded:
            all_records.extend(
                _gridded_records(
                    regional_challenger,
                    reference_name=reference_name,
                    aligned_reference=aligned_reference,
                    variables=_GRIDDED_VARIABLES,
                    context=context,
                    area_weighted=area_weighted,
                    depth_applicable=True,
                )
            )
        if include_mixed_layer_depth:
            all_records.extend(
                _gridded_records(
                    regional_challenger,
                    reference_name=reference_name,
                    aligned_reference=aligned_reference,
                    variables=_MIXED_LAYER_DEPTH_VARIABLES,
                    context=context,
                    area_weighted=area_weighted,
                    depth_applicable=False,
                    transform=compute_mixed_layer_depth,
                )
            )
        if include_geostrophic:
            all_records.extend(
                _gridded_records(
                    regional_challenger,
                    reference_name=reference_name,
                    aligned_reference=aligned_reference,
                    variables=_GEOSTROPHIC_VARIABLES,
                    context=context,
                    area_weighted=area_weighted,
                    depth_applicable=False,
                    transform=compute_geostrophic_currents,
                )
            )

    if include_class4:
        class4_records, class4_flag = _class4_records(regional_challenger, region, context, resolved_observation_opener)
        all_records.extend(class4_records)
        if class4_flag is not None:
            flags.append(class4_flag)

    if include_lagrangian:
        for reference_name in references:
            try:
                all_records.extend(
                    _lagrangian_records(
                        regional_challenger,
                        reference_name=reference_name,
                        reference_openers=resolved_reference_openers,
                        region=region,
                        context=context,
                    )
                )
            except Exception as error:  # noqa: BLE001 - one metric must not abort the whole run
                flags.append(f"lagrangian_{reference_name} skipped: {error}")

    if not area_weighted:
        flags.append("gridded RMSD computed UNWEIGHTED (test-only mode; production default is area-weighted).")

    scores = records.records_to_dataframe(all_records)
    parquet_path = os.path.join(output_root, challenger, str(year), region, "scores.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    scores.to_parquet(parquet_path, index=False)
    return RunResult(parquet_path=parquet_path, scores=scores, flags=list(dict.fromkeys(flags)))


def run_challenger(
    challenger: str,
    region: str = GLOBAL_REGION_NAME,
    year: int = 2024,
    **options,
) -> str:
    """Score ``challenger`` and return the path to the written ``scores.parquet``."""
    return run_challenger_scores(challenger, region, year, **options).parquet_path
