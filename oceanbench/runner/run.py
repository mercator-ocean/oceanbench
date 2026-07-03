# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The scoring runner: compute the metric functions directly (no papermill /
no notebook) and write per-start long-format records to
``runs/<challenger>/<year>/<region>/scores.parquet``.

Gridded RMSD (variables, mixed layer depth, geostrophic) and the Lagrangian
deviation are emitted per forecast start date. Class-4 RMSD is emitted
aggregate-only (``start_date`` null) and flagged: its published value is a
single RMSD pooled over every observation at a given lead day across the whole
year, which does not decompose into a per-start mean.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import os

import pandas
import xarray

import oceanbench.datasets.challenger as challenger_datasets
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
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

_REFERENCE_OPENERS = {
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


def _gridded_records(
    regional_challenger: xarray.Dataset,
    *,
    reference_name: str,
    variables: list[Variable],
    region: str,
    context: records.RunContext,
    area_weighted: bool,
    depth_applicable: bool,
    transform: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
) -> list[dict]:
    reference = subset_dataset_to_region(_REFERENCE_OPENERS[reference_name](regional_challenger), region)
    challenger_input = transform(regional_challenger) if transform is not None else regional_challenger
    reference_input = transform(reference) if transform is not None else reference
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
) -> tuple[list[dict], str | None]:
    from oceanbench.core.classIV import rmsd_class4_validation
    from oceanbench.core.references.observations import ObservationDataUnavailableError, observations

    try:
        observation_dataset = subset_dataset_to_region(observations(regional_challenger), region)
    except ObservationDataUnavailableError as error:
        return [], f"class4_rmsd unavailable: {error}"
    frame = rmsd_class4_validation(regional_challenger, observation_dataset, variables=_CLASS4_VARIABLES)
    if frame.empty or "Message" in frame.columns:
        return [], "class4_rmsd produced no rows"
    return records.class4_records(frame, context=context, start_date=None), None


def _lagrangian_records(
    regional_challenger: xarray.Dataset,
    *,
    reference_name: str,
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

    reference = subset_dataset_to_region(_REFERENCE_OPENERS[reference_name](regional_challenger), region)
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


def run_challenger_scores(
    challenger: str,
    region: str = GLOBAL_REGION_NAME,
    year: int = 2024,
    *,
    references: tuple[str, ...] = ("glorys", "glo12"),
    include_gridded: bool = True,
    include_class4: bool = True,
    include_lagrangian: bool = True,
    area_weighted: bool = True,
    challenger_version: str = "0.2.1",
    output_root: str = "runs",
) -> RunResult:
    """Score ``challenger`` on ``region``/``year`` and write per-start records to parquet."""
    dataset = _open_challenger(challenger)
    regional_challenger = subset_dataset_to_region(dataset, region)
    context = records.RunContext(
        challenger=challenger,
        challenger_version=challenger_version,
        year=year,
        region=region,
        oceanbench_version=OCEANBENCH_VERSION,
    )

    all_records: list[dict] = []
    flags: list[str] = []

    if include_gridded:
        for reference_name in references:
            all_records.extend(
                _gridded_records(
                    regional_challenger,
                    reference_name=reference_name,
                    variables=_GRIDDED_VARIABLES,
                    region=region,
                    context=context,
                    area_weighted=area_weighted,
                    depth_applicable=True,
                )
            )
            all_records.extend(
                _gridded_records(
                    regional_challenger,
                    reference_name=reference_name,
                    variables=_MIXED_LAYER_DEPTH_VARIABLES,
                    region=region,
                    context=context,
                    area_weighted=area_weighted,
                    depth_applicable=False,
                    transform=compute_mixed_layer_depth,
                )
            )
            all_records.extend(
                _gridded_records(
                    regional_challenger,
                    reference_name=reference_name,
                    variables=_GEOSTROPHIC_VARIABLES,
                    region=region,
                    context=context,
                    area_weighted=area_weighted,
                    depth_applicable=False,
                    transform=compute_geostrophic_currents,
                )
            )

    if include_class4:
        class4_records, class4_flag = _class4_records(regional_challenger, region, context)
        all_records.extend(class4_records)
        flags.append(
            "class4_rmsd emitted aggregate-only (start_date null): the published value pools all "
            "observations per lead day into one RMSD, which does not decompose into a per-start mean."
        )
        if class4_flag is not None:
            flags.append(class4_flag)

    if include_lagrangian:
        for reference_name in references:
            try:
                all_records.extend(
                    _lagrangian_records(
                        regional_challenger,
                        reference_name=reference_name,
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
    return RunResult(parquet_path=parquet_path, scores=scores, flags=flags)


def run_challenger(
    challenger: str,
    region: str = GLOBAL_REGION_NAME,
    year: int = 2024,
    **options,
) -> str:
    """Score ``challenger`` and return the path to the written ``scores.parquet``."""
    return run_challenger_scores(challenger, region, year, **options).parquet_path
