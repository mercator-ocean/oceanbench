# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Class-4 observation match-up artifact (contracts.md §4, ``class4-matchups`` parquet).

One row per observation point: the observation value, the model value interpolated
to that point, its latitude / longitude / depth / time, the CF standard-name
variable, the 1-based ``lead_day``, the forecast ``start_date``, and — for sea level —
the SLA shift already folded into ``model_value``.

The model-at-observation values are the very ones the Class-4 metric consumes. This
writer does not touch the numerical core: it composes the same public core
functions (``create_class4_observations_dataframe``,
``prepare_class4_model_variable``, ``interpolate_class4_model_to_observations``) that
``oceanbench.core.classIV`` uses, so the RMSD recomputed from the match-up parquet by
:func:`recompute_class4_rmsd` equals the Class-4 metric output exactly (see
``tests/runner/test_matchups.py``).
"""

import math
import os

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    create_class4_observations_dataframe,
    interpolate_class4_model_to_observations,
    mean_sea_surface_height_shift,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.runner.records import RunContext

MATCHUP_COLUMNS = [
    "challenger",
    "challenger_version",
    "year",
    "region",
    "variable",
    "depth",
    "depth_bin",
    "lead_day",
    "start_date",
    "time",
    "latitude",
    "longitude",
    "observation_value",
    "model_value",
    "sla_shift",
]


def _observation_frame_with_model(
    challenger: xarray.Dataset,
    observations: xarray.Dataset,
    variable_key: str,
    lead_days_count: int,
    challenger_slug: str | None = None,
) -> pandas.DataFrame:
    observation_frame = create_class4_observations_dataframe(
        observations,
        variable_key,
        variable_key,
        lead_days_count,
    )
    if observation_frame.empty:
        return observation_frame
    observation_frame = observation_frame.dropna(subset=["observation_value"])
    model_variable = prepare_class4_model_variable(challenger[variable_key], variable_key, challenger_slug)
    observation_frame = observation_frame.assign(
        model_value=interpolate_class4_model_to_observations(model_variable, observation_frame)
    )
    return observation_frame


def _shaped_matchups(
    observation_frame: pandas.DataFrame,
    variable_key: str,
    context: RunContext,
    challenger_slug: str | None = None,
) -> pandas.DataFrame:
    is_sea_surface_height = variable_key == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    return pandas.DataFrame(
        {
            "challenger": context.challenger,
            "challenger_version": context.challenger_version,
            "year": context.year,
            "region": context.region,
            "variable": variable_key,
            "depth": observation_frame[Dimension.DEPTH.key()].to_numpy(),
            "depth_bin": observation_frame["depth_bin"].to_numpy(),
            "lead_day": observation_frame["lead_day"].to_numpy().astype("int64") + 1,
            "start_date": observation_frame["first_day"].to_numpy(),
            "time": observation_frame[Dimension.TIME.key()].to_numpy(),
            "latitude": observation_frame[Dimension.LATITUDE.key()].to_numpy(),
            "longitude": observation_frame[Dimension.LONGITUDE.key()].to_numpy(),
            "observation_value": observation_frame["observation_value"].to_numpy(),
            "model_value": observation_frame["model_value"].to_numpy(),
            "sla_shift": mean_sea_surface_height_shift(challenger_slug) if is_sea_surface_height else numpy.nan,
        },
        columns=MATCHUP_COLUMNS,
    )


def class4_matchups(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    variables: list[Variable],
    *,
    context: RunContext,
) -> pandas.DataFrame:
    """Build the Class-4 match-up dataframe (one row per observation point) for ``variables``.

    Column ``model_value`` is the model interpolated to the observation exactly as the
    Class-4 metric computes it; for sea-surface-height rows it is already SLA-shifted and
    ``sla_shift`` records the applied constant.
    """
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    challenger_slug = context.challenger
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    per_variable = [
        _shaped_matchups(observation_frame, variable.key(), context, challenger_slug)
        for variable in variables
        for observation_frame in [
            _observation_frame_with_model(
                challenger, observation_dataset, variable.key(), lead_days_count, challenger_slug
            )
        ]
        if not observation_frame.empty
    ]
    if not per_variable:
        return pandas.DataFrame(columns=MATCHUP_COLUMNS)
    return pandas.concat(per_variable, ignore_index=True)


def recompute_class4_rmsd(matchups: pandas.DataFrame) -> pandas.DataFrame:
    """Recompute the Class-4 RMSD table from a match-up dataframe.

    Same reduction as ``oceanbench.core.classIV_support.compute_class4_rmsd_table``:
    ``sqrt(mean(squared_difference))`` over the observations of each
    ``(variable, depth_bin, lead_day)`` cell, dropping rows with a missing model or
    observation value. ``lead_day`` is the match-up's 1-based value.
    """
    valid = matchups.dropna(subset=["model_value", "observation_value"])
    grouped = (
        valid.assign(squared_difference=(valid["model_value"] - valid["observation_value"]) ** 2)
        .groupby(["variable", "depth_bin", "lead_day"], as_index=False)
        .agg(
            rmsd=("squared_difference", lambda values: math.sqrt(values.mean())),
            count=("squared_difference", "size"),
        )
    )
    grouped["count"] = grouped["count"].astype(int)
    return grouped[["variable", "depth_bin", "lead_day", "rmsd", "count"]]


def write_class4_matchups(matchups: pandas.DataFrame, output_path: str) -> tuple[str, int]:
    """Write the match-up dataframe to parquet, returning ``(path, bytes)``."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    matchups.to_parquet(output_path, index=False)
    return output_path, os.path.getsize(output_path)
