# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Score currents on the STRAT view, stratified by CURRENT_TEST.

Same machinery as score_rung.py (same observation patching, same model
interpolation, same 15 m target depth) with one change: the CURRENT_TEST code is
carried on every observation row, so the sums can be grouped by code. The depth
interpolation helper is copied from classIV_support with the code added to the
group keys, which cannot change any result because currents sit at a single depth
and a code is a property of the drifter, not of the depth level.

Emitted per (variable, code, latitude band, lead day, start date): sum of squared
residuals, sum of signed residuals and count. The wind slippage distribution per
code is measured separately, directly on the store rows.
"""
import argparse
import os
import time

import numpy
import pandas

import oceanbench
from oceanbench.core import classIV_support
from oceanbench.core.references import observations as observations_module
from oceanbench.core.classIV_support import (
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.regions import subset_dataset_to_region

VARIABLE_BY_NAME = {
    "uo": Variable.EASTWARD_SEA_WATER_VELOCITY,
    "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY,
}
TARGET_DEPTH = classIV_support.VELOCITY_TARGET_DEPTH_METERS

TIME_KEY = Dimension.TIME.key()
LATITUDE_KEY = Dimension.LATITUDE.key()
LONGITUDE_KEY = Dimension.LONGITUDE.key()
DEPTH_KEY = Dimension.DEPTH.key()


def patch_observation_source(root):
    def observation_path(day_datetime):
        return os.path.join(root, pandas.Timestamp(day_datetime).strftime("%Y%m%d") + ".zarr")

    observations_module.observation_path = observation_path
    observations_module._should_stage_observations_locally = lambda: False


def interpolate_to_target_depth(frame):
    """classIV_support._interpolate_observations_to_target_depth plus the code column."""
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["observation_value", DEPTH_KEY])
    if frame.empty:
        return frame
    group_keys = [TIME_KEY, LATITUDE_KEY, LONGITUDE_KEY, "first_day", "lead_day", "current_test"]
    target_columns = [
        "observation_value", TIME_KEY, LATITUDE_KEY, LONGITUDE_KEY,
        "first_day", DEPTH_KEY, "lead_day", "current_test",
    ]
    records = [
        record
        for record in (
            classIV_support._interpolated_observation_record_at_target_depth(
                group_key, group, group_keys, DEPTH_KEY, TARGET_DEPTH
            )
            for group_key, group in frame.groupby(group_keys, sort=False)
        )
        if record is not None
    ]
    if not records:
        return pandas.DataFrame(columns=target_columns)
    return pandas.DataFrame.from_records(records)[target_columns]


def observations_with_code(observation_dataset, variable_key, lead_days_count):
    base_frame, selected, observation_dimension_key = classIV_support._prepared_class4_observations(
        observation_dataset, lead_days_count
    )
    values = observation_dataset[variable_key].isel({observation_dimension_key: selected}).values
    codes = observation_dataset["current_test"].isel({observation_dimension_key: selected}).values
    finite = ~numpy.isnan(values)
    frame = base_frame.loc[finite].copy()
    frame["observation_value"] = values[finite]
    frame["current_test"] = codes[finite]
    frame = frame[[
        "observation_value", TIME_KEY, LATITUDE_KEY, LONGITUDE_KEY,
        "first_day", DEPTH_KEY, "lead_day", "current_test",
    ]]
    return interpolate_to_target_depth(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-root", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--region", default="global")
    parser.add_argument("--variables", default="uo,vo")
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    patch_observation_source(arguments.obs_root)
    started = time.time()

    challenger_dataset = getattr(oceanbench.datasets.challenger, arguments.challenger)()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    start_count = challenger_dataset.sizes[first_day_key]
    selected = numpy.array_split(numpy.arange(start_count), arguments.chunk_count)[arguments.chunk_index]
    if selected.size == 0:
        pandas.DataFrame(columns=[
            "variable", "current_test", "lat_band", "lead_day", "first_day",
            "sumsq", "sumres", "count",
        ]).to_csv(arguments.out, index=False)
        return
    challenger_dataset = challenger_dataset.isel(
        {first_day_key: slice(int(selected[0]), int(selected[-1]) + 1)}
    )
    print(f"chunk {arguments.chunk_index}/{arguments.chunk_count} "
          f"start dates {selected[0]}..{selected[-1]} region={arguments.region}", flush=True)

    challenger_dataset = subset_dataset_to_region(challenger_dataset, arguments.region)
    observation_dataset = subset_dataset_to_region(
        observations_module.observations(challenger_dataset), arguments.region
    )
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]

    rows = []
    for variable_name in arguments.variables.split(","):
        variable_key = VARIABLE_BY_NAME[variable_name].key()
        variable_started = time.time()
        frame = observations_with_code(observation_dataset, variable_key, lead_days_count)
        if frame.empty:
            print(f"  {variable_name}: no observations", flush=True)
            continue
        model_variable = prepare_class4_model_variable(challenger[variable_key], variable_key)
        frame["model_value"] = interpolate_class4_model_to_observations(model_variable, frame)
        valid = frame.dropna(subset=["model_value", "observation_value"]).copy()
        valid["residual"] = valid["model_value"] - valid["observation_value"]
        valid["squared"] = valid["residual"] ** 2
        valid["lat_band"] = (numpy.floor(valid[LATITUDE_KEY] / 20.0) * 20.0).astype(int)
        grouped = valid.groupby(
            ["current_test", "lat_band", "lead_day", "first_day"], as_index=False
        ).agg(
            sumsq=("squared", "sum"),
            sumres=("residual", "sum"),
            count=("residual", "size"),
        )
        grouped["variable"] = variable_name
        rows.append(grouped)
        print(f"  {variable_name}: obs_rows={len(frame)} matched={len(valid)} "
              f"in {time.time() - variable_started:.0f}s", flush=True)

    result = pandas.concat(rows, ignore_index=True) if rows else pandas.DataFrame(
        columns=["variable", "current_test", "lat_band", "lead_day", "first_day",
                 "sumsq", "sumres", "count"]
    )
    result = result[["variable", "current_test", "lat_band", "lead_day", "first_day",
                     "sumsq", "sumres", "count"]]
    os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
    result.to_csv(arguments.out, index=False)
    print(f"DONE {arguments.out} rows={len(result)} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
