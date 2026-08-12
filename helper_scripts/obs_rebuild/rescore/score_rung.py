# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Score one ladder rung for one challenger, one region, one chunk of start dates.

Mirrors oceanbench.core.classIV.rmsd_class4_validation exactly, but emits the
per-group sum of squared differences and count instead of the RMSD, so that
chunks can be combined without loss: rmsd = sqrt(sum(sumsq) / sum(count)),
which is what _compute_rmsd_table computes on the whole set.

The observation source is redirected to a local directory of daily zarrs by
patching observation_path. Observation staging is disabled so that no rung can
read another rung's cache; mean-dynamic-topography staging is left on.
"""
import argparse, os, sys, time

import numpy
import pandas

import oceanbench
from oceanbench.core import classIV_support
from oceanbench.core.references import observations as observations_module
from oceanbench.core.classIV_support import (
    create_class4_observations_dataframe,
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.regions import subset_dataset_to_region

VARIABLE_BY_NAME = {
    "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    "so": Variable.SEA_WATER_SALINITY,
    "sla": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    "uo": Variable.EASTWARD_SEA_WATER_VELOCITY,
    "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY,
}


def patch_observation_source(root: str) -> None:
    def observation_path(day_datetime):
        return os.path.join(root, pandas.Timestamp(day_datetime).strftime("%Y%m%d") + ".zarr")

    observations_module.observation_path = observation_path
    observations_module._should_stage_observations_locally = lambda: False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", required=True)
    parser.add_argument("--obs-root", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--region", default="global")
    parser.add_argument("--variables", default="thetao,so,sla,uo,vo")
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    patch_observation_source(arguments.obs_root)

    started_at = time.time()
    challenger_dataset = getattr(oceanbench.datasets.challenger, arguments.challenger)()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    start_date_count = challenger_dataset.sizes[first_day_key]
    selected = numpy.array_split(numpy.arange(start_date_count), arguments.chunk_count)[arguments.chunk_index]
    if selected.size == 0:
        print("empty chunk, nothing to do", flush=True)
        pandas.DataFrame(
            columns=["rung", "region", "variable", "depth_bin", "lead_day", "sumsq", "count"]
        ).to_csv(arguments.out, index=False)
        return
    challenger_dataset = challenger_dataset.isel({first_day_key: slice(int(selected[0]), int(selected[-1]) + 1)})
    print(
        f"[{arguments.rung}] chunk {arguments.chunk_index}/{arguments.chunk_count} "
        f"start dates {selected[0]}..{selected[-1]} region={arguments.region}",
        flush=True,
    )

    challenger_dataset = subset_dataset_to_region(challenger_dataset, arguments.region)
    observation_dataset = subset_dataset_to_region(
        observations_module.observations(challenger_dataset), arguments.region
    )

    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]

    rows = []
    for variable_name in arguments.variables.split(","):
        variable_key = VARIABLE_BY_NAME[variable_name].key()
        variable_started_at = time.time()
        observations_dataframe = create_class4_observations_dataframe(
            observation_dataset, variable_key, variable_key, lead_days_count
        )
        if observations_dataframe.empty:
            print(f"  {variable_name}: no observations", flush=True)
            continue
        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])
        model_variable = prepare_class4_model_variable(challenger[variable_key], variable_key)
        observations_dataframe["model_value"] = interpolate_class4_model_to_observations(
            model_variable, observations_dataframe
        )
        valid = observations_dataframe.dropna(subset=["model_value", "observation_value"])
        grouped = (
            valid.assign(squared_difference=(valid["model_value"] - valid["observation_value"]) ** 2)
            .groupby(["depth_bin", "lead_day"], as_index=False)
            .agg(sumsq=("squared_difference", "sum"), count=("squared_difference", "size"))
        )
        grouped["variable"] = variable_key
        rows.append(grouped)
        print(
            f"  {variable_name}: obs_rows={len(observations_dataframe)} matched={len(valid)} "
            f"in {time.time() - variable_started_at:.0f}s",
            flush=True,
        )

    if rows:
        result = pandas.concat(rows, ignore_index=True)
    else:
        result = pandas.DataFrame(columns=["variable", "depth_bin", "lead_day", "sumsq", "count"])
    result["rung"] = arguments.rung
    result["region"] = arguments.region
    result["chunk"] = arguments.chunk_index
    result = result[["rung", "region", "chunk", "variable", "depth_bin", "lead_day", "sumsq", "count"]]
    os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
    result.to_csv(arguments.out, index=False)
    print(f"DONE {arguments.out} rows={len(result)} in {time.time() - started_at:.0f}s", flush=True)


if __name__ == "__main__":
    main()
