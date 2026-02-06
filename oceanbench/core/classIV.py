# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import time
import numpy
import xarray
import pandas
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)

LEAD_DAYS_COUNT = 1
SELECTED_LEAD_DAYS_FOR_DISPLAY = [0, 2, 4, 6, 9]
REANALYSIS_MSSH_SHIFT = -0.1148
MSSH_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.zarr"


VARIABLE_DISPLAY_NAMES: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "surface height",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "temperature",
    Variable.SEA_WATER_SALINITY.key(): "salinity",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "northward velocity",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "eastward velocity",
    Variable.MIXED_LAYER_DEPTH.key(): "mixed layer depth",
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): "northward geostrophic velocity",
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): "eastward geostrophic velocity",
}

DEPTH_BINS_DEFAULT = {
    "surface": (-1, 1),
    "0-5m": (0, 5),
    "5-100m": (5, 100),
    "100-300m": (100, 300),
    "300-600m": (300, 600),
}

DEPTH_BINS_BY_VARIABLE = {
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): {"surface": (-5, 5)},
}

_MSSH_CACHE = None


def _load_mean_ssh() -> xarray.DataArray:
    global _MSSH_CACHE
    if _MSSH_CACHE is None:
        dataset = xarray.open_dataset(MSSH_URL, engine="zarr", chunks="auto")
        _MSSH_CACHE = dataset["mssh"]
    return _MSSH_CACHE


def _standardize_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    """
    Standardize dataset using CF standard names + manual fallback for NEMO names.
    """
    dataset = rename_dataset_with_standard_names(dataset)

    manual_mapping = {
        "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        "so": Variable.SEA_WATER_SALINITY.key(),
        "uo": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    }

    vars_to_rename = {old: new for old, new in manual_mapping.items() if old in dataset and new not in dataset}

    if vars_to_rename:
        dataset = dataset.rename(vars_to_rename)

    coordinate_mapping = {
        "lat": Dimension.LATITUDE.key(),
        "lon": Dimension.LONGITUDE.key(),
        "nav_lat": Dimension.LATITUDE.key(),
        "nav_lon": Dimension.LONGITUDE.key(),
        "deptht": Dimension.DEPTH.key(),
        "olevel": Dimension.DEPTH.key(),
        "lev": Dimension.DEPTH.key(),
    }

    coords_to_rename = {}
    for original, standardized in coordinate_mapping.items():
        if original in dataset.variables or original in dataset.dims:
            if standardized not in dataset:
                coords_to_rename[original] = standardized

    if coords_to_rename:
        dataset = dataset.rename(coords_to_rename)

    return dataset


def _prepare_obs_dataset(reference_dataset: xarray.Dataset, variable_key: str) -> xarray.Dataset:

    variables_to_keep = [
        variable_key,
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]

    if Dimension.DEPTH.key() in reference_dataset:
        variables_to_keep.append(Dimension.DEPTH.key())

    obs_dataset = reference_dataset[variables_to_keep].copy()

    if Dimension.DEPTH.key() in obs_dataset:
        obs_dataset = obs_dataset.set_coords(Dimension.DEPTH.key())

    # SSH Correction
    if variable_key == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        mean_ssh = _load_mean_ssh()
        # Interpolate lazy array
        mean_ssh_at_obs = mean_ssh.interp(
            latitude=obs_dataset[Dimension.LATITUDE.key()],
            longitude=obs_dataset[Dimension.LONGITUDE.key()],
            method="linear",
        )
        # Compute correction values only for obs points
        correction = mean_ssh_at_obs.compute().values
        obs_dataset[variable_key] = obs_dataset[variable_key] + correction + REANALYSIS_MSSH_SHIFT

    obs_time = obs_dataset.time
    first_day = obs_dataset[Dimension.FIRST_DAY_DATETIME.key()]

    # Calculate lead days
    lead_days = (obs_time - first_day).dt.days - 1

    obs_dataset = obs_dataset.assign_coords({Dimension.LEAD_DAY_INDEX.key(): lead_days})

    # Filter invalid data or out-of-bounds lead days
    # .compute() is required here for boolean masking on lazy arrays
    valid_mask = (
        (obs_dataset[Dimension.LEAD_DAY_INDEX.key()] >= 0)
        & (obs_dataset[Dimension.LEAD_DAY_INDEX.key()] < LEAD_DAYS_COUNT)
        & obs_dataset[variable_key].notnull()
    ).compute()

    return obs_dataset.where(valid_mask, drop=True)


def _align_model_to_obs(
    challenger_dataset: xarray.Dataset, obs_dataset: xarray.Dataset, variable_key: str
) -> xarray.DataArray:

    obs_dimension = list(obs_dataset.dims)[0]

    # Helper to strip metadata/coords that cause conflicts (like 'depth')
    def to_indexer(data_array):
        return xarray.DataArray(data_array.values, dims=obs_dimension)

    # 1. Temporal Selection
    temporal_indexers = {
        Dimension.FIRST_DAY_DATETIME.key(): to_indexer(obs_dataset[Dimension.FIRST_DAY_DATETIME.key()]),
        Dimension.LEAD_DAY_INDEX.key(): to_indexer(obs_dataset[Dimension.LEAD_DAY_INDEX.key()]),
    }

    model_subset = challenger_dataset[variable_key].sel(temporal_indexers, method="nearest")

    # 2. Spatial Interpolation
    spatial_indexers = {
        Dimension.LATITUDE.key(): to_indexer(obs_dataset[Dimension.LATITUDE.key()]),
        Dimension.LONGITUDE.key(): to_indexer(obs_dataset[Dimension.LONGITUDE.key()]),
    }

    # Only include depth if both model and obs have it
    model_has_depth = Dimension.DEPTH.key() in model_subset.dims
    obs_has_depth = Dimension.DEPTH.key() in obs_dataset

    if model_has_depth and obs_has_depth:
        spatial_indexers[Dimension.DEPTH.key()] = to_indexer(obs_dataset[Dimension.DEPTH.key()])

    return model_subset.interp(spatial_indexers, method="linear")


def _compute_rmsd_agg(
    squared_difference: xarray.DataArray, obs_dataset: xarray.Dataset, variable_key: str
) -> list[dict]:

    results = []
    depth_bins = DEPTH_BINS_BY_VARIABLE.get(variable_key, DEPTH_BINS_DEFAULT)
    grouped_by_lead = squared_difference.groupby(Dimension.LEAD_DAY_INDEX.key())

    obs_depths = obs_dataset[Dimension.DEPTH.key()] if Dimension.DEPTH.key() in obs_dataset else None

    for lead_day, indices in grouped_by_lead.groups.items():
        diff_subset = squared_difference.isel({squared_difference.dims[0]: indices})

        if obs_depths is not None:
            depth_subset = obs_depths.isel({obs_depths.dims[0]: indices})

        for bin_name, (depth_min, depth_max) in depth_bins.items():
            if obs_depths is not None:
                # Force compute on mask to allow drop=True with Dask
                depth_mask = ((depth_subset >= depth_min) & (depth_subset < depth_max)).compute()
                valid_data = diff_subset.where(depth_mask, drop=True)
            else:
                if bin_name != "surface":
                    continue
                valid_data = diff_subset

            if valid_data.size == 0:
                continue

            # Compute the RMSD value (triggers Dask computation for this scalar)
            rmsd_value = numpy.sqrt(valid_data.mean()).values.item()

            results.append(
                {
                    "variable": variable_key,
                    "depth_bin": bin_name,
                    "lead_day": int(lead_day),
                    "rmsd": rmsd_value,
                    "count": valid_data.size,
                }
            )

    return results


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    use_ssh_correction: bool = True,
) -> pandas.DataFrame:
    """
    Compute Class-4 validation metrics (Point-to-Grid) using vectorized Xarray operations.
    """
    start_time = time.time()

    print("Standardizing datasets...", flush=True)
    challenger_dataset = _standardize_dataset(challenger_dataset)
    reference_dataset = _standardize_dataset(reference_dataset)

    all_results = []

    for variable_enum in variables:
        variable_key = variable_enum.key()

        # Check if variable exists in standardized obs
        if variable_key not in reference_dataset:
            continue

        print(f"Processing {variable_key}...", flush=True)

        # Prepare Obs (Compute Lead Days, SSH correction, etc.)
        obs_dataset = _prepare_obs_dataset(reference_dataset, variable_key)

        if obs_dataset.sizes.get("obs", 0) == 0:
            continue

        # Align Model to Obs (Vectorized selection + interpolation)
        model_at_obs = _align_model_to_obs(challenger_dataset, obs_dataset, variable_key)

        # Compute Squared Difference (Lazy Dask graph)
        # .values triggers computation/streaming
        squared_difference = (model_at_obs.values - obs_dataset[variable_key]) ** 2

        squared_difference = squared_difference.assign_coords(
            {Dimension.LEAD_DAY_INDEX.key(): obs_dataset[Dimension.LEAD_DAY_INDEX.key()]}
        )

        # Aggregate Results
        all_results.extend(_compute_rmsd_agg(squared_difference, obs_dataset, variable_key))

    if not all_results:
        return pandas.DataFrame()

    dataframe = pandas.DataFrame(all_results)

    # Map to display names
    dataframe["variable_display"] = dataframe["variable"].map(VARIABLE_DISPLAY_NAMES)

    # Create Pivot Table
    pivot = dataframe.pivot_table(
        values="rmsd",
        index=["variable_display", "depth_bin"],
        columns="lead_day",
    ).reset_index()

    # Add counts
    counts = dataframe[dataframe["lead_day"] == 0][["variable_display", "depth_bin", "count"]]

    if not counts.empty:
        pivot = pivot.merge(counts, on=["variable_display", "depth_bin"], how="left")
        pivot = pivot.rename(columns={"count": "Number of observations"})

    pivot.columns.name = None
    pivot = pivot.rename(columns=lead_day_labels(1, LEAD_DAYS_COUNT))

    elapsed = time.time() - start_time
    print(f"Validation completed in {elapsed:.1f}s")

    return pivot
