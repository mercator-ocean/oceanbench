# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas as pd
import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4, perform_matchup
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.rmsd import rmsd
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import obs_insitu_dataset

from oceanbench.core.lagrangian_trajectory import (
    Zone,
    deviation_of_lagrangian_trajectories,
)


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=glorys_reanalysis_dataset(challenger_dataset),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    # Charger les observations
    reference_dataset = obs_insitu_dataset(challenger_dataset)

    # Mapping: Variable enum -> (nom challenger, nom observations)
    variable_mapping = {
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: ("zos", "SLEV"),
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE: ("thetao", "TEMP"),
        Variable.SEA_WATER_SALINITY: ("so", "PSAL"),
        Variable.EASTWARD_SEA_WATER_VELOCITY: ("uo", "EWCT"),
        Variable.NORTHWARD_SEA_WATER_VELOCITY: ("vo", "NSCT"),
    }

    all_results = []

    for var, (challenger_var_name, obs_var_name) in variable_mapping.items():

        # Vérifier que la variable existe
        if obs_var_name not in reference_dataset:
            print(f"⚠️ Variable {obs_var_name} not in observations, skipping")
            continue

        # Extraire directement les valeurs
        obs_data = reference_dataset[obs_var_name].values
        time_data = reference_dataset["time"].values
        lat_data = reference_dataset["latitude"].values
        lon_data = reference_dataset["longitude"].values
        depth_data = reference_dataset["depth"].values
        first_day_data = reference_dataset["first_day_datetime"].values

        # Créer DataFrame
        obs_df = pd.DataFrame(
            {
                challenger_var_name: obs_data,
                "time": time_data,
                "lat": lat_data,
                "lon": lon_data,
                "depth": depth_data,
                "first_day_datetime": first_day_data,
            }
        )

        # Supprimer les NaN
        obs_df = obs_df.dropna(subset=[challenger_var_name])

        print(f"✓ {obs_var_name} -> {challenger_var_name}: {len(obs_df)} valid observations")

        if obs_df.empty:
            continue

        # Faire le matchup
        matchup_df = perform_matchup(challenger_dataset, obs_df, challenger_var_name)

        if matchup_df.empty:
            print(f"⚠️ No matchups for {challenger_var_name}")
            continue

        # Calculer RMSD par depth bin
        rmsd_result = rmsd_class4(matchup_df, challenger_var_name)
        all_results.append(rmsd_result)

    # Combiner tous les résultats
    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Formatter le résultat en tableau pivote
    # Pour l'instant on garde juste lead_day=1
    table = combined[combined["lead_day"] == 1].copy()

    # Créer le tableau formaté
    pivot = table.pivot_table(values="rmsd", index=["variable", "depth_bin"], aggfunc="first").reset_index()

    # Ajouter aussi le count
    pivot_count = table.pivot_table(values="count", index=["variable", "depth_bin"], aggfunc="first").reset_index()

    pivot["count"] = pivot_count["count"]

    # Renommer les variables pour affichage
    var_display_names = {
        "zos": "SSH",
        "thetao": "Temperature",
        "so": "Salinity",
        "uo": "Zonal current",
        "vo": "Meridional current",
    }

    pivot["variable"] = pivot["variable"].map(var_display_names)

    # Définir l'ordre correct des variables
    var_order = ["Temperature", "Salinity", "SSH", "Zonal current", "Meridional current"]

    # Définir l'ordre correct des profondeurs
    depth_order = ["surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]

    # Créer des colonnes de tri
    pivot["var_sort"] = pivot["variable"].map({var: i for i, var in enumerate(var_order)})
    pivot["depth_sort"] = pivot["depth_bin"].map({depth: i for i, depth in enumerate(depth_order)})

    # Trier par variable puis par profondeur
    pivot = pivot.sort_values(["var_sort", "depth_sort"])

    # Supprimer les colonnes de tri
    pivot = pivot.drop(["var_sort", "depth_sort"], axis=1)

    # Formatter les valeurs RMSD
    pivot["RMSE"] = pivot["rmsd"].apply(lambda x: f"{x:.3f}")
    pivot["Count"] = pivot["count"].astype(int)

    # Résultat final
    result = pivot[["variable", "depth_bin", "RMSE", "Count"]]
    result.columns = ["Variable", "Depth Range", "RMSE (lead=1 day)", "N observations"]

    print("\n" + "=" * 70)
    print("RMSE BY VARIABLE AND DEPTH (Lead day = 1)")
    print("=" * 70)
    print(result.to_string(index=False))
    print("=" * 70)

    return result


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glorys_reanalysis_dataset(challenger_dataset)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glorys_reanalysis_dataset(challenger_dataset)),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=glorys_reanalysis_dataset(challenger_dataset),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=glo12_analysis_dataset(challenger_dataset),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


"""from oceanbench.core.climate_forecast_standard_names import (
    VARIABLE_TO_OBSERVATION_MAPPING,
)"""

# from oceanbench.core.metrics import rmsd_class4, perform_matchup
# from oceanbench.core.metrics import VARIABLE_LABELS
"""
def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    #Compute RMSD of challenger dataset against observations using Class 4 methodology.
    obs_dict = obs_insitu_dataset(challenger_dataset)
    rmsd_results = {}

    for standard_var, (obs_source, obs_column) in VARIABLE_TO_OBSERVATION_MAPPING.items():
        var_key = standard_var.value

        if var_key not in challenger_dataset:
            continue

        obs_source_key = obs_source.value

        if obs_source_key not in obs_dict:
            continue

        obs_df = obs_dict[obs_source_key]

        if obs_column not in obs_df.columns:
            continue

        try:
            matchup_df = perform_matchup(challenger=challenger_dataset, obs_df=obs_df, var_name=var_key)

            if matchup_df.empty:
                continue

            rmsd_df = rmsd_class4(matchup_df=matchup_df, var_name=obs_column)

            variable_label = VARIABLE_LABELS.get(var_key, var_key)
            rmsd_results[variable_label] = rmsd_df["rmsd"].values

        except Exception:
            continue

    if not rmsd_results:
        return pandas.DataFrame()

    return pandas.DataFrame(rmsd_results, index=lead_day_labels(1, LEAD_DAYS_COUNT)).T
"""


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glo12_analysis_dataset(challenger_dataset)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glo12_analysis_dataset(challenger_dataset)),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=glo12_analysis_dataset(challenger_dataset),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )
