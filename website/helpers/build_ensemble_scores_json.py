# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Turn the ensemble evaluation aggregates into the JSON the ensemble page bakes in.

The aggregates are produced once by the evaluation campaigns and are not readable from the website
build, so this converter is run by hand and its output is committed next to the page.
"""

import argparse
import json
import os

import pandas as pd

DEFAULT_AGGREGATE_ROOT = "/Users/jseillade/projects/probax-report"
DEFAULT_GRIDDED_GLOENS_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-gloens.parquet"
DEFAULT_GRIDDED_ICP_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-icp.parquet"
DEFAULT_DETERMINISTIC_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-det-glonet2.parquet"
DEFAULT_OBSERVATIONS_GLOENS_PATH = f"{DEFAULT_AGGREGATE_ROOT}/01-observations/data-gloens/aggregate.parquet"
DEFAULT_OBSERVATIONS_ICP_PATH = f"{DEFAULT_AGGREGATE_ROOT}/01-observations/data-icp/aggregate.parquet"

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(SCRIPT_DIRECTORY), "data", "ensemble-scores.json")

EVALUATION_YEAR = 2024
FULL_START_COUNT = 52

OBSERVATION_LEAD_DAYS = [1, 3, 5, 7, 9]
GRIDDED_LEAD_DAYS = [1, 3, 5, 7, 9, 10]

GLOENS = "gloens"
ICP = "glonet2-ens-icp"
DETERMINISTIC = "glonet2"

SYSTEMS = {
    DETERMINISTIC: {
        "label": "GLONET2 (deterministic)",
        "kind": "Deterministic",
        "description": "Single member GLONET2 forecast, 52 weekly starts.",
    },
    GLOENS: {
        "label": "GloEns",
        "kind": "Ensemble",
        "description": "Mercator Ocean physics ensemble, 50 members, Thursday starts.",
    },
    ICP: {
        "label": "GloNet2-ens-icp",
        "kind": "Ensemble",
        "description": "GloNet2 machine learning ensemble, 8 members, Wednesday starts.",
    },
}

SYSTEM_ORDER = [DETERMINISTIC, GLOENS, ICP]

STREAM_LABELS = {
    "drifter_sst": "Drifter SST",
    "profiles_t": "Profile temperature",
    "profiles_s": "Profile salinity",
    "sla": "Sea level anomaly",
    "currents_u": "Eastward current",
    "currents_v": "Northward current",
}

STREAM_ORDER = list(STREAM_LABELS)

STREAM_DECIMALS = {
    "drifter_sst": 3,
    "profiles_t": 3,
    "profiles_s": 3,
    "sla": 4,
    "currents_u": 3,
    "currents_v": 3,
}

# The temperature streams are scored in kelvin and the deterministic class 4 aggregate in degrees
# Celsius: an error is a difference, so both are numerically the same.
STREAM_UNITS = {
    "drifter_sst": "K",
    "profiles_t": "K",
    "profiles_s": "psu",
    "sla": "m",
    "currents_u": "m s-1",
    "currents_v": "m s-1",
}

DEPTH_BAND_LABELS = {
    "all": "All depths",
    "0-100": "0-100 m",
    "100-500": "100-500 m",
    "500+": "500+ m",
}

DEPTH_BAND_ORDER = ["all", "0-100", "100-500", "500+"]

# The streams that carry no depth structure are scored at the depth of their observations.
SINGLE_DEPTH_STREAM_LABELS = {
    "drifter_sst": "Surface",
    "sla": "Surface",
    "currents_u": "15 m",
    "currents_v": "15 m",
}

# The deterministic class 4 aggregate uses its own depth bins, which do not match the ensemble bands.
DETERMINISTIC_STREAMS = {
    ("sea_water_potential_temperature", "surface"): "drifter_sst",
    ("sea_water_potential_temperature", "0-5m"): "profiles_t",
    ("sea_water_potential_temperature", "5-100m"): "profiles_t",
    ("sea_water_potential_temperature", "100-300m"): "profiles_t",
    ("sea_water_potential_temperature", "300-600m"): "profiles_t",
    ("sea_water_salinity", "0-5m"): "profiles_s",
    ("sea_water_salinity", "5-100m"): "profiles_s",
    ("sea_water_salinity", "100-300m"): "profiles_s",
    ("sea_water_salinity", "300-600m"): "profiles_s",
    ("sea_surface_height_above_geoid", "surface"): "sla",
    ("eastward_sea_water_velocity", "15m"): "currents_u",
    ("northward_sea_water_velocity", "15m"): "currents_v",
}

DETERMINISTIC_DEPTH_BIN_LABELS = {
    "surface": "Surface",
    "15m": "15 m",
    "0-5m": "0-5 m",
    "5-100m": "5-100 m",
    "100-300m": "100-300 m",
    "300-600m": "300-600 m",
}

DETERMINISTIC_DEPTH_BIN_ORDER = ["surface", "15m", "0-5m", "5-100m", "100-300m", "300-600m"]

GRIDDED_VARIABLE_LABELS = {
    "sea_water_potential_temperature": "Temperature",
    "sea_water_salinity": "Salinity",
    "sea_surface_height_above_geoid": "Sea surface height",
}

GRIDDED_VARIABLE_ORDER = list(GRIDDED_VARIABLE_LABELS)

GRIDDED_VARIABLE_DECIMALS = {
    "sea_water_potential_temperature": 3,
    "sea_water_salinity": 3,
    "sea_surface_height_above_geoid": 4,
}

SPREAD_ERROR_RATIO_DECIMALS = 2


def _depth_sort_key(depth: str) -> float:
    if depth == "surface":
        return -1.0
    return float(depth.removesuffix("m"))


def _rounded(value, decimals: int):
    if value is None or pd.isna(value):
        return None
    return round(float(value), decimals)


def _row(
    system_key: str,
    variable_label: str,
    depth_label: str,
    unit: str,
    decimals: int,
    values: list,
    reduced_leads: list,
) -> dict:
    return {
        "system": system_key,
        "system_label": SYSTEMS[system_key]["label"],
        "variable": variable_label,
        "depth_band": depth_label,
        "unit": unit,
        "decimals": decimals,
        "values": values,
        "reduced_start_leads": reduced_leads,
    }


def gridded_rows(frame: pd.DataFrame, system_key: str, metric: str, is_ratio: bool) -> list[dict]:
    """Read the year mean rows of one gridded aggregate for one metric."""
    selected = frame[(frame["aggregation"] == "year_mean") & (frame["metric"] == metric)]
    rows = []
    for variable in GRIDDED_VARIABLE_ORDER:
        variable_frame = selected[selected["variable"] == variable]
        if variable_frame.empty:
            continue
        decimals = SPREAD_ERROR_RATIO_DECIMALS if is_ratio else GRIDDED_VARIABLE_DECIMALS[variable]
        for depth in sorted(variable_frame["depth"].unique(), key=_depth_sort_key):
            depth_frame = variable_frame[variable_frame["depth"] == depth].set_index("lead_day")
            values = []
            reduced_leads = []
            for lead_day in GRIDDED_LEAD_DAYS:
                if lead_day not in depth_frame.index:
                    values.append(None)
                    continue
                entry = depth_frame.loc[lead_day]
                values.append(_rounded(entry["value"], decimals))
                if int(entry["start_count"]) < FULL_START_COUNT:
                    reduced_leads.append(lead_day)
            unit = "" if is_ratio else str(depth_frame["unit"].iloc[0])
            depth_label = "Surface" if depth == "surface" else str(depth).replace("m", " m")
            rows.append(
                _row(
                    system_key,
                    GRIDDED_VARIABLE_LABELS[variable],
                    depth_label,
                    unit,
                    decimals,
                    values,
                    reduced_leads,
                )
            )
    return rows


def observation_rows(frame: pd.DataFrame, system_key: str, column: str, is_ratio: bool) -> list[dict]:
    """Read the global rows of one observation space aggregate for one metric column."""
    selected = frame[frame["region"] == "global"]
    rows = []
    for stream in STREAM_ORDER:
        stream_frame = selected[selected["stream"] == stream]
        if stream_frame.empty:
            continue
        decimals = SPREAD_ERROR_RATIO_DECIMALS if is_ratio else STREAM_DECIMALS[stream]
        available_bands = [band for band in DEPTH_BAND_ORDER if band in set(stream_frame["depth_band"])]
        for band in available_bands:
            band_frame = stream_frame[stream_frame["depth_band"] == band].set_index("lead_day")
            values = []
            reduced_leads = []
            for lead_day in OBSERVATION_LEAD_DAYS:
                if lead_day not in band_frame.index:
                    values.append(None)
                    continue
                entry = band_frame.loc[lead_day]
                values.append(_rounded(entry[column], decimals))
                if int(entry["n_inits"]) < FULL_START_COUNT:
                    reduced_leads.append(lead_day)
            unit = "" if is_ratio else STREAM_UNITS[stream]
            depth_label = SINGLE_DEPTH_STREAM_LABELS.get(stream, DEPTH_BAND_LABELS[band])
            rows.append(
                _row(
                    system_key,
                    STREAM_LABELS[stream],
                    depth_label,
                    unit,
                    decimals,
                    values,
                    reduced_leads,
                )
            )
    return rows


def deterministic_rows(frame: pd.DataFrame) -> list[dict]:
    """Read the deterministic class 4 aggregate, which keeps its own depth bins."""
    rows = []
    for (variable, depth_bin), stream in DETERMINISTIC_STREAMS.items():
        bin_frame = frame[(frame["variable"] == variable) & (frame["depth_bin"] == depth_bin)]
        if bin_frame.empty:
            continue
        bin_frame = bin_frame.set_index("lead_day_number")
        values = []
        reduced_leads = []
        for lead_day in OBSERVATION_LEAD_DAYS:
            if lead_day not in bin_frame.index:
                values.append(None)
                continue
            entry = bin_frame.loc[lead_day]
            values.append(_rounded(entry["rmsd"], STREAM_DECIMALS[stream]))
            if int(entry["start_count"]) < FULL_START_COUNT:
                reduced_leads.append(lead_day)
        rows.append(
            _row(
                DETERMINISTIC,
                STREAM_LABELS[stream],
                DETERMINISTIC_DEPTH_BIN_LABELS[depth_bin],
                STREAM_UNITS[stream],
                STREAM_DECIMALS[stream],
                values,
                reduced_leads,
            )
        )
    return rows


def _sorted_observation_rows(rows: list[dict]) -> list[dict]:
    stream_labels = list(STREAM_LABELS.values())
    depth_labels = list(DEPTH_BAND_LABELS.values()) + [
        DETERMINISTIC_DEPTH_BIN_LABELS[depth_bin] for depth_bin in DETERMINISTIC_DEPTH_BIN_ORDER
    ]
    return sorted(
        rows,
        key=lambda row: (
            stream_labels.index(row["variable"]),
            SYSTEM_ORDER.index(row["system"]),
            depth_labels.index(row["depth_band"]),
        ),
    )


def build_ensemble_scores(
    gridded_gloens: pd.DataFrame,
    gridded_icp: pd.DataFrame,
    deterministic: pd.DataFrame,
    observations_gloens: pd.DataFrame,
    observations_icp: pd.DataFrame,
) -> dict:
    observation_rmsd = deterministic_rows(deterministic)
    observation_rmsd += observation_rows(observations_gloens, GLOENS, "rmsd_ensemble_mean", is_ratio=False)
    observation_rmsd += observation_rows(observations_icp, ICP, "rmsd_ensemble_mean", is_ratio=False)

    observation_crps = observation_rows(observations_gloens, GLOENS, "crps_fair", is_ratio=False)
    observation_crps += observation_rows(observations_icp, ICP, "crps_fair", is_ratio=False)

    observation_ratio = observation_rows(observations_gloens, GLOENS, "ssr_add", is_ratio=True)
    observation_ratio += observation_rows(observations_icp, ICP, "ssr_add", is_ratio=True)

    blocks = {
        "observations_rmsd": {
            "title": "Root mean square error against observations",
            "note": "Ensemble mean error for the ensembles, single member error for GLONET2.",
            "lead_days": OBSERVATION_LEAD_DAYS,
            "rows": _sorted_observation_rows(observation_rmsd),
        },
        "gridded_rmsd": {
            "title": "Ensemble mean root mean square difference against GLORYS",
            "note": "Quarter degree GLORYS reanalysis reference.",
            "lead_days": GRIDDED_LEAD_DAYS,
            "rows": gridded_rows(gridded_gloens, GLOENS, "ensemble_mean_rmsd", is_ratio=False)
            + gridded_rows(gridded_icp, ICP, "ensemble_mean_rmsd", is_ratio=False),
        },
        "gridded_crps": {
            "title": "Fair continuous ranked probability score against GLORYS",
            "note": "Lower is better, in the unit of the variable.",
            "lead_days": GRIDDED_LEAD_DAYS,
            "rows": gridded_rows(gridded_gloens, GLOENS, "crps_fair", is_ratio=False)
            + gridded_rows(gridded_icp, ICP, "crps_fair", is_ratio=False),
        },
        "gridded_spread_error_ratio": {
            "title": "Spread error ratio against GLORYS",
            "note": "One is a reliable ensemble, below one is under dispersive.",
            "lead_days": GRIDDED_LEAD_DAYS,
            "rows": gridded_rows(gridded_gloens, GLOENS, "spread_error_ratio", is_ratio=True)
            + gridded_rows(gridded_icp, ICP, "spread_error_ratio", is_ratio=True),
        },
        "observations_crps": {
            "title": "Fair continuous ranked probability score against observations",
            "note": "Lower is better, in the unit of the variable.",
            "lead_days": OBSERVATION_LEAD_DAYS,
            "rows": _sorted_observation_rows(observation_crps),
        },
        "observations_spread_error_ratio": {
            "title": "Additive spread error ratio against observations",
            "note": "The observation error variance is added to the ensemble variance before the ratio.",
            "lead_days": OBSERVATION_LEAD_DAYS,
            "rows": _sorted_observation_rows(observation_ratio),
        },
    }

    return {
        "year": EVALUATION_YEAR,
        "full_start_count": FULL_START_COUNT,
        "systems": SYSTEMS,
        "system_order": SYSTEM_ORDER,
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gridded-gloens", default=DEFAULT_GRIDDED_GLOENS_PATH)
    parser.add_argument("--gridded-icp", default=DEFAULT_GRIDDED_ICP_PATH)
    parser.add_argument("--deterministic", default=DEFAULT_DETERMINISTIC_PATH)
    parser.add_argument("--observations-gloens", default=DEFAULT_OBSERVATIONS_GLOENS_PATH)
    parser.add_argument("--observations-icp", default=DEFAULT_OBSERVATIONS_ICP_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    scores = build_ensemble_scores(
        pd.read_parquet(arguments.gridded_gloens),
        pd.read_parquet(arguments.gridded_icp),
        pd.read_parquet(arguments.deterministic),
        pd.read_parquet(arguments.observations_gloens),
        pd.read_parquet(arguments.observations_icp),
    )

    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    with open(arguments.output, "w") as file:
        json.dump(scores, file, indent=2)
        file.write("\n")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
