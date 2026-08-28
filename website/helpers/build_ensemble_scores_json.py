# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Turn the ensemble evaluation aggregates into the JSON the ensemble page bakes in.

The aggregates are produced once by the evaluation campaigns and are not readable from the website
build, so this converter is run by hand and its output is committed next to the page.
"""

import argparse
import json
import math
import os

import pandas as pd

DEFAULT_AGGREGATE_ROOT = "/Users/jseillade/projects/probax-report"
DEFAULT_GRIDDED_GLOENS_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-gloens.parquet"
DEFAULT_GRIDDED_ICP_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-icp.parquet"
DEFAULT_DETERMINISTIC_GLONET_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-det-glonet.parquet"
DEFAULT_DETERMINISTIC_GLO12_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-det-glo12.parquet"
DEFAULT_OBSERVATIONS_GLOENS_PATH = f"{DEFAULT_AGGREGATE_ROOT}/01-observations/data-gloens/aggregate.parquet"
DEFAULT_OBSERVATIONS_ICP_PATH = f"{DEFAULT_AGGREGATE_ROOT}/01-observations/data-icp/aggregate.parquet"

# The ensemble means are also scored through the class 4 route the deterministic systems go through,
# so the error against observations reads on one matchup and one set of depth bins for every system.
# The superob aggregates above stay the source of the probabilistic scores, which need the members.
DEFAULT_CLASS4_GLOENS_MEAN_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-det-gloens-mean.parquet"
DEFAULT_CLASS4_ICP_MEAN_PATH = f"{DEFAULT_AGGREGATE_ROOT}/03-library-year/aggregate-det-icp-mean.parquet"

# The GloEns year on the depth axis was run subsurface only, because its surface fields had already
# been scored by the earlier campaign and were deliberately not scored again. That frozen record is
# read here so the surface band of the table is filled from it rather than left empty.
DEFAULT_GLOENS_SURFACE_PATH = f"{DEFAULT_AGGREGATE_ROOT}/02-gridded-glorys/scores-gloens-surface.csv"

# A campaign wave that scored a stream after the fact writes it next to the aggregate instead of into
# it, so an observation aggregate is read together with whatever sidecar sits beside it.
OBSERVATION_SIDECAR_NAME = "aggregate_currents.parquet"
OBSERVATION_ROW_KEY = ["stream", "region", "depth_band", "lead_day"]

# The gridded fill waves scored the cells the first waves left empty, the two velocity components on
# every level and salinity at the surface, and they follow the same rule as the observation sidecar:
# the fill is written beside the aggregate it completes rather than into it, so the aggregate stays
# the frozen artifact its campaign produced and the fill stays separately auditable.
GRIDDED_FILL_SUFFIX = "-fill.parquet"
GRIDDED_ROW_KEY = ["variable", "depth", "lead_day", "metric"]

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(SCRIPT_DIRECTORY), "data", "ensemble-scores.json")

EVALUATION_YEAR = 2024
FULL_START_COUNT = 52

# Every table of the page offers the same lead days, the ones the deterministic page shows plus the
# lead day 9 the shorter of the two ensembles ends on. A system missing one of them leaves the cell
# empty, and a lead day no system reaches at all is dropped when the table is drawn.
OBSERVATION_LEAD_DAYS = [1, 3, 5, 7, 9, 10]
GRIDDED_LEAD_DAYS = [1, 3, 5, 7, 9, 10]

GLOENS = "gloens"
ICP = "glonet2-ens-icp"
GLONET = "glonet"
GLO12 = "glo12"

DETERMINISTIC_SYSTEMS = [GLONET, GLO12]

SYSTEMS = {
    GLONET: {
        "label": "GLONET (deterministic)",
        "kind": "Deterministic",
        "description": "Single member GLONET forecast, 52 weekly starts.",
    },
    GLO12: {
        "label": "GLO12 (deterministic)",
        "kind": "Deterministic",
        "description": "Single member GLO12 forecast, 52 weekly starts.",
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

SYSTEM_ORDER = [*DETERMINISTIC_SYSTEMS, GLOENS, ICP]

STREAM_LABELS = {
    "drifter_sst": "Drifter SST",
    "profiles_t": "Profile temperature",
    "profiles_s": "Profile salinity",
    "sla": "Sea level anomaly",
    "currents_u": "Eastward current",
    "currents_v": "Northward current",
}

STREAM_ORDER = list(STREAM_LABELS)

# The digits a stored value keeps, which is a storage choice alone: see :func:`_rounded`.
STORED_SIGNIFICANT_DIGITS = 4

# The temperature streams are scored in kelvin and the deterministic class 4 aggregate in degrees
# Celsius: an error is a difference, so both are numerically the same.
STREAM_UNITS = {
    "drifter_sst": "K",
    "profiles_t": "K",
    "profiles_s": "PSU",
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
    "eastward_sea_water_velocity": "Eastward current",
    "northward_sea_water_velocity": "Northward current",
}

GRIDDED_VARIABLE_ORDER = list(GRIDDED_VARIABLE_LABELS)

# The frozen surface record holds both references and both sea level bases. Only the GLORYS rows
# belong next to the depth aggregate, and of the two sea level bases only the datum aligned one is
# comparable: GloEns carries a sea level datum of its own, while the other system and the reference
# share theirs, so the raw basis would show a constant offset instead of a forecast error.
GRIDDED_REFERENCE = "glorys"
GLOENS_SURFACE_DEPTH = "surface"
GLOENS_DATUM_ALIGNED_DEPTH = "surface-datum-aligned"
FROZEN_METRIC_COLUMNS = [
    "crps_biased",
    "crps_fair",
    "ensemble_mean_rmsd",
    "ensemble_spread",
    "member_rmsd",
    "spread_error_ratio",
]
RATIO_METRIC = "spread_error_ratio"
RATIO_UNIT = "1"
# Neither the frozen surface record nor a fill sidecar carries a unit column, so the units of the
# variables they carry are restated here, exactly as the depth aggregate spells them.
GRIDDED_VARIABLE_UNITS = {
    "sea_water_potential_temperature": "°C",
    "sea_surface_height_above_geoid": "m",
    "sea_water_salinity": "PSU",
    "eastward_sea_water_velocity": "m s-1",
    "northward_sea_water_velocity": "m s-1",
}


def _depth_sort_key(depth: str) -> float:
    if depth == "surface":
        return -1.0
    return float(depth.removesuffix("m"))


def _rounded(value):
    """Round a stored value to its significant digits, which keeps the committed JSON short.

    This is a storage choice, not a display choice, and it is the same for every variable: the
    page formats an ensemble cell through the same function as a deterministic cell, so the two
    views of the scores page can never drift apart, and neither of them reads a precision from
    the data. The digits kept here are more than that function shows, because the percent
    differences the page also computes are read from these values rather than from the source.
    """
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if number == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(number)))
    return round(number, STORED_SIGNIFICANT_DIGITS - 1 - magnitude)


def _row(
    system_key: str,
    variable_label: str,
    depth_label: str,
    unit: str,
    values: list,
    reduced_leads: list,
) -> dict:
    return {
        "system": system_key,
        "system_label": SYSTEMS[system_key]["label"],
        "variable": variable_label,
        "depth_band": depth_label,
        "unit": unit,
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
        for depth in sorted(variable_frame["depth"].unique(), key=_depth_sort_key):
            depth_frame = variable_frame[variable_frame["depth"] == depth].set_index("lead_day")
            values = []
            reduced_leads = []
            for lead_day in GRIDDED_LEAD_DAYS:
                if lead_day not in depth_frame.index:
                    values.append(None)
                    continue
                entry = depth_frame.loc[lead_day]
                values.append(_rounded(entry["value"]))
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
                    values,
                    reduced_leads,
                )
            )
    return rows


def gloens_surface_frame(frozen: pd.DataFrame) -> pd.DataFrame:
    """Shape the frozen GloEns surface record like a year mean slice of the depth aggregate."""
    against_reference = frozen[frozen["reference"] == GRIDDED_REFERENCE]
    sea_level = against_reference["variable"] == "sea_surface_height_above_geoid"
    datum_aligned = against_reference["depth"] == GLOENS_DATUM_ALIGNED_DEPTH
    kept = against_reference[(sea_level & datum_aligned) | (~sea_level & ~datum_aligned)].copy()
    kept["depth"] = GLOENS_SURFACE_DEPTH

    long = kept.melt(
        id_vars=["variable", "depth", "lead_day", "start_count"],
        value_vars=FROZEN_METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )
    long["aggregation"] = "year_mean"
    long["unit"] = [
        RATIO_UNIT if metric == RATIO_METRIC else GRIDDED_VARIABLE_UNITS[variable]
        for metric, variable in zip(long["metric"], long["variable"])
    ]
    return long


def with_gloens_surface(depth_frame: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Add the frozen surface band to the subsurface only GloEns aggregate."""
    if GLOENS_SURFACE_DEPTH in set(depth_frame["depth"]):
        raise ValueError("the GloEns depth aggregate already carries a surface band, so the frozen record is stale")
    return pd.concat([depth_frame, gloens_surface_frame(frozen)], ignore_index=True)


def gridded_fill_frame(fill: pd.DataFrame) -> pd.DataFrame:
    """Shape a wide gridded fill record like a year mean slice of the depth aggregate."""
    long = fill.melt(
        id_vars=["variable", "depth", "lead_day", "start_count"],
        value_vars=FROZEN_METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )
    long["aggregation"] = "year_mean"
    long["unit"] = [
        RATIO_UNIT if metric == RATIO_METRIC else GRIDDED_VARIABLE_UNITS[variable]
        for metric, variable in zip(long["metric"], long["variable"])
    ]
    return long


def with_gridded_fill(frame: pd.DataFrame, fill: pd.DataFrame) -> pd.DataFrame:
    """Append the rows of a fill, refusing a fill that repeats rows of the aggregate it completes."""
    shaped = gridded_fill_frame(fill)
    year_mean = frame[frame["aggregation"] == "year_mean"]
    repeated = year_mean.merge(shaped[GRIDDED_ROW_KEY].drop_duplicates(), on=GRIDDED_ROW_KEY)
    if not repeated.empty:
        raise ValueError("the gridded fill repeats rows of the aggregate it completes, so they cannot be concatenated")
    return pd.concat([frame, shaped], ignore_index=True)


def with_gridded_fill_beside(aggregate_path: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Add the fill a later campaign wave may have written beside this aggregate.

    The frame is passed in rather than read here because the GloEns aggregate has its frozen
    surface band added first: that step refuses a frame which already carries a surface band, and
    the fill carries one, so the two have to be applied in this order.
    """
    fill_path = aggregate_path.removesuffix(".parquet") + GRIDDED_FILL_SUFFIX
    if not os.path.exists(fill_path):
        return frame
    return with_gridded_fill(frame, pd.read_parquet(fill_path))


def observation_rows(frame: pd.DataFrame, system_key: str, column: str, is_ratio: bool) -> list[dict]:
    """Read the global rows of one observation space aggregate for one metric column."""
    selected = frame[frame["region"] == "global"]
    rows = []
    for stream in STREAM_ORDER:
        stream_frame = selected[selected["stream"] == stream]
        if stream_frame.empty:
            continue
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
                values.append(_rounded(entry[column]))
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
                    values,
                    reduced_leads,
                )
            )
    return rows


def class4_rows(frame: pd.DataFrame, system_key: str) -> list[dict]:
    """Read one class 4 aggregate, of a deterministic system or of an ensemble mean, with its own depth bins."""
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
            values.append(_rounded(entry["rmsd"]))
            if int(entry["start_count"]) < FULL_START_COUNT:
                reduced_leads.append(lead_day)
        rows.append(
            _row(
                system_key,
                STREAM_LABELS[stream],
                DETERMINISTIC_DEPTH_BIN_LABELS[depth_bin],
                STREAM_UNITS[stream],
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
    deterministic_glonet: pd.DataFrame,
    deterministic_glo12: pd.DataFrame,
    class4_gloens_mean: pd.DataFrame,
    class4_icp_mean: pd.DataFrame,
    observations_gloens: pd.DataFrame,
    observations_icp: pd.DataFrame,
) -> dict:
    observation_rmsd = class4_rows(deterministic_glonet, GLONET)
    observation_rmsd += class4_rows(deterministic_glo12, GLO12)
    observation_rmsd += class4_rows(class4_gloens_mean, GLOENS)
    observation_rmsd += class4_rows(class4_icp_mean, ICP)

    observation_crps = observation_rows(observations_gloens, GLOENS, "crps_fair", is_ratio=False)
    observation_crps += observation_rows(observations_icp, ICP, "crps_fair", is_ratio=False)

    observation_ratio = observation_rows(observations_gloens, GLOENS, "ssr_add", is_ratio=True)
    observation_ratio += observation_rows(observations_icp, ICP, "ssr_add", is_ratio=True)

    blocks = {
        "observations_rmsd": {
            "title": "Root mean square error against observations",
            "note": (
                "Ensemble mean error for the ensembles, single member error for the two deterministic "
                "references, every one of them scored through the same class 4 matchup."
            ),
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


def with_observation_sidecar(frame: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Append the streams of a sidecar aggregate, refusing a sidecar that repeats rows of the main aggregate."""
    repeated = frame.merge(sidecar[OBSERVATION_ROW_KEY].drop_duplicates(), on=OBSERVATION_ROW_KEY)
    if not repeated.empty:
        raise ValueError("the sidecar aggregate repeats rows of the main aggregate, so the two cannot be concatenated")
    return pd.concat([frame, sidecar], ignore_index=True)


def read_observation_aggregate(path: str) -> pd.DataFrame:
    """Read one observation space aggregate, plus the sidecar a later campaign wave may have written next to it."""
    frame = pd.read_parquet(path)
    sidecar_path = os.path.join(os.path.dirname(path), OBSERVATION_SIDECAR_NAME)
    if not os.path.exists(sidecar_path):
        return frame
    return with_observation_sidecar(frame, pd.read_parquet(sidecar_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gridded-gloens", default=DEFAULT_GRIDDED_GLOENS_PATH)
    parser.add_argument("--gridded-icp", default=DEFAULT_GRIDDED_ICP_PATH)
    parser.add_argument("--deterministic-glonet", default=DEFAULT_DETERMINISTIC_GLONET_PATH)
    parser.add_argument("--deterministic-glo12", default=DEFAULT_DETERMINISTIC_GLO12_PATH)
    parser.add_argument("--class4-gloens-mean", default=DEFAULT_CLASS4_GLOENS_MEAN_PATH)
    parser.add_argument("--class4-icp-mean", default=DEFAULT_CLASS4_ICP_MEAN_PATH)
    parser.add_argument("--gloens-surface", default=DEFAULT_GLOENS_SURFACE_PATH)
    parser.add_argument("--observations-gloens", default=DEFAULT_OBSERVATIONS_GLOENS_PATH)
    parser.add_argument("--observations-icp", default=DEFAULT_OBSERVATIONS_ICP_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    scores = build_ensemble_scores(
        with_gridded_fill_beside(
            arguments.gridded_gloens,
            with_gloens_surface(pd.read_parquet(arguments.gridded_gloens), pd.read_csv(arguments.gloens_surface)),
        ),
        with_gridded_fill_beside(arguments.gridded_icp, pd.read_parquet(arguments.gridded_icp)),
        pd.read_parquet(arguments.deterministic_glonet),
        pd.read_parquet(arguments.deterministic_glo12),
        pd.read_parquet(arguments.class4_gloens_mean),
        pd.read_parquet(arguments.class4_icp_mean),
        read_observation_aggregate(arguments.observations_gloens),
        read_observation_aggregate(arguments.observations_icp),
    )

    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    with open(arguments.output, "w") as file:
        json.dump(scores, file, indent=2)
        file.write("\n")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
