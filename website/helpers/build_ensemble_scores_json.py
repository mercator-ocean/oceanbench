# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Turn the scores of the ensemble helper scripts into the JSON the ensemble page bakes in.

Every score comes from the ``scores.parquet`` a helper script of this branch writes, the class 4
scorer of ``helper_scripts/ensemble_class4`` on the observation axis and the gridded scorers of
``helper_scripts/ensemble_gridded`` against GLORYS. Those outputs are not readable from the website
build, so this converter is run by hand, with one explicit path per input, and its output is
committed next to the page.
"""

import argparse
import json
import math
import os

import pandas as pd

# The GloEns year on the depth axis was scored in three runs: the subsurface levels, a fill with the
# two velocity components on every level and salinity at the surface, and the surface temperature
# and sea level. The three are joined here, and a run repeating the rows of another is refused.
GRIDDED_ROW_KEY = ["variable", "depth", "lead_day", "metric"]

# The helper scores of both axes carry every region and, on the gridded axis, every reference they
# were run against. The page reads the global rows against GLORYS alone.
HELPER_REGION = "global"


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
GLOWENS = "glowens"
GLONET = "glonet"
GLO12 = "glo12"

DETERMINISTIC_SYSTEMS = [GLONET, GLO12]
ENSEMBLE_SYSTEMS = [GLOENS, GLOWENS]

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
    GLOWENS: {
        "label": "GLOW-ens",
        "kind": "Ensemble",
        "description": "GLOW machine learning ensemble, 16 members, Wednesday starts.",
    },
}

SYSTEM_ORDER = [*DETERMINISTIC_SYSTEMS, GLOENS, GLOWENS]

# The rank histograms are read in the member dressing, the primary diagnostic of the class 4 helper,
# where every member rather than the observation carries the observation error draw.
RANK_HISTOGRAM_DRESSING_MODE = "member"
RANK_HISTOGRAM_ROW_KEY = ["challenger", "variable", "depth", "lead_day", "dressing_mode"]
# GloEns has 50 members and GLOW-ens 16, so their histograms have 51 and 17 bins. The GloEns ranks
# are merged three at a time onto the 17 bins of GLOW-ens, so both are drawn with the same bars.
RANK_HISTOGRAM_BIN_COUNT = 17
RANK_HISTOGRAM_SOURCE_BIN_COUNTS = {GLOENS: 51, GLOWENS: 17}
# The histograms are offered pooled over lead days 1 to 9 and for each of those days alone. The
# lead day 10 of GloEns is left out, so both ensembles are compared on the same days.
RANK_HISTOGRAM_LEAD_DAYS = list(range(1, 10))
RANK_HISTOGRAM_LEAD_DAY_BANDS = [
    {"label": "All days", "lead_days": RANK_HISTOGRAM_LEAD_DAYS},
    *({"label": str(lead_day), "lead_days": [lead_day]} for lead_day in RANK_HISTOGRAM_LEAD_DAYS),
]

STREAM_LABELS = {
    "drifter_sst": "Drifter SST",
    "profiles_t": "Profile temperature",
    "profiles_s": "Profile salinity",
    "sla": "Sea level anomaly",
    "currents_u": "Eastward current",
    "currents_v": "Northward current",
}

# The digits a stored value keeps, which is a storage choice alone: see :func:`_rounded`.
STORED_SIGNIFICANT_DIGITS = 4

# The temperature streams are labelled in kelvin whatever unit the class 4 helper writes: an error
# is a difference, so kelvin and degrees Celsius are numerically the same.
STREAM_UNITS = {
    "drifter_sst": "K",
    "profiles_t": "K",
    "profiles_s": "PSU",
    "sla": "m",
    "currents_u": "m s-1",
    "currents_v": "m s-1",
}

# One depth axis for every observation space row, ensemble and deterministic alike. These are the
# bins of DEPTH_BINS_DEFAULT in oceanbench.core.dataset_utils, plus the surface bin the drifting
# buoys and the altimeter are scored in and the 15 m bin the currents are interpolated to. The
# earlier campaign bands, 0-100, 100-500, 500+ and an unbounded all depths, pooled a different
# ocean than the deterministic table and could not be read beside it.
DEPTH_BAND_LABELS = {
    "surface": "Surface",
    "15m": "15 m",
    "0-5m": "0-5 m",
    "5-100m": "5-100 m",
    "100-300m": "100-300 m",
    "300-600m": "300-600 m",
}

DEPTH_BAND_ORDER = list(DEPTH_BAND_LABELS)

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

GRIDDED_VARIABLE_LABELS = {
    "sea_water_potential_temperature": "Temperature",
    "sea_water_salinity": "Salinity",
    "sea_surface_height_above_geoid": "Sea surface height",
    "eastward_sea_water_velocity": "Eastward current",
    "northward_sea_water_velocity": "Northward current",
}

GRIDDED_VARIABLE_ORDER = list(GRIDDED_VARIABLE_LABELS)

# The GloEns surface scores hold both sea level bases. Only the GLORYS rows belong next to the
# depth scores, and of the two sea level bases only the datum aligned one is
# comparable: GloEns carries a sea level datum of its own, while the other system and the reference
# share theirs, so the raw basis would show a constant offset instead of a forecast error.
GRIDDED_REFERENCE = "glorys"
GLOENS_SURFACE_DEPTH = "surface"
GLOENS_DATUM_ALIGNED_DEPTH = "surface-datum-aligned"
# The biased CRPS the ensemble scores also carry is left out: it is the fair estimator without its
# finite-ensemble correction, so it rewards a small ensemble, and no table on the page reads it.
ENSEMBLE_METRIC_COLUMNS = [
    "crps_fair",
    "ensemble_mean_rmsd",
    "ensemble_spread",
    "member_rmsd",
    "spread_error_ratio",
]
# A deterministic system is scored on this axis as an ensemble of one, and an ensemble of one has
# no spread and no fair CRPS, so its scores carry the root mean square difference alone.
DETERMINISTIC_METRIC_COLUMNS = ["ensemble_mean_rmsd"]
RATIO_METRIC = "spread_error_ratio"
RATIO_UNIT = "1"
# The gridded helper scores carry no unit column, so the units of their variables are restated here.
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
    reduced_start_counts: dict,
) -> dict:
    return {
        "system": system_key,
        "system_label": SYSTEMS[system_key]["label"],
        "variable": variable_label,
        "depth_band": depth_label,
        "unit": unit,
        "values": values,
        "reduced_start_counts": reduced_start_counts,
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
            reduced_start_counts = {}
            for lead_day in GRIDDED_LEAD_DAYS:
                if lead_day not in depth_frame.index:
                    values.append(None)
                    continue
                entry = depth_frame.loc[lead_day]
                values.append(_rounded(entry["value"]))
                if int(entry["start_count"]) < FULL_START_COUNT:
                    reduced_start_counts[str(lead_day)] = int(entry["start_count"])
            unit = "" if is_ratio else str(depth_frame["unit"].iloc[0])
            depth_label = "Surface" if depth == "surface" else str(depth).replace("m", " m")
            rows.append(
                _row(
                    system_key,
                    GRIDDED_VARIABLE_LABELS[variable],
                    depth_label,
                    unit,
                    values,
                    reduced_start_counts,
                )
            )
    return rows


def gridded_long_frame(wide: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    """Shape wide gridded scores, one column per metric, into one year mean row per metric."""
    long = wide.melt(
        id_vars=["variable", "depth", "lead_day", "start_count"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    long["aggregation"] = "year_mean"
    long["unit"] = [
        RATIO_UNIT if metric == RATIO_METRIC else GRIDDED_VARIABLE_UNITS[variable]
        for metric, variable in zip(long["metric"], long["variable"])
    ]
    return long


def _against_glorys(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[(scores["reference"] == GRIDDED_REFERENCE) & (scores["region"] == HELPER_REGION)]


def helper_gridded_frame(scores: pd.DataFrame) -> pd.DataFrame:
    """Read the gridded helper scores of an ensemble against GLORYS."""
    return gridded_long_frame(_against_glorys(scores), ENSEMBLE_METRIC_COLUMNS)


def deterministic_gridded_frame(scores: pd.DataFrame) -> pd.DataFrame:
    """Read the gridded helper scores of a one-member system against GLORYS."""
    return gridded_long_frame(_against_glorys(scores), DETERMINISTIC_METRIC_COLUMNS)


def gloens_surface_frame(scores: pd.DataFrame) -> pd.DataFrame:
    """Read the GloEns surface helper scores, keeping the datum aligned sea level alone."""
    against_reference = _against_glorys(scores)
    sea_level = against_reference["variable"] == "sea_surface_height_above_geoid"
    datum_aligned = against_reference["depth"] == GLOENS_DATUM_ALIGNED_DEPTH
    kept = against_reference[(sea_level & datum_aligned) | (~sea_level & ~datum_aligned)].copy()
    kept["depth"] = GLOENS_SURFACE_DEPTH
    return gridded_long_frame(kept, ENSEMBLE_METRIC_COLUMNS)


def with_gridded_scores(frame: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    """Append gridded rows, refusing rows that repeat rows of the frame they complete."""
    repeated = frame.merge(added[GRIDDED_ROW_KEY].drop_duplicates(), on=GRIDDED_ROW_KEY)
    if not repeated.empty:
        raise ValueError("the gridded scores repeat rows of the scores they complete, so they cannot be concatenated")
    return pd.concat([frame, added], ignore_index=True)


def gloens_gridded_frame(depth: pd.DataFrame, depth_fill: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    """Join the GloEns subsurface, fill and surface helper scores into one frame."""
    return with_gridded_scores(
        with_gridded_scores(helper_gridded_frame(depth), helper_gridded_frame(depth_fill)),
        gloens_surface_frame(surface),
    )


def helper_start_counts(scores: pd.DataFrame) -> dict[tuple, int]:
    """The scored starts of every observation group, counted on the per-start rows beside the pooled ones.

    The class 4 helper writes one record per start next to the record pooling them, so the start
    count the page caveats on is read here rather than carried by the pooled row itself.
    """
    per_start = scores[scores["start_date"].notna()]
    counted = per_start.groupby(["variable", "depth", "lead_day"])["start_date"].nunique()
    return {key: int(count) for key, count in counted.items()}


def helper_observation_rows(scores: pd.DataFrame, system_key: str, metric: str, is_ratio: bool) -> list[dict]:
    """Read the pooled global rows of one class 4 helper scores frame for one metric."""
    pooled = scores[(scores["region"] == HELPER_REGION) & scores["start_date"].isna() & (scores["metric"] == metric)]
    start_counts = helper_start_counts(scores)
    rows = []
    for (variable, depth), stream in DETERMINISTIC_STREAMS.items():
        bin_frame = pooled[(pooled["variable"] == variable) & (pooled["depth"] == depth)]
        if bin_frame.empty:
            continue
        bin_frame = bin_frame.set_index("lead_day")
        values = []
        reduced_start_counts = {}
        for lead_day in OBSERVATION_LEAD_DAYS:
            if lead_day not in bin_frame.index:
                values.append(None)
                continue
            values.append(_rounded(bin_frame.loc[lead_day, "value"]))
            start_count = start_counts.get((variable, depth, lead_day))
            if start_count is not None and start_count < FULL_START_COUNT:
                reduced_start_counts[str(lead_day)] = start_count
        rows.append(
            _row(
                system_key,
                STREAM_LABELS[stream],
                DEPTH_BAND_LABELS[depth],
                "" if is_ratio else STREAM_UNITS[stream],
                values,
                reduced_start_counts,
            )
        )
    return rows


def _sorted_observation_rows(rows: list[dict]) -> list[dict]:
    stream_labels = list(STREAM_LABELS.values())
    depth_labels = [DEPTH_BAND_LABELS[band] for band in DEPTH_BAND_ORDER]
    return sorted(
        rows,
        key=lambda row: (
            stream_labels.index(row["variable"]),
            SYSTEM_ORDER.index(row["system"]),
            depth_labels.index(row["depth_band"]),
        ),
    )


def systems_gridded_rows(
    helper_gridded: dict[str, pd.DataFrame],
    system_keys: list[str],
    metric: str,
    is_ratio: bool,
) -> list[dict]:
    """Read one gridded metric for every listed system."""
    return [
        row
        for system_key in system_keys
        for row in gridded_rows(helper_gridded[system_key], system_key, metric, is_ratio)
    ]


def with_rank_histogram_override(frame: pd.DataFrame, override: pd.DataFrame) -> pd.DataFrame:
    """Replace the rows of a rank histogram frame by the rows of a later run carrying the same key."""
    replaced_keys = override[RANK_HISTOGRAM_ROW_KEY].drop_duplicates()
    kept = frame.merge(replaced_keys, on=RANK_HISTOGRAM_ROW_KEY, how="left", indicator=True)
    kept = kept[kept["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([kept, override], ignore_index=True)


def merged_rank_bins(frame: pd.DataFrame, system_key: str) -> pd.DataFrame:
    """Sum the member dressed rank bins of one system onto the common 17 bins."""
    source_bin_count = RANK_HISTOGRAM_SOURCE_BIN_COUNTS[system_key]
    member = frame[frame["dressing_mode"] == RANK_HISTOGRAM_DRESSING_MODE]
    bin_counts = member.groupby(["variable", "depth", "lead_day"])["rank_bin"].agg(["nunique", "min", "max"])
    assert (
        bin_counts["nunique"] == source_bin_count
    ).all(), f"{system_key} histograms must have {source_bin_count} bins"
    assert (bin_counts["min"] == 0).all() and (bin_counts["max"] == source_bin_count - 1).all()
    merged = member.assign(rank_bin=member["rank_bin"] // (source_bin_count // RANK_HISTOGRAM_BIN_COUNT))
    return merged.groupby(["variable", "depth", "lead_day", "rank_bin"], as_index=False)["frequency"].sum()


def _rank_histogram_densities(merged: pd.DataFrame, variable: str, depth: str, lead_days: list[int]) -> list | None:
    """The pooled histogram of some lead days as a density, one for a flat histogram."""
    selected = merged[
        (merged["variable"] == variable) & (merged["depth"] == depth) & merged["lead_day"].isin(lead_days)
    ]
    if selected.empty:
        return None
    frequencies = selected.groupby("rank_bin")["frequency"].sum().reindex(range(RANK_HISTOGRAM_BIN_COUNT), fill_value=0)
    return [_rounded(frequency / frequencies.sum() * RANK_HISTOGRAM_BIN_COUNT) for frequency in frequencies]


def rank_histograms(frames: dict[str, pd.DataFrame]) -> dict:
    """The rank histograms of the ensembles, one panel per variable and depth, pooled over lead day bands."""
    merged = {system_key: merged_rank_bins(frame, system_key) for system_key, frame in frames.items()}
    panels = []
    for (variable, depth), stream in DETERMINISTIC_STREAMS.items():
        densities = {
            system_key: [
                _rank_histogram_densities(merged[system_key], variable, depth, band["lead_days"])
                for band in RANK_HISTOGRAM_LEAD_DAY_BANDS
            ]
            for system_key in ENSEMBLE_SYSTEMS
            if system_key in merged
        }
        if all(band is None for bands in densities.values() for band in bands):
            continue
        panels.append(
            {
                "variable": STREAM_LABELS[stream],
                "depth_band": DEPTH_BAND_LABELS[depth],
                "densities": densities,
            }
        )
    return {
        "bin_count": RANK_HISTOGRAM_BIN_COUNT,
        "lead_day_bands": [band["label"] for band in RANK_HISTOGRAM_LEAD_DAY_BANDS],
        "panels": panels,
    }


def build_ensemble_scores(
    helper_observations: dict[str, pd.DataFrame],
    helper_gridded: dict[str, pd.DataFrame],
    rank_histogram_frames: dict[str, pd.DataFrame] = {},
) -> dict:
    observation_rmsd = [
        row
        for system_key in SYSTEM_ORDER
        for row in helper_observation_rows(
            helper_observations[system_key], system_key, "ensemble_mean_rmsd", is_ratio=False
        )
    ]
    observation_crps = [
        row
        for system_key in ENSEMBLE_SYSTEMS
        for row in helper_observation_rows(helper_observations[system_key], system_key, "crps_fair", is_ratio=False)
    ]
    observation_ratio = [
        row
        for system_key in ENSEMBLE_SYSTEMS
        for row in helper_observation_rows(helper_observations[system_key], system_key, "ssr_add", is_ratio=True)
    ]

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
            "title": "Root mean square difference against GLORYS",
            "note": (
                "Quarter degree GLORYS reanalysis reference. Ensemble mean difference for the "
                "ensembles, single member difference for the two deterministic references. "
                "Systems built from or initialised by the GLORYS family are favoured by a GLORYS "
                "reference; weight the observation tables."
            ),
            "lead_days": GRIDDED_LEAD_DAYS,
            "rows": systems_gridded_rows(
                helper_gridded, [*ENSEMBLE_SYSTEMS, *DETERMINISTIC_SYSTEMS], "ensemble_mean_rmsd", is_ratio=False
            ),
        },
        "gridded_crps": {
            "title": "Fair continuous ranked probability score against GLORYS",
            "note": "Lower is better, in the unit of the variable.",
            "lead_days": GRIDDED_LEAD_DAYS,
            "rows": systems_gridded_rows(helper_gridded, ENSEMBLE_SYSTEMS, "crps_fair", is_ratio=False),
        },
        "gridded_spread_error_ratio": {
            "title": "Spread error ratio against GLORYS",
            "note": (
                "Spread of the ensemble over its mean difference to GLORYS, without observation or "
                "analysis error, so it reads low; agreement with the analysis, not a calibration test."
            ),
            "lead_days": GRIDDED_LEAD_DAYS,
            "rows": systems_gridded_rows(helper_gridded, ENSEMBLE_SYSTEMS, "spread_error_ratio", is_ratio=True),
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
        "rank_histograms": rank_histograms(rank_histogram_frames),
    }


def _helper_destination(prefix: str, system_key: str) -> str:
    return f"{prefix}-{system_key}".replace("-", "_")


def _add_helper_arguments(parser: argparse.ArgumentParser, prefix: str, system_keys: list[str]) -> None:
    for system_key in system_keys:
        parser.add_argument(f"--{prefix}-{system_key}", dest=_helper_destination(prefix, system_key), required=True)


def _helper_frames(arguments: argparse.Namespace, prefix: str, system_keys: list[str], read) -> dict[str, pd.DataFrame]:
    return {system_key: read(getattr(arguments, _helper_destination(prefix, system_key))) for system_key in system_keys}


def _helper_gridded_frames(arguments: argparse.Namespace) -> dict[str, pd.DataFrame]:
    return {
        GLOENS: gloens_gridded_frame(
            pd.read_parquet(arguments.helper_gridded_gloens_depth),
            pd.read_parquet(arguments.helper_gridded_gloens_depth_fill),
            pd.read_parquet(arguments.helper_gridded_gloens_surface),
        ),
        GLOWENS: helper_gridded_frame(pd.read_parquet(arguments.helper_gridded_glowens)),
        **_helper_frames(
            arguments,
            "helper-gridded",
            DETERMINISTIC_SYSTEMS,
            lambda path: deterministic_gridded_frame(pd.read_parquet(path)),
        ),
    }


def _rank_histogram_frames(arguments: argparse.Namespace) -> dict[str, pd.DataFrame]:
    frames = _helper_frames(arguments, "helper-rank-histograms", ENSEMBLE_SYSTEMS, pd.read_parquet)
    if arguments.helper_rank_histograms_gloens_override is None:
        return frames
    override = pd.read_parquet(arguments.helper_rank_histograms_gloens_override)
    return {**frames, GLOENS: with_rank_histogram_override(frames[GLOENS], override)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_helper_arguments(parser, "helper-observations", SYSTEM_ORDER)
    _add_helper_arguments(
        parser,
        "helper-gridded",
        [*DETERMINISTIC_SYSTEMS, "gloens-depth", "gloens-depth-fill", "gloens-surface", GLOWENS],
    )
    _add_helper_arguments(parser, "helper-rank-histograms", ENSEMBLE_SYSTEMS)
    parser.add_argument("--helper-rank-histograms-gloens-override", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    scores = build_ensemble_scores(
        _helper_frames(arguments, "helper-observations", SYSTEM_ORDER, pd.read_parquet),
        _helper_gridded_frames(arguments),
        _rank_histogram_frames(arguments),
    )

    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    with open(arguments.output, "w") as file:
        json.dump(scores, file, indent=2)
        file.write("\n")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
