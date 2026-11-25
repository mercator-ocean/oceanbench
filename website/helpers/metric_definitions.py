# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

_METRICS = [
    {"key": "rmsd_variables", "title": "Forecasted variables", "function": "rmsd_of_variables", "has_depths": True},
    {
        "key": "rmsd_mld",
        "title": "RMSD of Mixed Layer Depth",
        "function": "rmsd_of_mixed_layer_depth",
        "has_depths": False,
    },
    {
        "key": "rmsd_geostrophic",
        "title": "RMSD of Geostrophic Currents",
        "function": "rmsd_of_geostrophic_currents",
        "has_depths": False,
    },
    {
        "key": "lagrangian",
        "title": "Lagrangian Trajectory Deviation",
        "function": "deviation_of_lagrangian_trajectories",
        "has_depths": False,
    },
]

_REFERENCES = [
    {"key": "reanalysis", "suffix": "glorys", "function_suffix": "compared_to_glorys_reanalysis"},
    {"key": "analysis", "suffix": "glo12", "function_suffix": "compared_to_glo12_analysis"},
]

_OBSERVATIONS_METRIC_KEY = "rmsd_variables_observations"
_CONTROLLED_FUNCTION_PREFIX = "oceanbench.metrics.smoothing_controlled_rmsd_of_variables"

# The smoothing-controlled RMSD of each reference, shown in place of the RMSD of variables by a switch on its table,
# with the fitted width and slope read from the fields of its scores file rows: (reference suffix, function suffix).
_CONTROLLED_REFERENCES = [(reference["suffix"], reference["function_suffix"]) for reference in _REFERENCES] + [
    ("observations", "compared_to_observations")
]


def _controlled_metrics(suffix: str) -> dict[str, str]:
    return {
        "value": f"smoothing_controlled_rmsd_variables_{suffix}",
        "width": f"smoothing_width_{suffix}",
        "slope": f"smoothing_slope_{suffix}",
    }


CONTROLLED_METRICS = {suffix: _controlled_metrics(suffix) for suffix, _ in _CONTROLLED_REFERENCES}

# Row fields of the smoothing-controlled RMSD: (key in its controlled metrics, row field, title, unit).
DIAGNOSTIC_FIELDS = [
    ("width", "smoothing_width_km", "Fitted smoothing width", "km"),
    ("slope", "smoothing_slope", "Fitted anomaly slope", "1"),
]

# Each controlled metric key with its diagnostic metric keys by row field.
DIAGNOSTIC_METRICS = {
    metrics["value"]: [(metrics[name], field, unit) for name, field, _, unit in DIAGNOSTIC_FIELDS]
    for metrics in CONTROLLED_METRICS.values()
}

METRIC_PATTERNS = (
    {
        f"{metric['key']}_{reference['suffix']}": f"oceanbench.metrics.{metric['function']}_{reference['function_suffix']}"
        for metric in _METRICS
        for reference in _REFERENCES
    }
    | {
        _OBSERVATIONS_METRIC_KEY: "oceanbench.metrics.rmsd_of_variables_compared_to_observations",
    }
    | {
        CONTROLLED_METRICS[suffix]["value"]: f"{_CONTROLLED_FUNCTION_PREFIX}_{function_suffix}"
        for suffix, function_suffix in _CONTROLLED_REFERENCES
    }
)

DEPTH_VARIABLE_METRICS = (
    {
        f"{metric['key']}_{reference['suffix']}"
        for metric in _METRICS
        for reference in _REFERENCES
        if metric["has_depths"]
    }
    | {_OBSERVATIONS_METRIC_KEY}
    | {metric_key for metrics in CONTROLLED_METRICS.values() for metric_key in metrics.values()}
)

METRIC_TITLES = (
    {f"{metric['key']}_{reference['suffix']}": metric["title"] for metric in _METRICS for reference in _REFERENCES}
    | {_OBSERVATIONS_METRIC_KEY: "Forecasted variables"}
    | {metrics["value"]: "Smoothing-controlled RMSD" for metrics in CONTROLLED_METRICS.values()}
    | {metrics[name]: title for metrics in CONTROLLED_METRICS.values() for name, _, title, _ in DIAGNOSTIC_FIELDS}
)

_OBSERVATION_DEPTH_GROUPS = [
    {
        "depths": ["0-5m", "5-100m", "100-300m", "300-600m"],
        "variables": ["temperature", "salinity"],
        "variable_sensors": {"temperature": "argo-profiler", "salinity": "argo-profiler"},
    },
    {
        "depths": ["Surface"],
        "variables": ["temperature", "sea level anomaly"],
        "variable_sensors": {"temperature": "drifter-buoy", "sea level anomaly": "satellite"},
        "show_depth_label": True,
    },
    {
        "depths": ["15m"],
        "variables": ["zonal current", "meridional current"],
        "variable_sensors": {"zonal current": "drifter-buoy", "meridional current": "drifter-buoy"},
        "show_depth_label": True,
    },
]

SECTIONS = {
    reference["key"]: {
        "depth_metric": next(f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if metric["has_depths"]),
        "flat_metrics": [f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if not metric["has_depths"]],
        "controlled_metrics": CONTROLLED_METRICS[reference["suffix"]],
    }
    for reference in _REFERENCES
} | {
    "observations": {
        "depth_metric": _OBSERVATIONS_METRIC_KEY,
        "flat_metrics": [],
        "depth_groups": _OBSERVATION_DEPTH_GROUPS,
        "controlled_metrics": CONTROLLED_METRICS["observations"],
    }
}
