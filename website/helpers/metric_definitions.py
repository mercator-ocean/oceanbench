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
_CONTROLLED_METRIC_KEY = "smoothing_controlled_rmsd_variables_observations"
_CONTROLLED_FUNCTION = "oceanbench.metrics.smoothing_controlled_rmsd_of_variables_compared_to_observations"

# Row diagnostics of the smoothing-controlled RMSD, read from the fields of its scores file rows.
# Each is shown as a table shaped like the score: (metric key, row field, title, unit).
DIAGNOSTIC_METRICS = [
    ("smoothing_width_observations", "smoothing_width_km", "Fitted smoothing width", "km"),
    ("smoothing_slope_observations", "smoothing_slope", "Fitted anomaly slope", "1"),
]

METRIC_PATTERNS = {
    f"{metric['key']}_{reference['suffix']}": f"oceanbench.metrics.{metric['function']}_{reference['function_suffix']}"
    for metric in _METRICS
    for reference in _REFERENCES
} | {
    _OBSERVATIONS_METRIC_KEY: "oceanbench.metrics.rmsd_of_variables_compared_to_observations",
    _CONTROLLED_METRIC_KEY: _CONTROLLED_FUNCTION,
}

DEPTH_VARIABLE_METRICS = {
    f"{metric['key']}_{reference['suffix']}" for metric in _METRICS for reference in _REFERENCES if metric["has_depths"]
} | {_OBSERVATIONS_METRIC_KEY, _CONTROLLED_METRIC_KEY}
DEPTH_VARIABLE_METRICS |= {metric_key for metric_key, _, _, _ in DIAGNOSTIC_METRICS}

METRIC_TITLES = (
    {f"{metric['key']}_{reference['suffix']}": metric["title"] for metric in _METRICS for reference in _REFERENCES}
    | {
        _OBSERVATIONS_METRIC_KEY: "Forecasted variables",
        _CONTROLLED_METRIC_KEY: "Smoothing-controlled RMSD",
    }
    | {metric_key: title for metric_key, _, title, _ in DIAGNOSTIC_METRICS}
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
    }
    for reference in _REFERENCES
} | {
    observation_section: {"depth_metric": metric_key, "flat_metrics": [], "depth_groups": _OBSERVATION_DEPTH_GROUPS}
    for observation_section, metric_key in [
        ("observations", _OBSERVATIONS_METRIC_KEY),
        ("observations_controlled", _CONTROLLED_METRIC_KEY),
    ]
    + [(metric_key, metric_key) for metric_key, _, _, _ in DIAGNOSTIC_METRICS]
}
