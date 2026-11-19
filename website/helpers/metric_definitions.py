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

METRIC_PATTERNS = {
    f"{metric['key']}_{reference['suffix']}": f"oceanbench.metrics.{metric['function']}_{reference['function_suffix']}"
    for metric in _METRICS
    for reference in _REFERENCES
} | {_OBSERVATIONS_METRIC_KEY: "oceanbench.metrics.rmsd_of_variables_compared_to_observations"}

DEPTH_VARIABLE_METRICS = {
    f"{metric['key']}_{reference['suffix']}" for metric in _METRICS for reference in _REFERENCES if metric["has_depths"]
} | {_OBSERVATIONS_METRIC_KEY}

METRIC_TITLES = {
    f"{metric['key']}_{reference['suffix']}": metric["title"] for metric in _METRICS for reference in _REFERENCES
} | {_OBSERVATIONS_METRIC_KEY: "Forecasted variables"}

SECTIONS = {
    reference["key"]: {
        "depth_metric": next(f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if metric["has_depths"]),
        "flat_metrics": [f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if not metric["has_depths"]],
    }
    for reference in _REFERENCES
} | {
    "observations": {
        "depth_metric": _OBSERVATIONS_METRIC_KEY,
        "flat_metrics": [],
        "depth_groups": [
            {
                "depths": ["0-5m", "5-100m", "100-300m", "300-600m"],
                "variables": ["temperature", "salinity"],
                "variable_sensors": {"temperature": "argo-profiler", "salinity": "argo-profiler"},
            },
            {
                "depths": ["Surface"],
                "variables": ["temperature", "sea level anomaly", "high resolution sea level anomaly (swot)"],
                "variable_sensors": {
                    "temperature": "drifter-buoy",
                    "sea level anomaly": "satellite",
                    "high resolution sea level anomaly (swot)": "satellite",
                },
                "show_depth_label": True,
            },
            {
                "depths": ["15m"],
                "variables": ["zonal current", "meridional current"],
                "variable_sensors": {"zonal current": "drifter-buoy", "meridional current": "drifter-buoy"},
                "show_depth_label": True,
            },
        ],
    }
}
