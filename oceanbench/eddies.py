# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes mesoscale-eddy diagnostics for OceanBench demonstrations.
"""

from oceanbench.core.eddies import (
    default_eddy_detection_parameters,
    detect_mesoscale_eddies,
    filter_mesoscale_eddy_detections_by_contours,
    match_mesoscale_eddies,
    mesoscale_eddy_concentration_from_contours,
    mesoscale_eddy_contours_from_detections,
    mesoscale_eddy_concentration_from_detections,
    mesoscale_eddy_summary,
    mesoscale_eddy_summary_from_detections,
    surface_ssh_anomaly_field,
    surface_ssh_field,
)

__all__ = [
    "default_eddy_detection_parameters",
    "surface_ssh_field",
    "surface_ssh_anomaly_field",
    "detect_mesoscale_eddies",
    "filter_mesoscale_eddy_detections_by_contours",
    "match_mesoscale_eddies",
    "mesoscale_eddy_concentration_from_contours",
    "mesoscale_eddy_contours_from_detections",
    "mesoscale_eddy_concentration_from_detections",
    "mesoscale_eddy_summary",
    "mesoscale_eddy_summary_from_detections",
]
