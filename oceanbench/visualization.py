# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes plotting helpers for OceanBench diagnostics.
"""

from oceanbench.core.visualization import (
    plot_class4_drifter_trajectory_comparison,
    plot_class4_scatter_gallery,
    plot_lagrangian_trajectory_comparison,
    plot_mesoscale_eddy_concentration_gallery,
    plot_mesoscale_eddy_overlay_gallery,
    plot_spatial_rmse_gallery,
    plot_surface_field_comparison_gallery,
    plot_surface_field_gallery,
    plot_zonal_longitude_psd_comparison_gallery,
)

__all__ = [
    "plot_surface_field_gallery",
    "plot_spatial_rmse_gallery",
    "plot_zonal_longitude_psd_comparison_gallery",
    "plot_mesoscale_eddy_overlay_gallery",
    "plot_mesoscale_eddy_concentration_gallery",
    "plot_class4_scatter_gallery",
    "plot_class4_drifter_trajectory_comparison",
    "plot_lagrangian_trajectory_comparison",
    "plot_surface_field_comparison_gallery",
]
