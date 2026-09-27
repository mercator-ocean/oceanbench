# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import matplotlib.pyplot
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.observation_support import ObservationSupportReport, observation_support

__all__ = ["ObservationSupportReport", "observation_support", "plot_observation_coverage"]


def plot_observation_coverage(
    report: ObservationSupportReport,
    variable: str | Variable = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    depth_bin: str = "surface",
) -> Figure:
    """Plot sampled spatial support, distinguishing unobserved and sparse cells.

    Cell classifications use identified profile groups and the report's
    descriptive minimum_profiles threshold. The plot deliberately has no ocean
    mask, inferred coastal outline, density extrapolation, or ocean fraction.
    Longitude bounds may be unwrapped beyond 180 degrees for dateline regions.
    """
    variable = variable.key() if isinstance(variable, Variable) else variable
    selected = report.spatial_coverage.loc[
        (report.spatial_coverage["variable"] == variable) & (report.spatial_coverage["depth_bin"] == depth_bin)
    ]
    if selected.empty:
        raise ValueError(f"No coverage cells for variable={variable!r}, depth_bin={depth_bin!r}.")
    statuses = ["unobserved", "sparse", "observed", "identity_unknown"]
    colors = ["#e5e7eb", "#f5b65c", "#397c96", "#9c78b8"]
    selected = selected.assign(_status=selected["status"].map(dict(zip(statuses, range(len(statuses))))))
    grid = selected.pivot(index="latitude_min", columns="longitude_min", values="_status").sort_index()
    longitude_edges = [*grid.columns, selected["longitude_max"].max()]
    latitude_edges = [*grid.index, selected["latitude_max"].max()]
    figure, axis = matplotlib.pyplot.subplots(figsize=(11, 6))
    figure.subplots_adjust(bottom=0.27, top=0.85)
    axis.pcolormesh(
        longitude_edges,
        latitude_edges,
        grid.values,
        cmap=ListedColormap(colors),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], len(colors)),
        shading="flat",
    )
    axis.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
    axis.set_title(
        f"{variable} · {depth_bin} · {report.region['display_name']}\n"
        f"Sampled forecast dates: {report.evaluated_dates[0]} to {report.evaluated_dates[-1]}"
    )
    axis.legend(
        handles=[
            Patch(facecolor=colors[0], label="Unobserved"),
            Patch(facecolor=colors[1], label=f"Sparse: fewer than {report.minimum_profiles} identified groups"),
            Patch(facecolor=colors[2], label=f"Observed: ≥{report.minimum_profiles} identified groups"),
            Patch(facecolor=colors[3], label="Measurements present; profile identity unknown"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.025,
        "Geographic cells include land. Sparse is descriptive; no ocean coverage fraction.",
        ha="center",
    )
    return figure
