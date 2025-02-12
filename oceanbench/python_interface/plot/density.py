from pathlib import Path

from oceanbench.core.plot.density_core import plot_density_core


def plot_density(density_dataset_path: Path, plot_output_path: Path, show_plot: bool):
    return plot_density_core(
        dataset_path=density_dataset_path,
        plot_output_path=plot_output_path,
        show_plot=show_plot,
    )
