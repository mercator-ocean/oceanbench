import click

from oceanbench.command_line_interface.plot.density import plot_density
from oceanbench.command_line_interface.plot.geo import plot_geo
from oceanbench.command_line_interface.plot.mld import plot_mld
from oceanbench.command_line_interface.plot.rmse import (
    plot_pointwise_evaluation,
    plot_pointwise_evaluation_depth_for_average_time,
    plot_pointwise_evaluation_for_average_depth,
)


@click.group()
def plot():
    click.echo("Using plot command...")


plot.add_command(plot_pointwise_evaluation)
plot.add_command(plot_pointwise_evaluation_for_average_depth)
plot.add_command(plot_pointwise_evaluation_depth_for_average_time)
plot.add_command(plot_mld)
plot.add_command(plot_geo)
plot.add_command(plot_density)
