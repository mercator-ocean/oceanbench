from pathlib import Path
import click


def show_plot_option(function):
    return click.option(
        "-show",
        "--show-plot",
        is_flag=True,
        default=False,
        help="A flag to directly show the plot on screen.",
    )(function)


def latitude_option(function):
    return click.option(
        "-lat",
        "--latitude",
        type=click.FloatRange(min=-90, max=-90),
        help="A float between -90 and 90",
    )(function)


def longitude_option(function):
    return click.option(
        "-lon",
        "--longitude",
        type=click.FloatRange(min=-90, max=-90),
        help="A float between -90 and 90",
    )(function)


def lead_option(function):
    return click.option("--lead", type=click.INT, help="TODO: write helper")(function)


def output_path_option(function):
    return click.option(
        "-o",
        "--output-path",
        type=click.Path(path_type=Path),
        default=Path("./output.nc"),
        help="The filepath where to write the output dataset",
    )(function)
