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


def minimum_latitude_option(function):
    return click.option(
        "-min-lat",
        "--minimum-latitude",
        type=click.FloatRange(min=-90, max=90),
        required=True,
        help="A float between -90 and 90",
    )(function)


def maximum_latitude_option(function):
    return click.option(
        "-max-lat",
        "--maximum-latitude",
        type=click.FloatRange(min=-90, max=90),
        required=True,
        help="A float between -90 and 90",
    )(function)


def minimum_longitude_option(function):
    return click.option(
        "-min-lon",
        "--minimum-longitude",
        type=click.FloatRange(min=-180, max=180),
        required=True,
        help="A float between -180 and 180",
    )(function)


def maximum_longitude_option(function):
    return click.option(
        "-max-lon",
        "--maximum-longitude",
        type=click.FloatRange(min=-180, max=180),
        required=True,
        help="A float between -180 and 180",
    )(function)


def lead_option(function):
    return click.option("--lead", type=click.INT, required=True, help="TODO: write helper")(function)


def output_path_option(function):
    return click.option(
        "-o",
        "--output-path",
        type=click.Path(path_type=Path),
        default=Path("./output.nc"),
        help="The filepath where to write the output dataset",
    )(function)
