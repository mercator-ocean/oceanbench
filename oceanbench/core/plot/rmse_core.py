from typing import Any

import numpy
import seaborn
from matplotlib import pyplot


def plot_temporal_rmse_for_depth(
    rmse_dataarray: numpy.ndarray[Any, Any], depth: int
):
    seaborn.reset_defaults()
    seaborn.set_context("talk", font_scale=0.7)

    _, ax = pyplot.subplots(2, 3, figsize=(15, 6))

    variables = ["uo", "vo", "so", "thetao", "zos"]
    for index, variable in enumerate(variables):
        i = index // 3
        j = index % 3
        dataset_v = numpy.array(rmse_dataarray.item()[variable])[
            depth if variable != "zos" else 0
        ]

        ax[i, j].plot(dataset_v, label="dataset", linestyle="-")
        ax[i, j].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[i, j].legend()
        ax[i, j].set_title(variable)
        ax[i, j].set_xlabel("Forecast Day")
        ax[i, j].set_ylabel("RMSE [m/s]")
        ax[i, j].set_xticks([0, 2, 4, 6, 8])
        ax[i, j].set_xticklabels(["1 ", "3 ", "5 ", "7 ", "9 "], rotation=0)

    pyplot.tight_layout()
    pyplot.show()


def plot_temporal_rmse_for_average_depth(rmse_dataarray: numpy.ndarray[Any]):
    seaborn.reset_defaults()
    seaborn.set_context("talk", font_scale=0.7)
    _, ax = pyplot.subplots(2, 3, figsize=(15, 6))

    variables = ["uo", "vo", "so", "thetao", "zos"]
    for index, variable in enumerate(variables):
        i = index // 3
        j = index % 3
        dataset_v = numpy.array(rmse_dataarray.item()[variable]).mean(axis=0)

        ax[i, j].plot(dataset_v, label="dataset", linestyle="-")
        ax[i, j].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[i, j].legend()
        ax[i, j].set_title(variable)
        ax[i, j].set_xlabel("Forecast Day")
        ax[i, j].set_ylabel("RMSE [m/s]")
        ax[i, j].set_xticks([0, 2, 4, 6, 8])
        ax[i, j].set_xticklabels(["1 ", "3 ", "5 ", "7 ", "9 "], rotation=0)

    pyplot.tight_layout()
    pyplot.show()


def plot_depth_rmse_average_on_time(
    rmse_dataarray: numpy.ndarray[Any],
    dataset_depth_values: numpy.ndarray,
):
    _, ax = pyplot.subplots(1, 4, figsize=(12, 4), sharey=True)

    variables = ["uo", "vo", "so", "thetao"]
    for index, variable in enumerate(variables):
        dataset_v = numpy.array(rmse_dataarray.item()[variable]).mean(axis=1)
        ax[index].plot(
            dataset_v, dataset_depth_values, label="dataset", linestyle="-"
        )
        ax[index].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[index].set_xlabel(
            "RMSE [m/s]" if variable != "thetao" else "RMSE [°C]"
        )
        ax[index].set_ylabel("Depth [$m$]")
        ax[index].set_title(f"{variable}")
        ax[index].legend()
        ax[index].invert_yaxis()

    pyplot.tight_layout()
    pyplot.show()


def plot_euclidean_distance_core(e_d):
    _, ax = pyplot.subplots(1, 1, figsize=(4, 4))

    ax.plot(e_d, label="dataset", linestyle="-")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.set_title("euclidean distance")
    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("RMSE [km]")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels(["1 ", "3 ", "5 ", "7 ", "9 "], rotation=0)

    pyplot.tight_layout()
    pyplot.show()


def plot_energy_cascade_core(dataset_sc):
    _, ax = pyplot.subplots(1, 1, figsize=(8, 3))

    ax.plot(dataset_sc, label="dataset")
    # ax[1].legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # ax[0].set_title(var)
    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("Small-Scale Energy")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels(["1 ", "3 ", "5 ", "7 ", "9 "], rotation=0)
    ax.set_title("Global")

    pyplot.tight_layout()
    pyplot.show()
