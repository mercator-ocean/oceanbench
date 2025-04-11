from matplotlib import pyplot


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
