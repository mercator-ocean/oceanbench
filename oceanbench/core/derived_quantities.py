import xarray

from oceanbench.core import mixed_layer_depth


def add_mixed_layer_depth(
    datasets: list[xarray.Dataset],
) -> list[xarray.Dataset]:
    return list(map(mixed_layer_depth.add_mixed_layer_depth, datasets))
