from xarray import Dataset

from oceanbench.core.dataset_utils import Dimension, get_dimension

QUARTER_DEGREE = 0.25


def is_quarter_degree_dataset(dataset: Dataset) -> bool:
    return _check_dataset_resolution(dataset, QUARTER_DEGREE)


def _check_dataset_resolution(dataset: Dataset, resolution: float) -> bool:
    return _check_dimension_resolution(dataset, resolution, Dimension.LATITUDE) and _check_dimension_resolution(
        dataset, resolution, Dimension.LONGITUDE
    )


def _check_dimension_resolution(dataset: Dataset, resolution: float, dimension: Dimension) -> bool:
    return (
        round(
            float(
                get_dimension(
                    dataset,
                    dimension,
                )
                .diff(dimension.dimension_name_from_dataset(dataset))
                .mean()
            ),
            5,
        )
        == resolution
    )
