from datetime import datetime, timedelta
from typing import List
from xarray import Dataset, open_dataset
import copernicusmarine
import logging

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _glorys_subset(start_datetime: datetime) -> Dataset:
    return copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
        dataset_version="202311",
        variables=["thetao", "zos", "uo", "vo", "so"],
        start_datetime=start_datetime,
        end_datetime=start_datetime + timedelta(days=10),
    )


def _to_1_4(glorys_dataset: Dataset) -> Dataset:
    initial_datetime = datetime.fromisoformat(str(glorys_dataset["time"][0].values)) - timedelta(days=1)
    initial_datetime_string = initial_datetime.strftime("%Y-%m-%d")
    return open_dataset(
        f"https://minio.dive.edito.eu/project-oceanbench/public/glorys14/{initial_datetime_string}.zarr",
        engine="zarr",
    )


def _glorys_datasets(candidate_dataset: Dataset) -> Dataset:
    start_datetime = datetime.fromisoformat(str(candidate_dataset["time"][0].values))
    return _to_1_4(_glorys_subset(start_datetime))


def glorys_datasets(candidate_datasets: List[Dataset]) -> List[Dataset]:
    return list(map(_glorys_datasets, candidate_datasets))
