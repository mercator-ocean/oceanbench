# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timedelta
from pathlib import Path

import numpy
import xarray

from oceanbench.core import classIV
from oceanbench.core.interpolate import interpolate_1_degree
from oceanbench.core.references import glo12 as glo12_reference
from oceanbench.core.references import glorys as glorys_reference
from oceanbench.core.references import observations as observations_reference

SAMPLE_FIRST_DAY = datetime.fromisoformat("2024-01-03")
SAMPLE_FIRST_DAY_KEY = SAMPLE_FIRST_DAY.strftime("%Y%m%d")
EDDY_SAMPLE_VARIABLES = ["zos"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def demo_data_dir() -> Path:
    return project_root() / "demo_data" / "glonet_sample"


def metadata_path() -> Path:
    return demo_data_dir() / "metadata.json"


def eddy_demo_data_dir() -> Path:
    return project_root() / "demo_data" / "glonet_eddy_sample"


def eddy_metadata_path() -> Path:
    return eddy_demo_data_dir() / "metadata.json"


def challenger_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> Path:
    return demo_data_dir() / "challenger" / f"{first_day_key}_1degree.zarr"


def glorys_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> Path:
    return demo_data_dir() / "references" / f"glorys_1degree_{first_day_key}.zarr"


def eddy_challenger_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> Path:
    return eddy_demo_data_dir() / "challenger" / f"{first_day_key}_quarter_degree_ssh.zarr"


def eddy_glorys_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> Path:
    return eddy_demo_data_dir() / "references" / f"glorys_quarter_degree_ssh_{first_day_key}.zarr"


def eddy_glo12_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> Path:
    return eddy_demo_data_dir() / "references" / f"glo12_quarter_degree_ssh_{first_day_key}.zarr"


def glo12_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> Path:
    return demo_data_dir() / "references" / f"glo12_1degree_{first_day_key}.zarr"


def observation_path(day_key: str) -> Path:
    return demo_data_dir() / "observations" / f"{day_key}.zarr"


def mean_sea_surface_height_path() -> Path:
    return demo_data_dir() / "support" / "glorys12_mean_sea_surface_height_2024.zarr"


def remote_glonet_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> str:
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/{first_day_key}.zarr"


def remote_glorys_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> str:
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glorys_1degree_2024_V2/{first_day_key}.zarr"


def remote_glorys_quarter_degree_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> str:
    return f"https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/{first_day_key}.zarr"


def remote_glo12_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> str:
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo12_1degree_2024_V2/{first_day_key}.zarr"


def remote_glo12_quarter_degree_path(first_day_key: str = SAMPLE_FIRST_DAY_KEY) -> str:
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo14/{first_day_key}.zarr"


def remote_observation_path(day_key: str) -> str:
    return f"https://minio.dive.edito.eu/project-oceanbench/public/observations2024/{day_key}.zarr"


def remote_mean_sea_surface_height_path() -> str:
    return "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mean_sea_surface_height_2024.zarr"


def _date_key(value) -> str:
    return numpy.datetime_as_string(numpy.datetime64(value), unit="D").replace("-", "")


def observation_day_keys() -> list[str]:
    final_day = SAMPLE_FIRST_DAY + timedelta(days=10)
    day_count = (final_day - SAMPLE_FIRST_DAY).days + 1
    return [(SAMPLE_FIRST_DAY + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(day_count)]


def ensure_local_demo_data_exists() -> None:
    required_paths = [
        challenger_path(),
        glorys_path(),
        glo12_path(),
        mean_sea_surface_height_path(),
        *[observation_path(day_key) for day_key in observation_day_keys()],
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path.relative_to(project_root())) for path in missing_paths)
        raise FileNotFoundError(
            "Local GLONET demo data is missing. "
            "Run `python3 -m oceanbench.demo.fetch_glonet_demo_data` first. "
            f"Missing: {missing_display}"
        )


def ensure_local_eddy_demo_data_exists() -> None:
    required_paths = [
        eddy_challenger_path(),
        eddy_glorys_path(),
        eddy_glo12_path(),
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path.relative_to(project_root())) for path in missing_paths)
        raise FileNotFoundError(
            "Local GLONET eddy demo data is missing. "
            "Run `python3 -m oceanbench.demo.fetch_glonet_eddy_demo_data` first. "
            f"Missing: {missing_display}"
        )


def load_remote_challenger_dataset() -> xarray.Dataset:
    dataset = xarray.open_zarr(remote_glonet_path(), consolidated=True)
    dataset = dataset.rename({"time": "lead_day_index"})
    dataset = dataset.assign_coords({"lead_day_index": range(dataset.sizes["lead_day_index"])})
    dataset = dataset.expand_dims({"first_day_datetime": [numpy.datetime64(SAMPLE_FIRST_DAY)]})
    dataset = interpolate_1_degree(dataset)
    return dataset.chunk(
        {
            "first_day_datetime": 1,
            "lead_day_index": 1,
            "depth": 1,
            "latitude": 170,
            "longitude": 360,
        }
    )


def load_remote_native_challenger_dataset() -> xarray.Dataset:
    dataset = xarray.open_zarr(remote_glonet_path(), consolidated=True)
    dataset = dataset.rename({"time": "lead_day_index"})
    dataset = dataset.assign_coords({"lead_day_index": range(dataset.sizes["lead_day_index"])})
    dataset = dataset.expand_dims({"first_day_datetime": [numpy.datetime64(SAMPLE_FIRST_DAY)]})
    return dataset.chunk(
        {
            "first_day_datetime": 1,
            "lead_day_index": 1,
            "lat": 672,
            "lon": 1440,
        }
    )


def load_remote_reference_dataset(remote_path: str) -> xarray.Dataset:
    dataset = xarray.open_zarr(remote_path, consolidated=True)
    return dataset.chunk(
        {dimension: size for dimension, size in {"time": 1, "depth": 1}.items() if dimension in dataset.dims}
    )


def load_remote_observation_dataset(day_key: str) -> xarray.Dataset:
    return xarray.open_zarr(
        remote_observation_path(day_key),
        consolidated=True,
        decode_cf=False,
    )


def load_remote_mean_sea_surface_height_dataset() -> xarray.Dataset:
    return xarray.open_zarr(remote_mean_sea_surface_height_path(), consolidated=True)


def load_local_challenger_dataset() -> xarray.Dataset:
    ensure_local_demo_data_exists()
    return xarray.open_zarr(challenger_path(), consolidated=True)


def load_local_eddy_challenger_dataset() -> xarray.Dataset:
    ensure_local_eddy_demo_data_exists()
    return xarray.open_zarr(eddy_challenger_path(), consolidated=True)


def load_local_eddy_glorys_dataset() -> xarray.Dataset:
    ensure_local_eddy_demo_data_exists()
    dataset = xarray.open_zarr(eddy_glorys_path(), consolidated=True)
    dataset = dataset.rename({"time": "lead_day_index"})
    dataset = dataset.assign_coords({"lead_day_index": range(dataset.sizes["lead_day_index"])})
    return dataset.expand_dims({"first_day_datetime": [numpy.datetime64(SAMPLE_FIRST_DAY)]})


def load_local_eddy_glo12_dataset() -> xarray.Dataset:
    ensure_local_eddy_demo_data_exists()
    dataset = xarray.open_zarr(eddy_glo12_path(), consolidated=True)
    dataset = dataset.rename({"time": "lead_day_index"})
    dataset = dataset.assign_coords({"lead_day_index": range(dataset.sizes["lead_day_index"])})
    return dataset.expand_dims({"first_day_datetime": [numpy.datetime64(SAMPLE_FIRST_DAY)]})


def configure_local_reference_paths() -> None:
    glorys_reference._glorys_1_degree_path = lambda first_day_datetime: str(glorys_path(_date_key(first_day_datetime)))
    glo12_reference._glo12_1_degree_path = lambda first_day_datetime: str(glo12_path(_date_key(first_day_datetime)))
    observations_reference.observation_path = lambda day_datetime: str(observation_path(_date_key(day_datetime)))
    classIV.MEAN_SEA_SURFACE_HEIGHT_URL = str(mean_sea_surface_height_path())
