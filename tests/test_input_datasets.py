# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_input_modules():
    oceanbench_package = types.ModuleType("oceanbench")
    oceanbench_package.__path__ = [str(PROJECT_ROOT / "oceanbench")]
    sys.modules["oceanbench"] = oceanbench_package

    core_package = types.ModuleType("oceanbench.core")
    core_package.__path__ = [str(PROJECT_ROOT / "oceanbench" / "core")]
    sys.modules["oceanbench.core"] = core_package

    datasets_package = types.ModuleType("oceanbench.datasets")
    datasets_package.__path__ = [str(PROJECT_ROOT / "oceanbench" / "datasets")]
    sys.modules["oceanbench.datasets"] = datasets_package

    _load_module(
        "oceanbench.core.climate_forecast_standard_names",
        "oceanbench/core/climate_forecast_standard_names.py",
    )
    _load_module("oceanbench.core.dataset_utils", "oceanbench/core/dataset_utils.py")
    _load_module("oceanbench.core.datetime_utils", "oceanbench/core/datetime_utils.py")
    input_datasets = _load_module("oceanbench.core.input_datasets", "oceanbench/core/input_datasets.py")
    core_package.input_datasets = input_datasets
    datasets_input = _load_module("oceanbench.datasets.input", "oceanbench/datasets/input.py")
    datasets_package.input = datasets_input

    return input_datasets, datasets_input


def test_glo12_additional_forcing_dataset_path_uses_expected_bucket_layout() -> None:
    input_datasets, _datasets_input = _load_input_modules()
    path = input_datasets._glo12_forcing_dataset_path(datetime(2023, 1, 4))

    assert path == "s3://oceanbench-bucket/dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R20230104.zarr"


def test_ifs_additional_forcing_dataset_path_uses_expected_bucket_layout() -> None:
    input_datasets, _datasets_input = _load_input_modules()
    path = input_datasets._ifs_forcing_zarr_dataset_path(datetime(2023, 1, 3))

    assert path == "s3://oceanbench-bucket/dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R20230103.zarr"


def test_public_input_api_forwards_glo12_additional_forcings(monkeypatch) -> None:
    input_datasets, datasets_input = _load_input_modules()
    sentinel = object()
    monkeypatch.setattr(input_datasets, "glo12_forcings", lambda: sentinel)

    assert datasets_input.glo12_forcings() is sentinel


def test_public_input_api_forwards_ifs_additional_forcings_zarr(monkeypatch) -> None:
    input_datasets, datasets_input = _load_input_modules()
    sentinel = object()
    monkeypatch.setattr(input_datasets, "ifs_forcings_zarr", lambda: sentinel)

    assert datasets_input.ifs_forcings_zarr() is sentinel
