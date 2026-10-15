# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core import ocean_mask as ocean_mask_module
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.ocean_mask import (
    OCEAN_MASK_DEPTHS,
    OCEAN_MASK_STAGE_NAME,
    OCEAN_MASK_VARIABLE,
    OceanMaskChecksumError,
    ocean_mask,
    ocean_mask_checksum,
)
from oceanbench.core.runtime_configuration import RuntimeConfiguration

LATITUDES = numpy.array([0.0, 1.0])
LONGITUDES = numpy.array([10.0, 11.0])


def _synthetic_ocean_mask() -> xarray.DataArray:
    values = numpy.ones((len(OCEAN_MASK_DEPTHS), len(LATITUDES), len(LONGITUDES)), dtype=bool)
    values[-1, 0, 0] = False
    return xarray.DataArray(
        values,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): LATITUDES,
            Dimension.LONGITUDE.key(): LONGITUDES,
        },
        name=OCEAN_MASK_VARIABLE,
    )


@pytest.fixture
def stored_ocean_mask(tmp_path, monkeypatch) -> xarray.DataArray:
    mask = _synthetic_ocean_mask()
    store_path = tmp_path / "ocean-mask.zarr"
    mask.to_dataset(name=OCEAN_MASK_VARIABLE).to_zarr(store_path, mode="w", consolidated=True)
    monkeypatch.setattr(ocean_mask_module, "OCEAN_MASK_URL", str(store_path))
    ocean_mask_module._ocean_mask.cache_clear()
    yield mask
    ocean_mask_module._ocean_mask.cache_clear()


def test_ocean_mask_loads_the_artefact_pointed_at_by_the_mask_url(stored_ocean_mask, monkeypatch) -> None:
    monkeypatch.setattr(ocean_mask_module, "OCEAN_MASK_SHA256", ocean_mask_checksum(stored_ocean_mask))

    loaded_mask = ocean_mask()

    assert loaded_mask.dtype == bool
    xarray.testing.assert_equal(loaded_mask, stored_ocean_mask)


def test_ocean_mask_refuses_an_artefact_that_does_not_match_the_pinned_checksum(
    stored_ocean_mask,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ocean_mask_module, "OCEAN_MASK_SHA256", "0" * 64)

    with pytest.raises(OceanMaskChecksumError, match="does not match the checksum pinned"):
        ocean_mask()


def test_ocean_mask_checksum_changes_when_a_single_cell_changes(stored_ocean_mask) -> None:
    flipped_mask = stored_ocean_mask.copy()
    flipped_mask.values[0, 0, 0] = False

    assert ocean_mask_checksum(flipped_mask) != ocean_mask_checksum(stored_ocean_mask)


def _use_stage_directory(monkeypatch, stage_directory) -> None:
    monkeypatch.setattr(
        runtime_configuration,
        "_runtime_configuration",
        RuntimeConfiguration(staged_components=("all",), stage_directory=str(stage_directory)),
    )


def _forbid_remote_reads(monkeypatch) -> None:
    def fail_remote_read(*_args, **_kwargs):
        raise AssertionError("The ocean mask must not be read remotely when it is staged.")

    monkeypatch.setattr(ocean_mask_module, "OCEAN_MASK_URL", "https://example.invalid/ocean-mask.zarr")
    monkeypatch.setattr(ocean_mask_module, "open_remote_zarr", fail_remote_read)


def test_ocean_mask_is_read_from_the_stage_directory_without_network(
    stored_ocean_mask,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(ocean_mask_module, "OCEAN_MASK_SHA256", ocean_mask_checksum(stored_ocean_mask))
    stage_directory = tmp_path / "stage"
    stored_ocean_mask.to_dataset(name=OCEAN_MASK_VARIABLE).to_zarr(stage_directory / OCEAN_MASK_STAGE_NAME)
    _use_stage_directory(monkeypatch, stage_directory)
    _forbid_remote_reads(monkeypatch)

    loaded_mask = ocean_mask()

    xarray.testing.assert_equal(loaded_mask, stored_ocean_mask)


def test_ocean_mask_is_staged_once_then_reused(stored_ocean_mask, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ocean_mask_module, "OCEAN_MASK_SHA256", ocean_mask_checksum(stored_ocean_mask))
    stage_directory = tmp_path / "stage"
    _use_stage_directory(monkeypatch, stage_directory)

    staged_mask = ocean_mask()

    assert (stage_directory / OCEAN_MASK_STAGE_NAME).is_dir()
    xarray.testing.assert_equal(staged_mask, stored_ocean_mask)

    ocean_mask_module._ocean_mask.cache_clear()
    _forbid_remote_reads(monkeypatch)

    reused_mask = ocean_mask()

    xarray.testing.assert_equal(reused_mask, stored_ocean_mask)
