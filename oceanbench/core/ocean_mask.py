# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
The OceanBench ocean mask.

The mask says, for each of the six OceanBench standard depths and for the first native level below
600 metres, which cells of the twelfth of a degree grid are ocean for OceanBench. A cell is wet
when it is wet in both official Copernicus Marine static masks, the GLO12 analysis and forecast one
and the GLORYS12 reanalysis one, so the scored population never depends on which reference a metric
happens to use. The two masks differ by a few thousand cells per depth, almost all of them in the
Arctic.

The artefact is built once from the two static datasets and pinned by the checksum of its array
bytes, so a silent change of the upstream static files or of the stored file is an error rather
than a quiet shift of the scores.
"""

import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import copernicusmarine
import numpy
import xarray
from xarray import DataArray, Dataset

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.remote_http import open_remote_zarr, with_remote_http_retries

# The six OceanBench standard depths, on the native twelfth of a degree levels.
OCEAN_MASK_STANDARD_DEPTHS = numpy.array(
    [
        0.494025,
        47.37369,
        92.32607,
        222.47520,
        318.12741,
        541.08893,
    ]
)

# The first native level below 600 metres, the bottom of the deepest Class IV depth bin, so the Class
# IV gate brackets every observation it scores. Only the Class IV gate uses it.
OCEAN_MASK_CLASS4_BOTTOM_DEPTH = 643.56677

OCEAN_MASK_DEPTHS = numpy.append(OCEAN_MASK_STANDARD_DEPTHS, OCEAN_MASK_CLASS4_BOTTOM_DEPTH)

# Indices of the mask depths in the fifty native levels, identical in both static datasets.
OCEAN_MASK_LEVEL_INDEXES = (0, 17, 21, 26, 28, 31, 32)

GLO12_STATIC_DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_static"
GLORYS_STATIC_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_static"
STATIC_MASK_VARIABLE = "mask"

# The two static files store the same grid in single precision, with rounding differences well
# below a hundredth of a cell, so they are compared with a tolerance and GLO12 is taken as canonical.
STATIC_GRID_ALIGNMENT_ATOL = 1e-4

OCEAN_MASK_VARIABLE = "ocean_mask"

# Public copy of the ocean mask artefact.
OCEAN_MASK_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ocean_mask/oceanbench-ocean-mask-v2.zarr"

# SHA256 of the boolean array bytes, C order, depth then latitude then longitude.
OCEAN_MASK_SHA256 = "9aa92afe12a1d35f35b9ce0ad6631d835cbd85f783075beea07d4188e869c52d"


class OceanMaskChecksumError(ValueError):
    pass


def ocean_mask_checksum(mask: DataArray) -> str:
    """Checksum of the mask array bytes, so the pin does not depend on the storage format."""
    return hashlib.sha256(numpy.ascontiguousarray(mask.values, dtype=bool).tobytes()).hexdigest()


def _static_mask(dataset_id: str) -> DataArray:
    static_dataset = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=[STATIC_MASK_VARIABLE],
    )
    return (
        static_dataset[STATIC_MASK_VARIABLE]
        .isel({Dimension.DEPTH.key(): list(OCEAN_MASK_LEVEL_INDEXES)})
        .astype(bool)
        .compute()
    )


def build_ocean_mask() -> DataArray:
    """
    Build the mask from the two official Copernicus Marine static datasets.

    Needs Copernicus Marine credentials. The two static grids are the same twelfth of a degree grid,
    which is checked here rather than assumed.
    """
    glo12_mask = _static_mask(GLO12_STATIC_DATASET_ID)
    glorys_mask = _static_mask(GLORYS_STATIC_DATASET_ID)

    for coordinate_name in (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()):
        glo12_coordinate = glo12_mask[coordinate_name].values
        glorys_coordinate = glorys_mask[coordinate_name].values
        if glo12_coordinate.shape != glorys_coordinate.shape or not numpy.allclose(
            glo12_coordinate,
            glorys_coordinate,
            atol=STATIC_GRID_ALIGNMENT_ATOL,
        ):
            raise ValueError(
                f"The GLO12 and GLORYS12 static masks disagree on the {coordinate_name} coordinate, "
                "so they cannot be combined cell by cell."
            )

    combined = numpy.logical_and(glo12_mask.values, glorys_mask.values)
    return DataArray(
        combined,
        dims=(Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): glo12_mask[Dimension.LATITUDE.key()].values,
            Dimension.LONGITUDE.key(): glo12_mask[Dimension.LONGITUDE.key()].values,
        },
        name=OCEAN_MASK_VARIABLE,
        attrs={
            "long_name": "OceanBench ocean mask",
            "description": (
                "True where both the GLO12 and the GLORYS12 official static masks are wet, "
                "on the six OceanBench standard depths and the first native level below 600 metres."
            ),
            "glo12_static_dataset_id": GLO12_STATIC_DATASET_ID,
            "glorys_static_dataset_id": GLORYS_STATIC_DATASET_ID,
        },
    )


def write_ocean_mask(path: Path) -> str:
    """Build the mask, write it as a zarr store and return its checksum."""
    mask = build_ocean_mask()
    mask.to_dataset(name=OCEAN_MASK_VARIABLE).to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={OCEAN_MASK_VARIABLE: {"chunks": (1,) + mask.shape[1:]}},
    )
    return ocean_mask_checksum(mask)


def _open_ocean_mask_dataset(path: str) -> Dataset:
    if path.startswith("http://") or path.startswith("https://"):
        return with_remote_http_retries("ocean mask open", lambda: open_remote_zarr(path))
    return xarray.open_dataset(path, engine="zarr")


def _verified_ocean_mask(mask: DataArray, path: str) -> DataArray:
    found_checksum = ocean_mask_checksum(mask)
    if found_checksum != OCEAN_MASK_SHA256:
        raise OceanMaskChecksumError(
            f"The ocean mask read from {path} does not match the checksum pinned in "
            f"{__name__}.OCEAN_MASK_SHA256. Expected {OCEAN_MASK_SHA256}, found {found_checksum}. "
            "Rebuild the artefact with `python -m oceanbench.core.ocean_mask` and update the constant "
            "if the change is intended."
        )
    return mask


@lru_cache(maxsize=1)
def _ocean_mask(path: str) -> DataArray:
    mask_dataset = _open_ocean_mask_dataset(path)
    mask = mask_dataset[OCEAN_MASK_VARIABLE].astype(bool).compute()
    return _verified_ocean_mask(mask, path)


def ocean_mask() -> DataArray:
    """
    Load the OceanBench ocean mask, boolean, with depth, latitude and longitude coordinates.

    The artefact is read from OCEAN_MASK_URL and its checksum is verified against OCEAN_MASK_SHA256.
    """
    return _ocean_mask(OCEAN_MASK_URL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the OceanBench ocean mask and print its checksum.")
    parser.add_argument("path", type=Path, help="Path of the zarr store to write.")
    arguments = parser.parse_args()
    print(write_ocean_mask(arguments.path))


if __name__ == "__main__":
    main()
