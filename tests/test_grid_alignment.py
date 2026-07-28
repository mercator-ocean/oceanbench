# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Grid alignment guard for the gridded metrics (issue #305)."""

import numpy
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.grid_alignment import (
    GridAlignmentError,
    align_reference_to_challenger_grid,
)
from oceanbench.core.rmsd import rmsd_per_start_date

LATITUDE_KEY = Dimension.LATITUDE.key()
LONGITUDE_KEY = Dimension.LONGITUDE.key()
DEPTH_KEY = Dimension.DEPTH.key()
LEAD_DAY_KEY = Dimension.LEAD_DAY_INDEX.key()
START_KEY = Dimension.FIRST_DAY_DATETIME.key()

SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()


def _field_dataset(latitudes: numpy.ndarray, longitudes: numpy.ndarray, *, offset: float = 0.0) -> xarray.Dataset:
    """A small, fully populated sea-surface-height field on the given grid."""
    latitude_grid, longitude_grid = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    values = numpy.sin(numpy.deg2rad(longitude_grid)) * numpy.cos(numpy.deg2rad(latitude_grid)) + offset
    return xarray.Dataset(
        {SEA_SURFACE_HEIGHT_KEY: ((LATITUDE_KEY, LONGITUDE_KEY), values)},
        coords={LATITUDE_KEY: latitudes, LONGITUDE_KEY: longitudes},
    )


def _global_grid(spacing: float = 1.0) -> tuple[numpy.ndarray, numpy.ndarray]:
    latitudes = numpy.arange(-80.0, 80.0 + spacing, spacing)
    longitudes = numpy.arange(-180.0, 180.0, spacing)
    return latitudes, longitudes


def test_identical_grids_are_returned_untouched():
    latitudes, longitudes = _global_grid()
    challenger = _field_dataset(latitudes, longitudes)
    reference = _field_dataset(latitudes, longitudes, offset=0.1)

    aligned_reference, alignment = align_reference_to_challenger_grid(challenger, reference)

    assert alignment.snapped is False
    assert alignment.coverage == 1.0
    assert aligned_reference is reference


def test_float32_encoding_difference_is_snapped_not_dropped():
    """The XiHe / GLORYS case: same nominal grid, different float32 longitude encoding.

    CMEMS serves the reanalysis and the analysis with encodings that differ in the last
    float32 digit (179.91667 against 179.91669). Under an inner join only the coordinates
    that happen to round-trip identically survive.
    """
    latitudes, longitudes = _global_grid()
    challenger = _field_dataset(latitudes, longitudes)
    reference = _field_dataset(latitudes, longitudes, offset=0.1)
    encoding_noise = numpy.full(longitudes.shape, 2e-5)
    reference = reference.assign_coords({LONGITUDE_KEY: longitudes + encoding_noise})

    inner_join_cells = (challenger - reference)[SEA_SURFACE_HEIGHT_KEY].size
    assert inner_join_cells == 0, "the raw subtraction should discard the whole field here"

    aligned_reference, alignment = align_reference_to_challenger_grid(challenger, reference)

    assert alignment.snapped is True
    assert alignment.coverage == 1.0
    assert alignment.maximum_offset_degrees == pytest.approx(2e-5)
    assert aligned_reference[SEA_SURFACE_HEIGHT_KEY].shape == challenger[SEA_SURFACE_HEIGHT_KEY].shape
    numpy.testing.assert_array_equal(aligned_reference[LONGITUDE_KEY].values, challenger[LONGITUDE_KEY].values)
    numpy.testing.assert_allclose(
        aligned_reference[SEA_SURFACE_HEIGHT_KEY].values,
        _field_dataset(latitudes, longitudes, offset=0.1)[SEA_SURFACE_HEIGHT_KEY].values,
        atol=1e-6,
    )


def test_reference_missing_a_latitude_row_still_aligns():
    """The LangYa case: the challenger grid omits a row the reference carries."""
    latitudes, longitudes = _global_grid()
    challenger = _field_dataset(latitudes[:-1], longitudes)
    reference = _field_dataset(latitudes, longitudes, offset=0.1)

    aligned_reference, alignment = align_reference_to_challenger_grid(challenger, reference)

    assert alignment.coverage == 1.0
    assert aligned_reference.sizes[LATITUDE_KEY] == challenger.sizes[LATITUDE_KEY]


def test_half_cell_offset_raises_instead_of_scoring_a_subsample():
    latitudes, longitudes = _global_grid()
    challenger = _field_dataset(latitudes, longitudes)
    reference = _field_dataset(latitudes + 0.5, longitudes + 0.5, offset=0.1)

    with pytest.raises(GridAlignmentError, match="do not match"):
        align_reference_to_challenger_grid(challenger, reference)


def test_partially_overlapping_grid_raises():
    """A reference covering only part of the challenger must not be scored silently."""
    latitudes, longitudes = _global_grid()
    challenger = _field_dataset(latitudes, longitudes)
    reference = _field_dataset(latitudes, longitudes[: len(longitudes) // 4], offset=0.1)

    with pytest.raises(GridAlignmentError, match="do not match"):
        align_reference_to_challenger_grid(challenger, reference)


def test_mismatched_longitude_conventions_raise_a_specific_error():
    latitudes, longitudes = _global_grid()
    challenger = _field_dataset(latitudes, longitudes)
    reference = _field_dataset(latitudes, numpy.arange(0.0, 360.0, 1.0), offset=0.1)

    with pytest.raises(GridAlignmentError, match="longitude conventions"):
        align_reference_to_challenger_grid(challenger, reference)


def _scoreable_dataset(latitudes: numpy.ndarray, longitudes: numpy.ndarray, *, offset: float = 0.0) -> xarray.Dataset:
    """A dataset shaped the way the RMSD path expects: (start, lead day, depth, lat, lon)."""
    from oceanbench.core.dataset_utils import DepthLevel

    depths = [depth_level.value for depth_level in DepthLevel]
    latitude_grid, longitude_grid = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    surface = numpy.sin(numpy.deg2rad(longitude_grid)) * numpy.cos(numpy.deg2rad(latitude_grid)) + offset
    values = numpy.broadcast_to(surface, (1, 2, len(depths), len(latitudes), len(longitudes))).copy()
    return xarray.Dataset(
        {SEA_SURFACE_HEIGHT_KEY: ((START_KEY, LEAD_DAY_KEY, DEPTH_KEY, LATITUDE_KEY, LONGITUDE_KEY), values)},
        coords={
            START_KEY: numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            LEAD_DAY_KEY: [0, 1],
            DEPTH_KEY: depths,
            LATITUDE_KEY: latitudes,
            LONGITUDE_KEY: longitudes,
        },
    )


def test_rmsd_is_unchanged_by_a_float32_encoding_difference():
    """The score must not depend on how the reference encodes its coordinates."""
    latitudes, longitudes = _global_grid(spacing=2.0)
    challenger = _scoreable_dataset(latitudes, longitudes)
    reference = _scoreable_dataset(latitudes, longitudes, offset=0.1)
    reference_with_encoding_noise = reference.assign_coords({LONGITUDE_KEY: longitudes + 2e-5})

    variables = [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID]
    clean_frames = rmsd_per_start_date(challenger, reference, variables)
    noisy_frames = rmsd_per_start_date(challenger, reference_with_encoding_noise, variables)

    assert list(clean_frames) == list(noisy_frames)
    for start_date, clean_frame in clean_frames.items():
        numpy.testing.assert_allclose(clean_frame.values, noisy_frames[start_date].values, rtol=1e-9)
        assert numpy.all(numpy.isfinite(clean_frame.values))


def test_rmsd_refuses_a_genuinely_different_grid():
    latitudes, longitudes = _global_grid(spacing=2.0)
    challenger = _scoreable_dataset(latitudes, longitudes)
    reference = _scoreable_dataset(latitudes + 1.0, longitudes + 1.0, offset=0.1)

    with pytest.raises(GridAlignmentError):
        rmsd_per_start_date(challenger, reference, [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID])
