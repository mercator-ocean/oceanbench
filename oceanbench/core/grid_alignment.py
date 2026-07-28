# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Explicit challenger/reference grid alignment for the gridded metrics (issue #305).

xarray subtracts two datasets on the *intersection* of their coordinate labels
(``arithmetic_join="inner"``), silently discarding every cell whose latitude or longitude
is not bit-for-bit equal in both operands. Two datasets on the same nominal grid therefore
score on a fraction of the ocean without any error being raised: a 1/12-degree challenger
carrying the CMEMS ``anfc`` float32 longitude encoding (179.91669) keeps only 21.6% of its
cells against a reference carrying the ``my`` encoding (179.91667), and that surviving
sample is geographically skewed, so the resulting RMSD is wrong by several percent.

``align_reference_to_challenger_grid`` replaces that silent intersection with an explicit,
auditable step: the reference is snapped onto the challenger's own coordinate labels when
every challenger cell finds a reference cell within a tolerance far below the grid spacing,
and the run fails loudly when it does not. Scoring stays on the challenger's native grid
(contracts.md section 1), and no value is ever interpolated: snapping only rewrites
coordinate *labels* that already denote the same cell.
"""

from dataclasses import dataclass

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension

# Coordinate labels closer than this denote the same grid cell. Float32 round-off on a
# global grid reaches ~2e-5 degrees near 180, so the tolerance must exceed it; the finest
# grid OceanBench scores is 1/36 degree (0.0278), so it must stay well below half of that.
# 1e-3 degrees (~110 m at the equator) sits two orders of magnitude inside both bounds.
GRID_SNAP_TOLERANCE_DEGREES = 1e-3

# A reference legitimately missing a row or two (LangYa's grid omits the latitude=90 row)
# still covers 99.95% of the challenger cells. Anything below this is a real grid mismatch.
MINIMUM_GRID_COVERAGE = 0.999


class GridAlignmentError(ValueError):
    """Raised when a reference grid cannot be aligned onto a challenger grid."""


@dataclass(frozen=True)
class GridAlignment:
    """What aligning a reference onto a challenger grid required, and what it cost.

    ``coverage`` is the fraction of challenger cells that found a reference cell within the
    tolerance; it is 1.0 for two datasets describing the same grid. ``snapped`` records
    whether the coordinate labels had to be rewritten at all, and ``maximum_offset_degrees``
    the largest label discrepancy that was absorbed.
    """

    coverage: float
    matched_cell_count: int
    challenger_cell_count: int
    maximum_offset_degrees: float
    snapped: bool

    def describe(self) -> str:
        if not self.snapped:
            return "grids identical"
        return (
            f"reference snapped onto the challenger grid "
            f"(max label offset {self.maximum_offset_degrees:.2e} degrees, "
            f"coverage {self.coverage:.4%})"
        )


def _axis_values(dataset: xarray.Dataset, dimension: Dimension) -> numpy.ndarray:
    key = dimension.key()
    if key not in dataset.coords and key not in dataset.variables:
        raise GridAlignmentError(f"Dataset has no {key!r} coordinate; cannot align grids.")
    return numpy.asarray(dataset[key].values, dtype="float64")


def _nearest_neighbour_offsets(
    challenger_values: numpy.ndarray,
    reference_values: numpy.ndarray,
) -> numpy.ndarray:
    """Distance from each challenger coordinate to the closest reference coordinate."""
    sorted_reference = numpy.sort(reference_values)
    insertion_points = numpy.searchsorted(sorted_reference, challenger_values)
    last_index = len(sorted_reference) - 1
    lower_neighbours = sorted_reference[numpy.clip(insertion_points - 1, 0, last_index)]
    upper_neighbours = sorted_reference[numpy.clip(insertion_points, 0, last_index)]
    return numpy.minimum(
        numpy.abs(challenger_values - lower_neighbours),
        numpy.abs(challenger_values - upper_neighbours),
    )


def _uses_positive_longitudes(longitude_values: numpy.ndarray) -> bool:
    return bool(longitude_values.min() >= 0 and longitude_values.max() > 180)


def _check_longitude_conventions(
    challenger_longitudes: numpy.ndarray,
    reference_longitudes: numpy.ndarray,
) -> None:
    if _uses_positive_longitudes(challenger_longitudes) == _uses_positive_longitudes(reference_longitudes):
        return
    raise GridAlignmentError(
        "Challenger and reference use different longitude conventions "
        f"(challenger spans [{challenger_longitudes.min():.3f}, {challenger_longitudes.max():.3f}], "
        f"reference spans [{reference_longitudes.min():.3f}, {reference_longitudes.max():.3f}]). "
        "Convert one of them to the other convention before scoring: aligning them here would "
        "silently discard a hemisphere."
    )


def _axis_is_identical(challenger_values: numpy.ndarray, reference_values: numpy.ndarray) -> bool:
    return challenger_values.shape == reference_values.shape and bool(
        numpy.array_equal(challenger_values, reference_values)
    )


def align_reference_to_challenger_grid(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    *,
    tolerance_degrees: float = GRID_SNAP_TOLERANCE_DEGREES,
    minimum_coverage: float = MINIMUM_GRID_COVERAGE,
) -> tuple[xarray.Dataset, GridAlignment]:
    """Put ``reference_dataset`` on the challenger's exact latitude/longitude labels.

    Returns the aligned reference and a :class:`GridAlignment` report. Datasets already
    sharing bit-identical coordinates are returned untouched. Otherwise every challenger
    cell must find a reference cell within ``tolerance_degrees``, and the fraction that does
    must reach ``minimum_coverage``; a lower coverage means the two grids genuinely differ
    (a real offset, extent or convention mismatch) and raises :class:`GridAlignmentError`
    rather than scoring a skewed subsample.
    """
    challenger_latitudes = _axis_values(challenger_dataset, Dimension.LATITUDE)
    challenger_longitudes = _axis_values(challenger_dataset, Dimension.LONGITUDE)
    reference_latitudes = _axis_values(reference_dataset, Dimension.LATITUDE)
    reference_longitudes = _axis_values(reference_dataset, Dimension.LONGITUDE)

    if _axis_is_identical(challenger_latitudes, reference_latitudes) and _axis_is_identical(
        challenger_longitudes, reference_longitudes
    ):
        cell_count = challenger_latitudes.size * challenger_longitudes.size
        return reference_dataset, GridAlignment(
            coverage=1.0,
            matched_cell_count=cell_count,
            challenger_cell_count=cell_count,
            maximum_offset_degrees=0.0,
            snapped=False,
        )

    _check_longitude_conventions(challenger_longitudes, reference_longitudes)

    latitude_offsets = _nearest_neighbour_offsets(challenger_latitudes, reference_latitudes)
    longitude_offsets = _nearest_neighbour_offsets(challenger_longitudes, reference_longitudes)
    matched_latitudes = latitude_offsets <= tolerance_degrees
    matched_longitudes = longitude_offsets <= tolerance_degrees

    challenger_cell_count = challenger_latitudes.size * challenger_longitudes.size
    matched_cell_count = int(matched_latitudes.sum()) * int(matched_longitudes.sum())
    coverage = matched_cell_count / challenger_cell_count if challenger_cell_count else 0.0
    maximum_matched_offset = max(
        float(latitude_offsets[matched_latitudes].max()) if matched_latitudes.any() else 0.0,
        float(longitude_offsets[matched_longitudes].max()) if matched_longitudes.any() else 0.0,
    )

    if coverage < minimum_coverage:
        raise GridAlignmentError(
            f"Challenger and reference grids do not match: only {coverage:.2%} of the "
            f"{challenger_cell_count} challenger cells find a reference cell within "
            f"{tolerance_degrees} degrees (latitude {int(matched_latitudes.sum())}/{matched_latitudes.size}, "
            f"longitude {int(matched_longitudes.sum())}/{matched_longitudes.size}). "
            "Scoring on that subsample would report a wrong RMSD over a skewed part of the "
            "ocean, so the run stops here. Regrid the reference onto the challenger grid, or "
            "score against a reference published on the same grid."
        )

    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    aligned_reference = reference_dataset.reindex(
        {
            latitude_key: challenger_dataset[latitude_key],
            longitude_key: challenger_dataset[longitude_key],
        },
        method="nearest",
        tolerance=tolerance_degrees,
    )
    return aligned_reference, GridAlignment(
        coverage=coverage,
        matched_cell_count=matched_cell_count,
        challenger_cell_count=challenger_cell_count,
        maximum_offset_degrees=maximum_matched_offset,
        snapped=True,
    )
