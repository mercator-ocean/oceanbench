# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Inspect exact xarray grid alignment for OceanBench 1/12 degree comparisons.

Run from a notebook:

    %run helper_scripts/inspect_twelfth_degree_grid_alignment.py --date 20240103

Or import and call:

    from helper_scripts.inspect_twelfth_degree_grid_alignment import run_alignment_report
    summary_df, coordinate_differences_df, component_summary_df = run_alignment_report("20240103")
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy
import pandas
import xarray


TWELFTH_DEGREE_ATOL = 1e-4
LATITUDE_CANDIDATES = ("latitude", "lat")
LONGITUDE_CANDIDATES = ("longitude", "lon")


@dataclass(frozen=True)
class ZarrDatasetSpec:
    name: str
    url_template: str


@dataclass(frozen=True)
class CopernicusDatasetSpec:
    name: str
    dataset_id: str
    variables: tuple[str, ...]


CHALLENGERS: tuple[ZarrDatasetSpec, ...] = (
    ZarrDatasetSpec("GLO12_challenger", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12/{date}.zarr"),
    ZarrDatasetSpec("XIHE_challenger", "https://minio.dive.edito.eu/project-oceanbench/public/XIHE/{date}.zarr"),
    ZarrDatasetSpec("WENHAI_challenger", "https://minio.dive.edito.eu/project-oceanbench/public/WENHAI/{date}.zarr"),
    ZarrDatasetSpec(
        "GLONET_challenger",
        "https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/{date}.zarr",
    ),
)


REFERENCES: tuple[CopernicusDatasetSpec, ...] = (
    CopernicusDatasetSpec("GLORYS_ref", "cmems_mod_glo_phy_my_0.083deg_P1D-m", ("thetao",)),
    CopernicusDatasetSpec("GLO12_ref", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ("thetao",)),
)


GLO12_REFERENCE_COMPONENTS: tuple[CopernicusDatasetSpec, ...] = (
    CopernicusDatasetSpec("GLO12_ref_thetao", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ("thetao",)),
    CopernicusDatasetSpec("GLO12_ref_so", "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ("so",)),
    CopernicusDatasetSpec("GLO12_ref_cur", "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ("uo", "vo")),
    CopernicusDatasetSpec("GLO12_ref_zos", "cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ("zos",)),
)


def _parse_date(date: str) -> pandas.Timestamp:
    if len(date) == 8 and date.isdigit():
        return pandas.Timestamp(datetime.strptime(date, "%Y%m%d"))
    return pandas.Timestamp(date)


def _date_key(date: str) -> str:
    return _parse_date(date).strftime("%Y%m%d")


def _start_end_datetimes(date: str) -> tuple[str, str]:
    start = _parse_date(date)
    end = start + pandas.Timedelta(days=9)
    return start.strftime("%Y-%m-%dT00:00:00"), end.strftime("%Y-%m-%dT00:00:00")


def _open_zarr_dataset(spec: ZarrDatasetSpec, date: str) -> xarray.Dataset:
    return xarray.open_dataset(spec.url_template.format(date=_date_key(date)), engine="zarr", chunks={})


def _zarr_source(spec: ZarrDatasetSpec, date: str) -> str:
    return spec.url_template.format(date=_date_key(date))


def _open_copernicus_dataset(spec: CopernicusDatasetSpec, date: str) -> xarray.Dataset:
    import copernicusmarine

    start_datetime, end_datetime = _start_end_datetimes(date)
    return copernicusmarine.open_dataset(
        dataset_id=spec.dataset_id,
        variables=list(spec.variables),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def _copernicus_source(spec: CopernicusDatasetSpec) -> str:
    return f"{spec.dataset_id} variables={','.join(spec.variables)}"


def _find_coordinate_name(dataset: xarray.Dataset, candidates: Iterable[str], standard_name: str) -> str:
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.variables:
            return candidate
    for name in dataset.variables:
        if getattr(dataset[name], "standard_name", None) == standard_name:
            return name
    raise KeyError(f"Could not find {standard_name!r} coordinate in dataset variables: {list(dataset.variables)}")


def _coordinate_values(dataset: xarray.Dataset, standard_name: str) -> tuple[str, numpy.ndarray]:
    if standard_name == "latitude":
        coordinate_name = _find_coordinate_name(dataset, LATITUDE_CANDIDATES, standard_name)
    elif standard_name == "longitude":
        coordinate_name = _find_coordinate_name(dataset, LONGITUDE_CANDIDATES, standard_name)
    else:
        raise ValueError(f"Unsupported coordinate {standard_name!r}")
    return coordinate_name, dataset[coordinate_name].values


def _spacing(values: numpy.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return abs(float(values[1] - values[0]))


def _coordinate_hash(values: numpy.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def _dataset_grid_summary(name: str, dataset: xarray.Dataset, source: str) -> dict[str, object]:
    latitude_name, latitude_values = _coordinate_values(dataset, "latitude")
    longitude_name, longitude_values = _coordinate_values(dataset, "longitude")
    return {
        "dataset": name,
        "source": source,
        "dims": dict(dataset.sizes),
        "latitude_name": latitude_name,
        "latitude_size": len(latitude_values),
        "latitude_dtype": str(latitude_values.dtype),
        "latitude_first": float(latitude_values[0]),
        "latitude_last": float(latitude_values[-1]),
        "latitude_spacing": _spacing(latitude_values),
        "latitude_sha256": _coordinate_hash(latitude_values),
        "longitude_name": longitude_name,
        "longitude_size": len(longitude_values),
        "longitude_dtype": str(longitude_values.dtype),
        "longitude_first": float(longitude_values[0]),
        "longitude_last": float(longitude_values[-1]),
        "longitude_spacing": _spacing(longitude_values),
        "longitude_sha256": _coordinate_hash(longitude_values),
    }


def _coordinate_alignment(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    coordinate: str,
    tolerance: float,
) -> dict[str, object]:
    challenger_coordinate_name, challenger_values = _coordinate_values(challenger_dataset, coordinate)
    reference_coordinate_name, reference_values = _coordinate_values(reference_dataset, coordinate)

    exact_values = numpy.intersect1d(challenger_values, reference_values)
    common_size = min(len(challenger_values), len(reference_values))
    same_size = len(challenger_values) == len(reference_values)

    positional_difference = challenger_values[:common_size] - reference_values[:common_size]
    positional_mismatch_indexes = numpy.where(positional_difference != 0)[0]
    first_mismatch_index = None
    first_challenger_value = None
    first_reference_value = None
    first_difference = None
    last_mismatch_index = None
    last_challenger_value = None
    last_reference_value = None
    last_difference = None

    if len(positional_mismatch_indexes):
        first_mismatch_index = int(positional_mismatch_indexes[0])
        first_challenger_value = float(challenger_values[first_mismatch_index])
        first_reference_value = float(reference_values[first_mismatch_index])
        first_difference = float(positional_difference[first_mismatch_index])
        last_mismatch_index = int(positional_mismatch_indexes[-1])
        last_challenger_value = float(challenger_values[last_mismatch_index])
        last_reference_value = float(reference_values[last_mismatch_index])
        last_difference = float(positional_difference[last_mismatch_index])

    return {
        "coordinate": coordinate,
        "challenger_coordinate_name": challenger_coordinate_name,
        "reference_coordinate_name": reference_coordinate_name,
        "challenger_size": len(challenger_values),
        "reference_size": len(reference_values),
        "exact_common": len(exact_values),
        "ignored_by_coordinate": len(challenger_values) - len(exact_values),
        "exact_ratio": len(exact_values) / len(challenger_values) if len(challenger_values) else float("nan"),
        "same_size": same_size,
        "array_equal": same_size and numpy.array_equal(challenger_values, reference_values),
        "allclose_tolerance": same_size
        and numpy.allclose(challenger_values, reference_values, rtol=0.0, atol=tolerance),
        "max_abs_positional_difference": (
            float(numpy.max(numpy.abs(positional_difference))) if common_size else float("nan")
        ),
        "positional_mismatch_count": int(len(positional_mismatch_indexes)),
        "first_mismatch_index": first_mismatch_index,
        "first_challenger_value": first_challenger_value,
        "first_reference_value": first_reference_value,
        "first_difference": first_difference,
        "last_mismatch_index": last_mismatch_index,
        "last_challenger_value": last_challenger_value,
        "last_reference_value": last_reference_value,
        "last_difference": last_difference,
    }


def _pair_alignment_summary(
    challenger_name: str,
    challenger_dataset: xarray.Dataset,
    reference_name: str,
    reference_dataset: xarray.Dataset,
    tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    latitude = _coordinate_alignment(challenger_dataset, reference_dataset, "latitude", tolerance)
    longitude = _coordinate_alignment(challenger_dataset, reference_dataset, "longitude", tolerance)

    challenger_cells = latitude["challenger_size"] * longitude["challenger_size"]
    exact_common_cells = latitude["exact_common"] * longitude["exact_common"]
    ignored_cells = challenger_cells - exact_common_cells

    summary = {
        "challenger": challenger_name,
        "reference": reference_name,
        "latitude_exact": f"{latitude['exact_common']}/{latitude['challenger_size']}",
        "longitude_exact": f"{longitude['exact_common']}/{longitude['challenger_size']}",
        "cells_used_by_xarray_inner_join": exact_common_cells,
        "challenger_cells": challenger_cells,
        "cells_ignored_by_xarray_inner_join": ignored_cells,
        "cells_used_ratio": exact_common_cells / challenger_cells if challenger_cells else float("nan"),
        "cells_ignored_ratio": ignored_cells / challenger_cells if challenger_cells else float("nan"),
        "array_equal_latitude": latitude["array_equal"],
        "array_equal_longitude": longitude["array_equal"],
        "allclose_latitude": latitude["allclose_tolerance"],
        "allclose_longitude": longitude["allclose_tolerance"],
        "max_abs_latitude_difference": latitude["max_abs_positional_difference"],
        "max_abs_longitude_difference": longitude["max_abs_positional_difference"],
    }

    details = []
    for alignment in (latitude, longitude):
        details.append({"challenger": challenger_name, "reference": reference_name, **alignment})

    return summary, details


def _component_alignment_summary(
    component_datasets: dict[str, xarray.Dataset],
    tolerance: float,
) -> pandas.DataFrame:
    names = list(component_datasets)
    base_name = names[0]
    base_dataset = component_datasets[base_name]
    rows = []
    for name in names:
        dataset = component_datasets[name]
        for coordinate in ("latitude", "longitude"):
            alignment = _coordinate_alignment(base_dataset, dataset, coordinate, tolerance)
            rows.append(
                {
                    "base_component": base_name,
                    "component": name,
                    "coordinate": coordinate,
                    "exact_common": alignment["exact_common"],
                    "base_size": alignment["challenger_size"],
                    "array_equal": alignment["array_equal"],
                    "allclose_tolerance": alignment["allclose_tolerance"],
                    "max_abs_positional_difference": alignment["max_abs_positional_difference"],
                }
            )
    return pandas.DataFrame(rows)


def _print_table(title: str, dataframe: pandas.DataFrame) -> None:
    print(f"\n{title}")
    if dataframe.empty:
        print("(empty)")
    else:
        print(dataframe.to_string(index=False))


def run_alignment_report(
    date: str = "20240103",
    tolerance: float = TWELFTH_DEGREE_ATOL,
    include_glonet: bool = False,
    check_glo12_components: bool = True,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """Open 1/12 degree datasets lazily and report exact coordinate alignment."""
    challengers = {}
    sources = {}
    for spec in CHALLENGERS:
        if spec.name == "GLONET_challenger" and not include_glonet:
            continue
        challengers[spec.name] = _open_zarr_dataset(spec, date)
        sources[spec.name] = _zarr_source(spec, date)

    references = {}
    for spec in REFERENCES:
        references[spec.name] = _open_copernicus_dataset(spec, date)
        sources[spec.name] = _copernicus_source(spec)

    print(f"Date: {_date_key(date)}")
    print(f"xarray arithmetic_join: {xarray.get_options()['arithmetic_join']}")
    print(f"allclose tolerance: {tolerance:g}")

    grid_summary_df = pandas.DataFrame(
        [_dataset_grid_summary(name, dataset, sources[name]) for name, dataset in {**challengers, **references}.items()]
    )
    _print_table("Dataset grids", grid_summary_df)

    summary_rows = []
    detail_rows = []
    for challenger_name, challenger_dataset in challengers.items():
        for reference_name, reference_dataset in references.items():
            summary, details = _pair_alignment_summary(
                challenger_name,
                challenger_dataset,
                reference_name,
                reference_dataset,
                tolerance,
            )
            summary_rows.append(summary)
            detail_rows.extend(details)

    summary_df = pandas.DataFrame(summary_rows)
    coordinate_differences_df = pandas.DataFrame(detail_rows)

    summary_columns = [
        "challenger",
        "reference",
        "latitude_exact",
        "longitude_exact",
        "cells_used_by_xarray_inner_join",
        "challenger_cells",
        "cells_ignored_by_xarray_inner_join",
        "cells_used_ratio",
        "cells_ignored_ratio",
        "array_equal_latitude",
        "array_equal_longitude",
        "allclose_latitude",
        "allclose_longitude",
        "max_abs_latitude_difference",
        "max_abs_longitude_difference",
    ]
    detail_columns = [
        "challenger",
        "reference",
        "coordinate",
        "challenger_coordinate_name",
        "reference_coordinate_name",
        "challenger_size",
        "reference_size",
        "exact_common",
        "ignored_by_coordinate",
        "exact_ratio",
        "array_equal",
        "allclose_tolerance",
        "max_abs_positional_difference",
        "positional_mismatch_count",
        "first_mismatch_index",
        "first_challenger_value",
        "first_reference_value",
        "first_difference",
        "last_mismatch_index",
        "last_challenger_value",
        "last_reference_value",
        "last_difference",
    ]

    _print_table("Pair summary", summary_df[summary_columns])
    _print_table("Coordinate differences", coordinate_differences_df[detail_columns])

    if check_glo12_components:
        component_datasets = {spec.name: _open_copernicus_dataset(spec, date) for spec in GLO12_REFERENCE_COMPONENTS}
        component_summary_df = _component_alignment_summary(component_datasets, tolerance)
        _print_table("GLO12 reference component coordinate check", component_summary_df)
    else:
        component_summary_df = pandas.DataFrame()

    print("\nInterpretation")
    print(
        "The RMSD subtraction uses xarray label alignment. With arithmetic_join='inner', "
        "cells_used_by_xarray_inner_join is the number of horizontal cells kept before the mean."
    )
    print(
        "If array_equal is false but allclose_tolerance is true, the grids are nominally the same "
        "but float coordinate labels differ enough for xarray to drop cells."
    )

    return summary_df, coordinate_differences_df, component_summary_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20240103", help="Forecast start date, e.g. 20240103 or 2024-01-03.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TWELFTH_DEGREE_ATOL,
        help="Absolute tolerance used for allclose diagnostics, not for exact xarray alignment.",
    )
    parser.add_argument(
        "--include-glonet",
        action="store_true",
        help="Also open GLONET. It is usually not 1/12 degree in this dataset and is included only for diagnostics.",
    )
    parser.add_argument(
        "--skip-glo12-components",
        action="store_true",
        help="Skip the internal coordinate check across GLO12 reference component datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary_df, coordinate_differences_df, component_summary_df = run_alignment_report(
        date=args.date,
        tolerance=args.tolerance,
        include_glonet=args.include_glonet,
        check_glo12_components=not args.skip_glo12_components,
    )
