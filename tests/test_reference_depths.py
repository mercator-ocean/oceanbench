# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.dataset_source import DatasetSource, get_dataset_source, with_dataset_source
from oceanbench.core.lagrangian_support import _lagrangian_domain_stage_variant, _lagrangian_stage_directory
from oceanbench.core.reference_depths import (
    REFERENCE_DEPTH_GRID_HASH_ATTRIBUTE,
    REFERENCE_DEPTH_GRID_ROUNDING_ATTRIBUTE,
    REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS,
    REFERENCE_TARGET_DEPTHS_ATTRIBUTE,
    reference_depth_grid_stage_variant,
    with_reference_depth_grid_metadata,
)
from oceanbench.core.references import glo12, glorys
from oceanbench.core.weekly_stage import _weekly_stage_directory, maybe_stage_weekly_dataset


def _challenger_dataset(depths: list[float]) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0, 1],
            Dimension.DEPTH.key(): depths,
        }
    )


def _spatial_dataset(latitudes: list[float], longitudes: list[float]) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.TIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        }
    )


def test_reference_depth_grid_stage_variant_uses_rounded_depths() -> None:
    xihe_depths = numpy.array([0.494, 2.6457, 5.0782, 7.9296])
    wenhai_depths = numpy.array([0.494025, 2.645669, 5.078224, 7.929560])
    glo12_depths = numpy.array([0.494025, 47.37369, 92.32607, 155.8507])

    assert reference_depth_grid_stage_variant(xihe_depths) == reference_depth_grid_stage_variant(wenhai_depths)
    assert reference_depth_grid_stage_variant(xihe_depths) != reference_depth_grid_stage_variant(glo12_depths)


def test_with_reference_depth_grid_metadata_records_auditable_depth_grid() -> None:
    target_depths = numpy.array([0.494025, 2.645669])
    dataset = xarray.Dataset()

    dataset_with_metadata = with_reference_depth_grid_metadata(dataset, target_depths)

    assert dataset_with_metadata.attrs[REFERENCE_DEPTH_GRID_HASH_ATTRIBUTE]
    assert (
        dataset_with_metadata.attrs[REFERENCE_DEPTH_GRID_ROUNDING_ATTRIBUTE] == REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS
    )
    assert dataset_with_metadata.attrs[REFERENCE_TARGET_DEPTHS_ATTRIBUTE] == [0.494, 2.646]


def test_glorys_twelfth_degree_reference_stage_variant_depends_on_target_depth_grid(monkeypatch) -> None:
    captured_variants = []

    def fake_maybe_stage_weekly_dataset(**kwargs):
        captured_variants.append((kwargs["resolution"], kwargs["stage_variant"]))
        return xarray.Dataset()

    monkeypatch.setattr(glorys, "maybe_stage_weekly_dataset", fake_maybe_stage_weekly_dataset)

    glorys._glorys_reanalysis_dataset_1_12(_challenger_dataset([0.494, 2.6457]))
    glorys._glorys_reanalysis_dataset_1_12(_challenger_dataset([0.494025, 2.645669]))
    glorys._glorys_reanalysis_dataset_1_12(_challenger_dataset([0.494025, 47.37369]))

    assert captured_variants[0][0] == "twelfth_degree"
    assert captured_variants[0][1] == captured_variants[1][1]
    assert captured_variants[0][1] != captured_variants[2][1]


def test_glo12_twelfth_degree_reference_stage_variant_depends_on_target_depth_grid(monkeypatch) -> None:
    captured_variants = []

    def fake_maybe_stage_weekly_dataset(**kwargs):
        captured_variants.append((kwargs["resolution"], kwargs["stage_variant"]))
        return xarray.Dataset()

    monkeypatch.setattr(glo12, "maybe_stage_weekly_dataset", fake_maybe_stage_weekly_dataset)

    glo12._glo12_analysis_dataset_1_12(_challenger_dataset([0.494, 2.6457]))
    glo12._glo12_analysis_dataset_1_12(_challenger_dataset([0.494025, 2.645669]))
    glo12._glo12_analysis_dataset_1_12(_challenger_dataset([0.494025, 47.37369]))

    assert captured_variants[0][0] == "twelfth_degree"
    assert captured_variants[0][1] == captured_variants[1][1]
    assert captured_variants[0][1] != captured_variants[2][1]


def test_weekly_stage_variant_is_added_to_stage_path_without_changing_resolution(monkeypatch) -> None:
    monkeypatch.setattr("oceanbench.core.weekly_stage.local_stage_directory", lambda: Path("/tmp/oceanbench-stage"))

    stage_directory = _weekly_stage_directory(
        dataset_kind="reference",
        dataset_name="glo12",
        resolution="twelfth_degree",
        stage_variant="depths-23-abc",
        lead_days_count=10,
    )

    assert str(stage_directory) == "/tmp/oceanbench-stage/reference-glo12-twelfth_degree-depths-23-abc-10d"


def test_weekly_stage_variant_is_preserved_in_non_staged_source_metadata(monkeypatch) -> None:
    monkeypatch.setattr("oceanbench.core.weekly_stage.should_stage_locally", lambda _stage_key: False)

    dataset = maybe_stage_weekly_dataset(
        stage_key="references",
        dataset_kind="reference",
        dataset_name="glo12",
        first_day_datetimes=numpy.array([numpy.datetime64("2024-01-03")]),
        lead_days_count=10,
        open_week_dataset=lambda _first_day_datetime: xarray.Dataset(),
        open_remote_dataset=xarray.Dataset,
        resolution="twelfth_degree",
        stage_variant="depths-23-abc",
    )

    assert get_dataset_source(dataset) == DatasetSource(
        kind="reference",
        name="glo12",
        resolution="twelfth_degree",
        variant="depths-23-abc",
    )


def test_dataset_source_keeps_internal_variant_separate_from_resolution() -> None:
    dataset = with_dataset_source(
        xarray.Dataset(),
        kind="reference",
        name="glo12",
        resolution="twelfth_degree",
        variant="depths-23-abc",
    )

    dataset_source = get_dataset_source(dataset)

    assert dataset_source == DatasetSource(
        kind="reference",
        name="glo12",
        resolution="twelfth_degree",
        variant="depths-23-abc",
    )


def test_lagrangian_stage_path_uses_source_variant(monkeypatch) -> None:
    monkeypatch.setattr(
        "oceanbench.core.lagrangian_support.local_stage_directory", lambda: Path("/tmp/oceanbench-stage")
    )

    stage_directory = _lagrangian_stage_directory(
        DatasetSource(
            kind="reference",
            name="glo12",
            resolution="twelfth_degree",
            variant="depths-23-abc",
        ),
        lead_days_count=10,
        domain_variant="domain-2041x4320-def",
    )

    assert (
        str(stage_directory)
        == "/tmp/oceanbench-stage/lagrangian-reference-glo12-twelfth_degree-depths-23-abc-domain-2041x4320-def-10d"
    )


def test_lagrangian_domain_stage_variant_depends_on_spatial_grid() -> None:
    global_domain_variant = _lagrangian_domain_stage_variant(
        _spatial_dataset(
            latitudes=[-89.5, 0.5, 89.5],
            longitudes=[0.5, 1.5, 2.5],
        )
    )
    regional_domain_variant = _lagrangian_domain_stage_variant(
        _spatial_dataset(
            latitudes=[40.5, 41.5],
            longitudes=[-8.5, -7.5, -6.5],
        )
    )

    assert global_domain_variant.startswith("domain-3x3-")
    assert regional_domain_variant.startswith("domain-2x3-")
    assert global_domain_variant != regional_domain_variant


def test_lagrangian_domain_stage_variant_uses_rounded_coordinates() -> None:
    first_domain_variant = _lagrangian_domain_stage_variant(
        _spatial_dataset(
            latitudes=[40.5000001, 41.4999999],
            longitudes=[-8.4999999, -7.5000001],
        )
    )
    second_domain_variant = _lagrangian_domain_stage_variant(
        _spatial_dataset(
            latitudes=[40.5, 41.5],
            longitudes=[-8.5, -7.5],
        )
    )

    assert first_domain_variant == second_domain_variant
