# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core import challenger_datasets, glo36v1, metrics
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references import glo36


def _raw_glo36_week_dataset() -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    values_with_depth = numpy.zeros((1, 3, 2, 2, 2), dtype="float32")
    values_surface = numpy.zeros((1, 3, 2, 2), dtype="float32")
    return xarray.Dataset(
        {
            "zos": ((first_day_key, lead_day_key, "lat", "lon"), values_surface),
            "thetao": (
                (first_day_key, lead_day_key, "depth", "lat", "lon"),
                values_with_depth,
            ),
            "so": (
                (first_day_key, lead_day_key, "depth", "lat", "lon"),
                values_with_depth,
            ),
            "uo": (
                (first_day_key, lead_day_key, "depth", "lat", "lon"),
                values_with_depth,
            ),
            "vo": (
                (first_day_key, lead_day_key, "depth", "lat", "lon"),
                values_with_depth,
            ),
        },
        coords={
            first_day_key: [numpy.datetime64("2023-01-04")],
            lead_day_key: [2, 3, 4],
            "depth": [0.0, 1.0],
            "lat": [0.0, 1.0 / 12.0],
            "lon": [10.0, 10.0 + 1.0 / 12.0],
        },
    )


def _challenger_dataset(first_day_datetimes: list[str], lead_days_count: int = 10) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(first_day_datetimes, dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count),
            Dimension.LATITUDE.key(): [0.0, 1.0 / 12.0],
            Dimension.LONGITUDE.key(): [10.0, 10.0 + 1.0 / 12.0],
        }
    )


def test_prepare_glo36v1_week_dataset_renames_dimensions_and_sets_standard_names() -> None:
    prepared = glo36v1.prepare_glo36v1_week_dataset(
        _raw_glo36_week_dataset(),
        lead_days_count=2,
        operation_name="test",
    )

    assert Dimension.LATITUDE.key() in prepared.dims
    assert Dimension.LONGITUDE.key() in prepared.dims
    assert prepared[Dimension.LEAD_DAY_INDEX.key()].values.tolist() == [0, 1]
    assert prepared["zos"].attrs["standard_name"] == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    assert prepared["thetao"].attrs["standard_name"] == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    assert prepared["so"].attrs["standard_name"] == Variable.SEA_WATER_SALINITY.key()
    assert prepared["uo"].attrs["standard_name"] == Variable.EASTWARD_SEA_WATER_VELOCITY.key()
    assert prepared["vo"].attrs["standard_name"] == Variable.NORTHWARD_SEA_WATER_VELOCITY.key()


def test_prepare_glo36v1_week_dataset_accepts_time_dimension_without_first_day() -> None:
    raw_dataset = _raw_glo36_week_dataset().isel(
        {Dimension.FIRST_DAY_DATETIME.key(): 0},
        drop=True,
    )
    raw_dataset = raw_dataset.rename({Dimension.LEAD_DAY_INDEX.key(): Dimension.TIME.key()})

    prepared = glo36v1.prepare_glo36v1_week_dataset(
        raw_dataset,
        lead_days_count=2,
        operation_name="test",
        first_day_datetime=numpy.datetime64("2023-01-04"),
    )

    assert list(prepared[Dimension.FIRST_DAY_DATETIME.key()].values.astype("datetime64[D]")) == [
        numpy.datetime64("2023-01-04")
    ]
    assert prepared[Dimension.LEAD_DAY_INDEX.key()].values.tolist() == [0, 1]


def test_glonet_high_resolution_dataset_path_uses_edito_bucket_path() -> None:
    assert (
        glo36v1.glonet_high_resolution_dataset_path(numpy.datetime64("2023-01-04"))
        == "https://minio.dive.edito.eu/moiai-octo-bucket/public/octo/v0/ai-gallery/octo-glonet-hr-p1d/20230104.zarr"
    )


def test_glonet_high_resolution_challenger_loader_uses_super_resolution_track(
    monkeypatch,
) -> None:
    captured = {}

    def fake_maybe_stage_weekly_dataset(**kwargs):
        captured.update(kwargs)
        return xarray.Dataset()

    monkeypatch.setattr(
        challenger_datasets,
        "maybe_stage_weekly_dataset",
        fake_maybe_stage_weekly_dataset,
    )
    monkeypatch.setattr(
        challenger_datasets,
        "with_remote_http_retries",
        lambda _operation_name, callback: callback(),
    )

    challenger_datasets.glonet_high_resolution()

    assert captured["dataset_kind"] == "challenger"
    assert captured["dataset_name"] == "glonet_high_resolution"
    assert captured["resolution"] == "super_resolution"
    assert captured["lead_days_count"] == glo36v1.GLO36V1_LEAD_DAYS_COUNT


def test_matching_glo36v1_first_day_datetimes_keeps_only_available_dates() -> None:
    challenger_dataset = _challenger_dataset(
        [
            "2022-12-28",
            "2023-01-04",
            "2024-01-03",
            "2024-01-10",
        ]
    )

    matching_dates = glo36v1.matching_glo36v1_first_day_datetimes(challenger_dataset)

    assert list(matching_dates.astype("datetime64[D]")) == [
        numpy.datetime64("2023-01-04"),
        numpy.datetime64("2024-01-03"),
    ]


def test_glo36v1_reference_dataset_uses_matching_dates_and_available_lead_days(
    monkeypatch,
) -> None:
    glo36._GLO36V1_REFERENCE_DATASET_CACHE.clear()
    captured = {}

    def fake_maybe_stage_weekly_dataset(**kwargs):
        captured.update(kwargs)
        return xarray.Dataset()

    monkeypatch.setattr(glo36, "maybe_stage_weekly_dataset", fake_maybe_stage_weekly_dataset)
    monkeypatch.setattr(glo36, "with_remote_http_retries", lambda _operation_name, callback: callback())
    challenger_dataset = with_dataset_source(
        _challenger_dataset(["2023-01-04", "2024-01-10"], lead_days_count=10),
        kind="challenger",
        name="glo36v1",
    )

    glo36.glo36v1_reference_dataset(challenger_dataset)

    assert captured["dataset_kind"] == "reference"
    assert captured["dataset_name"] == "glo36v1"
    assert captured["resolution"] == "super_resolution"
    assert captured["lead_days_count"] == glo36v1.GLO36V1_LEAD_DAYS_COUNT
    assert list(captured["first_day_datetimes"].astype("datetime64[D]")) == [numpy.datetime64("2023-01-04")]


def test_super_resolution_detection_uses_source_name_or_thirty_sixth_grid() -> None:
    twelfth_degree_dataset = _challenger_dataset(["2023-01-04"])
    thirty_sixth_degree_dataset = xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): [0.0, 1.0 / 36.0],
            Dimension.LONGITUDE.key(): [10.0, 10.0 + 1.0 / 36.0],
        }
    )

    assert not glo36v1.is_super_resolution_dataset(twelfth_degree_dataset)
    assert glo36v1.is_super_resolution_dataset(
        with_dataset_source(twelfth_degree_dataset, kind="challenger", name="glo36v1")
    )
    assert glo36v1.is_super_resolution_dataset(thirty_sixth_degree_dataset)


def test_glorys_and_glo12_scores_are_skipped_for_super_resolution_challengers(
    monkeypatch,
) -> None:
    def fail_reference_loader(_dataset):
        raise AssertionError("reference loader should not be called")

    monkeypatch.setattr(metrics, "glorys_reanalysis_dataset", fail_reference_loader)
    monkeypatch.setattr(metrics, "glo12_analysis_dataset", fail_reference_loader)
    challenger_dataset = with_dataset_source(
        _challenger_dataset(["2023-01-04"], lead_days_count=1),
        kind="challenger",
        name="glonet_high_resolution",
    )

    metric_functions = [
        metrics.rmsd_of_variables_compared_to_glorys_reanalysis,
        metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis,
        metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis,
        metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis,
        metrics.rmsd_of_variables_compared_to_glo12_analysis,
        metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis,
        metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis,
        metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis,
    ]

    for metric_function in metric_functions:
        scores = metric_function(challenger_dataset)
        assert list(scores.columns) == ["Message"]
        assert "GLO36V1 reference scores" in scores["Message"].iloc[0]
