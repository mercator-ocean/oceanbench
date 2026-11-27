# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The climatology baseline and the reference challengers: their kind, shift and self-comparison."""

from datetime import datetime

import numpy
import pytest
import xarray

from oceanbench.core import challenger_datasets
from oceanbench.core.references import glorys as glorys_references
from oceanbench.core.dataset_source import (
    get_challenger_description,
    get_challenger_kind,
    get_challenger_own_reference,
    with_challenger_kind,
)
from oceanbench.core.references import glo12 as glo12_references
from oceanbench.core.dataset_utils import Dimension, Variable

_START = datetime(2024, 1, 3)


def _sea_surface_height() -> xarray.DataArray:
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array([_START], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.LATITUDE.key(): numpy.arange(-10.0, 10.0, 1.0),
        Dimension.LONGITUDE.key(): numpy.arange(-20.0, 20.0, 1.0),
    }
    return xarray.DataArray(
        numpy.full((1, 1, 20, 40), 1.0),
        dims=list(coordinates),
        coords=coordinates,
        name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    )


@pytest.fixture
def opened_paths(monkeypatch) -> list[str]:
    paths: list[str] = []

    def open_multizarr(path_from_start, *, first_day_datetimes=None, lead_days_count=10):
        paths.append(path_from_start(first_day_datetimes[0]))
        return _sea_surface_height().to_dataset()

    monkeypatch.setattr(challenger_datasets, "_open_multizarr_forecasts_as_challenger_dataset", open_multizarr)
    monkeypatch.setattr(challenger_datasets, "interpolate_1_degree", lambda dataset: dataset.copy())
    return paths


def test_the_climatology_opens_the_baseline_forecasts_as_a_baseline(opened_paths) -> None:
    for climatology in (challenger_datasets.climatology, challenger_datasets.climatology_1_degree):
        dataset = climatology([_START])
        assert get_challenger_kind(dataset) == "baseline"
        assert get_challenger_description(dataset) == "Daily climatology derived from GLORYS, years 1993 to 2019"
    assert opened_paths[0].endswith("/public/baseline-forecasts/climatology/20240103.zarr")


def test_the_glorys_challengers_wrap_the_reference_openers_as_references(monkeypatch) -> None:
    templates = []

    def reference_opener(template: xarray.Dataset) -> xarray.Dataset:
        templates.append(template)
        return _sea_surface_height().to_dataset()

    monkeypatch.setattr(challenger_datasets.glorys_references, "_glorys_reanalysis_dataset_1_4", reference_opener)
    monkeypatch.setattr(challenger_datasets.glorys_references, "_glorys_reanalysis_dataset_1_degree", reference_opener)
    for glorys in (challenger_datasets.glorys, challenger_datasets.glorys_1_degree):
        dataset = glorys([_START])
        assert get_challenger_kind(dataset) == "reference"
        assert get_challenger_description(dataset).startswith("Reanalysis; assimilates")
    for template in templates:
        assert template.sizes == {Dimension.FIRST_DAY_DATETIME.key(): 1, Dimension.LEAD_DAY_INDEX.key(): 10}


_FORECAST_DEPTHS = numpy.array([0.494025, 1.541375, 5727.917], dtype="float32")


@pytest.fixture
def twelfth_degree_templates(monkeypatch) -> list[xarray.Dataset]:
    templates: list[xarray.Dataset] = []

    def reference_opener(template: xarray.Dataset) -> xarray.Dataset:
        templates.append(template)
        return _sea_surface_height().to_dataset()

    forecast_week = xarray.Dataset(coords={Dimension.DEPTH.key(): _FORECAST_DEPTHS})
    monkeypatch.setattr(challenger_datasets, "_open_glo12_forecast_week", lambda _first_day: forecast_week)
    monkeypatch.setattr(glorys_references, "_glorys_reanalysis_dataset_1_12", reference_opener)
    monkeypatch.setattr(glo12_references, "_glo12_analysis_dataset_1_12", reference_opener)
    return templates


def test_the_twelfth_degree_references_open_with_the_reference_openers_on_the_glo12_forecast_depths(
    twelfth_degree_templates,
) -> None:
    glorys = challenger_datasets.glorys_twelfth_degree([_START])
    analysis = challenger_datasets.glo12_analysis([_START])

    assert get_challenger_kind(glorys) == get_challenger_kind(analysis) == "reference"
    assert get_challenger_description(glorys).startswith("Reanalysis; assimilates")
    assert get_challenger_description(analysis).startswith("Analysis of the GLO12 system; assimilates")
    assert get_challenger_own_reference(glorys) == "glorys_reanalysis"
    assert get_challenger_own_reference(analysis) == "glo12_analysis"
    for template in twelfth_degree_templates:
        assert template.sizes == {
            Dimension.FIRST_DAY_DATETIME.key(): 1,
            Dimension.LEAD_DAY_INDEX.key(): 10,
            Dimension.DEPTH.key(): 3,
        }
        numpy.testing.assert_array_equal(template[Dimension.DEPTH.key()].values, _FORECAST_DEPTHS)


def test_only_the_glorys_twelfth_degree_reference_uses_the_glorys_shift(twelfth_degree_templates) -> None:
    from oceanbench.core.classIV_support import (
        GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
        REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
        sea_surface_height_shift,
    )

    assert sea_surface_height_shift(challenger_datasets.glorys_twelfth_degree([_START])) == (
        GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    )
    assert sea_surface_height_shift(challenger_datasets.glo12_analysis([_START])) == (
        REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    )


def test_a_reference_without_its_own_reference_is_glorys_and_other_kinds_have_none() -> None:
    dataset = _sea_surface_height().to_dataset()
    assert get_challenger_own_reference(with_challenger_kind(dataset, "reference", "")) == "glorys_reanalysis"
    assert get_challenger_own_reference(with_challenger_kind(dataset, "baseline", "")) is None
    assert get_challenger_own_reference(dataset) is None


def test_a_challenger_without_a_kind_is_a_model_and_unknown_kinds_are_rejected() -> None:
    dataset = _sea_surface_height().to_dataset()
    assert get_challenger_kind(dataset) == "model"
    assert get_challenger_description(dataset) is None
    with pytest.raises(ValueError):
        with_challenger_kind(dataset, "oracle", "")


def test_glorys_against_its_own_gridded_reference_is_exactly_zero_at_every_lead(monkeypatch) -> None:
    lead_days = 10
    zos = xarray.concat([_sea_surface_height() + lead for lead in range(lead_days)], dim="time")
    week = (
        zos.isel({Dimension.LEAD_DAY_INDEX.key(): 0}, drop=True)
        .squeeze(Dimension.FIRST_DAY_DATETIME.key(), drop=True)
        .to_dataset()
    )
    week = week.assign_coords(time=numpy.arange(lead_days))
    monkeypatch.setattr(glorys_references, "open_remote_zarr", lambda _path: week)
    monkeypatch.setattr(
        glorys_references,
        "open_remote_multizarr",
        lambda paths, **kwargs: xarray.concat(
            [kwargs["preprocess"](week) for _ in paths], dim=Dimension.FIRST_DAY_DATETIME.key()
        ),
    )
    challenger = challenger_datasets.glorys_1_degree([_START])
    reference = glorys_references._glorys_reanalysis_dataset_1_degree(challenger)
    key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()

    difference = (challenger[key] - reference[key]).max(
        [Dimension.FIRST_DAY_DATETIME.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    )
    assert difference.sizes[Dimension.LEAD_DAY_INDEX.key()] == lead_days
    numpy.testing.assert_array_equal(difference.values, 0.0)


def test_glorys_based_challengers_use_the_glorys_shift_and_every_other_challenger_the_glo12_one(
    opened_paths, monkeypatch
) -> None:
    from oceanbench.core.classIV_support import (
        GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
        REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
        sea_surface_height_shift,
    )

    assert REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT == -0.1148
    assert GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT == pytest.approx(-0.1148 + 0.0502)
    monkeypatch.setattr(
        glorys_references, "_glorys_reanalysis_dataset_1_4", lambda _t: _sea_surface_height().to_dataset()
    )
    monkeypatch.setattr(
        glorys_references, "_glorys_reanalysis_dataset_1_degree", lambda _t: _sea_surface_height().to_dataset()
    )
    for opener in (
        challenger_datasets.glorys,
        challenger_datasets.glorys_1_degree,
        challenger_datasets.climatology,
        challenger_datasets.climatology_1_degree,
    ):
        assert sea_surface_height_shift(opener([_START])) == GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    for opener in (challenger_datasets.glo12_persistence, challenger_datasets.glonet, challenger_datasets.xihe):
        assert sea_surface_height_shift(opener([_START])) == REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    for kind in ("model", "baseline", "reference"):
        plain = with_challenger_kind(_sea_surface_height().to_dataset(), kind, "other")
        assert sea_surface_height_shift(plain) == REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
