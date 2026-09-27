# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import nbformat
import pandas
import pytest
import xarray

import oceanbench
from oceanbench.core import metrics
from oceanbench.core.python2jupyter import generate_evaluation_notebook_file
from oceanbench.core.regions import region_to_dict, resolve_region


MARINE_HEATWAVE_METRICS = (
    "marine_heatwave_diagnostics_compared_to_glorys_reanalysis",
    "marine_heatwave_diagnostics_compared_to_glo12_analysis",
)


def _generate_notebook(tmp_path, filename, region=None, spacing=0.25):
    challenger_path = tmp_path / filename
    challenger_path.write_text(
        "import xarray\n\n"
        "challenger_dataset = xarray.Dataset(coords={\n"
        f"    'latitude': [0.0, {spacing!r}],\n"
        f"    'longitude': [0.0, {spacing!r}],\n"
        "    'first_day_datetime': ['2024-01-01'],\n"
        "})\n",
        encoding="utf-8",
    )
    notebook_path = tmp_path / "report.ipynb"
    generate_evaluation_notebook_file(str(challenger_path), str(notebook_path), region=region)
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace = {}
    for index in (0, 2, 4):
        exec(notebook.cells[index].source, namespace)
    return notebook, namespace


def _metric_cell(notebook, metric_name):
    matching_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and f".{metric_name}(" in cell.source]
    assert len(matching_cells) == 1
    return matching_cells[0]


@pytest.mark.parametrize("filename", ["forecast.py", "custom_challenger.py", "forecast_1_degree.py"])
@pytest.mark.parametrize("region_identifier", ["global", "ibi", "custom"])
def test_generated_notebook_includes_both_diagnostics_with_region(tmp_path, monkeypatch, filename, region_identifier):
    region = (
        oceanbench.regions.custom(
            identifier="test_region",
            display_name="Test region",
            minimum_latitude=-1.0,
            maximum_latitude=1.0,
            minimum_longitude=-1.0,
            maximum_longitude=1.0,
        )
        if region_identifier == "custom"
        else region_identifier
    )
    notebook, namespace = _generate_notebook(tmp_path, filename, region)
    expected_region = resolve_region(region)

    assert notebook.metadata["oceanbench"]["region"] == region_to_dict(expected_region)
    assert resolve_region(namespace["region"]) == expected_region
    for metric_name in MARINE_HEATWAVE_METRICS:
        calls = []

        def record_call(challenger_dataset, *, region):
            calls.append((challenger_dataset, region))

        monkeypatch.setattr(oceanbench.metrics, metric_name, record_call)
        exec(_metric_cell(notebook, metric_name).source, namespace)

        assert len(calls) == 1
        assert calls[0][0] is namespace["challenger_dataset"]
        assert calls[0][1] is namespace["region"]

    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert "Marine Heatwave diagnostics compared to GLORYS reanalysis" in markdown
    assert "Marine Heatwave diagnostics compared to GLO12 analysis" in markdown


@pytest.mark.parametrize("filename", ["custom_forecast.py", "forecast_1_degree.py"])
@pytest.mark.parametrize("metric_name", MARINE_HEATWAVE_METRICS)
def test_one_degree_notebook_diagnostics_report_unavailability_without_remote_io(
    tmp_path, monkeypatch, filename, metric_name
):
    notebook, namespace = _generate_notebook(tmp_path, filename, spacing=1.0)

    def unexpected_remote_io(*args, **kwargs):
        pytest.fail("Unsupported Marine Heatwave diagnostics attempted to load remote data")

    for loader_name in (
        "glorys_reanalysis_dataset",
        "glo12_analysis_dataset",
        "glorys_reanalysis_history_dataset",
        "glo12_analysis_history_dataset",
        "marine_heatwave_climatology_mean_and_percentile_90",
    ):
        monkeypatch.setattr(metrics, loader_name, unexpected_remote_io)
    monkeypatch.setattr(xarray, "open_zarr", unexpected_remote_io)

    result = eval(_metric_cell(notebook, metric_name).source, namespace)

    assert list(result.columns) == ["Message"]
    assert "unavailable" in result.iloc[0]["Message"]
    assert "grid resolution" in result.iloc[0]["Message"]


@pytest.mark.parametrize("filename", ["custom_forecast.py", "forecast_1_degree.py"])
@pytest.mark.parametrize("metric_name", MARINE_HEATWAVE_METRICS)
@pytest.mark.parametrize("spacing", [0.25, 1.0 / 12.0])
def test_supported_notebook_diagnostics_use_dataset_resolution(tmp_path, monkeypatch, filename, metric_name, spacing):
    notebook, namespace = _generate_notebook(tmp_path, filename, spacing=spacing)
    reference_datasets = []
    expected_result = pandas.DataFrame({"score": [1.0]})

    def load_reference(challenger_dataset):
        reference_datasets.append(challenger_dataset)
        return challenger_dataset

    def compute_diagnostics(*args):
        return expected_result

    def load_history(challenger_dataset, history_days):
        return challenger_dataset

    monkeypatch.setattr(metrics, "glorys_reanalysis_dataset", load_reference)
    monkeypatch.setattr(metrics, "glo12_analysis_dataset", load_reference)
    monkeypatch.setattr(metrics, "glorys_reanalysis_history_dataset", load_history)
    monkeypatch.setattr(metrics, "glo12_analysis_history_dataset", load_history)
    monkeypatch.setattr(metrics, "_marine_heatwave_diagnostics_against_reference", compute_diagnostics)

    result = eval(_metric_cell(notebook, metric_name).source, namespace)

    assert result is expected_result
    assert len(reference_datasets) == 1
    xarray.testing.assert_equal(reference_datasets[0], namespace["challenger_dataset"])
