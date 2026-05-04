# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import warnings
from pathlib import Path

import nbformat
import numpy
import xarray

import oceanbench.visualization
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.python2jupyter import generate_evaluation_notebook_file


def _map_dataset(values_by_variable: dict[str, numpy.ndarray]) -> xarray.Dataset:
    first_day_count = next(iter(values_by_variable.values())).shape[0]
    depth_count = max((values.shape[2] for values in values_by_variable.values() if values.ndim == 5), default=0)
    return xarray.Dataset(
        {
            variable_name: (
                (
                    [
                        Dimension.FIRST_DAY_DATETIME.key(),
                        Dimension.LEAD_DAY_INDEX.key(),
                        Dimension.DEPTH.key(),
                        Dimension.LATITUDE.key(),
                        Dimension.LONGITUDE.key(),
                    ]
                    if values.ndim == 5
                    else [
                        Dimension.FIRST_DAY_DATETIME.key(),
                        Dimension.LEAD_DAY_INDEX.key(),
                        Dimension.LATITUDE.key(),
                        Dimension.LONGITUDE.key(),
                    ]
                ),
                values,
            )
            for variable_name, values in values_by_variable.items()
        },
        coords={
            **(
                {
                    Dimension.DEPTH.key(): numpy.array(
                        [0.5, 10.0, 47.4, 92.3, 155.9, 222.5, 318.1, 541.1][:depth_count],
                        dtype=float,
                    )
                }
                if depth_count
                else {}
            ),
            Dimension.FIRST_DAY_DATETIME.key(): numpy.datetime64("2024-01-03")
            + numpy.arange(first_day_count).astype("timedelta64[D]"),
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(next(iter(values_by_variable.values())).shape[1]),
            Dimension.LATITUDE.key(): [-1.0, 1.0],
            Dimension.LONGITUDE.key(): [10.0, 12.0, 14.0],
        },
    )


def test_plot_surface_comparison_explorer_returns_self_contained_html() -> None:
    sea_surface_height = numpy.arange(36, dtype=float).reshape(2, 3, 2, 3)
    depth_template = numpy.arange(72, dtype=float).reshape(2, 3, 2, 2, 3)
    temperature = depth_template + 10.0
    salinity = depth_template + 30.0
    eastward_velocity = depth_template / 100.0
    northward_velocity = eastward_velocity + 0.2
    challenger_dataset = _map_dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): temperature,
            Variable.SEA_WATER_SALINITY.key(): salinity,
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): eastward_velocity,
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): northward_velocity,
        }
    )
    reference_dataset = _map_dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height + 1.0,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): temperature + 0.5,
            Variable.SEA_WATER_SALINITY.key(): salinity + 0.2,
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): eastward_velocity + 0.1,
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): northward_velocity - 0.1,
        }
    )

    html_output = oceanbench.visualization.plot_surface_comparison_explorer(
        challenger_dataset,
        reference_dataset,
        "Reference",
        height_pixels=500,
    )

    assert "<iframe" in html_output.data
    assert "Sea surface height" in html_output.data
    assert "Temperature" in html_output.data
    assert "Salinity" in html_output.data
    assert "Zonal current" in html_output.data
    assert "Meridional current" in html_output.data
    assert "Signed error" in html_output.data
    assert "Absolute error" in html_output.data
    assert "RMSE over dates" in html_output.data
    assert "rmse_over_dates" in html_output.data
    assert "data:image/webp;base64," in html_output.data
    assert "Lead day" in html_output.data
    assert "ob-map-variable-buttons" in html_output.data
    assert "ob-map-depth-buttons" in html_output.data
    assert "ob-map-layer-buttons" in html_output.data
    assert "ob-map-loading" not in html_output.data
    assert "0.5 m" in html_output.data
    assert "10 m" in html_output.data
    assert "depths" in html_output.data
    assert "ob-map-secondary-row" in html_output.data
    assert "font-variant-numeric: tabular-nums" in html_output.data
    assert "white-space: nowrap" in html_output.data
    assert "decodeInt16" not in html_output.data
    assert "allow-scripts" in html_output.data
    assert "height:500px" in html_output.data


def test_plot_surface_comparison_explorer_skips_missing_default_variables() -> None:
    sea_surface_height = numpy.arange(18, dtype=float).reshape(1, 3, 2, 3)
    challenger_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height})
    reference_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height + 1.0})

    html_output = oceanbench.visualization.plot_surface_comparison_explorer(
        challenger_dataset,
        reference_dataset,
        "Reference",
    )

    assert "Sea surface height" in html_output.data
    assert "Temperature" not in html_output.data
    assert "Salinity" not in html_output.data
    assert "Zonal current" not in html_output.data
    assert "Meridional current" not in html_output.data


def test_plot_surface_comparison_explorer_uses_demo_depths_by_default() -> None:
    values = numpy.arange(48, dtype=float).reshape(1, 1, 8, 2, 3)
    challenger_dataset = _map_dataset({Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): values})
    reference_dataset = _map_dataset({Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): values + 1.0})

    html_output = oceanbench.visualization.plot_surface_comparison_explorer(
        challenger_dataset,
        reference_dataset,
        "Reference",
    )

    assert "0.5 m" in html_output.data
    assert "47 m" in html_output.data
    assert "92 m" in html_output.data
    assert "222 m" in html_output.data
    assert "318 m" in html_output.data
    assert "541 m" in html_output.data
    assert "10 m" not in html_output.data
    assert "156 m" not in html_output.data


def test_plot_multi_reference_surface_comparison_explorer_uses_one_viewer() -> None:
    sea_surface_height = numpy.arange(18, dtype=float).reshape(1, 3, 2, 3)
    challenger_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height})
    first_reference_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height + 1.0})
    second_reference_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height + 2.0})

    html_output = oceanbench.visualization.plot_multi_reference_surface_comparison_explorer(
        challenger_dataset,
        {
            "First reference": first_reference_dataset,
            "Second reference": second_reference_dataset,
        },
    )

    assert "First reference" in html_output.data
    assert "Second reference" in html_output.data
    assert "ob-map-reference-buttons" in html_output.data
    assert "challengerLayer" in html_output.data
    assert "references" in html_output.data
    assert "Signed error" in html_output.data
    assert "Spatial RMSE over lead days" in html_output.data
    assert "spatialRmse" in html_output.data
    assert "data:image/webp;base64," in html_output.data


def test_plot_multi_reference_zonal_psd_comparison_returns_compact_figure() -> None:
    sea_surface_height = numpy.arange(18, dtype=float).reshape(1, 3, 2, 3)
    temperature = numpy.arange(18, dtype=float).reshape(1, 3, 1, 2, 3) + 10.0
    challenger_dataset = _map_dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): temperature,
        }
    )
    reference_dataset = _map_dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height + 1.0,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): temperature + 0.5,
        }
    )

    figure = oceanbench.visualization.plot_multi_reference_zonal_psd_comparison(
        challenger_dataset,
        {"Reference": reference_dataset},
    )

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xscale() == "log"
    assert figure.axes[0].get_yscale() == "log"


def test_plot_multi_reference_zonal_psd_comparison_accepts_masked_rows_without_warning() -> None:
    sea_surface_height = numpy.arange(18, dtype=float).reshape(1, 3, 2, 3)
    sea_surface_height[:, :, 1, :] = numpy.nan
    challenger_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height})
    reference_dataset = _map_dataset({Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): sea_surface_height + 1.0})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        figure = oceanbench.visualization.plot_multi_reference_zonal_psd_comparison(
            challenger_dataset,
            {"Reference": reference_dataset},
            variables=[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID],
        )

    assert len(figure.axes) == 1


def test_plot_multi_reference_lagrangian_trajectory_explorer_returns_animated_html(monkeypatch) -> None:
    from oceanbench.core import visualization as core_visualization

    monkeypatch.setattr(
        core_visualization,
        "_lagrangian_payload",
        lambda **_: {
            "title": "Lagrangian trajectory divergence",
            "challengerName": "Challenger",
            "particleCount": 2,
            "timeLabels": ["1.0", "2.0"],
            "bounds": {
                "longitudeMinimum": 9.0,
                "longitudeMaximum": 15.0,
                "latitudeMinimum": -2.0,
                "latitudeMaximum": 2.0,
            },
            "landMask": {
                "longitude": [9.0, 10.0],
                "latitude": [0.0, 1.0],
                "land": [[0, 1], [1, 0]],
            },
            "separationScaleKilometers": 10.0,
            "challenger": {
                "longitude": [[10.0, 11.0], [12.0, 13.0]],
                "latitude": [[0.0, 0.5], [1.0, 1.5]],
                "initialLongitude": [10.0, 12.0],
                "initialLatitude": [0.0, 1.0],
            },
            "references": [
                {
                    "key": "reference",
                    "label": "Reference",
                    "track": {
                        "longitude": [[10.0, 10.5], [12.0, 12.5]],
                        "latitude": [[0.0, 0.2], [1.0, 1.2]],
                        "initialLongitude": [10.0, 12.0],
                        "initialLatitude": [0.0, 1.0],
                    },
                }
            ],
        },
    )

    html_output = oceanbench.visualization.plot_multi_reference_lagrangian_trajectory_explorer(
        xarray.Dataset(),
        {"Reference": xarray.Dataset()},
        height_pixels=510,
    )

    assert "<iframe" in html_output.data
    assert "height:510px" in html_output.data
    assert "Lagrangian trajectory divergence" in html_output.data
    assert "requestAnimationFrame" in html_output.data
    assert "Smooth visual interpolation between true daily particle positions" in html_output.data
    assert "drawLandMask" in html_output.data
    assert "Current separation distance" in html_output.data
    assert "Reached daily positions" in html_output.data


def test_plot_multi_reference_eddy_matching_explorer_returns_animated_html(monkeypatch) -> None:
    from oceanbench.core import visualization as core_visualization

    eddy_record = {
        "id": 1,
        "latitude": 0.0,
        "longitude": 10.0,
        "polarity": "cyclone",
        "contourLatitude": [0.0, 0.3, 0.0, -0.3],
        "contourLongitude": [9.7, 10.0, 10.3, 10.0],
    }
    monkeypatch.setattr(
        core_visualization,
        "_eddy_payload",
        lambda **_: {
            "title": "Mesoscale eddy matching",
            "bounds": {
                "longitudeMinimum": 9.0,
                "longitudeMaximum": 15.0,
                "latitudeMinimum": -2.0,
                "latitudeMaximum": 2.0,
            },
            "landMask": {
                "longitude": [9.0, 10.0],
                "latitude": [0.0, 1.0],
                "land": [[0, 1], [1, 0]],
            },
            "references": [
                {
                    "key": "reference",
                    "label": "Reference",
                    "frames": [
                        {
                            "leadDay": 1,
                            "matches": [
                                {
                                    "challenger": eddy_record,
                                    "reference": eddy_record,
                                    "distanceKilometers": 12.0,
                                }
                            ],
                            "spurious": [eddy_record],
                            "missed": [eddy_record],
                        }
                    ],
                }
            ],
        },
    )

    html_output = oceanbench.visualization.plot_multi_reference_eddy_matching_explorer(
        xarray.Dataset(),
        {"Reference": xarray.Dataset()},
        height_pixels=520,
    )

    assert "<iframe" in html_output.data
    assert "height:520px" in html_output.data
    assert "Mesoscale eddy matching" in html_output.data
    assert "Discrete SSH eddy detections per lead day" in html_output.data
    assert "Matched eddy" in html_output.data
    assert "Spurious challenger" in html_output.data
    assert "Missed reference" in html_output.data
    assert "window.setInterval" in html_output.data


def test_plot_class4_observation_error_explorer_returns_interactive_html(monkeypatch) -> None:
    from oceanbench.core import visualization as core_visualization

    monkeypatch.setattr(
        core_visualization,
        "_class4_payload",
        lambda **_: {
            "title": "Class IV observation error maps",
            "bounds": {
                "longitudeMinimum": 9.0,
                "longitudeMaximum": 15.0,
                "latitudeMinimum": -2.0,
                "latitudeMaximum": 2.0,
            },
            "landMask": {
                "longitude": [9.0, 10.0],
                "latitude": [0.0, 1.0],
                "land": [[0, 1], [1, 0]],
            },
            "variables": [
                {
                    "key": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
                    "label": "Sea level anomaly",
                    "unit": "m",
                    "depths": [
                        {
                            "key": "surface",
                            "label": "surface",
                            "signedScale": 0.2,
                            "absoluteScale": 0.2,
                            "frames": [
                                {
                                    "leadDay": 1,
                                    "totalCount": 2,
                                    "shownCount": 2,
                                    "longitude": [10.0, 11.0],
                                    "latitude": [0.0, 1.0],
                                    "error": [-0.1, 0.2],
                                    "absoluteError": [0.1, 0.2],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    html_output = oceanbench.visualization.plot_class4_observation_error_explorer(
        xarray.Dataset(),
        xarray.Dataset(),
        height_pixels=500,
    )

    assert "<iframe" in html_output.data
    assert "height:500px" in html_output.data
    assert "Class IV observation error maps" in html_output.data
    assert "Model minus Class IV observation errors" in html_output.data
    assert "Observation density" not in html_output.data
    assert "Signed error" in html_output.data


def test_generated_evaluation_notebook_contains_diagnostic_explorers(tmp_path: Path) -> None:
    challenger_path = tmp_path / "challenger.py"
    challenger_path.write_text("import xarray\n\nchallenger_dataset = xarray.Dataset()\n", encoding="utf-8")
    output_path = tmp_path / "report.ipynb"

    generate_evaluation_notebook_file(
        str(challenger_path),
        str(output_path),
        region="global",
    )

    notebook = nbformat.read(output_path, as_version=4)
    all_sources = "\n".join(cell.source for cell in notebook.cells)

    assert "oceanbench.visualization.plot_multi_reference_surface_comparison_explorer" in all_sources
    assert "glorys_reanalysis_dataset" in all_sources
    assert '"GLORYS reanalysis"' in all_sources
    assert "glo12_analysis_dataset" in all_sources
    assert '"GLO12 analysis"' in all_sources
    assert "from oceanbench.core.rmsd import rmsd" in all_sources
    assert "from dask.array.core import PerformanceWarning" in all_sources
    assert "forecast_comparison_explorer" in all_sources
    assert 'title="Forecast comparison maps"' in all_sources
    assert "forecast_comparison_variables" in all_sources
    assert "geostrophic_current_variables" in all_sources
    assert "dynamic_diagnostic_explorer" in all_sources
    assert 'title="Dynamic diagnostic maps"' in all_sources
    assert "dynamic_diagnostic_variables" in all_sources
    assert "lagrangian_trajectory_explorer" in all_sources
    assert "plot_multi_reference_lagrangian_trajectory_explorer" in all_sources
    assert "particle_count=300" in all_sources
    assert "eddy_matching_explorer" in all_sources
    assert "plot_multi_reference_eddy_matching_explorer" in all_sources
    assert "class4_observation_error_explorer" in all_sources
    assert "plot_class4_observation_error_explorer" in all_sources
    assert "class4_validation_dataframe" in all_sources
    assert "rmsd_class4_validation_dataframe" in all_sources
    assert "class4_observation_comparison_dataframe" in all_sources
    assert "ObservationDataUnavailableError" in all_sources
    assert "Eddy centers and contours are detected" in all_sources
    assert "plot_multi_reference_zonal_psd_comparison" in all_sources
    assert "zonal_psd_figure" in all_sources
    assert "zonal_psd_figure = oceanbench.visualization.plot_multi_reference_zonal_psd_comparison" in all_sources
    assert ")\nNone" in next(cell.source for cell in notebook.cells if "zonal_psd_figure =" in cell.source)
    assert "compute_mixed_layer_depth" in all_sources
    assert "compute_geostrophic_currents" in all_sources
    assert "Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID" in all_sources
    assert "Variable.SEA_WATER_POTENTIAL_TEMPERATURE" in all_sources
    assert "Variable.SEA_WATER_SALINITY" in all_sources
    assert "Variable.EASTWARD_SEA_WATER_VELOCITY" in all_sources
    assert "Variable.NORTHWARD_SEA_WATER_VELOCITY" in all_sources
    assert "Variable.MIXED_LAYER_DEPTH" in all_sources
    assert "Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY" in all_sources
    assert "Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY" in all_sources
    assert "forecast_comparison_explorer\n" in all_sources
    assert "dynamic_diagnostic_explorer\n" in all_sources
    assert "surface_comparison_explorer =" not in all_sources
    assert "\nsurface_comparison_explorer\n" not in all_sources
    assert "surface_comparison_variables" not in all_sources
    assert "warnings.filterwarnings" in all_sources
    assert "Increasing number of chunks" in all_sources
    assert "Geostrophic currents are masked near the equator" in all_sources
    assert all_sources.index("rmsd(\n    challenger_dataset=regional_challenger_dataset") < all_sources.index(
        "forecast_comparison_explorer ="
    )
    assert all_sources.index("glo12_geostrophic_dataset = compute_geostrophic_currents") < all_sources.index(
        "dynamic_diagnostic_explorer ="
    )
    assert all_sources.index(
        "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis"
    ) < all_sources.index("lagrangian_trajectory_explorer =")
    assert all_sources.index("class4_observation_error_explorer =") < all_sources.index(
        "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis"
    )
    assert all_sources.index("lagrangian_trajectory_explorer =") < all_sources.index("eddy_matching_explorer =")
    assert all_sources.index("eddy_matching_explorer =") < all_sources.index("forecast_comparison_explorer =")
    assert "reference_dataset=glorys_dataset" in all_sources
    assert "reference_dataset=glo12_dataset" in all_sources
    assert "reference_dataset=glorys_mld_dataset" in all_sources
    assert "reference_dataset=glo12_mld_dataset" in all_sources
    assert "reference_dataset=glorys_geostrophic_dataset" in all_sources
    assert "reference_dataset=glo12_geostrophic_dataset" in all_sources
    assert "oceanbench.metrics.rmsd_of_variables_compared_to_observations" not in all_sources
    assert "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis" in all_sources
    assert "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis" in all_sources
    assert "oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis" not in all_sources
    assert "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis" not in all_sources
    assert "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis" not in all_sources
    assert "oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis" not in all_sources
    assert "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis" not in all_sources
    assert "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis" not in all_sources
    assert "glorys_surface_comparison_explorer" not in all_sources
    assert "glo12_surface_comparison_explorer" not in all_sources
    assert "plot_surface_comparison_maps" not in all_sources
    assert "plot_spatial_rmse_gallery" not in all_sources
    assert "plot_lagrangian_trajectory_comparison" not in all_sources
