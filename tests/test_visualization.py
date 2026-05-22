# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import warnings
from pathlib import Path

import nbformat
import numpy
import pandas
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
    assert "Color scale:" in html_output.data
    assert "Maps are downsampled and compressed for display only" in html_output.data
    assert "Absolute error" in html_output.data
    assert "RMSE over dates" in html_output.data
    assert "rmse_over_dates" in html_output.data
    assert "data:image/webp;base64," in html_output.data
    assert "Lead day" in html_output.data
    assert "ob-map-variable-buttons" in html_output.data
    assert "ob-map-depth-buttons" in html_output.data
    assert "ob-map-layer-buttons" in html_output.data
    assert "ob-map-loading" not in html_output.data
    assert "max-width: none" in html_output.data
    assert "--ob-map-content-max-width: 1220px" in html_output.data
    assert "max-width: var(--ob-map-content-max-width)" in html_output.data
    assert "0.5 m" in html_output.data
    assert "10 m" in html_output.data
    assert "depths" in html_output.data
    assert "ob-map-secondary-row" in html_output.data
    assert "font-variant-numeric: tabular-nums" in html_output.data
    assert "white-space: nowrap" in html_output.data
    assert "decodeInt16" not in html_output.data
    assert "allow-scripts" in html_output.data
    assert "height:500px" in html_output.data


def test_map_image_uses_vertical_colorbar(monkeypatch) -> None:
    from oceanbench.core import visualization as core_visualization

    captured_colorbar_arguments = {}

    original_colorbar = core_visualization.plt.Figure.colorbar

    def recording_colorbar(self, *args, **kwargs):
        captured_colorbar_arguments.update(kwargs)
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(core_visualization.plt.Figure, "colorbar", recording_colorbar)
    field = xarray.DataArray(
        numpy.arange(6, dtype=float).reshape(2, 3),
        dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={Dimension.LATITUDE.key(): [-1.0, 1.0], Dimension.LONGITUDE.key(): [10.0, 12.0, 14.0]},
    )

    core_visualization._encoded_webp_image(
        field,
        core_visualization._positive_norm([field]),
        "viridis",
        "test value",
        "Test map",
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    )

    assert captured_colorbar_arguments["orientation"] == "vertical"


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
    assert "Spatial RMSE trend (auto-scaled)" in html_output.data
    assert "lead 1" in html_output.data
    assert "spatialRmse" in html_output.data
    assert "scaleLabel" in html_output.data
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


def test_plot_multi_reference_zonal_psd_comparison_explorer_returns_hoverable_html() -> None:
    longitude = numpy.linspace(0.0, 330.0, 12)
    sea_surface_height = numpy.sin(numpy.deg2rad(longitude))[None, None, None, :] + numpy.arange(3)[None, :, None, None]
    sea_surface_height = numpy.repeat(sea_surface_height, 2, axis=2)
    challenger_dataset = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                sea_surface_height,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(3),
            Dimension.LATITUDE.key(): [-1.0, 1.0],
            Dimension.LONGITUDE.key(): longitude,
        },
    )
    reference_dataset = challenger_dataset + 0.2

    html_output = oceanbench.visualization.plot_multi_reference_zonal_psd_comparison_explorer(
        challenger_dataset,
        {"Reference": reference_dataset},
        variables=[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID],
    )

    assert "ob-psd-explorer" in html_output.data
    assert "ob-psd-tooltip" in html_output.data
    assert "wavelengthKilometers" in html_output.data
    assert "Power spectral density" in html_output.data
    assert "minimumWavelengthKilometers" in html_output.data


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
                "paths": [
                    {
                        "longitude": [9.0, 10.0, 10.0, 9.0, 9.0],
                        "latitude": [0.0, 0.0, 1.0, 1.0, 0.0],
                    }
                ],
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
    assert "particles are sampled for display" in html_output.data
    assert "drawLandMask" in html_output.data
    assert "landMask.paths" in html_output.data
    assert "landMask.land" not in html_output.data
    assert "ob-lagrangian-zoom" in html_output.data
    assert "createMapViewport" in html_output.data
    assert "createMapBackgroundCache" in html_output.data
    assert "backgroundCache.draw(context, viewport.cacheKey())" in html_output.data
    assert "viewport.longitudeShiftsForPath(path)" in html_output.data
    assert "projection.xUnwrapped" in html_output.data
    assert "fill(&quot;evenodd&quot;)" in html_output.data
    assert "Current separation distance" in html_output.data
    assert "Reached daily positions" in html_output.data


def test_land_mask_payload_uses_model_derived_vector_paths() -> None:
    from oceanbench.core import visualization as core_visualization

    dataset = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
                numpy.array(
                    [
                        [1.0, numpy.nan, 1.0],
                        [1.0, numpy.nan, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ),
            )
        },
        coords={
            Dimension.LATITUDE.key(): [-1.0, 0.0, 1.0],
            Dimension.LONGITUDE.key(): [10.0, 11.0, 12.0],
        },
    )

    payload = core_visualization._lagrangian_land_mask_payload(dataset, first_day_index=0)

    assert "paths" in payload
    assert "land" not in payload
    assert payload["paths"]
    assert payload["paths"][0]["longitude"]
    assert payload["paths"][0]["latitude"]


def test_map_bounds_use_periodic_pacific_centered_longitude_for_global_domains() -> None:
    from oceanbench.core import visualization as core_visualization

    dataset = xarray.Dataset(
        coords={
            Dimension.LATITUDE.key(): [-77.5, 0.0, 89.5],
            Dimension.LONGITUDE.key(): [-179.5, -0.5, 0.5, 179.5],
        },
    )

    bounds = core_visualization._lagrangian_bounds(dataset)

    assert bounds["longitudeMinimum"] == 0.0
    assert bounds["longitudeMaximum"] == 360.0
    assert bounds["latitudeMinimum"] == -77.5
    assert bounds["latitudeMaximum"] == 89.5
    assert bounds["periodicLongitude"] is True
    assert bounds["longitudePeriod"] == 360.0


def test_map_bounds_keep_regional_domains_clamped_to_data_extent() -> None:
    from oceanbench.core import visualization as core_visualization

    dataset = xarray.Dataset(
        coords={
            Dimension.LATITUDE.key(): [-2.0, 0.0, 2.0],
            Dimension.LONGITUDE.key(): [10.0, 12.0, 14.0],
        },
    )

    bounds = core_visualization._lagrangian_bounds(dataset)

    assert bounds["longitudeMinimum"] == 10.0
    assert bounds["longitudeMaximum"] == 14.0
    assert bounds["latitudeMinimum"] == -2.0
    assert bounds["latitudeMaximum"] == 2.0
    assert bounds["periodicLongitude"] is False


def test_class4_sampler_uses_time_sorted_bucket_centers_for_profile_variables() -> None:
    from oceanbench.core import visualization as core_visualization

    group = pandas.DataFrame(
        {
            Dimension.TIME.key(): pandas.to_datetime(
                [
                    "2024-01-03T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-03T00:00:00",
                    "2024-01-02T00:00:00",
                ]
            ),
            Dimension.LONGITUDE.key(): [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            Dimension.LATITUDE.key(): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "lead_day": [0, 0, 0, 0, 0, 0],
            "error": [100.0, 10.0, 41.0, 11.0, 101.0, 40.0],
            "absolute_error": [100.0, 10.0, 41.0, 11.0, 101.0, 40.0],
        }
    )

    frame = core_visualization._class4_sampled_frame(
        group,
        maximum_points_per_frame=3,
        variable_key=Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    )

    assert frame["leadDay"] == 1
    assert frame["totalCount"] == 6
    assert frame["shownCount"] == 3
    assert frame["longitude"] == [1.0, 1.0, 1.0]
    assert frame["error"] == [11.0, 41.0, 101.0]


def test_class4_sampler_keeps_spaced_track_windows_for_satellite_variables() -> None:
    from oceanbench.core import visualization as core_visualization

    shuffled_indices = [9, 0, 5, 15, 6, 4, 16, 14, 19, 1, 2, 3, 7, 8, 10, 11, 12, 13, 17, 18]
    group = pandas.DataFrame(
        {
            Dimension.TIME.key(): pandas.to_datetime([f"2024-01-01T00:{minute:02d}:00" for minute in shuffled_indices]),
            Dimension.LONGITUDE.key(): [float(index) for index in range(len(shuffled_indices))],
            Dimension.LATITUDE.key(): [0.0] * len(shuffled_indices),
            "lead_day": [0] * len(shuffled_indices),
            "error": [float(index) for index in range(len(shuffled_indices))],
            "absolute_error": [float(index) for index in range(len(shuffled_indices))],
        }
    )

    frame = core_visualization._class4_sampled_frame(
        group,
        maximum_points_per_frame=6,
        variable_key=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    )

    assert frame["leadDay"] == 1
    assert frame["totalCount"] == 20
    assert frame["shownCount"] == 6
    assert frame["longitude"] == [4.0, 5.0, 6.0, 14.0, 15.0, 16.0]
    assert frame["error"] == [4.0, 5.0, 6.0, 14.0, 15.0, 16.0]


def test_eddy_contour_payload_uses_shape_preserving_point_budget() -> None:
    from oceanbench.core import visualization as core_visualization

    angles = numpy.linspace(0, 2 * numpy.pi, 200)
    longitudes = 10.0 + numpy.cos(angles)
    latitudes = 45.0 + 0.5 * numpy.sin(angles)

    contour_longitudes, contour_latitudes = core_visualization._simplified_contour_payload(
        longitudes,
        latitudes,
        maximum_points=20,
    )

    assert len(contour_longitudes) == len(contour_latitudes)
    assert len(contour_longitudes) <= 20
    assert contour_longitudes[0] == contour_longitudes[-1]
    assert contour_latitudes[0] == contour_latitudes[-1]


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
                "paths": [
                    {
                        "longitude": [9.0, 10.0, 10.0, 9.0, 9.0],
                        "latitude": [0.0, 0.0, 1.0, 1.0, 0.0],
                    }
                ],
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
    assert "contours are decimated for display" in html_output.data
    assert "Matched eddy" in html_output.data
    assert "Spurious challenger" in html_output.data
    assert "Missed reference" in html_output.data
    assert "Matched center offset" in html_output.data
    assert "Match displacement" not in html_output.data
    assert "ob-eddy-tooltip" in html_output.data
    assert "ob-eddy-polarity-buttons" in html_output.data
    assert "landMask.paths" in html_output.data
    assert "landMask.land" not in html_output.data
    assert "fill(&quot;evenodd&quot;)" in html_output.data
    assert "ob-eddy-zoom" in html_output.data
    assert "createMapViewport" in html_output.data
    assert "createMapBackgroundCache" in html_output.data
    assert "backgroundCache.draw(context, viewport.cacheKey())" in html_output.data
    assert "viewport.longitudeShiftsForPath(path)" in html_output.data
    assert "projection.xUnwrapped" in html_output.data
    assert "viewport.unwrappedLongitudes(candidate.contourLongitude" in html_output.data
    assert "viewport.nearestLongitudeCopy" in html_output.data
    assert "segmentTouchesCanvas" in html_output.data
    assert "Map zoom controls" in html_output.data
    assert "Zoom in" in html_output.data
    assert "Zoom out" in html_output.data
    assert "Reset zoom" in html_output.data
    assert "viewBounds" in html_output.data
    assert "Cyclones" in html_output.data
    assert "Anticyclones" in html_output.data
    assert "Cyclone contour" in html_output.data
    assert "Anticyclone contour" in html_output.data
    assert "Cyclones M/S/M" in html_output.data
    assert "Matched challenger" in html_output.data
    assert "Matched reference" in html_output.data
    assert "drawEddyHoverFocus" in html_output.data
    assert "drawPolarityMarker" not in html_output.data
    assert "ob-eddy-marker" not in html_output.data
    assert "pointerdown" in html_output.data
    assert "dragging" in html_output.data
    assert "constrainedSpan" in html_output.data
    assert "setLineDash" in html_output.data
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
                "paths": [
                    {
                        "longitude": [9.0, 10.0, 10.0, 9.0, 9.0],
                        "latitude": [0.0, 0.0, 1.0, 1.0, 0.0],
                    }
                ],
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
    assert "SLA points are sampled along observed satellite tracks for display" in html_output.data
    assert "metrics use all observations" in html_output.data
    assert "ob-class4-tooltip" in html_output.data
    assert "mask.paths" in html_output.data
    assert "mask.land" not in html_output.data
    assert "fill(&quot;evenodd&quot;)" in html_output.data
    assert "ob-class4-zoom" in html_output.data
    assert "createMapViewport" in html_output.data
    assert "createMapBackgroundCache" in html_output.data
    assert "backgroundCache.draw(context, viewport.cacheKey())" in html_output.data
    assert "viewport.longitudeShiftsForPath(path)" in html_output.data
    assert "project.xUnwrapped" in html_output.data
    assert "Map zoom controls" in html_output.data
    assert "Zoom in" in html_output.data
    assert "Zoom out" in html_output.data
    assert "Reset zoom" in html_output.data
    assert "viewBounds" in html_output.data
    assert "Absolute error" in html_output.data
    assert "Observation density" not in html_output.data
    assert "Signed error" in html_output.data
    assert "pointerdown" in html_output.data
    assert "dragging" in html_output.data
    assert "constrainedSpan" in html_output.data
    assert "robust 95%" in html_output.data


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

    assert notebook.cells[0].cell_type == "markdown"
    assert notebook.cells[0].source.startswith("### Report guide")
    assert "Report guide" in all_sources
    assert "Score tables provide the quantitative OceanBench evaluation." in all_sources
    assert "Interactive figures help diagnose the scores" in all_sources
    assert "metric scores are computed from the underlying datasets" in all_sources
    assert "from oceanbench.core.evaluation_report import prepare_evaluation_report" in all_sources
    assert "evaluation_report = prepare_evaluation_report(challenger_dataset, region=region)" in all_sources
    assert "evaluation_report.glorys_variable_rmsd" in all_sources
    assert "evaluation_report.glorys_mixed_layer_depth_rmsd" in all_sources
    assert "evaluation_report.glorys_geostrophic_current_rmsd" in all_sources
    assert "evaluation_report.glorys_lagrangian_trajectory_deviation" in all_sources
    assert "evaluation_report.glo12_variable_rmsd" in all_sources
    assert "evaluation_report.glo12_mixed_layer_depth_rmsd" in all_sources
    assert "evaluation_report.glo12_geostrophic_current_rmsd" in all_sources
    assert "evaluation_report.glo12_lagrangian_trajectory_deviation" in all_sources
    assert "evaluation_report.class4_observation.rmsd" in all_sources
    assert "evaluation_report.class4_observation_error_explorer" in all_sources
    assert "evaluation_report.lagrangian_trajectory_explorer" in all_sources
    assert "evaluation_report.eddy_matching_explorer" in all_sources
    assert "evaluation_report.forecast_comparison_explorer" in all_sources
    assert "evaluation_report.dynamic_diagnostic_explorer" in all_sources
    assert "evaluation_report.zonal_psd_explorer" in all_sources
    assert "Eddy centers and contours are detected" in all_sources
    assert "wavelength-band scale separation" in all_sources
    assert "Geostrophic currents are masked near the equator" in all_sources
    assert all_sources.index("evaluation_report.glorys_variable_rmsd") < all_sources.index(
        "evaluation_report.forecast_comparison_explorer"
    )
    assert all_sources.index("evaluation_report.glo12_geostrophic_current_rmsd") < all_sources.index(
        "evaluation_report.dynamic_diagnostic_explorer"
    )
    assert all_sources.index("evaluation_report.class4_observation_error_explorer") < all_sources.index(
        "evaluation_report.glorys_lagrangian_trajectory_deviation"
    )
    assert all_sources.index("evaluation_report.lagrangian_trajectory_explorer") < all_sources.index(
        "evaluation_report.eddy_matching_explorer"
    )
    assert all_sources.index("evaluation_report.eddy_matching_explorer") < all_sources.index(
        "evaluation_report.forecast_comparison_explorer"
    )
    assert "regional_challenger_dataset =" not in all_sources
    assert "glorys_dataset =" not in all_sources
    assert "glo12_dataset =" not in all_sources
    assert "challenger_mld_dataset" not in all_sources
    assert "challenger_geostrophic_dataset" not in all_sources
    assert "class4_observation_comparison_dataframe" not in all_sources
    assert "xarray.merge" not in all_sources
    assert "warnings.filterwarnings" not in all_sources
    assert "oceanbench.metrics.rmsd_of_variables_compared_to_observations" not in all_sources
    assert "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis" not in all_sources
    assert "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis" not in all_sources
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
