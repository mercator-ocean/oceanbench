# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import json
import os
import subprocess
import sys
import types

import numpy
import pandas
import pytest
import xarray

from oceanbench.core import eddy_census
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.publish.viewer_artifacts import _eddy_frame as _viewer_eddy_frame, encoded_eddy_frame

_LATITUDE = Dimension.LATITUDE.key()
_LONGITUDE = Dimension.LONGITUDE.key()
_KILOMETRES_PER_DEGREE = 111.195


def _field(latitudes: numpy.ndarray, longitudes: numpy.ndarray, values: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(values, coords={_LATITUDE: latitudes, _LONGITUDE: longitudes}, dims=(_LATITUDE, _LONGITUDE))


def _regular_field(cell_size_degrees: float, latitude_count: int = 12, longitude_count: int = 12) -> xarray.DataArray:
    latitudes = 30.0 + cell_size_degrees * (numpy.arange(latitude_count) + 0.5)
    longitudes = -10.0 + cell_size_degrees * (numpy.arange(longitude_count) + 0.5)
    values = numpy.arange(latitude_count * longitude_count, dtype=float).reshape(latitude_count, longitude_count)
    return _field(latitudes, longitudes, values)


def test_a_native_quarter_degree_grid_is_used_as_is() -> None:
    for cell_size in (0.25, 0.2, 0.3):
        field = _regular_field(cell_size)
        xarray.testing.assert_identical(eddy_census.detection_field(field), field)


def test_a_grid_coarser_than_the_census_limit_gets_no_census() -> None:
    field = _regular_field(0.5)
    assert eddy_census.detection_field(field) is None
    assert eddy_census.detection_grid_description(field.to_dataset(name="zos")) is None
    assert not eddy_census.has_census_resolution(field.to_dataset(name="zos"))
    assert eddy_census.has_census_resolution(_regular_field(0.25).to_dataset(name="zos"))


def test_a_twelfth_degree_grid_is_averaged_over_three_by_three_blocks() -> None:
    field = _regular_field(1 / 12, latitude_count=2041 // 10 * 3 + 1, longitude_count=4320 // 10)
    regridded = eddy_census.detection_field(field)

    assert regridded.shape == (field.shape[0] // 3, field.shape[1] // 3)
    numpy.testing.assert_allclose(numpy.diff(regridded[_LATITUDE].values), 0.25)
    numpy.testing.assert_allclose(numpy.diff(regridded[_LONGITUDE].values), 0.25)
    numpy.testing.assert_allclose(regridded[_LATITUDE].values[0], field[_LATITUDE].values[1])
    description = eddy_census.detection_grid_description(field.to_dataset(name="zos"))
    assert description["block_average_factor"] == [3, 3]
    assert description["cell_degrees"] == pytest.approx([0.25, 0.25])


def test_block_averages_are_weighted_by_cell_area() -> None:
    latitudes = numpy.array([0.0, 30.0, 60.0])
    longitudes = numpy.array([0.0, 0.1, 0.2])
    values = numpy.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    averaged = eddy_census._block_averaged(_field(latitudes, longitudes, values), 3, 3)

    weights = numpy.cos(numpy.deg2rad(latitudes))
    assert float(averaged.item()) == pytest.approx(float((weights * values[:, 0]).sum() / weights.sum()))
    assert float(averaged.item()) < 2.0


def test_land_cells_are_ignored_and_a_block_less_than_half_ocean_is_land() -> None:
    latitudes = numpy.arange(6) * 0.01
    longitudes = numpy.arange(3) * 0.01
    values = numpy.full((6, 3), numpy.nan)
    values[0, :] = 1.0
    values[1, 0] = 1.0
    values[3:5, :] = 2.0
    values[5, :2] = 2.0
    averaged = eddy_census._block_averaged(_field(latitudes, longitudes, values), 3, 3)

    assert numpy.isnan(averaged.values[0, 0])
    assert averaged.values[1, 0] == pytest.approx(2.0)


def test_descending_latitudes_are_sorted_before_detection() -> None:
    field = _regular_field(0.25)
    flipped = field.isel({_LATITUDE: slice(None, None, -1)})
    xarray.testing.assert_identical(eddy_census.detection_field(flipped), field)


class _FakeObservations:
    def __init__(self, longitudes, latitudes, amplitudes, contour_longitudes, contour_latitudes):
        self.lon = numpy.array(longitudes)
        self.lat = numpy.array(latitudes)
        self.amplitude = numpy.array(amplitudes)
        self.radius_e = numpy.full(len(longitudes), 80_000.0)
        self.radius_s = numpy.full(len(longitudes), 50_000.0)
        self.contour_lon_e = numpy.array(contour_longitudes)
        self.contour_lat_e = numpy.array(contour_latitudes)


class _FakeGrid:
    calls: list = []
    detections = None

    @classmethod
    def with_array(cls, coordinates, datas, variables_description=None, **keywords):
        cls.calls.append(("with_array", coordinates, sorted(datas), variables_description, keywords))
        return cls()

    def bessel_high_filter(self, grid_name, wave_length, order=1):
        self.calls.append(("bessel_high_filter", wave_length, order))

    def add_uv(self, grid_name):
        self.calls.append(("add_uv",))

    def eddy_identification(self, grid_name, u_name, v_name, date, **keywords):
        self.calls.append(("eddy_identification", date, keywords))
        if self.detections is not None:
            return self.detections
        anticyclones = _FakeObservations([359.9], [30.0], [0.05], [[359.5, 360.3, 359.9]], [[29.8, 30.0, 30.2]])
        cyclones = _FakeObservations(
            [10.0, 200.0], [-40.0, 20.0], [0.02, 0.01], [[9.8, 10.2, 10.0], [199.8, 200.2, 200.0]], [[0.0] * 3] * 2
        )
        return anticyclones, cyclones


@pytest.fixture
def fake_py_eddy_tracker(monkeypatch):
    _FakeGrid.calls = []
    _FakeGrid.detections = None
    package = types.ModuleType("py_eddy_tracker")
    package.__version__ = "0+fake"
    dataset_package = types.ModuleType("py_eddy_tracker.dataset")
    grid_module = types.ModuleType("py_eddy_tracker.dataset.grid")
    grid_module.RegularGridDataset = _FakeGrid
    for module in (package, dataset_package, grid_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return _FakeGrid


def test_py_eddy_tracker_is_called_with_the_atlas_settings(fake_py_eddy_tracker) -> None:
    eddy_census.detect_eddies_in_field(_regular_field(0.25), datetime(2024, 1, 3))

    calls = {call[0]: call[1:] for call in fake_py_eddy_tracker.calls}
    assert calls["with_array"][3]["centered"] is True
    assert calls["bessel_high_filter"] == (700.0, 1)
    assert calls["eddy_identification"][1] == {
        "step": 0.002,
        "shape_error": 70.0,
        "pixel_limit": (5, 1000),
        "precision": None,
        "mle": 1,
        "nb_step_min": 2,
    }
    assert eddy_census.detection_parameters()["minimum_amplitude_meters"] == 0.004
    assert eddy_census.py_eddy_tracker_version() == "0+fake"


def test_detected_eddies_follow_the_census_conventions(fake_py_eddy_tracker) -> None:
    eddies = eddy_census.detect_eddies_in_field(_regular_field(0.25), datetime(2024, 1, 3))

    assert list(eddies.columns) == eddy_census.EDDY_COLUMNS
    assert list(eddies[eddy_census.POLARITY_COLUMN]) == ["cyclone", "cyclone", "anticyclone"]
    numpy.testing.assert_allclose(eddies[eddy_census.LONGITUDE_COLUMN], [10.0, -160.0, -0.1])
    numpy.testing.assert_allclose(eddies[eddy_census.EFFECTIVE_RADIUS_COLUMN], 80.0)
    numpy.testing.assert_allclose(eddies[eddy_census.SPEED_RADIUS_COLUMN], 50.0)
    numpy.testing.assert_allclose(eddies[eddy_census.AMPLITUDE_COLUMN], [0.02, 0.01, 0.05])
    numpy.testing.assert_allclose(eddies[eddy_census.CONTOUR_LONGITUDES_COLUMN].iloc[2], [-0.5, 0.3, -0.1])
    numpy.testing.assert_allclose(eddies[eddy_census.CONTOUR_LONGITUDES_COLUMN].iloc[1], [-160.2, -159.8, -160.0])


def test_eddies_centred_near_the_equator_are_dropped() -> None:
    eddies = pandas.DataFrame({eddy_census.LATITUDE_COLUMN: [2.0, -2.0, 8.0, -8.0, 5.0]})

    kept = eddy_census._outside_equatorial_band(eddies)

    assert list(kept[eddy_census.LATITUDE_COLUMN]) == [8.0, -8.0, 5.0]
    assert eddy_census.detection_parameters()["minimum_absolute_centre_latitude_degrees"] == 5.0


def _no_observations():
    return _FakeObservations([], [], [], numpy.empty((0, 3)), numpy.empty((0, 3)))


def _assert_valid_empty_or_single_polarity(eddies, polarities: list[str]) -> None:
    assert list(eddies.columns) == eddy_census.EDDY_COLUMNS
    assert list(eddies[eddy_census.POLARITY_COLUMN]) == polarities
    frame = encoded_eddy_frame(_viewer_eddy_frame(eddies, 0, None))
    assert frame["count"] == len(polarities)


def test_a_field_without_anticyclones_gives_its_cyclones(fake_py_eddy_tracker) -> None:
    fake_py_eddy_tracker.detections = (
        _no_observations(),
        _FakeObservations([10.0], [-40.0], [0.02], [[9.8, 10.2, 10.0]], [[-40.0] * 3]),
    )
    eddies = eddy_census.detect_eddies_in_field(_regular_field(0.25), datetime(2024, 1, 3))
    _assert_valid_empty_or_single_polarity(eddies, ["cyclone"])


def test_a_field_without_eddies_gives_an_empty_census(fake_py_eddy_tracker) -> None:
    fake_py_eddy_tracker.detections = (_no_observations(), _no_observations())
    eddies = eddy_census.detect_eddies_in_field(_regular_field(0.25), datetime(2024, 1, 3))
    _assert_valid_empty_or_single_polarity(eddies, [])


def test_a_field_whose_eddies_are_all_equatorial_gives_an_empty_census(fake_py_eddy_tracker) -> None:
    fake_py_eddy_tracker.detections = (
        _FakeObservations([20.0], [3.0], [0.05], [[19.8, 20.2, 20.0]], [[3.0] * 3]),
        _FakeObservations([30.0], [-1.0], [0.02], [[29.8, 30.2, 30.0]], [[-1.0] * 3]),
    )
    eddies = eddy_census.detect_eddies_in_field(_regular_field(0.25), datetime(2024, 1, 3))
    _assert_valid_empty_or_single_polarity(eddies, [])


def test_a_coarse_field_never_reaches_py_eddy_tracker(fake_py_eddy_tracker) -> None:
    eddies = eddy_census.detect_eddies_in_field(_regular_field(1.0), datetime(2024, 1, 3))

    assert eddies.empty
    assert list(eddies.columns) == eddy_census.EDDY_COLUMNS
    assert fake_py_eddy_tracker.calls == []


def test_detect_eddies_selects_the_start_and_lead_day(fake_py_eddy_tracker) -> None:
    field = _regular_field(0.25)
    values = numpy.broadcast_to(field.values, (2, 3, *field.shape))
    dataset = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                ["first_day_datetime", "lead_day_index", _LATITUDE, _LONGITUDE],
                values,
            )
        },
        coords={
            "first_day_datetime": numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
            "lead_day_index": numpy.arange(3),
            _LATITUDE: field[_LATITUDE].values,
            _LONGITUDE: field[_LONGITUDE].values,
        },
    )
    eddy_census.detect_eddies(dataset, 1, 2)

    assert fake_py_eddy_tracker.calls[-1][1] == datetime(2024, 1, 12)


# The tests below run py-eddy-tracker itself on planted Gaussian eddies.
_PLANTED_EDDIES = [
    (30.0, 180.0, 0.2),
    (-35.0, 60.0, -0.15),
    (25.0, -100.0, 0.1),
    (-30.0, -20.0, 0.12),
    (40.0, 150.0, -0.2),
]
_PLANTED_RADIUS_KM = 60.0


def _planted_field(first_longitude: float, cell_size_degrees: float = 0.25) -> xarray.DataArray:
    latitudes = numpy.arange(-50.0 + cell_size_degrees / 2, 50.0, cell_size_degrees)
    longitudes = (
        first_longitude + cell_size_degrees / 2 + numpy.arange(round(360 / cell_size_degrees)) * cell_size_degrees
    )
    grid_latitude, grid_longitude = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    values = numpy.zeros(grid_latitude.shape)
    for latitude, longitude, amplitude in _PLANTED_EDDIES:
        longitude_offset = (grid_longitude - longitude + 180.0) % 360.0 - 180.0
        eastward_km = longitude_offset * _KILOMETRES_PER_DEGREE * numpy.cos(numpy.deg2rad(latitude))
        northward_km = (grid_latitude - latitude) * _KILOMETRES_PER_DEGREE
        values += amplitude * numpy.exp(-(eastward_km**2 + northward_km**2) / (2 * _PLANTED_RADIUS_KM**2))
    return _field(latitudes, longitudes, values)


def _pet_eddies(field: xarray.DataArray):
    pytest.importorskip(eddy_census.PY_EDDY_TRACKER_MODULE)
    return eddy_census.detect_eddies_in_field(field, datetime(2024, 1, 3))


def _sorted_centres(eddies) -> list[tuple]:
    return sorted(
        (polarity, round(latitude, 2), round(longitude, 2))
        for polarity, latitude, longitude in eddies[["polarity", "latitude", "longitude"]].itertuples(index=False)
    )


def test_py_eddy_tracker_finds_the_planted_eddies() -> None:
    eddies = _pet_eddies(_planted_field(-180.0))

    assert len(eddies) == len(_PLANTED_EDDIES)
    for latitude, longitude, amplitude in _PLANTED_EDDIES:
        distance = numpy.hypot(
            eddies["latitude"] - latitude, ((eddies["longitude"] - longitude + 180.0) % 360.0 - 180.0)
        )
        eddy = eddies.loc[distance.idxmin()]
        assert distance.min() < 0.2
        assert eddy["polarity"] == ("anticyclone" if amplitude > 0 else "cyclone")
        assert eddy["amplitude"] == pytest.approx(abs(amplitude), rel=0.15)
        assert eddy["speed_radius_km"] == pytest.approx(_PLANTED_RADIUS_KM, rel=0.1)
        assert eddy["effective_radius_km"] > eddy["speed_radius_km"]
        assert -180.0 <= eddy["longitude"] < 180.0


def test_an_eddy_on_the_dateline_keeps_a_continuous_contour_around_its_centre() -> None:
    eddies = _pet_eddies(_planted_field(0.0))
    dateline_eddy = eddies.loc[(eddies["latitude"] - 30.0).abs().idxmin()]
    contour_longitudes = numpy.asarray(dateline_eddy["contour_longitudes"])

    assert min(abs(dateline_eddy["longitude"] + 180.0), abs(dateline_eddy["longitude"] - 180.0)) < 0.01
    assert numpy.abs(numpy.diff(contour_longitudes)).max() < 1.0
    assert contour_longitudes.min() < dateline_eddy["longitude"] < contour_longitudes.max()
    assert numpy.abs(contour_longitudes - dateline_eddy["longitude"]).max() < 5.0


def test_a_0_to_360_grid_gives_the_census_of_a_minus_180_to_180_grid() -> None:
    assert _sorted_centres(_pet_eddies(_planted_field(0.0))) == _sorted_centres(_pet_eddies(_planted_field(-180.0)))


def test_a_twelfth_degree_field_gives_the_census_of_its_quarter_degree_version() -> None:
    fine = _pet_eddies(_planted_field(-180.0, 1 / 12))
    quarter = _pet_eddies(_planted_field(-180.0))

    assert len(fine) == len(quarter)
    numpy.testing.assert_allclose(sorted(fine["speed_radius_km"]), sorted(quarter["speed_radius_km"]), rtol=0.05)


def test_py_eddy_tracker_drops_a_planted_eddy_at_2_degrees_and_keeps_one_at_8_degrees() -> None:
    latitudes = numpy.arange(-20.0 + 0.125, 20.0, 0.25)
    longitudes = -180.0 + 0.125 + numpy.arange(1440) * 0.25
    grid_latitude, grid_longitude = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    values = numpy.zeros(grid_latitude.shape)
    for latitude, longitude, amplitude in ((2.0, -100.0, 0.15), (8.0, 60.0, 0.15)):
        eastward_km = (grid_longitude - longitude) * _KILOMETRES_PER_DEGREE * numpy.cos(numpy.deg2rad(latitude))
        northward_km = (grid_latitude - latitude) * _KILOMETRES_PER_DEGREE
        values += amplitude * numpy.exp(-(eastward_km**2 + northward_km**2) / (2 * _PLANTED_RADIUS_KM**2))

    eddies = _pet_eddies(_field(latitudes, longitudes, values))

    assert len(eddies) == 1
    assert eddies["polarity"].iloc[0] == "anticyclone"
    assert eddies["latitude"].iloc[0] == pytest.approx(8.0, abs=0.2)
    assert eddies["longitude"].iloc[0] == pytest.approx(60.0, abs=0.2)


def _planted_tropical_field(eddies) -> xarray.DataArray:
    latitudes = numpy.arange(-20.0 + 0.125, 20.0, 0.25)
    longitudes = -180.0 + 0.125 + numpy.arange(1440) * 0.25
    grid_latitude, grid_longitude = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    values = numpy.zeros(grid_latitude.shape)
    for latitude, longitude, amplitude in eddies:
        eastward_km = (grid_longitude - longitude) * _KILOMETRES_PER_DEGREE * numpy.cos(numpy.deg2rad(latitude))
        northward_km = (grid_latitude - latitude) * _KILOMETRES_PER_DEGREE
        values += amplitude * numpy.exp(-(eastward_km**2 + northward_km**2) / (2 * _PLANTED_RADIUS_KM**2))
    return _field(latitudes, longitudes, values)


def test_py_eddy_tracker_census_of_a_field_with_cyclones_only() -> None:
    eddies = _pet_eddies(_planted_tropical_field([(10.0, 60.0, -0.15)]))
    _assert_valid_empty_or_single_polarity(eddies, ["cyclone"])


def test_py_eddy_tracker_census_of_a_flat_field_is_empty() -> None:
    eddies = _pet_eddies(_planted_tropical_field([]))

    _assert_valid_empty_or_single_polarity(eddies, [])


def test_py_eddy_tracker_census_of_a_field_too_weak_for_any_contour_is_empty() -> None:
    eddies = _pet_eddies(_planted_tropical_field([(30.0, 60.0, 0.001)]))
    _assert_valid_empty_or_single_polarity(eddies, [])


def test_py_eddy_tracker_census_of_equatorial_eddies_only_is_empty() -> None:
    eddies = _pet_eddies(_planted_tropical_field([(2.0, -100.0, 0.15), (-3.0, 60.0, -0.15)]))
    _assert_valid_empty_or_single_polarity(eddies, [])


def test_census_workers_default_to_eight_capped_by_the_cpus(monkeypatch) -> None:
    monkeypatch.delenv("OCEANBENCH_EDDY_CENSUS_MAX_WORKERS", raising=False)
    monkeypatch.setattr(eddy_census, "_available_cpu_count", lambda: 3)
    assert eddy_census.census_max_workers() == 3
    monkeypatch.setattr(eddy_census, "_available_cpu_count", lambda: 64)
    assert eddy_census.census_max_workers() == eddy_census.DEFAULT_MAXIMUM_CENSUS_WORKERS == 8


def test_census_workers_follow_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OCEANBENCH_EDDY_CENSUS_MAX_WORKERS", "5")
    assert eddy_census.census_max_workers() == 5
    monkeypatch.setenv("OCEANBENCH_EDDY_CENSUS_MAX_WORKERS", "0")
    with pytest.raises(ValueError, match="OCEANBENCH_EDDY_CENSUS_MAX_WORKERS"):
        eddy_census.census_max_workers()


def test_a_single_worker_detects_in_process_in_input_order(fake_py_eddy_tracker) -> None:
    fields = [(_regular_field(0.25), datetime(2024, 1, day)) for day in (3, 4, 5)]
    results = list(eddy_census.detect_eddies_in_fields(iter(fields), max_workers=1))
    assert len(results) == 3
    dates = [call[1] for call in fake_py_eddy_tracker.calls if call[0] == "eddy_identification"]
    assert dates == [date for _, date in fields]


_FAKE_PY_EDDY_TRACKER_GRID = """
import os

import numpy


class _Observations:
    def __init__(self, longitudes, latitudes, amplitudes):
        self.lon = numpy.array(longitudes, dtype=float)
        self.lat = numpy.array(latitudes, dtype=float)
        self.amplitude = numpy.array(amplitudes, dtype=float)
        self.radius_e = numpy.full(len(self.lon), 80_000.0)
        self.radius_s = numpy.full(len(self.lon), 50_000.0)
        self.contour_lon_e = numpy.stack([self.lon - 0.2, self.lon + 0.2, self.lon])
        self.contour_lat_e = numpy.stack([self.lat, self.lat, self.lat + 0.2])
        self.contour_lon_e, self.contour_lat_e = self.contour_lon_e.T, self.contour_lat_e.T


class RegularGridDataset:
    @classmethod
    def with_array(cls, coordinates, datas, **keywords):
        grid = cls()
        grid.longitudes, grid.latitudes = datas[coordinates[0]], datas[coordinates[1]]
        grid.values = numpy.asarray(datas["sea_surface_height"])
        return grid

    def bessel_high_filter(self, grid_name, wave_length, order=1):
        pass

    def add_uv(self, grid_name):
        pass

    def eddy_identification(self, grid_name, u_name, v_name, date, **keywords):
        if date.year == 1999:
            raise ValueError("planted failure")
        with open(os.environ["FAKE_PY_EDDY_TRACKER_LOG"], "a") as log:
            log.write(f"{os.getpid()} {os.environ.get('OMP_NUM_THREADS')} {date.isoformat()}\\n")
        longitude_index, latitude_index = numpy.unravel_index(numpy.argmax(self.values), self.values.shape)
        longitude, latitude = self.longitudes[longitude_index], self.latitudes[latitude_index]
        anticyclones = _Observations([longitude], [latitude], [self.values.max() / 7.0 + date.day])
        cyclones = _Observations([longitude + 1.0, longitude - 1.0], [latitude, -latitude], [0.01, date.day / 3.0])
        return anticyclones, cyclones
"""


@pytest.fixture
def fake_py_eddy_tracker_on_disk(tmp_path, monkeypatch):
    package = tmp_path / "fake_site" / "py_eddy_tracker"
    (package / "dataset").mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0+fake-on-disk"\n')
    (package / "dataset" / "__init__.py").write_text("")
    (package / "dataset" / "grid.py").write_text(_FAKE_PY_EDDY_TRACKER_GRID)
    log_path = tmp_path / "calls.log"
    site = str(tmp_path / "fake_site")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [site, os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv("FAKE_PY_EDDY_TRACKER_LOG", str(log_path))
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.delenv("OPENCV_FOR_THREADS_NUM", raising=False)
    monkeypatch.syspath_prepend(site)
    for name in [name for name in sys.modules if name.startswith("py_eddy_tracker")]:
        monkeypatch.delitem(sys.modules, name)
    return log_path


def _varied_fields(count: int) -> list:
    fields = []
    for index in range(count):
        field = _regular_field(0.25)
        fields.append((field.copy(data=numpy.roll(field.values, index)), datetime(2024, 1, 1 + index)))
    return fields


def _frame(eddies: pandas.DataFrame) -> dict:
    return _viewer_eddy_frame(eddies, 0, None)


def _logged_calls(log_path) -> list[list[str]]:
    return [line.split() for line in log_path.read_text().splitlines()]


def test_workers_give_byte_identical_results_in_input_order(fake_py_eddy_tracker_on_disk) -> None:
    fields = _varied_fields(7)
    serial = list(eddy_census.detect_eddies_in_fields(iter(fields), max_workers=1))
    parallel = list(eddy_census.detect_eddies_in_fields(iter(fields), max_workers=3))
    assert len(parallel) == 7
    for serial_frame, parallel_frame in zip(serial, parallel):
        pandas.testing.assert_frame_equal(serial_frame, parallel_frame, check_exact=True)
        frames = [encoded_eddy_frame(_frame(frame)) for frame in (serial_frame, parallel_frame)]
        assert json.dumps(frames[0]) == json.dumps(frames[1])
    worker_calls = _logged_calls(fake_py_eddy_tracker_on_disk)[7:]
    assert {int(pid) for pid, _, _ in worker_calls} - {os.getpid()} == {int(pid) for pid, _, _ in worker_calls}
    assert len({pid for pid, _, _ in worker_calls}) == 3


def test_workers_are_single_threaded_and_the_parent_environment_is_untouched(fake_py_eddy_tracker_on_disk) -> None:
    results = eddy_census.detect_eddies_in_fields(iter(_varied_fields(4)), max_workers=2)
    next(results)
    assert os.environ["OMP_NUM_THREADS"] == "16"
    assert "OPENCV_FOR_THREADS_NUM" not in os.environ
    list(results)
    assert {threads for _, threads, _ in _logged_calls(fake_py_eddy_tracker_on_disk)} == {"1"}
    assert os.environ["OMP_NUM_THREADS"] == "16"


def test_a_worker_error_names_the_field(fake_py_eddy_tracker_on_disk) -> None:
    fields = _varied_fields(3) + [(_regular_field(0.25), datetime(1999, 5, 6))]
    with pytest.raises(RuntimeError, match="1999-05-06.*planted failure"):
        list(eddy_census.detect_eddies_in_fields(iter(fields), max_workers=2))


_UNGUARDED_SCRIPT = """
import datetime
import sys

import numpy
import xarray

from oceanbench.core import eddy_census

with open(sys.argv[1], "a") as runs:
    runs.write("script body ran\\n")
latitudes = 30.0 + 0.25 * numpy.arange(12)
longitudes = -10.0 + 0.25 * numpy.arange(12)
fields = [
    (
        xarray.DataArray(
            numpy.roll(numpy.arange(144.0).reshape(12, 12), index),
            coords={"latitude": latitudes, "longitude": longitudes},
            dims=("latitude", "longitude"),
        ),
        datetime.datetime(2024, 1, 1 + index),
    )
    for index in range(6)
]
results = list(eddy_census.detect_eddies_in_fields(fields, max_workers=3))
print("eddies", [len(result) for result in results])
"""


def test_a_script_without_main_guard_runs_its_body_exactly_once(fake_py_eddy_tracker_on_disk, tmp_path) -> None:
    script = tmp_path / "unguarded.py"
    script.write_text(_UNGUARDED_SCRIPT)
    runs = tmp_path / "runs.txt"
    completed = subprocess.run(
        [sys.executable, str(script), str(runs)], capture_output=True, text=True, cwd=tmp_path, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert "eddies [3, 3, 3, 3, 3, 3]" in completed.stdout
    assert runs.read_text().splitlines() == ["script body ran"]
    worker_pids = {pid for pid, _, _ in _logged_calls(fake_py_eddy_tracker_on_disk)}
    assert len(worker_pids) == 3
