# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Mesoscale eddy census of a challenger with py-eddy-tracker and the settings of the AVISO META3.2exp NRT atlas.

py-eddy-tracker (https://github.com/AntSimi/py-eddy-tracker, GPL-3.0) is an optional dependency that is not
installed with OceanBench. It is imported only when a census is computed.
"""

from collections import deque
from collections.abc import Iterable, Iterator
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy
import pandas
import xarray

import oceanbench
from oceanbench.core import eddy_census_worker
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable

PY_EDDY_TRACKER_MODULE = "py_eddy_tracker"
PY_EDDY_TRACKER_NAME = "py-eddy-tracker"

ATLAS_HANDBOOK_URL = (
    "https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/hdbk_eddytrajectory_META3.2exp_NRT.pdf"
)
# Detection settings of the META3.2exp NRT atlas (handbook above, sections 3.2.1 and 3.2.2): ADT high-passed
# by removing a first-order Lanczos low-pass of 700 km half-power cutoff (py-eddy-tracker's Bessel high-pass
# of order 1), contours every 0.2 cm, amplitude at least 2 steps (0.4 cm: the handbook prints "<= 0.4cm",
# but the atlas files' amplitude minimum is 0.004 m), shape error at most 70%, one extremum, between 5 and
# 1000 pixels. The handbook's contour range of -100 to 100 cm is not set: py-eddy-tracker spans the field.
HIGH_PASS_WAVELENGTH_KILOMETRES = 700.0
HIGH_PASS_FILTER_ORDER = 1
CONTOUR_STEP_METERS = 0.002
MINIMUM_AMPLITUDE_STEP_COUNT = 2
SHAPE_ERROR_PERCENT = 70.0
MAXIMUM_LOCAL_EXTREMUM_COUNT = 1
MINIMUM_PIXEL_COUNT = 5
MAXIMUM_PIXEL_COUNT = 1000

DETECTION_CELL_SIZE_DEGREES = 0.25
MINIMUM_NATIVE_DETECTION_CELL_SIZE_DEGREES = 0.2
MAXIMUM_CENSUS_CELL_SIZE_DEGREES = 0.3
MINIMUM_OCEAN_FRACTION = 0.5
REGIONAL_CENSUS_MARGIN_DEGREES = 10.0
# Geostrophy and the speed radius are undefined near the equator: same band as the previous detector.
MINIMUM_ABSOLUTE_CENTRE_LATITUDE_DEGREES = 5.0

CYCLONE = "cyclone"
ANTICYCLONE = "anticyclone"
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"
POLARITY_COLUMN = "polarity"
AMPLITUDE_COLUMN = "amplitude"
EFFECTIVE_RADIUS_COLUMN = "effective_radius_km"
SPEED_RADIUS_COLUMN = "speed_radius_km"
CONTOUR_LATITUDES_COLUMN = "contour_latitudes"
CONTOUR_LONGITUDES_COLUMN = "contour_longitudes"
EDDY_COLUMNS = [
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    POLARITY_COLUMN,
    AMPLITUDE_COLUMN,
    EFFECTIVE_RADIUS_COLUMN,
    SPEED_RADIUS_COLUMN,
    CONTOUR_LATITUDES_COLUMN,
    CONTOUR_LONGITUDES_COLUMN,
]

PY_EDDY_TRACKER_INSTALL_HINT = (
    "The eddy census needs py-eddy-tracker, which is not installed: see 'Eddies' in the viewer data "
    "documentation (https://oceanbench.edito.eu/documentation/viewer-data/) for how to install it."
)

# Fields are detected in a pool of processes, each limited to one thread. A worker peaks at about 1.5 GB
# on a global 1/4 degree field (a 1/12 degree field is detected on its 1/4 degree block average), so the
# default stays at 8 workers (about 12 GB), fewer when the memory left to this process is short.
DEFAULT_MAXIMUM_CENSUS_WORKERS = 8
CENSUS_WORKER_MEMORY_BYTES = 1.5e9
_SINGLE_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    # py-eddy-tracker uses OpenCV, whose thread pool otherwise takes every CPU.
    "OPENCV_FOR_THREADS_NUM",
)

_WORKER_MODULE = "oceanbench.core.eddy_census_worker"
_SEA_SURFACE_HEIGHT_NAME = "sea_surface_height"
_METERS_PER_KILOMETRE = 1000.0
_CELL_SIZE_TOLERANCE_DEGREES = 1e-6


def py_eddy_tracker_is_installed() -> bool:
    try:
        return importlib.util.find_spec(PY_EDDY_TRACKER_MODULE) is not None
    except ImportError:
        return False


def py_eddy_tracker_version() -> str:
    import py_eddy_tracker

    return str(getattr(py_eddy_tracker, "__version__", "unknown"))


def detection_parameters() -> dict:
    """The py-eddy-tracker settings of the census, as recorded in the census files."""
    return {
        "method": PY_EDDY_TRACKER_NAME,
        "reference": ATLAS_HANDBOOK_URL,
        "high_pass_wavelength_km": HIGH_PASS_WAVELENGTH_KILOMETRES,
        "high_pass_filter": f"bessel, order {HIGH_PASS_FILTER_ORDER}",
        "contour_step_meters": CONTOUR_STEP_METERS,
        "minimum_amplitude_meters": round(CONTOUR_STEP_METERS * MINIMUM_AMPLITUDE_STEP_COUNT, 6),
        "shape_error_percent": SHAPE_ERROR_PERCENT,
        "maximum_local_extremum_count": MAXIMUM_LOCAL_EXTREMUM_COUNT,
        "pixel_limit": [MINIMUM_PIXEL_COUNT, MAXIMUM_PIXEL_COUNT],
        "detection_grid_cell_degrees": DETECTION_CELL_SIZE_DEGREES,
        "minimum_absolute_centre_latitude_degrees": MINIMUM_ABSOLUTE_CENTRE_LATITUDE_DEGREES,
    }


def _median_spacing(values: numpy.ndarray) -> float:
    return float(numpy.median(numpy.abs(numpy.diff(numpy.asarray(values, dtype=float)))))


def _block_factor(cell_size_degrees: float) -> int | None:
    if cell_size_degrees > MAXIMUM_CENSUS_CELL_SIZE_DEGREES + _CELL_SIZE_TOLERANCE_DEGREES:
        return None
    if cell_size_degrees >= MINIMUM_NATIVE_DETECTION_CELL_SIZE_DEGREES - _CELL_SIZE_TOLERANCE_DEGREES:
        return 1
    return max(1, int(round(DETECTION_CELL_SIZE_DEGREES / cell_size_degrees)))


def has_census_resolution(dataset: xarray.Dataset) -> bool:
    """False on grids coarser than ``MAXIMUM_CENSUS_CELL_SIZE_DEGREES``, whose eddies span a few cells."""
    return _block_factor(_median_spacing(dataset[Dimension.LATITUDE.key()].values)) is not None


def _block_averaged(field: xarray.DataArray, latitude_factor: int, longitude_factor: int) -> xarray.DataArray:
    latitude, longitude = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    blocks = {latitude: latitude_factor, longitude: longitude_factor}
    cell_area = numpy.cos(numpy.deg2rad(field[latitude])) * xarray.ones_like(field)
    ocean_area = cell_area.where(field.notnull(), 0.0).coarsen(blocks, boundary="trim").sum()
    block_area = cell_area.coarsen(blocks, boundary="trim").sum()
    weighted_sum = (field * cell_area).coarsen(blocks, boundary="trim").sum(skipna=True)
    return (weighted_sum / ocean_area).where(ocean_area >= MINIMUM_OCEAN_FRACTION * block_area)


def detection_field(field: xarray.DataArray) -> xarray.DataArray | None:
    """
    A 2D sea surface height field on the census grid of about 1/4 degree, or None on a grid too coarse.

    A grid with cells between ``MINIMUM_NATIVE_DETECTION_CELL_SIZE_DEGREES`` and
    ``MAXIMUM_CENSUS_CELL_SIZE_DEGREES`` is used as is. A finer grid is averaged over blocks of whole cells
    (3 x 3 for 1/12 degree), weighted by cell area and ignoring land; a block less than
    ``MINIMUM_OCEAN_FRACTION`` ocean by area is land. Coordinates are sorted ascending.
    """
    latitude, longitude = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    sorted_field = field.sortby([latitude, longitude])
    latitude_factor = _block_factor(_median_spacing(sorted_field[latitude].values))
    longitude_factor = _block_factor(_median_spacing(sorted_field[longitude].values))
    if latitude_factor is None or longitude_factor is None:
        return None
    if latitude_factor == 1 and longitude_factor == 1:
        return sorted_field
    return _block_averaged(sorted_field, latitude_factor, longitude_factor)


def _surface_sea_surface_height(dataset: xarray.Dataset, first_day_index: int, lead_day_index: int) -> xarray.DataArray:
    field = rename_dataset_with_standard_names(dataset)[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()]
    return (
        field.isel(
            {Dimension.FIRST_DAY_DATETIME.key(): first_day_index, Dimension.LEAD_DAY_INDEX.key(): lead_day_index}
        )
        .squeeze(drop=True)
        .astype("float64")
        .compute()
    )


def _py_eddy_tracker_grid(field: xarray.DataArray):
    from py_eddy_tracker.dataset.grid import RegularGridDataset

    latitude, longitude = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    values = field.transpose(longitude, latitude).values
    return RegularGridDataset.with_array(
        coordinates=(longitude, latitude),
        datas={
            longitude: numpy.asarray(field[longitude].values, dtype="float64"),
            latitude: numpy.asarray(field[latitude].values, dtype="float64"),
            _SEA_SURFACE_HEIGHT_NAME: numpy.ma.masked_invalid(values),
        },
        variables_description={_SEA_SURFACE_HEIGHT_NAME: {"units": "m"}},
        centered=True,
    )


def _wrapped_longitude(longitude: numpy.ndarray) -> numpy.ndarray:
    return (longitude + 180.0) % 360.0 - 180.0


def _eddy_rows(observations, polarity: str) -> pandas.DataFrame:
    centre_longitudes = numpy.asarray(observations.lon, dtype=float)
    wrapped_longitudes = _wrapped_longitude(centre_longitudes)
    if not len(centre_longitudes):
        return pandas.DataFrame(columns=EDDY_COLUMNS)
    contour_longitudes = numpy.asarray(observations.contour_lon_e, dtype=float).reshape(len(centre_longitudes), -1)
    contour_latitudes = numpy.asarray(observations.contour_lat_e, dtype=float).reshape(len(centre_longitudes), -1)
    shifted_contours = contour_longitudes + (wrapped_longitudes - centre_longitudes)[:, None]
    return pandas.DataFrame(
        {
            LATITUDE_COLUMN: numpy.asarray(observations.lat, dtype=float),
            LONGITUDE_COLUMN: wrapped_longitudes,
            POLARITY_COLUMN: polarity,
            AMPLITUDE_COLUMN: numpy.abs(numpy.asarray(observations.amplitude, dtype=float)),
            EFFECTIVE_RADIUS_COLUMN: numpy.asarray(observations.radius_e, dtype=float) / _METERS_PER_KILOMETRE,
            SPEED_RADIUS_COLUMN: numpy.asarray(observations.radius_s, dtype=float) / _METERS_PER_KILOMETRE,
            CONTOUR_LATITUDES_COLUMN: list(contour_latitudes),
            CONTOUR_LONGITUDES_COLUMN: list(shifted_contours),
        },
        columns=EDDY_COLUMNS,
    )


def _has_relief(field: xarray.DataArray) -> bool:
    return bool(numpy.nanmax(field.values) - numpy.nanmin(field.values) >= CONTOUR_STEP_METERS)


def _outside_equatorial_band(eddies: pandas.DataFrame) -> pandas.DataFrame:
    keep = eddies[LATITUDE_COLUMN].abs() >= MINIMUM_ABSOLUTE_CENTRE_LATITUDE_DEGREES
    return eddies[keep].reset_index(drop=True)


def detect_eddies_in_field(field: xarray.DataArray, date: datetime) -> pandas.DataFrame:
    """
    Eddies of one 2D sea surface height field (metres) on a regular grid, detected by py-eddy-tracker.

    The field is put on the census grid first (see ``detection_field``); a grid coarser than
    ``MAXIMUM_CENSUS_CELL_SIZE_DEGREES`` gives no eddies. Model sea surface height stands in for the
    absolute dynamic topography of the atlas: the 700 km high-pass removes the large-scale offset between
    a model's dynamic sea level and ADT. Anticyclones are sea level highs and cyclones lows, in both
    hemispheres. Eddies centred within ``MINIMUM_ABSOLUTE_CENTRE_LATITUDE_DEGREES`` of the equator are
    dropped. Returns one row per eddy with ``EDDY_COLUMNS``: centre latitude and longitude (wrapped
    into [-180, 180)), polarity, amplitude (m), effective and speed radii (km) and the effective contour,
    whose longitudes are continuous around the centre and may overhang the dateline.
    """
    grid_field = detection_field(field)
    if grid_field is None:
        return pandas.DataFrame(columns=EDDY_COLUMNS)
    if not _has_relief(grid_field):
        return pandas.DataFrame(columns=EDDY_COLUMNS)
    grid = _py_eddy_tracker_grid(grid_field)
    grid.bessel_high_filter(_SEA_SURFACE_HEIGHT_NAME, HIGH_PASS_WAVELENGTH_KILOMETRES, order=HIGH_PASS_FILTER_ORDER)
    grid.add_uv(_SEA_SURFACE_HEIGHT_NAME)
    anticyclones, cyclones = grid.eddy_identification(
        _SEA_SURFACE_HEIGHT_NAME,
        "u",
        "v",
        date,
        step=CONTOUR_STEP_METERS,
        shape_error=SHAPE_ERROR_PERCENT,
        pixel_limit=(MINIMUM_PIXEL_COUNT, MAXIMUM_PIXEL_COUNT),
        precision=None,
        mle=MAXIMUM_LOCAL_EXTREMUM_COUNT,
        nb_step_min=MINIMUM_AMPLITUDE_STEP_COUNT,
    )
    rows = [table for table in (_eddy_rows(cyclones, CYCLONE), _eddy_rows(anticyclones, ANTICYCLONE)) if len(table)]
    eddies = (pandas.concat(rows, ignore_index=True) if rows else pandas.DataFrame(columns=EDDY_COLUMNS)).astype(
        {LATITUDE_COLUMN: float, LONGITUDE_COLUMN: float}
    )
    return _outside_equatorial_band(eddies)


def detect_eddies(dataset: xarray.Dataset, first_day_index: int, lead_day_index: int) -> pandas.DataFrame:
    """Eddies of one forecast start and lead day of a challenger dataset (see ``detect_eddies_in_field``)."""
    first_day = pandas.Timestamp(dataset[Dimension.FIRST_DAY_DATETIME.key()].values[first_day_index])
    date = (first_day + pandas.Timedelta(days=lead_day_index)).to_pydatetime()
    return detect_eddies_in_field(_surface_sea_surface_height(dataset, first_day_index, lead_day_index), date)


def detection_grid_description(dataset: xarray.Dataset) -> dict | None:
    """Cell size and block factor of the census grid of a challenger, for the census metadata."""
    latitude_spacing = _median_spacing(dataset[Dimension.LATITUDE.key()].values)
    longitude_spacing = _median_spacing(dataset[Dimension.LONGITUDE.key()].values)
    latitude_factor, longitude_factor = _block_factor(latitude_spacing), _block_factor(longitude_spacing)
    if latitude_factor is None or longitude_factor is None:
        return None
    return {
        "native_cell_degrees": [round(latitude_spacing, 6), round(longitude_spacing, 6)],
        "block_average_factor": [latitude_factor, longitude_factor],
        "cell_degrees": [round(latitude_spacing * latitude_factor, 6), round(longitude_spacing * longitude_factor, 6)],
        "block_average": "area-weighted mean of ocean cells; land where less than half of the block area is ocean",
    }


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
_CGROUP_V1_MEMORY_ROOT = Path("/sys/fs/cgroup/memory")


def _cgroup_memory_files() -> list[tuple[Path, Path]]:
    """(limit, usage) files of the memory cgroup of this process, its own directory first, then the mount root."""
    files = []
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as cgroup_file:
            lines = cgroup_file.read().splitlines()
    except OSError:
        return files
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        relative_path = path.lstrip("/")
        if controllers == "":
            # In a container the cgroup of the process is the mount root.
            for directory in (_CGROUP_V2_ROOT / relative_path, _CGROUP_V2_ROOT):
                files.append((directory / "memory.max", directory / "memory.current"))
        elif "memory" in controllers.split(","):
            for directory in (_CGROUP_V1_MEMORY_ROOT / relative_path, _CGROUP_V1_MEMORY_ROOT):
                files.append((directory / "memory.limit_in_bytes", directory / "memory.usage_in_bytes"))
    return files


def _cgroup_memory_headroom_bytes() -> float | None:
    """Memory limit minus usage of the cgroup of this process, None without a limit."""
    for limit_file, usage_file in _cgroup_memory_files():
        try:
            limit = limit_file.read_text().strip()
            usage = float(usage_file.read_text().strip())
        except (OSError, ValueError):
            continue
        if limit == "max" or float(limit) >= 2**60:
            return None
        return float(limit) - usage
    return None


def _system_available_memory_bytes() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _available_memory_bytes() -> float | None:
    """Memory this process can still use: the least of its cgroup headroom and the system available memory."""
    values = [
        value for value in (_cgroup_memory_headroom_bytes(), _system_available_memory_bytes()) if value is not None
    ]
    return min(values) if values else None


def census_max_workers() -> int:
    """
    Number of processes of the census: ``OCEANBENCH_EDDY_CENSUS_MAX_WORKERS`` if set, otherwise
    ``DEFAULT_MAXIMUM_CENSUS_WORKERS`` capped at the CPUs available to this process and at the number of
    ``CENSUS_WORKER_MEMORY_BYTES`` workers that fit in the memory it can still use (cgroup limit or system).
    """
    raw_value = os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_EDDY_CENSUS_MAX_WORKERS.value)
    if raw_value is None or raw_value.strip() == "":
        workers = min(DEFAULT_MAXIMUM_CENSUS_WORKERS, _available_cpu_count())
        available_memory = _available_memory_bytes()
        if available_memory is not None:
            workers = min(workers, int(available_memory // CENSUS_WORKER_MEMORY_BYTES))
        return max(1, workers)
    max_workers = int(raw_value)
    if max_workers < 1:
        raise ValueError(
            f"{OceanbenchEnvironmentVariable.OCEANBENCH_EDDY_CENSUS_MAX_WORKERS.value} must be at least 1."
        )
    return max_workers


def _worker_environment() -> dict:
    package_root = str(Path(oceanbench.__file__).resolve().parent.parent)
    python_path = os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))
    return {
        **os.environ,
        **{name: "1" for name in _SINGLE_THREAD_ENVIRONMENT_VARIABLES},
        "PYTHONPATH": python_path,
    }


def _start_worker(environment: dict) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-P", "-m", _WORKER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=environment,
        text=True,
        bufsize=1,
    )


def _stop_worker(worker: subprocess.Popen) -> None:
    if worker.stdin and not worker.stdin.closed:
        try:
            worker.stdin.close()
        except OSError:
            pass
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
    if worker.stdout:
        worker.stdout.close()


def _submit(worker: subprocess.Popen, task_path: str, field: xarray.DataArray, date: datetime) -> None:
    with open(task_path, "wb") as task_file:
        pickle.dump((field, date), task_file, protocol=pickle.HIGHEST_PROTOCOL)
    worker.stdin.write(task_path + "\n")
    worker.stdin.flush()


def _collect(worker: subprocess.Popen, task_path: str, date: datetime) -> pandas.DataFrame:
    status = worker.stdout.readline().strip()
    if status not in (eddy_census_worker.OK, eddy_census_worker.ERROR):
        raise RuntimeError(f"The eddy census worker stopped (exit code {worker.poll()}) on the field of {date}.")
    result_path = task_path + eddy_census_worker.RESULT_SUFFIX
    with open(result_path, "rb") as result_file:
        result = pickle.load(result_file)
    os.remove(task_path)
    os.remove(result_path)
    if status == eddy_census_worker.ERROR:
        raise RuntimeError(f"The eddy census failed on the field of {date}: {result}")
    return result


def detect_eddies_in_fields(
    fields: Iterable[tuple[xarray.DataArray, datetime]], max_workers: int | None = None
) -> Iterator[pandas.DataFrame]:
    """
    ``detect_eddies_in_field`` on each ``(field, date)``, yielded in input order.

    With more than one worker the fields are detected in worker processes started as
    ``python -m oceanbench.core.eddy_census_worker``, each limited to one thread. They never import the
    calling script, so a script without an ``if __name__ == "__main__"`` guard works. Fields go through
    temporary files; they are read lazily and at most two per worker are in flight, which bounds memory.
    The results are the same as with one worker.
    """
    workers = census_max_workers() if max_workers is None else max_workers
    if workers <= 1 or not sys.executable:
        for field, date in fields:
            yield detect_eddies_in_field(field, date)
        return
    environment = _worker_environment()
    with tempfile.TemporaryDirectory(prefix="oceanbench-eddy-census-") as directory:
        processes = []
        try:
            processes = [_start_worker(environment) for _ in range(workers)]
            pending = deque()
            for index, (field, date) in enumerate(fields):
                worker = processes[index % workers]
                task_path = os.path.join(directory, f"field-{index}.pickle")
                _submit(worker, task_path, field, date)
                pending.append((worker, task_path, date))
                if len(pending) >= 2 * workers:
                    yield _collect(*pending.popleft())
            while pending:
                yield _collect(*pending.popleft())
        finally:
            for worker in processes:
                _stop_worker(worker)


def challenger_fields(dataset: xarray.Dataset) -> Iterator[tuple[int, int, xarray.DataArray, datetime]]:
    """``(start_index, lead_day_index, field, date)`` of every forecast start and lead day, start by start."""
    for start_index, first_day_value in enumerate(dataset[Dimension.FIRST_DAY_DATETIME.key()].values):
        first_day = pandas.Timestamp(first_day_value)
        for lead_day_index in range(dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]):
            date = (first_day + pandas.Timedelta(days=lead_day_index)).to_pydatetime()
            yield start_index, lead_day_index, _surface_sea_surface_height(dataset, start_index, lead_day_index), date
