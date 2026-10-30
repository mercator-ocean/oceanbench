<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Regridding challengers onto the GLO12 grid

OceanBench scores challengers on regular latitude-longitude grids at 1, 1/4 or 1/12 degree.
These scripts bring a model on another grid (curvilinear, rotated, or regular at another spacing)
onto the GLO12 1/12 degree grid and its 50 depth levels, so that it can be scored against GLO12.
They are not part of the installed package.

The method follows the operational Mercator regridder (pyregrid): SCRIP bilinear weights
(Jones, 1999), land corners dropped and the remaining weights renormalised, and linear vertical interpolation.
It is built only on public libraries.

## Pipeline

| Step | Script | Model specific |
|---|---|---|
| 1 | `croco_to_standard.py` | yes (CROCO) |
| 2 | `regrid_to_glo12_grid.py` | no |
| 3 | `to_challenger_format.py` | no |

Each script reads one NetCDF file and writes one, and also exposes a Python function.

### 1. Model output to standard variables on the native grid

`croco_to_standard_dataset` turns a CROCO history file into daily `thetao`, `so`, `uo`, `vo` and `zos`:

- time decoded as seconds since 2000-01-01, the CROCO default
- land (`mask_rho == 0`, stored as 0 by CROCO) set to NaN
- depth of every s-level from the Vtransform 2 equation, using the free surface of each snapshot
- linear interpolation onto the 50 GLO12 depths: the top value is copied above the shallowest level,
  and values below the deepest level are NaN (no pit filling)
- `u` and `v` averaged from cell faces to cell centres, keeping the zero velocity of land faces,
  then rotated by `angle` into eastward and northward components
- daily means by the trapezoidal rule over snapshots from 00:00 to 00:00 of the next day
- outermost rows and columns dropped
- each output variable keeps the floating point precision of the CROCO variable it comes from

Another model needs its own version of this step, producing the same variables and coordinates.

### 2. Horizontal regridding

`regrid_to_glo12_grid` takes any dataset with 2-D `latitude` and `longitude` and puts it on the GLO12 points
inside its latitude-longitude range:

- pyresample finds the four source points around each target point, on the sphere
- the fractional position inside that quadrilateral is solved by Newton iteration;
  the fractional distances of pyresample alone are off by up to one source cell when a target point lies
  on a source cell edge, for example a source column on a GLO12 meridian
- corners that are NaN (land, below the seabed, outside the domain) are dropped and the remaining weights
  renormalised, so a target point is NaN only when all four corners are
- target coordinates are computed in float32 as GLO12 stores them, `float32(first) + float32(k) * float32(1/12)`,
  and are bit-identical to the GLO12 coordinates, which OceanBench matches within 1e-4 degree

### 3. Challenger format

`to_challenger_format` turns the daily `time` axis of one forecast into `lead_day_index` and `first_day_datetime`.

## Running it

The scripts need `pyresample`, from the `regridding` Poetry group (installed by `poetry install`).

```sh
python croco_to_standard.py croco_his.nc croco_standard.nc
python regrid_to_glo12_grid.py croco_standard.nc croco_glo12_grid.nc
```

Then, in Python, with a region that leaves out the sponge layer of the regional model:

```python
import oceanbench
import xarray
from oceanbench.regions import load_region_file
from to_challenger_format import to_challenger_format

challenger = to_challenger_format(xarray.open_dataset("croco_glo12_grid.nc"))
oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(
    challenger, region=load_region_file("agulhas_region.json")
)
```

## Agulhas test case

The scripts were developed on a 5-day CROCO forecast of the Agulhas region (SEA-FORWARD, from 2026-10-21):
a curvilinear 159 by 99 grid of about 1/12 degree, rotated by up to 4 degrees, with 50 s-levels.

- a linear field of latitude and longitude is reproduced to within float32 precision on every target cell
- no regridded value leaves the range of the source values, and the domain-mean sea surface temperature changes
  by 0.02 degrees C at most
- no target point whose nearest CROCO point is ocean is left empty
- the RMSD against GLO12 over `agulhas_region.json` runs with 0.4 percent of ocean cells missing at most

`agulhas_region.json` leaves out about 10 grid points along the open boundaries, where the damped small-scale
variability of the model shows its sponge layer. The width is estimated from the model output and still
needs confirming against the model configuration.

## Not covered yet

- comparison against pyregrid on the same input
- conservative remapping, needed when a challenger grid is finer than 1/12 degree
- filling coastal GLO12 ocean points that the challenger leaves empty (the pyregrid drowning step)
- processing a global 1/12 degree file in depth chunks to bound memory
