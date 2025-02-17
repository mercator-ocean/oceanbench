# Oceanbench

## Usage examples

### RMSE
Plot temporal RMSE for depth 2:
```python
import xarray
import oceanbench

glonet_dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
glorys_dataset = xarray.open_dataset("data/glorys14/2024-01-03.nc")
nparray = oceanbench.evaluate.pointwise_evaluation(
    glonet_datasets=[glonet_dataset],
    glorys_datasets=[glorys_dataset],
)
oceanbench.plot.plot_pointwise_evaluation(rmse_dataarray=nparray, depth=2)
```

Plot temporal RMSE averaged over all depths:
```python
import xarray
import oceanbench

glonet_dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
glorys_dataset = xarray.open_dataset("data/glorys14/2024-01-03.nc")
nparray = oceanbench.evaluate.pointwise_evaluation(
    glonet_datasets=[glonet_dataset],
    glorys_datasets=[glorys_dataset],
)
oceanbench.plot.plot_pointwise_evaluation_for_average_depth(rmse_dataarray=nparray)
```

Plot depth rmses (averaged over all lead time):
```python
import xarray
import oceanbench

glonet_dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
glorys_dataset = xarray.open_dataset("data/glorys14/2024-01-03.nc")
nparray = oceanbench.evaluate.pointwise_evaluation(
    glonet_datasets=[glonet_dataset],
    glorys_datasets=[glorys_dataset],
)
dataset_depth_values = glonet_dataset.depth.values
oceanbench.plot.plot_pointwise_evaluation_depth_for_average_time(rmse_dataarray=nparray, dataset_depth_values=dataset_depth_values)
```

### MLD
```python
import xarray
import oceanbench

dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
dataset = oceanbench.process.calc_mld(dataset=dataset, lead=1)
oceanbench.plot.plot_mld(dataset=dataset)
```

### Geo
```python
import xarray
import oceanbench

dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
dataset = oceanbench.process.calc_geo(dataset=dataset, lead=1, variable="zos")
oceanbench.plot.plot_geo(dataset=dataset)
```

### Density
Get density:
```python
import xarray
import oceanbench

dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
dataarray = oceanbench.process.calc_density(
    dataset=dataset,
    lead=1,
    minimum_longitude=-100,
    maximum_longitude=-40,
    minimum_latitude=-15,
    maximum_latitude=50,
)
oceanbench.plot.plot_density(dataarray=dataarray)
```

### Euclidean distance
```python
from oceanbench.evaluate import get_euclidean_distance
from oceanbench.plot import plot_euclidean_distance

import xarray

glonet_dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
glorys_dataset = xarray.open_dataset("data/glorys14/2024-01-03.nc")

euclidean_distance = get_euclidean_distance(
    first_dataset=glonet_dataset,
    second_dataset=glorys_dataset,
    minimum_latitude=466,
    maximum_latitude=633,
    minimum_longitude=400,
    maximum_longitude=466,
)

plot_euclidean_distance(euclidean_distance)
```

### Energy cascading
```python
import xarray
from oceanbench.evaluate import analyze_energy_cascade
from oceanbench.plot import plot_energy_cascade

glonet = xarray.open_dataset("data/glonet/2024-01-03.nc")
_, gglonet_sc = analyze_energy_cascade(glonet, "uo", 0, 1 / 4)

plot_energy_cascade(gglonet_sc)
```

### Kinetic energy
```python
import xarray

from oceanbench.plot import plot_kinetic_energy

glonet = xarray.open_dataset("data/glonet/2024-01-03.nc")
plot_kinetic_energy(glonet)
```

### Vorticity
```python
import xarray

from oceanbench.plot import plot_vortocity

glonet = xarray.open_dataset("data/glonet/2024-01-03.nc")
plot_vortocity(glonet)
```

### Mass conservation
```python
import xarray

from oceanbench.process import mass_conservation

glonet = xarray.open_dataset("data/glonet/2024-01-03.nc")
mean_div_time_series = mass_conservation(glonet, 0, deg_resolution=0.25)  # should be close to zero
print(mean_div_time_series.data)  #time-dependent scores
```

## Proposed architecture for the package

```sh
oceanbench/
├── core # Set of core domain functions and utilities
│   ├── evaluate
│   │   └── rmse_core.py
│   ├── __init__.py
│   ├── plot
│   │   ├── density_core.py
│   │   ├── geo_core.py
│   │   ├── mld_core.py
│   │   └── rmse_core.py
│   └── process
│       ├── calc_density_core.py
│       ├── calc_geo_core.py
│       └── calc_mld_core.py
├── __init__.py
├── evaluate.py # Python interface for evaluate module
├── plot.py # Python interface for plot module
└── process.py # Python interface for process module
```

The `evaluate.py`, `plot.py` and `process.py` files are the entry-points of each module.
