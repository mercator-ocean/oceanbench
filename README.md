# Oceanbench

## Usage examples

### RMSE
Get and plot RMSE:
```python
import oceanbench
from pathlib import Path

nparray = oceanbench.evaluate.pointwise_evaluation(
    glonet_datasets_path="~/Downloads/glonet-data/glonet",
    glorys_datasets_path="~/Downloads/glonet-data/glorys14",
)
oceanbench.plot.plot_pointwise_evaluation(nparray, 2, Path("output.png"), True)
```

### Density
Get density:
```python
import oceanbench
from pathlib import Path

dataarray = oceanbench.process.calc_density(
    glonet_dataset_path=Path("~/Downloads/glonet-data/glonet/2024-01-03.nc"),
    lead=1,
    minimum_longitude=-100,
    maximum_longitude=-40,
    minimum_latitude=-15,
    maximum_latitude=50,
)
oceanbench.plot.plot_density(
    density_dataarray=dataarray, plot_output_path="plot.png", show_plot=True
)
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
