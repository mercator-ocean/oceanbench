# Contribution guidelines

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
├── metrics.py # Python interface for metrics module
├── plot.py # Python interface for plot module
└── derived_quantities.py # Python interface for derived_quantities module
```

The `metrics.py`, `plot.py` and `derived_quantities.py` files are the entry-points of each module.
