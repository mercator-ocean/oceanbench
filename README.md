# Oceanbench

## Usage examples

### RMSE
Get RMSE:
```sh
oceanbench evaluate pointwise-evaluation --glonet-datasets-path ~/Downloads/glonet-data/glonet --glorys-datasets-path ~/Downloads/glonet-data/glorys14
```
Each folder needs to have a least one dataset, with the format `YYYY-MM-DD.nc`

Then you can plot the RMSE with:
```sh
oceanbench plot plot-pointwise-evaluation --rmse-path glonet.npy --depth 2
```

### Density
Get density:
```sh
oceanbench process calc-density --glonet-dataset-path ~/Downloads/glonet-data/glonet/2024-01-03.nc --minimum-longitude -100 --maximum-longitude -40 --minimum-latitude 15 --maximum-latitude 50 --lead 0
```

Plot density:
```sh
oceanbench plot plot-density --density-dataset-path output.nc
```

## Proposed architecture for the package

```sh
oceanbench/
├── cli.py  # Entry point to reference our commands
├── command_line_interface/ # Each folder is a subcommand
│   ├── common_options.py # A list of reusable Click options
│   ├── evaluate/ # Each subcommand is an entrypoint and some other subcommands
│   │   ├── evaluate.py # Entrypoint
│   │   └── rmse.py # Subcommand(s)
│   ├── plot/
│   │   ├── plot.py
│   │   └── rmse.py
│   └── process/
└── core/   # Set of functions and utilities that can be used at multiple places
    ├── evaluate
    │   └── rmse_core.py
    ├── plot
    │   ├── density_core.py
    │   ├── geo_core.py
    │   ├── mld_core.py
    │   └── rmse_core.py
    └── process
        ├── calc_density_core.py
        ├── calc_geo_core.py
        └── calc_mld_core.py
```

Each folder of the `oceanbench/command_line_interface/` corresponds to an `oceanbench` command (`evaluate`, `plot`, `process`). In these folders, we define the command line interface with `Click` that calls a core function in `oceanbench/core/`. The idea is to decoupled the logic from the CLI to the core domain.
