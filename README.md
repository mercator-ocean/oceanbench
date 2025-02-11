# Oceanbench

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

### The common_options.py file

Contains de list of reusable options for `Click`.
Each option is named `<OPTION_NAME>_option` and can be called with `@<OPTION_NAME>_option` as a decorator of a command. Then pass the `<OPTION_NAME>` to the parameters of the command function.

For example, define your option like this:
```python
def foo_bar_option(function):
    return click.option(
        "--foo-bar",
        help="A useful helper.",
    )(function)
```

And call it like that:
```python
@foo_bar_option
def my_command(foo_bar):
    ...
```
