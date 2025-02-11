# oceanbench

## Proposed architecture for the package

```sh
oceanbench/
├── cli.py  # Entry point to reference our commands
├── command_line_interface/ # Each folder is a subcommand
│   ├── evaluate/ # Each subcommand is an entrypoint and some other subcommands
│   │   ├── evaluate.py # Entrypoint
│   │   └── rmse.py # Subcommand(s)
│   ├── plot/
│   │   ├── plot.py
│   │   └── rmse.py
│   └── process/
└── core/   # Set of functions and utilities that can be used at multiple places
    ├── plot/
    │   └── rmse.py
    └── pointwise_evaluation.py
```
