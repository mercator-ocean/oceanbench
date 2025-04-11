# OceanBench

## Installation

```bash
git clone git@github.com:mercator-ocean/oceanbench.git && cd oceanbench/ && pip install --editable .
```

### Dependence on the Copernicus Marine Service

Running Oceanbench uses the [Copernicus Marine Toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox/) and hence requires authentication to the [Copernicus Marine Service](https://marine.copernicus.eu/).

> If you're running Oceanbench in a non-interactive way, please follow the [Copernicus Marine Toolbox documentation](https://toolbox-docs.marine.copernicus.eu/en/v2.0.1/usage/quickoverview.html#copernicus-marine-toolbox-login) to login to the Copernicus Marine Service before running the bench.

## Example of OceanBench evaluation against GLONET sample

[This notebook](https://github.com/mercator-ocean/oceanbench/blob/main/assets/glonet_sample.report.ipynb) is the evaluation report of a sample (single forecast) of the GLONET system.
