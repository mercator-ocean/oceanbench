# OceanBench

OceanBench is a benchmarking tool to evaluate ocean models against reference ocean analysis datasets as well as observations.

## Score table and model comparison

The official score table is available on the OceanBench website.

## Definitions of evaluation methods

The definitions of the methods used to evaluate models are available on the OceanBench website and in the tool documentation.

## Evaluate your model with OceanBench

Checkout [this notebook](https://github.com/mercator-ocean/oceanbench/blob/main/assets/glonet_sample.report.ipynb) that evaluates a sample (single forecast) of the GLONET system on OceanBench.
The resulting executed notebook is used as the evaluation report of the model, and its content is used to fulfil the OceanBench score table.

You can replace the cell that open the challenger datasets with your code and execute the notebook.


### Installation

```bash
git clone git@github.com:mercator-ocean/oceanbench.git && cd oceanbench/ && pip install --editable .
```

### Dependency on the Copernicus Marine Service

Running OceanBench uses the [Copernicus Marine Toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox/) and hence requires authentication to the [Copernicus Marine Service](https://marine.copernicus.eu/).

> If you're running OceanBench in a non-interactive way, please follow the [Copernicus Marine Toolbox documentation](https://toolbox-docs.marine.copernicus.eu/en/v2.0.1/usage/quickoverview.html#copernicus-marine-toolbox-login) to login to the Copernicus Marine Service before running the bench.

## Citation
N/A

## Contribution

Your help to improve OceanBench is welcome.
Please first read contribution instructions [here](CONTRIBUTION.md).

## License
N/A
