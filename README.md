<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

<div align="center">
  <img src="https://minio.dive.edito.eu/project-oceanbench/public/oceanbench-logo-with-name.svg" alt="OceanBench logo" height="100"/>
</div>

# OceanBench: Evaluating ocean forecasting systems

[![The latest version of OceanBench can be found on PyPI.](https://img.shields.io/pypi/v/oceanbench.svg)](https://pypi.org/project/oceanbench)
[![Link to discover EDITO](https://dive.edito.eu/badges/Powered-by-EDITO.svg)](https://dive.edito.eu)
[![Information on what versions of Python OceanBench supports can be found on PyPI.](https://img.shields.io/pypi/pyversions/oceanbench.svg)](https://pypi.org/project/oceanbench)
[![Information on what kind of operating systems OceanBench can be installed on.](https://img.shields.io/badge/platform-linux-lightgrey)](https://en.wikipedia.org/wiki/Linux)
[![Information on the OceanBench licence.](https://img.shields.io/badge/licence-EUPL-lightblue)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
[![REUSE status](https://api.reuse.software/badge/github.com/mercator-ocean/oceanbench/)](https://api.reuse.software/info/github.com/mercator-ocean/oceanbench/)
[![Documentation](https://img.shields.io/readthedocs/oceanbench/latest?logo=readthedocs)](https://oceanbench.readthedocs.io)

OceanBench is a benchmarking tool to evaluate ocean forecasting systems against reference ocean analysis datasets (such as 2024 [GLORYS reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030) and [GLO12 analysis](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024)) as well as observations.

## Citation

OceanBench's scientific paper is published in [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/121394) and is accessible at [https://openreview.net/forum?id=wZGe1Kqs8G](https://openreview.net/forum?id=wZGe1Kqs8G).
```
@inproceedings{
  aouni2025oceanbench,
  title={OceanBench: A Benchmark for Data-Driven Global Ocean Forecasting systems},
  author={Anass El Aouni and Quentin Gaudel and Juan Emmanuel Johnson and REGNIER Charly and Julien Le Sommer and van Gennip and Ronan Fablet and Marie Drevillon and Yann DRILLET and Pierre Yves Le Traon},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=wZGe1Kqs8G}
}
```

## Score table and system comparison

The official score table is available on the [OceanBench website](https://oceanbench.lab.dive.edito.eu).

## Open GLORYS dataset to train your ocean forecasting system

You can train your model with [GLORYS reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030).
From an environment with OceanBench installed, run:

```
import oceanbench
oceanbench.datasets.reference.glorys_reanalysis()
```

to open GLORYS dataset as a [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html):
```
<xarray.Dataset> Size: 5TB
Dimensions:    (depth: 50, latitude: 2041, longitude: 4320, time: 366)
Coordinates:
  * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
  * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
  * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
  * time       (time) datetime64[ns] 3kB 2024-01-01 2024-01-02 ... 2024-12-31
Data variables:
    thetao     (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
    so         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
    uo         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
    vo         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
    zos        (time, latitude, longitude) float64 26GB dask.array<chunksize=(28, 512, 2048), meta=np.ndarray>
Attributes:
    source:       MERCATOR GLORYS12V1
    institution:  MERCATOR OCEAN
    comment:      CMEMS product
    title:        daily mean fields from Global Ocean Physics Analysis and Fo...
    references:   http://www.mercator-ocean.fr
    history:      2023/06/01 16:20:05 MERCATOR OCEAN Netcdf creation
    Conventions:  CF-1.4
```

## Evaluate your system with OceanBench

The evaluation of a system consists of the sequential execution of a Python notebook that runs several evaluation methods against a set of forecasts (produced by the system), namely the _challenger dataset_, opened as an [xarray Dataset](https://xarray.pydata.org/en/v2023.11.0/generated/xarray.Dataset.html).

The OceanBench documentation describes [the shape a challenger dataset](https://oceanbench.readthedocs.io/en/latest/shape-of-the-challenger-dataset.html) must have, as well as [the definitions of the methods used to evaluate systems](https://oceanbench.readthedocs.io/en/latest/evaluation-methods.html).

### Official evaluation

All official challenger notebooks are maintained and remain executable in order to update the scores with new OceanBench versions (all official challengers are re-evaluated with each new version).

To officially submit your system to OceanBench, please open an issue on this repository attaching one of the following:

1. The executed notebook resulting from an [interactive](#interactive-evaluation) or [programmatic](#programmatic-evaluation) evaluation.
2. A way to access the system output data in a standard format (e.g. Zarr or NetCDF).
3. A way to execute the system code or container along with clear instructions for how to run it (e.g., input/output format, required dependencies, etc.).

In addition, please provide the following metadata:
- The organization that leads the construction or operation of the system.
- A link to the reference paper of the system.
- The system method. For example, "Physics-based", "ML-based" or "Hybrid".
- The system type. For example, "Forecast (deterministic)" or "Forecast (ensemble)".
- The system initial conditions. For example, "GLO12/IFS".
- The approximate horizontal resolution of the system. For example, "1/12°" or "1/4°".

### Interactive evaluation

Checkout [this notebook](https://github.com/mercator-ocean/oceanbench/blob/main/assets/glonet_sample.report.ipynb) that evaluates a sample (two forecasts) of the GLONET system on OceanBench.
The resulting executed notebook is used as the evaluation report of the system, and its content is used to fulfill the OceanBench score table.

You can replace the cell that opens the challenger datasets with your code and execute the notebook.

#### Execute on your own resources

You will need to install OceanBench manually in your environment.

##### Installation

###### Using pip via PyPI

```bash
pip install oceanbench
```

###### From sources

```bash
git clone git@github.com:mercator-ocean/oceanbench.git && cd oceanbench/ && pip install --editable .
```

#### Execute on EDITO

You can open and manually execute the example notebook in EDITO datalab by clicking here:
[![Link to open resource in EDITO](https://dive.edito.eu/badges/Open-in-EDITO.svg)](https://datalab.dive.edito.eu/launcher/ocean-modelling/jupyter-python-ocean-science?name=jupyter-oceanbench&resources.requests.cpu=«4000m»&resources.requests.memory=«8Gi»&resources.limits.cpu=«7200m»&resources.limits.memory=«28Gi»&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2Fmercator-ocean%2Foceanbench%2Frefs%2Fheads%2Fmain%2Fedito%2Fopen-jupyter-notebook-url-edito.sh»&init.personalInitArgs=«https%3A%2F%2Fraw.githubusercontent.com%2Fmercator-ocean%2Foceanbench%2Frefs%2Fheads%2Fmain%2Fassets%2Fglonet_sample.report.ipynb»)

### Programmatic evaluation

#### Python

Once [installed](#installation), you can evaluate your system using python with the following code:

```python
import oceanbench

oceanbench.evaluate_challenger("path/to/file/opening/the/challenger/datasets.py", "notebook_report_name.ipynb")
```

More details in the [documentation](https://oceanbench.readthedocs.io/en/latest/source/oceanbench.html#oceanbench.evaluate_challenger).

### Dependency on the Copernicus Marine Service

Running OceanBench to evaluate systems with 1/12° resolution uses the [Copernicus Marine Toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox/) and therefore requires authentication with the [Copernicus Marine Service](https://marine.copernicus.eu/).

> If you're running OceanBench in a non-interactive way, please follow the [Copernicus Marine Toolbox documentation](https://toolbox-docs.marine.copernicus.eu) to login to the Copernicus Marine Service before running the bench.

## Contribution

Your help to improve OceanBench is welcome.
Please first read contribution instructions [here](CONTRIBUTION.md).

## License

Licensed under the [EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12) license.

## About

Implemented by:

<a href="https://mercator-ocean.eu"><img src="https://www.nemo-ocean.eu/wp-content/uploads/MOI.png" alt="Mercator Ocean logo" height="100"/></a>

As part of a fruitful collaboration with:

<a href="https://www.ocean-climat.fr"><img src="https://oceansconnectes.org/wp-content/uploads/2023/02/VISU-PPRsmol.jpg" alt="PPR logo" height="100"/></a>
<a href="https://www.imt-atlantique.fr"><img src="https://www.imt-atlantique.fr/sites/default/files/ecole/IMT_Atlantique_logo.png" alt="IMTA logo" height="100"/></a>
<a href="https://www.univ-grenoble-alpes.fr"><img src="https://www.grenoble-inp.fr/medias/photo/logo-uga-carrousel_1575017090994-png" alt="UGA logo" height="100"/></a>
<a href="https://igeo.ucm-csic.es/"><img src="https://igeo.ufrj.br/wp-content/uploads/2022/10/image-1.png" alt="IGEO logo" height="100"/></a>

Powered by:

<a href="https://dive.edito.eu"><img src="https://datalab.dive.edito.eu/custom-resources/logos/Full_EU_DTO_Banner.jpeg" alt="EU DTO banner" height="100"/></a>
