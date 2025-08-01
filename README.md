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

## Score table and system comparison

The official score table is available on the [OceanBench website](https://oceanbench.lab.dive.edito.eu).

## Evaluate your system with OceanBench

The evaluation of a system consists in the sequential execution of a Python notebook run several evaluation methods against a set of forecasts (produced by the system), namely the _challenger dataset_, opened as an [xarray Dataset](https://xarray.pydata.org/en/v2023.11.0/generated/xarray.Dataset.html).

The OceanBench documentation describes [the shape a challenger dataset](https://oceanbench.readthedocs.io/en/latest/shape-of-the-challenger-dataset.html) must have, as well of [the definitions of the methods used to evaluate systems](https://oceanbench.readthedocs.io/en/latest/evaluation-methods.html).

### Official evaluation

All official challenger notebooks are maintained and re-executable in order to update the scores with new OceanBench versions (all official challengers are re-evaluated at each new version).

To officially submit your system to OceanBench, please open an issue on this repository attaching one of the following:

1. The executed notebook resulting from an [interactive](#interactive-evalution) or [programmatic](#programmatic-evaluation) evaluation.
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

You can replace the cell that open the challenger datasets with your code and execute the notebook.

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

Running OceanBench to evaluate systems with 1/12° resolution uses the [Copernicus Marine Toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox/) and hence requires authentication to the [Copernicus Marine Service](https://marine.copernicus.eu/).

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

<a href="https://www.ocean-climat.fr"><img src="https://www.cnrs.fr/sites/default/files/image/VISU-PPRsmol.jpg" alt="PPR logo" height="100"/></a>
<a href="https://www.imt-atlantique.fr"><img src="https://www.imt-atlantique.fr/sites/default/files/ecole/IMT_Atlantique_logo.png" alt="IMTA logo" height="100"/></a>
<a href="https://www.univ-grenoble-alpes.fr"><img src="https://www.grenoble-inp.fr/medias/photo/logo-uga-carrousel_1575017090994-png" alt="UGA logo" height="100"/></a>
<a href="https://igeo.ucm-csic.es/"><img src="https://igeo.ufrj.br/wp-content/uploads/2022/10/image-1.png" alt="IGEO logo" height="100"/></a>

Powered by:

<a href="https://dive.edito.eu"><img src="https://datalab.dive.edito.eu/custom-resources/logos/Full_EU_DTO_Banner.jpeg" alt="EU DTO banner" height="100"/></a>
