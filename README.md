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

OceanBench is a benchmarking tool to evaluate ocean forecasting systems against reference ocean analysis datasets as well as observations.

## Score table and model comparison

The official score table is available on the [OceanBench website](https://oceanbench.lab.dive.edito.eu).

## Definitions of evaluation methods

The definitions of the methods used to evaluate models are available on the [OceanBench website](https://oceanbench.lab.dive.edito.eu) and in the tool [documentation](https://oceanbench.readthedocs.io).

## Evaluate your model with OceanBench

### Interactively

Checkout [this notebook](https://github.com/mercator-ocean/oceanbench/blob/main/assets/glonet_sample.report.ipynb) that evaluates a sample (two forecasts) of the GLONET system on OceanBench.
The resulting executed notebook is used as the evaluation report of the model, and its content is used to fulfil the OceanBench score table.

You can replace the cell that open the challenger datasets with your code and execute the notebook.

#### Run on your own resources

You will need to install OceanBench manually in your environment.

##### Installation

###### Using PyPI

```bash
pip install oceanbench
```

###### From sources

```bash
git clone git@github.com:mercator-ocean/oceanbench.git && cd oceanbench/ && pip install --editable .
```

#### Run on EDITO

You can open the example notebook in EDITO by clicking here:
[![Link to open resource in EDITO](https://dive.edito.eu/badges/Open-in-EDITO.svg)](https://datalab.dive.edito.eu/launcher/ocean-modelling/jupyter-python-ocean-science?name=jupyter-oceanbench&resources.requests.cpu=«4000m»&resources.requests.memory=«8Gi»&resources.limits.cpu=«7200m»&resources.limits.memory=«28Gi»&init.personalInit=«https%3A%2F%2Fraw.githubusercontent.com%2Fmercator-ocean%2Foceanbench%2Frefs%2Fheads%2Fmain%2Fedito%2Fopen-jupyter-notebook-url-edito.sh»&init.personalInitArgs=«https%3A%2F%2Fraw.githubusercontent.com%2Fmercator-ocean%2Foceanbench%2Frefs%2Fheads%2Fmain%2Fassets%2Fglonet_sample.report.ipynb»)

## Dependency on the Copernicus Marine Service

Running OceanBench to evaluate models with 1/12° resolution uses the [Copernicus Marine Toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox/) and hence requires authentication to the [Copernicus Marine Service](https://marine.copernicus.eu/).

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
