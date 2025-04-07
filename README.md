# OceanBench

## Installation

```bash
git clone git@github.com:mercator-ocean/oceanbench.git && cd oceanbench/ && pip install --editable .
```

### Dependence on the Copernicus Marine Service

Running Oceanbench uses the [Copernicus Marine Toolbox](https://github.com/mercator-ocean/copernicus-marine-toolbox/) and hence requires authentication to the [Copernicus Marine Service](https://marine.copernicus.eu/).

> If you're running Oceanbench in a non-interactive way, please follow the [Copernicus Marine Toolbox documentation](https://toolbox-docs.marine.copernicus.eu/en/v2.0.1/usage/quickoverview.html#copernicus-marine-toolbox-login) to login to the Copernicus Marine Service before running the bench.

## Example of OceanBench evaluation against GLONET sample

The following section exposes the evaluation code of the `oceanbench` library, against a sample of the GLONET system.
Its content is available as a notebook to download [here](https://raw.githubusercontent.com/mercator-ocean/oceanbench/refs/heads/main/assets/glonet_sample.ipynb), or you can [launch it in EDITO](https://datalab.dive.edito.eu/launcher/ocean-modelling/jupyter-python-ocean-science?name=oceanbench&s3=region-bb0d481d&resources.requests.cpu=%C2%AB4000m%C2%BB&resources.requests.memory=%C2%AB4Gi%C2%BB&resources.limits.cpu=%C2%AB7200m%C2%BB&resources.limits.memory=%C2%AB28Gi%C2%BB&init.personalInit=%C2%ABhttps%3A%2F%2Fgitlab.mercator-ocean.fr%2Fpub%2Fedito-infra%2Fconfiguration%2F-%2Fraw%2Fmain%2Fscripts%2Fopen-jupyter-notebook-url.sh%C2%BB&init.personalInitArgs=%C2%ABhttps%3A%2F%2Fraw.githubusercontent.com%2Fmercator-ocean%2Foceanbench%2Frefs%2Fheads%2Fmain%2Fassets%2Fglonet_sample.ipynb%C2%BB&persistence.size=%C2%AB30Gi%C2%BB&git.repository=«https%3A%2F%2Fgithub.com%2Fmercator-ocean%2Foceanbench.git»&autoLaunch=true).

<!-- BEGINNING of a block automatically generated with make update-readme -->
```
import oceanbench
```

### Open candidate datasets

> Insert here the code that opens the candidate datasets as `candidate_datasets: xarray.Dataset`


```
# Open GLONET forecast sample with xarray
import xarray

candidate_datasets = [
    xarray.open_dataset(
        "https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/2024-01-03.zarr",
        engine="zarr",
    )
]

```

### Evaluation of candidate datasets using OceanBench

#### Root Mean Square Error (RMSE) compared to GLORYS


```
oceanbench.metrics.rmse_to_glorys(
    candidate_datasets=candidate_datasets,
)
```

#### Mixed Layer Depth (MLD) analysis


```
oceanbench.derived_quantities.mld(
    candidate_datasets=candidate_datasets,
)
```

#### Geostrophic current analysis


```
oceanbench.derived_quantities.geostrophic_currents(
    candidate_datasets=candidate_datasets,
)
```

#### Density analysis


```
oceanbench.derived_quantities.density(
    candidate_datasets=candidate_datasets,
)
```

#### Euclidean distance to GLORYS reference


```
oceanbench.metrics.euclidean_distance_to_glorys(
    candidate_datasets=candidate_datasets,
)
```

#### Energy cascading analysis


```
oceanbench.metrics.energy_cascade(candidate_datasets)
```

#### Kinetic energy analysis


```
oceanbench.derived_quantities.kinetic_energy(candidate_datasets)
```

#### Vorticity analysis


```
oceanbench.derived_quantities.vorticity(candidate_datasets)
```

#### Mass conservation analysis


```
oceanbench.derived_quantities.mass_conservation(candidate_datasets)
```
<!-- END of a block automatically generated with make update-readme -->
