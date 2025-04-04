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

candidate_dataset = xarray.open_dataset(
    "https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/2024-01-03.zarr",
    engine="zarr",
)

```

### Evaluation of candidate datasets using OceanBench

#### Root Mean Square Error (RMSE) compared to GLORYS


```
nparray = oceanbench.metrics.rmse_to_glorys(
    candidate_datasets=[candidate_dataset],
)
oceanbench.plot.plot_rmse(rmse_dataarray=nparray, depth=2)
oceanbench.plot.plot_rmse_for_average_depth(rmse_dataarray=nparray)
oceanbench.plot.plot_rmse_depth_for_average_time(
    rmse_dataarray=nparray, dataset_depth_values=candidate_dataset.depth.values
)
```

#### Mixed Layer Depth (MLD) analysis


```
dataset = oceanbench.derived_quantities.mld(
    candidate_dataset=candidate_dataset,
    lead=1,
)
oceanbench.plot.plot_mld(dataset=dataset)
```

#### Geostrophic current analysis


```
dataset = oceanbench.derived_quantities.geostrophic_currents(
    candidate_dataset=candidate_dataset,
    lead=1,
    variable="zos",
)
oceanbench.plot.plot_geo(dataset=dataset)
```

#### Density analysis


```
dataarray = oceanbench.derived_quantities.density(
    candidate_dataset=candidate_dataset,
    lead=1,
    minimum_longitude=-100,
    maximum_longitude=-40,
    minimum_latitude=-15,
    maximum_latitude=50,
)
oceanbench.plot.plot_density(dataarray=dataarray)
```

#### Euclidean distance to GLORYS reference


```
euclidean_distance = oceanbench.metrics.euclidean_distance_to_glorys(
    candidate_dataset=candidate_dataset,
    minimum_latitude=466,
    maximum_latitude=633,
    minimum_longitude=400,
    maximum_longitude=466,
)
oceanbench.plot.plot_euclidean_distance(euclidean_distance)
```

#### Energy cascading analysis


```
_, gglonet_sc = oceanbench.metrics.energy_cascade(candidate_dataset, "uo", 0, 1 / 4)
oceanbench.plot.plot_energy_cascade(gglonet_sc)
```

#### Kinetic energy analysis


```
oceanbench.derived_quantities.kinetic_energy(candidate_dataset)
oceanbench.plot.plot_kinetic_energy(candidate_dataset)
```

#### Vorticity analysis


```
oceanbench.derived_quantities.vorticity(candidate_dataset)
oceanbench.plot.plot_vorticity(candidate_dataset)
```

#### Mass conservation analysis


```
mean_div_time_series = oceanbench.derived_quantities.mass_conservation(
    candidate_dataset, 0, deg_resolution=0.25
)  # should be close to zero
print(mean_div_time_series.data)  # time-dependent scores
```
<!-- END of a block automatically generated with make update-readme -->
