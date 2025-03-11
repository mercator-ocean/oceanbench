# Oceanbench

The following section exposes usage examples of the `oceanbench` library.
Its content is available as a notebook to download [here](https://raw.githubusercontent.com/mercator-ocean/oceanbench/refs/heads/main/assets/glonet-example.ipynb), or you can [launch it in EDITO](https://datalab.dive.edito.eu/launcher/ocean-modelling/jupyter-python-ocean-science?name=oceanbench&s3=region-bb0d481d&resources.requests.cpu=%C2%AB4000m%C2%BB&resources.requests.memory=%C2%AB4Gi%C2%BB&resources.limits.cpu=%C2%AB7200m%C2%BB&resources.limits.memory=%C2%AB28Gi%C2%BB&init.personalInit=%C2%ABhttps%3A%2F%2Fgitlab.mercator-ocean.fr%2Fpub%2Fedito-infra%2Fconfiguration%2F-%2Fraw%2Fmain%2Fscripts%2Fopen-jupyter-notebook-url.sh%C2%BB&init.personalInitArgs=%C2%ABhttps%3A%2F%2Fraw.githubusercontent.com%2Fmercator-ocean%2Foceanbench%2Frefs%2Fheads%2Fmain%2Fassets%2Fglonet-example.ipynb%C2%BB&persistence.size=%C2%AB30Gi%C2%BB&git.repository=«https%3A%2F%2Fgithub.com%2Fmercator-ocean%2Foceanbench.git»&autoLaunch=true).

<!-- BEGINNING of a block automatically generated with make update-readme -->
## Oceanbench analysis example with GLONET forecasts

### Installation


```python
!git clone git@github.com:mercator-ocean/oceanbench.git
```


```python
!cd oceanbench/ && pip install --editable .
```


```python
import oceanbench
import xarray
```

### Fetch dataset to benchmark and reference dataset (GLORYS)


```python
!mc cp -r s3/project-oceanbench/glo data/
```


```python
glonet_dataset = xarray.open_dataset("data/glonet/2024-01-03.nc")
glorys_dataset = xarray.open_dataset("data/glorys14/2024-01-03.nc")
```

### RMSE analysis


```python
nparray = oceanbench.metrics.rmse(
    glonet_datasets=[glonet_dataset],
    glorys_datasets=[glorys_dataset],
)
oceanbench.plot.plot_rmse(rmse_dataarray=nparray, depth=2)
oceanbench.plot.plot_rmse_for_average_depth(rmse_dataarray=nparray)
oceanbench.plot.plot_rmse_depth_for_average_time(rmse_dataarray=nparray, dataset_depth_values=glonet_dataset.depth.values)
```

### MLD analysis


```python
dataset = oceanbench.derived_quantities.mld(
    dataset=glonet_dataset,
    lead=1,
)
oceanbench.plot.plot_mld(dataset=dataset)
```

### Geostrophic current analysis


```python
dataset = oceanbench.derived_quantities.geostraphic_currents(
    dataset=glonet_dataset,
    lead=1,
    variable="zos",
)
oceanbench.plot.plot_geo(dataset=dataset)
```

### Density analysis


```python
dataarray = oceanbench.derived_quantities.density(
    dataset=glonet_dataset,
    lead=1,
    minimum_longitude=-100,
    maximum_longitude=-40,
    minimum_latitude=-15,
    maximum_latitude=50,
)
oceanbench.plot.plot_density(dataarray=dataarray)
```

### Euclidean distance analysis


```python
euclidean_distance = oceanbench.metrics.euclidean_distance(
    first_dataset=glonet_dataset,
    second_dataset=glorys_dataset,
    minimum_latitude=466,
    maximum_latitude=633,
    minimum_longitude=400,
    maximum_longitude=466,
)

oceanbench.plot.plot_euclidean_distance(euclidean_distance)
```

### Energy cascading analysis


```python
_, gglonet_sc = oceanbench.metrics.energy_cascade(glonet_dataset, "uo", 0, 1 / 4)

oceanbench.plot.plot_energy_cascade(gglonet_sc)
```

### Kinetic energy analysis


```python
oceanbench.derived_quantities.kinetic_energy(glonet_dataset)
oceanbench.plot.plot_kinetic_energy(glonet_dataset)
```

### Vorticity analysis


```python
oceanbench.derived_quantities.vortocity(glonet_dataset)
oceanbench.plot.plot_vortocity(glonet_dataset)
```

### Mass conservation analysis


```python
mean_div_time_series = oceanbench.derived_quantities.mass_conservation(glonet_dataset, 0, deg_resolution=0.25)  # should be close to zero
print(mean_div_time_series.data)  #time-dependent scores
```
<!-- END of a block automatically generated with make update-readme -->
