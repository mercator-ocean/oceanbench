# Oceanbench

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
nparray = oceanbench.evaluate.pointwise_evaluation(
    glonet_datasets=[glonet_dataset],
    glorys_datasets=[glorys_dataset],
)
oceanbench.plot.plot_pointwise_evaluation(rmse_dataarray=nparray, depth=2)
oceanbench.plot.plot_pointwise_evaluation_for_average_depth(rmse_dataarray=nparray)
oceanbench.plot.plot_pointwise_evaluation_depth_for_average_time(rmse_dataarray=nparray, dataset_depth_values=glonet_dataset.depth.values)
```

### MLD analysis


```python
dataset = oceanbench.process.calc_mld(
    dataset=glonet_dataset,
    lead=1,
)
oceanbench.plot.plot_mld(dataset=dataset)
```

### Geo analysis


```python
dataset = oceanbench.process.calc_geo(
    dataset=glonet_dataset,
    lead=1,
    variable="zos",
)
oceanbench.plot.plot_geo(dataset=dataset)
```

### Density analysis


```python
dataarray = oceanbench.process.calc_density(
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
euclidean_distance = oceanbench.evaluate.get_euclidean_distance(
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
_, gglonet_sc = oceanbench.evaluate.analyze_energy_cascade(glonet_dataset, "uo", 0, 1 / 4)

oceanbench.plot.plot_energy_cascade(gglonet_sc)
```

### Kinetic energy analysis


```python
oceanbench.plot.plot_kinetic_energy(glonet_dataset)
```

### Vorticity analysis


```python
oceanbench.plot.plot_vortocity(glonet_dataset)
```

### Mass conservation analysis


```python
mean_div_time_series = oceanbench.process.mass_conservation(glonet_dataset, 0, deg_resolution=0.25)  # should be close to zero
print(mean_div_time_series.data)  #time-dependent scores
```
<!-- END of a block automatically generated with make update-readme -->
