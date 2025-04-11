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
```python
import oceanbench

oceanbench.__version__
```




    '0.0.1a0'



### Open candidate datasets

> Insert here the code that opens the candidate datasets as `candidate_datasets: List[xarray.Dataset]`


```python
# Open GLONET forecast sample with xarray
from functools import reduce
import xarray
from typing import List

LATITUDE_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "latitude",
    "long_name": "Latitude",
    "units": "degrees_north",
    "units_long": "Degrees North",
    "axis": "Y",
}

LONGITUDE_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "longitude",
    "long_name": "Longitude",
    "units": "degrees_east",
    "units_long": "Degrees East",
    "axis": "X",
}

DEPTH_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "depth",
    "long_name": "Depth",
    "units": "m",
    "units_long": "Meters",
}

TIME_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "time",
    "long_name": "Time",
    "axis": "T",
}


def _update_variable_attributes(
    dataset: xarray.Dataset,
    variable_name_and_attributes: tuple[str, dict[str, str]],
) -> xarray.Dataset:
    variable_name, attributes = variable_name_and_attributes
    dataset[variable_name].attrs = attributes
    return dataset


def _add_climate_forecast_attributes(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    return reduce(
        _update_variable_attributes,
        zip(
            ["lat", "lon", "depth", "time"],
            [
                LATITUDE_CLIMATE_FORECAST_ATTRIBUTES,
                LONGITUDE_CLIMATE_FORECAST_ATTRIBUTES,
                DEPTH_CLIMATE_FORECAST_ATTRIBUTES,
                TIME_CLIMATE_FORECAST_ATTRIBUTES,
            ],
        ),
        dataset,
    )


candidate_datasets: List[xarray.Dataset] = [
    _add_climate_forecast_attributes(
        xarray.open_dataset(
            "https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/2024-01-03.zarr",
            engine="zarr",
        )
    )
]

```

### Evaluation of candidate datasets using OceanBench

#### Root Mean Square Error (RMSE) compared to GLORYS


```python
oceanbench.metrics.rmse_to_glorys(candidate_datasets)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lead day 1</th>
      <th>Lead day 2</th>
      <th>Lead day 3</th>
      <th>Lead day 4</th>
      <th>Lead day 5</th>
      <th>Lead day 6</th>
      <th>Lead day 7</th>
      <th>Lead day 8</th>
      <th>Lead day 9</th>
      <th>Lead day 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Surface temperature</th>
      <td>0.659611</td>
      <td>0.675301</td>
      <td>0.725245</td>
      <td>0.724560</td>
      <td>0.805426</td>
      <td>0.807227</td>
      <td>0.888620</td>
      <td>0.869715</td>
      <td>0.944387</td>
      <td>0.925018</td>
    </tr>
    <tr>
      <th>50m temperature</th>
      <td>0.966279</td>
      <td>0.981624</td>
      <td>1.003170</td>
      <td>1.022845</td>
      <td>1.060496</td>
      <td>1.097911</td>
      <td>1.136152</td>
      <td>1.187038</td>
      <td>1.216651</td>
      <td>1.282137</td>
    </tr>
    <tr>
      <th>Surface salinity</th>
      <td>0.792587</td>
      <td>0.790434</td>
      <td>0.795696</td>
      <td>0.792178</td>
      <td>0.787018</td>
      <td>0.781515</td>
      <td>0.780387</td>
      <td>0.776530</td>
      <td>0.778576</td>
      <td>0.771986</td>
    </tr>
    <tr>
      <th>50m salinity</th>
      <td>0.350590</td>
      <td>0.350406</td>
      <td>0.358510</td>
      <td>0.359323</td>
      <td>0.364514</td>
      <td>0.365943</td>
      <td>0.370236</td>
      <td>0.371366</td>
      <td>0.375886</td>
      <td>0.378222</td>
    </tr>
    <tr>
      <th>Surface northward velocity</th>
      <td>0.116657</td>
      <td>0.119014</td>
      <td>0.122878</td>
      <td>0.123277</td>
      <td>0.125830</td>
      <td>0.127524</td>
      <td>0.132315</td>
      <td>0.133787</td>
      <td>0.139171</td>
      <td>0.139889</td>
    </tr>
    <tr>
      <th>50m northward velocity</th>
      <td>0.107193</td>
      <td>0.107600</td>
      <td>0.106828</td>
      <td>0.107995</td>
      <td>0.109449</td>
      <td>0.111846</td>
      <td>0.114489</td>
      <td>0.117098</td>
      <td>0.118402</td>
      <td>0.120405</td>
    </tr>
    <tr>
      <th>Surface eastward velocity</th>
      <td>0.117798</td>
      <td>0.120017</td>
      <td>0.123027</td>
      <td>0.124172</td>
      <td>0.128817</td>
      <td>0.130287</td>
      <td>0.134783</td>
      <td>0.134989</td>
      <td>0.139812</td>
      <td>0.143858</td>
    </tr>
    <tr>
      <th>50m eastward velocity</th>
      <td>0.108636</td>
      <td>0.109575</td>
      <td>0.109599</td>
      <td>0.110640</td>
      <td>0.112196</td>
      <td>0.114510</td>
      <td>0.117032</td>
      <td>0.119348</td>
      <td>0.122307</td>
      <td>0.126167</td>
    </tr>
  </tbody>
</table>
</div>



#### Mixed Layer Depth (MLD) analysis


```python
oceanbench.derived_quantities.mld(candidate_datasets)
```



![png](glonet_sample.report_files/glonet_sample.report_6_0.png)



#### Geostrophic current analysis


```python
oceanbench.derived_quantities.geostrophic_currents(candidate_datasets)
```

    /home/github-runner/actions-runner/_work/oceanbench/oceanbench/oceanbench/core/process/calc_geo_core.py:26: RuntimeWarning: divide by zero encountered in divide
      u_geo = -g / f[:, numpy.newaxis] * dssh_dy
    /home/github-runner/actions-runner/_work/oceanbench/oceanbench/oceanbench/core/process/calc_geo_core.py:26: RuntimeWarning: invalid value encountered in multiply
      u_geo = -g / f[:, numpy.newaxis] * dssh_dy
    /home/github-runner/actions-runner/_work/oceanbench/oceanbench/oceanbench/core/process/calc_geo_core.py:27: RuntimeWarning: divide by zero encountered in divide
      v_geo = g / f[:, numpy.newaxis] * dssh_dx
    /home/github-runner/actions-runner/_work/oceanbench/oceanbench/oceanbench/core/process/calc_geo_core.py:27: RuntimeWarning: invalid value encountered in multiply
      v_geo = g / f[:, numpy.newaxis] * dssh_dx




![png](glonet_sample.report_files/glonet_sample.report_8_1.png)



#### Density analysis


```python
oceanbench.derived_quantities.density(candidate_datasets)
```



![png](glonet_sample.report_files/glonet_sample.report_10_0.png)



#### Euclidean distance to GLORYS reference


```python
oceanbench.metrics.euclidean_distance_to_glorys(candidate_datasets)
```

    466
    start


    INFO: Output files are stored in tst.zarr.


    <class 'numpy.ndarray'>


    466
    start


    INFO: Output files are stored in tst.zarr.


    <class 'numpy.ndarray'>




![png](glonet_sample.report_files/glonet_sample.report_12_6.png)



#### Energy cascading analysis


```python
oceanbench.metrics.energy_cascade(candidate_datasets)
```



![png](glonet_sample.report_files/glonet_sample.report_14_0.png)



#### Kinetic energy analysis


```python
oceanbench.derived_quantities.kinetic_energy(candidate_datasets)
```



![png](glonet_sample.report_files/glonet_sample.report_16_0.png)



#### Vorticity analysis


```python
oceanbench.derived_quantities.vorticity(candidate_datasets)
```



![png](glonet_sample.report_files/glonet_sample.report_18_0.png)



#### Mass conservation analysis


```python
oceanbench.derived_quantities.mass_conservation(candidate_datasets)
```

    [-1.03391335e-08 -1.28428394e-08 -1.16979252e-08 -1.41945821e-08
     -1.54307116e-08 -9.51230823e-09 -1.20808228e-08 -1.08712131e-08
     -1.52097767e-08 -1.30613604e-08]

<!-- END of a block automatically generated with make update-readme -->
