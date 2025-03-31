import oceanbench

# ### Open candidate datasets

# > Insert here the code that opens the candidate datasets as `candidate_datasets: xarray.Dataset`

import xarray

candidate_dataset: xarray.Dataset = ...

# ### Evaluation of candidate datasets using OceanBench

# #### Root Mean Square Error (RMSE) compared to GLORYS

nparray = oceanbench.metrics.rmse_to_glorys(
    candidate_datasets=[candidate_dataset],
)
oceanbench.plot.plot_rmse(rmse_dataarray=nparray, depth=2)
oceanbench.plot.plot_rmse_for_average_depth(rmse_dataarray=nparray)
oceanbench.plot.plot_rmse_depth_for_average_time(
    rmse_dataarray=nparray, dataset_depth_values=candidate_dataset.depth.values
)

# #### Mixed Layer Depth (MLD) analysis

dataset = oceanbench.derived_quantities.mld(
    candidate_dataset=candidate_dataset,
    lead=1,
)
oceanbench.plot.plot_mld(dataset=dataset)

# #### Geostrophic current analysis

dataset = oceanbench.derived_quantities.geostrophic_currents(
    candidate_dataset=candidate_dataset,
    lead=1,
    variable="zos",
)
oceanbench.plot.plot_geo(dataset=dataset)

# #### Density analysis

dataarray = oceanbench.derived_quantities.density(
    candidate_dataset=candidate_dataset,
    lead=1,
    minimum_longitude=-100,
    maximum_longitude=-40,
    minimum_latitude=-15,
    maximum_latitude=50,
)
oceanbench.plot.plot_density(dataarray=dataarray)

# #### Euclidean distance to GLORYS reference

euclidean_distance = oceanbench.metrics.euclidean_distance_to_glorys(
    candidate_dataset=candidate_dataset,
    minimum_latitude=466,
    maximum_latitude=633,
    minimum_longitude=400,
    maximum_longitude=466,
)
oceanbench.plot.plot_euclidean_distance(euclidean_distance)

# #### Energy cascading analysis

_, gglonet_sc = oceanbench.metrics.energy_cascade(
    candidate_dataset, "uo", 0, 1 / 4
)
oceanbench.plot.plot_energy_cascade(gglonet_sc)

# #### Kinetic energy analysis

oceanbench.derived_quantities.kinetic_energy(candidate_dataset)
oceanbench.plot.plot_kinetic_energy(candidate_dataset)

# #### Vorticity analysis
oceanbench.derived_quantities.vorticity(candidate_dataset)
oceanbench.plot.plot_vorticity(candidate_dataset)

# #### Mass conservation analysis

mean_div_time_series = oceanbench.derived_quantities.mass_conservation(
    candidate_dataset, 0, deg_resolution=0.25
)  # should be close to zero
print(mean_div_time_series.data)  # time-dependent scores
