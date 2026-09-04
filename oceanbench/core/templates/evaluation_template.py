import oceanbench
from IPython.display import display
from oceanbench.core.dataset_source import get_dataset_source
from oceanbench.core.glo36v1 import is_super_resolution_dataset

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation configuration

region = "global"

# ### Evaluation track

dataset_source = get_dataset_source(challenger_dataset)
is_super_resolution_track = is_super_resolution_dataset(challenger_dataset)
is_glonet_high_resolution_track = dataset_source is not None and dataset_source.name in {
    "glonet_high_resolution",
    "glonet_hr",
    "glonet_super_resolution",
}
evaluate_glo36v1_reference = is_super_resolution_track and not is_glonet_high_resolution_track

# ### Evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of variables compared to observations

display(
    oceanbench.metrics.rmsd_of_variables_compared_to_observations(
        challenger_dataset,
        region=region,
    )
)

# #### Deviation of Lagrangian trajectories compared to GLORYS reanalysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Deviation of Lagrangian trajectories compared to GLO12 analysis

if not is_super_resolution_track:
    display(
        oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO36V1 reference

if evaluate_glo36v1_reference:
    display(
        oceanbench.metrics.rmsd_of_variables_compared_to_glo36v1_reference(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO36V1 reference

if evaluate_glo36v1_reference:
    display(
        oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo36v1_reference(
            challenger_dataset,
            region=region,
        )
    )

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO36V1 reference

if evaluate_glo36v1_reference:
    display(
        oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo36v1_reference(
            challenger_dataset,
            region=region,
        )
    )

# #### Deviation of Lagrangian trajectories compared to GLO36V1 reference

if evaluate_glo36v1_reference:
    display(
        oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo36v1_reference(
            challenger_dataset,
            region=region,
        )
    )
