# Open local GLONET demo sample over a regional subset with xarray
import xarray
import oceanbench
from oceanbench.demo import glonet

glonet.ensure_local_demo_data_exists()
glonet.configure_local_reference_paths()

region_definition = oceanbench.regions.NORTH_ATLANTIC

challenger_dataset: xarray.Dataset = oceanbench.regions.subset_dataset_to_region(
    glonet.load_local_challenger_dataset(),
    region_definition,
)
challenger_dataset
