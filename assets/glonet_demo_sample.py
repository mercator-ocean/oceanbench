# Open local GLONET demo sample with xarray
import xarray
from oceanbench.demo import glonet

glonet.ensure_local_demo_data_exists()
glonet.configure_local_reference_paths()

challenger_dataset: xarray.Dataset = glonet.load_local_challenger_dataset()
challenger_dataset
