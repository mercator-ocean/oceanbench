# Open local GLONET quarter-degree SSH demo sample with xarray
import xarray
from oceanbench.demo import glonet

glonet.ensure_local_eddy_demo_data_exists()

challenger_dataset: xarray.Dataset = glonet.load_local_eddy_challenger_dataset()
challenger_dataset
