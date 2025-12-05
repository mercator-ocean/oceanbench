# Open GLONET forecasts with xarray
import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.glonet()

challenger_dataset
