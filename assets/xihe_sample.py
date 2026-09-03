# Open XIHE forecasts with oceanbench, for a single first day
from datetime import datetime
import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.xihe([datetime.fromisoformat("2024-01-03")])

challenger_dataset
