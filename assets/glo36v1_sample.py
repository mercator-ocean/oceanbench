# Open GLONET forecast sample with xarray 
from datetime import datetime 
import xarray 

challenger_dataset: xarray.Dataset = xarray.open_mfdataset(     
   [         
      "https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/20230104.zarr",         
      "https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/20230111.zarr"     
   ],     
   engine="zarr",     
   combine="nested",     
   concat_dim="first_day_datetime",     
   parallel=True, 
).assign(     
   {         
   "first_day_datetime": [             
      datetime.fromisoformat("2023-01-04"),             
      datetime.fromisoformat("2023-01-11")         
      ]     
   } 
)
challenger_dataset 

print(challenger_dataset)
