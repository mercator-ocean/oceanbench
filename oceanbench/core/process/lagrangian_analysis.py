from datetime import timedelta
import numpy
from parcels import (
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleSet,
)
import xarray


def get_particle_file_core(dataset: xarray.Dataset, latzone, lonzone) -> xarray.Dataset:
    variables = {
        "U": "uo",
        "V": "vo",
    }
    dimensions = {"lat": "lat", "lon": "lon", "time": "time"}
    fieldset = FieldSet.from_xarray_dataset(dataset, variables, dimensions)
    print(latzone[0])
    lon = dataset[dimensions["lon"]].data[latzone[0] : latzone[1]]
    lat = dataset[dimensions["lat"]].data[lonzone[0] : lonzone[1]]
    lon_mesh, lat_mesh = numpy.meshgrid(lon, lat)
    lats = lat_mesh.flatten()

    lons = lon_mesh.flatten()

    pset = ParticleSet.from_list(
        fieldset=fieldset,  # the fields on which the particles are advected
        pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
        lon=lons,  # a vector of release longitudes
        lat=lats,
        # time=data_source.time[-1] # backwar
        time=dataset.time[0],
    )
    print("start")

    output_file = pset.ParticleFile(name="tst.zarr", outputdt=timedelta(hours=24))
    pset.execute(
        AdvectionRK4,
        runtime=timedelta(days=9),
        # dt=-timedelta(minutes=60),#backward
        dt=timedelta(minutes=60),
        output_file=output_file,
        verbose_progress=False,
    )
    ds = xarray.open_zarr("tst.zarr")
    plats = ds.lat.values
    plons = ds.lon.values
    x = plats.reshape(lonzone[1] - lonzone[0], latzone[1] - latzone[0], 9).transpose(2, 0, 1)
    y = plons.reshape(lonzone[1] - lonzone[0], latzone[1] - latzone[0], 9).transpose(2, 0, 1)
    print(type(x))

    ds = xarray.Dataset(
        {
            "x": (["time", "lat", "lon"], x),
            "y": (["time", "lat", "lon"], y),
            "thetao": (
                ["lat", "lon"],
                dataset.thetao[-1, lonzone[0] : lonzone[1], latzone[0] : latzone[1]].values,
            ),
            "so": (
                ["lat", "lon"],
                dataset.so[-1, lonzone[0] : lonzone[1], latzone[0] : latzone[1]].values,
            ),
        },
        coords={"lat": lat, "lon": lon, "time": dataset.time[0:9]},
    )

    return ds
