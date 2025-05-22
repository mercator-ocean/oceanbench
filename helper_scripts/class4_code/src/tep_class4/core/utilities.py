import numpy as np
import matplotlib.pyplot as plt
import os, sys, re
from glob import glob
import xarray as xr


if sys.version_info[0] >= 3:
    import configparser as ConfigParser
else:
    import ConfigParser


def load_db(_DBFILE_):
    db = ConfigParser.ConfigParser()
    db.read(os.path.expanduser(_DBFILE_))
    if not db.sections():
        print("Configuration file not found !")
        sys.exit(1)
    return db


def conv_geojson(file):
    """
    Convert a text position file to a geojson file with the same name but different extension
    """
    filename = os.path.basename(file).split(".")[0] + ".geojson"
    data = open(file, "r")
    liste_position = []
    pos = np.ndarray(shape=(2), dtype=float, order="F")
    for line in data:
        lon = float(line.split()[0])
        lat = float(line.split()[1])
        pos[0] = lon
        pos[1] = lat
        liste_position.append(tuple(pos))
    with open(filename, "w") as outfile:
        json.dump(MultiPoint(liste_position), outfile)


def pull_names(name_data):
    """
    Convenient name retrieval function
    """
    names = []
    if isinstance(name_data, np.ma.masked_array):
        for letters in name_data:
            value = b"".join(letters.data).strip().decode("utf-8")
            # names.append(str(b''.join(letters.data)).strip())
            names.append(value)
    else:
        for letters in name_data:
            value = b"".join(letters).strip().decode("utf-8")
            names.append(value)
            # names.append(str(b''.join(letters)).strip())
    return names


def discrete_cmap(self, N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap, N)
    color_list = base(np.linspace(0, 1))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def searchcell(array, value):
    """
    look for cell where elements should be inserted
    """
    idx = np.searchsorted(array, value, side="right")
    if idx == 0 or idx == len(array) or np.ma.is_masked(value):
        return np.nan
    else:
        return idx - 1


def compute_last_day_month(y, m):
    return (dt.date(y + int(m / 12), m % 12 + 1, 1) - dt.date(y, m, 1)).days


def find_index_month(date):
    dateval = pd.to_datetime(date)
    index = dateval.month
    return index


def lon_lat_to_cartesian(lon, lat, R=1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


class nf(float):
    def __repr__(self):
        s = f"{self:.1f}"
        return f"{self:.0f}" if s[-1] == "0" else s


def findListFiles(path, pfx, sfx, time, dt_start=np.nan, dt_end=np.nan):
    liste_file = []
    for t in time:
        ffile = pfx + t.strftime("%Y%m%d") + "*" + sfx + "*.nc"
        file_mod = sorted(glob(path + ffile))
        if len(file_mod) == 0:
            print(f"File missing for time {pfx} {t.strftime('%Y%m%d')} {sfx} {path}")
            sys.exit(1)
        if np.isnan(dt_start) and np.isnan(dt_end):
            if len(file_mod) > 0 and file_mod[0] not in liste_file:
                liste_file.append(file_mod[0])
        else:
            if (
                len(file_mod) > 0
                and file_mod[0] not in liste_file
                and (int(t.strftime("%Y%m")) >= dt_start and int(t.strftime("%Y%m")) <= dt_end)
            ):
                liste_file.append(file_mod[0])
    return liste_file


def readDataSet(list_f, data_vars="all"):
    try:
        datasets = [xr.open_dataset(fname, chunks={}) for fname in list_f]
        mfdataset = xr.concat(datasets, dim="time_counter", data_vars=data_vars)
        time = mfdataset.variables["time_counter"][:].values
    except Exception as exs:
        log.error("Executation mfdataset failed: %s", exs)
        raise
    return mfdataset, time
