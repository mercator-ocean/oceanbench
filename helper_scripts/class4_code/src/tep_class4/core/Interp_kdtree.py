import os, sys
import numpy as np
import numpy.ma as ma

# Timer
import time

try:
    from pykdtree.kdtree import KDTree

    kd_tree_name = "pykdtree"
except ImportError:
    try:
        from scipy.spatial import cKDTree

        kd_tree_name = "scipy.spatial"
    except ImportError:
        raise ImportError("Either pykdtree or scipy must be available")
import xarray as xr
from tep_class4.core.utils import *
from . import utilities as utils
import netCDF4
from .Logger import Logger

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle


class interp_KDtree(object):
    """
    Class to compute weight file using kdtree
    """

    def __init__(self, log):
        self.kdname = kd_tree_name
        self.log = log

    def timeit(method):
        """
        Function to time the execution of a function
        """

        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if "log_time" in kw:
                name = kw.get("log_name", method.__name__.upper())
                kw["log_time"][name] = int((te - ts) * 1000)
            else:
                print("%r  %2.2f s" % (method.__name__, (te - ts)))
            return result

        return timed

    # @timeit
    def tree_and_query(self, X1, Y1, lon_out, lat_out, nbpts_interp, weight_file="", save=False, mask=None):
        """
        Compute tree and search weight files
        """

        if len(X1.shape) > 1:
            X1, Y1 = X1, Y1
            # self.log.debug('Shape 1 > 1')
            # self.log.debug('Shape 1 {}'.format(np.shape(X1)))
        else:
            X1, Y1 = np.meshgrid(X1, Y1)
            self.log.debug("Shape 1 = x {} y {}".format(np.shape(X1), np.shape(Y1)))
        if isinstance(lon_out, np.float64) or isinstance(lon_out, np.float32):
            self.log.debug("lon_out is a float")
        else:
            ll_onevalue = False
            # if len(lon_out)> 1 and lon_out.ndim < 2:
            if len(lon_out) > 2 and lon_out.ndim < 2:
                try:
                    if ma.is_masked(lon_out):
                        lon_out_tmp = lon_out[~lon_out.mask]
                    else:
                        lon_out_tmp = lon_out
                        lat_out_tmp = lat_out
                    global_step = float(
                        "{0:.3f}".format((lon_out_tmp[len(lon_out_tmp) - 1] - lon_out_tmp[0]) / (len(lon_out_tmp) - 1))
                    )
                    global_step_lat = float(
                        "{0:.3f}".format((lat_out_tmp[len(lat_out_tmp) - 1] - lat_out_tmp[0]) / (len(lat_out_tmp) - 1))
                    )
                    first_step = float("{0:.3f}".format(lon_out_tmp[1] - lon_out_tmp[0]))
                    first_step_lat = float("{0:.3f}".format(lat_out_tmp[1] - lat_out_tmp[0]))
                    self.log.debug(f"Step sizes {global_step} {first_step}")
                    self.log.debug(f"Step sizes lat {global_step_lat} {first_step_lat}")
                except Exception as e:
                    raise ("Problem during formatting {}".format(e))
            else:
                first_step = 1.0
                global_step = 2.0
                ll_onevalue = True
                self.log.debug("Only one step")

        # 1D cases
        if lon_out.ndim == 0 or ll_onevalue == True:
            self.log.debug("1D not regular")
            X2, Y2 = lon_out, lat_out
        elif lon_out.ndim < 2:
            self.log.debug(f"lon_out.ndim < 2 {global_step} {first_step}")
            self.log.debug(f"lon_out.ndim < 2 {global_step_lat} {first_step_lat}")
            # self.log.debug(f'{lon_out_tmp[0:100]}')
            if global_step == first_step and first_step > 0 and global_step_lat == first_step_lat:
                # Regular
                self.log.debug(f"1D regular => convert to meshgrid {global_step} {first_step}")
                X2, Y2 = np.meshgrid(lon_out, lat_out)
            else:
                # Irregular
                self.log.debug("1D not regular")
                X2, Y2 = lon_out, lat_out
        # 2D case
        else:
            self.log.debug("2D case")
            X2, Y2 = lon_out, lat_out

        # self.log.debug('Shape 2 = x {} y {}'.format(np.shape(X2), np.shape(Y2)))
        # Convert to cartesian
        xt, yt, zt = utils.lon_lat_to_cartesian(X2.flatten(), Y2.flatten())
        ### Convert to cartesian
        xs, ys, zs = utils.lon_lat_to_cartesian(X1.flatten(), Y1.flatten())
        if self.kdname == "pykdtree":
            coords = np.zeros((xs.size, 3), dtype=float)
            coords[:, 0] = xs
            coords[:, 1] = ys
            coords[:, 2] = zs
            coords2 = np.zeros((xt.size, 3), dtype=float)
            coords2[:, 0] = xt
            coords2[:, 1] = yt
            coords2[:, 2] = zt
        ## Compute weight file
        if os.path.exists(weight_file):
            # print ("load existing tree :%s "%(weight_file))
            if sys.version_info[0] >= 3:
                with open(weight_file, "rb") as f:
                    zip_var = pickle.load(f)
                    d, inds_idw = list(zip(*zip_var))
            else:
                f = file(str(weight_file), "r")
                zip_var = cPickle.load(f)
                d, inds_idw = zip(*zip_var)
            d1 = np.array(d)
            inds_idw1 = np.array(inds_idw)
            f.close
        else:
            ll_tree = True
            if sys.version_info[0] >= 3:
                if self.kdname == "pykdtree":
                    # --- pykdtree method ---
                    tree = KDTree(coords)
                    d1, inds_idw1 = tree.query(coords2, k=nbpts_interp, mask=mask)
                else:
                    # --- cKDTree method ---
                    tree = cKDTree(list(zip(xs, ys, zs)))
                    d1, inds_idw1 = tree.query(list(zip(xt, yt, zt)), k=nbpts_interp)
                if save:
                    with open(weight_file, "wb") as f:
                        pickle.dump(list(zip(d1, inds_idw1)), f, pickle.HIGHEST_PROTOCOL)
                        f.close
            else:
                if self.kdname == "pykdtree":
                    # --- pykdtree method ---
                    tree = KDTree(coords)
                    d1, inds_idw1 = tree.query(coords2, k=nbpts_interp)
                else:
                    # --- cKDTree method ---
                    tree = cKDTree(zip(xs, ys, zs))
                    d1, inds_idw1 = tree.query(zip(xt, yt, zt), k=nbpts_interp)
                if save:
                    with open(weight_file, "wb") as f:
                        cPickle.dump(zip(d1, inds_idw1), f, protocol=cPickle.HIGHEST_PROTOCOL)
                        f.close

        return d1, inds_idw1

    def get_weight(self, d1, max_weight=1e10):

        if np.any(d1 != 0):
            # weight = 1.0 / d1**2
            weight = np.where(d1 != 0, 1.0 / d1**2, max_weight)
            # weight = np.where(d1 != 0, 1.0 / d1**2, 0)
        else:
            # Handle the case when d1 is zero, e.g., set weight to a default value or take alternative action
            weight = np.nan

        return weight

    @timeit
    def get_ilonlat(self, lon_in, lat_in, inds_idw1):

        ilat = []
        ilon = []
        for ind in inds_idw1:
            inds = np.where(lon_in == lon_in.flatten()[ind])
            if np.size(inds[0][:]) > 1:
                w = np.where(lat_in[inds] == lat_in.flatten()[ind])
                ilat.append(inds[0][w][0])
                ilon.append(inds[1][w][0])
            else:
                ilat.append(inds[0][0])
                ilon.append(inds[1][0])
        return ilat, ilon

    # @timeit
    def compute_interp(self, d1, inds_idw1, X2, nbpts_interp, var_in):
        """
        Compute the interpolation
        """
        # Interpolate using inverse distance weighting, using n nearest neighbours (k=n)
        if d1 != 0:
            w = 1.0 / d1**2
        else:
            # Handle the case when d1 is zero, e.g., set weight to a default value or take alternative action
            weight = np.nan
        if nbpts_interp > 1:
            ## Inverse Distance Weight interpolation
            tab_interp = np.sum(w * np.array(var_in).flatten()[inds_idw1], axis=1) / np.sum(w, axis=1)
            tab_interp.shape = X2.shape
        else:
            ## Nearest neighbour interpolation
            # print ("Interp nearest")
            tab_interp = ma.array(var_in).flatten()[inds_idw1].reshape(X2.shape)

        return tab_interp


def interpolator_2D(
    x1, x2, y1, y2, input_file, output_grid, weight_file, nbpts_interp, variable, dpth=0, time=0, save=False
):
    """
    -------------------
    2D interpolator

    Parameters
    --------------------
    x1 : name of input longitude array
    x2 : name of output longitude array
    y1 : name of input latitude array
    y2 : name of output latitude array
    input_file : input file
    output_grid : output file
    weight_file : weight file
    nbpts_interp : number points for interpolation
        =>  1 = nearest
        =>  >1 IDW method
    tab_in : input variable array
    return interp tab
    """

    # Test the input
    if not os.path.exists(input_file):
        print("input grid doesn t exist : %s" % (input_file))
        sys.exit(1)
    if not os.path.exists(output_grid):
        print("output grid doesn t exist")
        sys.exit(1)
    ## Read output grid
    dataset_out = netCDF4.Dataset(output_grid, "r")
    lon_out = dataset_out.variables[x2]
    lat_out = dataset_out.variables[y2]
    ## Test size
    if len(lon_out.shape) > 1:
        X2, Y2 = np.array(lon_out), np.array(lat_out)
    else:
        X2, Y2 = np.meshgrid(lon_out, lat_out)

    ## Read input grid
    dataset_in = xr.open_dataset(input_file)
    lon_in = dataset_in[x1]
    lat_in = dataset_in[y1]
    var_in = dataset_in[variable]

    ## Define the depth to consider
    if dpth == 0:
        ind_dep = 0
    else:
        dep_tmp = dataset_in["deptht"]
        depthw = [0]
        for jk in np.arange(len(dep_tmp)):
            depthw.append(depthw[jk] + (dep_tmp[jk] - depthw[jk]) * 2)

        ind_dep = utils.searchcell(depthw, dpth)

    if len(var_in.shape) == 3:
        var_in = var_in[ind_dep, :, :]
    elif len(var_in.shape) == 4:
        var_in = var_in[time, ind_dep, :, :]

    ## Test size
    if len(lon_in.shape) > 1:
        X1, Y1 = np.array(lon_in), np.array(lat_in)
    else:
        X1, Y1 = np.meshgrid(lon_in, lat_in)

    ## Compute tree object and interp
    TreeObj = interp_KDtree()
    d1, inds_idw1 = TreeObj.tree_and_query(X1, Y1, X2, Y2, nbpts_interp, weight_file=weight_file, save=save)
    tab_interp = TreeObj.compute_interp(d1, inds_idw1, X2, nbpts_interp, var_in)

    return tab_interp, X2, Y2
