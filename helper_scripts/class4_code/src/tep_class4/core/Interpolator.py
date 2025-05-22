import os
import re
import sys
import numpy as np
import numpy.ma as ma
import json
from scipy.spatial import cKDTree
import xarray as xr
import vertical
from .reader import ReadModel
from tep_class4.core.utils import find_index_month
from .inputdata_loader import inputdata_loader
from collections import OrderedDict
from collections import Counter
from .Interp_kdtree import interp_KDtree
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from .Writer import Writer
from glob import glob
from .reader import read_GODAE_profile
from .Selector import Selector
from netCDF4 import Dataset
import netCDF4
from timeit import default_timer as timer
#import dask
#from dask.distributed import Client, SSHCluster
from joblib import Parallel, delayed
import warnings

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle

class BroadcastError(Exception):
        pass

class Interpolator(object):
    """
        Interpolator Class File : Interpolation with KDtree method
    """

    def __init__(self, log, dask=False):
        warnings.filterwarnings("ignore")
        self.log = log
        self.lonnames = ['lon', 'Lon', 'longitude', 'longitudes', 'Longitude', 'Longitudes']
        self.latnames = ['lat', 'Lat', 'latitude', 'latitudes', 'Latitude', 'Latitudes']
        self.missing_value = netCDF4.default_fillvals['f4']
        self.list_ssh = ["sossheig", "ssh", "zos"]
        self.ll_UVcorr = False
        self.compute_corr = False
        self.model_var = ['uo', 'vo', 'vozocrtx', 'vomecrty']
        self.listUV = ['drifter', 'drifter_filtr']
        self.list_insitu = ['BGC_ARGO', 'NC_CORIO_ORIGIN', 'NC_CORIO']
        self.tab_interp = {}
        if dask:
            self.log.debug("Launch dask server")
            #client = Client(n_workers=2, threads_per_worker=2, processes=False)
            #client
            #   worker_options={"nthreads": 2},
            cluster = SSHCluster( ["fidjim05-sihpc","fidjim06-sihpc","fidjim07-sihpc","fidjim08-sihpc"],
                                    connect_options={"known_hosts": "/home/smer/regnierc/.ssh/known_hosts" },
                                   scheduler_options={"port": 0, "dashboard_address": ":8797"})
            client = Client(cluster)
            client
        else:
            self.log.debug("Dask server not defined")

    def find_lon_lat_names(self, ds):
        coords = ds.coords.keys()
        lon_name = next((name for name in coords if name in self.lonnames), None)
        lat_name = next((name for name in coords if name in self.latnames), None)

        if lon_name is None or lat_name is None:
            raise ValueError("Could not find both latitude and longitude coordinate names.")

        return lon_name, lat_name
        
    def getneardepth(self, array, value):
       idx = (np.abs(array-value)).argmin()
       return idx

    def getnearpos(self, array, value):
       if array == 0 :
           idx1 = 0
           idx2 = 0
       else:
          idx = (np.abs(array-value)).argmin()
          if value[idx] >= array:
             idx1 = idx - 1
             idx2 = idx
          else:
             idx1 = idx
             idx2 = idx + 1
       return idx1, idx2

    def getweight(self, depth1, depth2, depth):
        dist_tot = abs(depth2 - depth1)
        w1 = abs(1 -(abs(depth1 - depth) / dist_tot))
        w2 = abs(1- (abs(depth2 - depth) / dist_tot))

        if w1 + w2 > 1.0001 :
           self.log.error(f"Pb with weight files {w1} + {w2} > 1.00001 => {w1+w2}")
           sys.exit(1)
        return w1, w2

    def read_input_modeldata(self, param_dict, date, leadtime, lead_int):
        self.log.info(f"Selection of input model {param_dict['data_type']} {param_dict['data_format']}")
        self.log.info(f"{leadtime=}")
        self.log.info('========================================================')
        self.list_input_file, self.ll_miss = Selector(param_dict['data_format'], param_dict['data_type'], self.log).\
            factory("GLOBAL_INPUT_MODEL_DATA").run(param_dict, date, leadtime, lead_int)
        self.log.debug(f'Select model ok {self.list_input_file}')
        self.nb_best, self.nb_fcst, self.nb_pers, self.nb_clim, self.nb_nwct, self.nb_clim, self.nb_bathy = \
                    self.check_dimensions(self.list_input_file)
        self.log.debug(f"Dimensions best : {self.nb_best} nwct: {self.nb_nwct} "\
                      f"fcst : {self.nb_fcst} pers : {self.nb_pers} clim : {self.nb_clim}"\
                      f"bathy : {self.nb_bathy}")

    def search_dims(self, filename):
        """
        Find the lon/lat dimensions of a file
        """
        nc = Dataset(filename,'r')
        list_lon = ['x','X','lon','long','longitude','longitudes']
        list_lat = ['y','Y','lat','lati','latitude','latitudes']
        nlon = None
        nlat = None
        for dim in nc.dimensions.keys():
            if dim.lower() in list_lon: nlon = nc.dimensions[dim].size
            if dim.lower() in list_lat: nlat = nc.dimensions[dim].size

        return nlon, nlat

    def compute_weight(self, ind, TreeObj, lon_mod, lat_mod, lon_obs, lat_obs,nb_ptsinterp, mask):
        if np.isnan(lon_obs).all() and np.isnan(lat_obs).all():
            length = len(lon_obs)  # Assuming lon_obs and lat_obs are the same length
            weight = np.full(length, np.nan)
            inds = np.full(length, np.nan)
        else:
            ## Compute weight at each level
            d1, inds = TreeObj.tree_and_query(lon_mod, lat_mod,
                                              np.array(lon_obs), np.array(lat_obs), nb_ptsinterp,\
                                              weight_file='test.p',save=False, mask=mask)
            weight = TreeObj.get_weight(d1)

        return weight, inds


    def check_dimensions(self, list_input_file):
        """
        Check the existence and the dimensions of different lead times
        Arguments
        --------------------
        list_input_file : dictionnary with all the files to interp
        Returns
        --------------------
        len(best), len(fcst), len(pers), len(clim): len of different lead time files
        """
        best = {k for k, v in list_input_file.items() if "hdct" in k}
        clim = {k for k, v in list_input_file.items() if "clim" in k}
        #fcst = {k for k, v in list_input_file.items() if "fcst" in k}
        fcst = {k for k, v in list_input_file.items() if k.startswith("fcst")}
        pers = {k for k, v in list_input_file.items() if "pers" in k}
        nwct = {k for k, v in list_input_file.items() if "nwct" in k}
        clim = {k for k, v in list_input_file.items() if "clim" in k}
        bathy = {k for k, v in list_input_file.items() if "bathy" in k}
        return len(best), len(fcst), len(pers), len(clim), len(nwct), len(clim), len(bathy)

    def interp_class4(self, obs_data , model_data, variable, param_dict, datevalue=None, leadtime=None):
        """
        Read and Interp data files
        Read input file, compute coloc value and interp with KDtree
        Arguments
        --------------------
        obs_data : tuple for the observations
        model_data : input xarray file or zarr array
        variable: name of the variable
        param_dict : dictionnary with all the parameters
        Returns
        --------------------
        tab_results : dictionnary with interpolated arrays
        """
        # Initialisation
        tab_results = {}
        nb_ptsinterp = param_dict['nb_points']
        tuple_obs = (np.shape(obs_data.lon)[0], np.shape(obs_data.lat)[0])
        tab_results['lon'] = obs_data.lon
        tab_results['lat'] = obs_data.lat
 
        if 'varname_select' in param_dict.keys():
            tab_results['varname_select'] = param_dict['varname_select']
        if 'qc' in param_dict.keys():
            tab_results['qc'] = param_dict['qc']
        # Get lon_mod / lat_mod
        lon_name, lat_name = self.find_lon_lat_names(model_data)
        lon_mod = model_data[lon_name].values
        lat_mod = model_data[lat_name].values
        #print(list(model_data.data_vars))
        if param_dict['cl_typemod'].lower() == "pgs" or param_dict['cl_typemod'].lower() == "pgs_ai":
            self.log.debug("PGS coordinates")    
            lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('PGS_coords').\
                read_pos(param_dict['coordinates'])
        if 'depth' in list(model_data.data_vars):
            depth_mod = model_data['depth'].values
        ll_singleobs = False
        same_grid = False
        ll_light = False
        ll_clim = False
        ll_case2 = False
        ll_nan = False
        ll_uniq = False
        is_2D = False
        if len(np.unique(obs_data.depth)) == 1 or len(np.shape(obs_data.depth))==1 :
            is_2D = True
            nb_obs = np.shape(obs_data.depth)[0]
            self.log.info(f"Dimensions obs {nb_obs}")
            if len(variable)==1 :
                reshaped_obs = obs_data.obs_value.reshape((*obs_data.obs_value.shape, 1, 1))
                tab_results['depth'] = np.expand_dims(obs_data.depth, axis=0)
            else:
                reshaped_obs = obs_data.obs_value.reshape((*obs_data.obs_value.shape, 1))
                tab_results['depth'] = np.expand_dims(obs_data.depth, axis=1)
            
            tab_results['observation'] = reshaped_obs
            ll_singleobs = True
            ll_uniq = True
            #tab_results[leadtime] = {}
            self.log.debug("Uniq depth for all the observations")
            if ~np.isnan(obs_data.lon).all() and ~np.isnan(obs_data.lat).all():
                self.log.debug("Compute tree")
                TreeObj = interp_KDtree(self.log) #level=self.log.level)
                d1, indexes = TreeObj.tree_and_query(lon_mod, lat_mod,
                                                     obs_data.lon, obs_data.lat, nb_ptsinterp,
                                                     weight_file='',save=False)
                weights = TreeObj.get_weight(d1)
                self.log.debug("Compute tree ok")
                if len(np.shape(obs_data.depth)) == 1 :
                    depthval = obs_data.depth[0]
                elif len(np.shape(obs_data.depth)) == 2 :
                    depthval =  obs_data.depth[0,0]
                else:
                    self.log.error("Case not known")
                self.log.debug(f"Interp 2D {variable=} {depthval}")
                array_results = np.full(
                                        shape=(nb_obs, len(variable), 1),
                                        #fill_value=netCDF4.default_fillvals['f4'],
                                        fill_value=np.nan,
                                        dtype=float,
                                        order='F'
                                        )
                for var in range(len(variable)):
                    varname = variable[var] 
                    interp_values = self.interp_2D(weights, indexes,
                                                   model_data,
                                                   nb_ptsinterp,
                                                   varname, param_dict,
                                                   depth=depthval, fillvalue=obs_data.FillValue,
                                                   time=0, nblon=tuple_obs[0], nblat=tuple_obs[1],
                                                   lon_mod=lon_mod, lat_mod=lat_mod,
                                                   #missing_value=self.missing_value,
                                                   missing_value=np.nan,
                                                   same_grid=same_grid)
                    self.log.debug(f"Interp value {interp_values[0:20]}{var}")
                    array_results[:, var, 0] = interp_values
                tab_results[leadtime] = array_results
                    #tab_results[leadtime][varname] = np.expand_dims(interp_values, axis=0)
                ##    reshaped = interp_values.reshape((*interp_values.shape, 1, 1))
                ##    tab_results[leadtime] = reshaped
                    #tab_results[leadtime] = np.expand_dims(interp_values, axis=0)

            else:
                self.log.error("Missing lon/lat")
                sys.exit(1)
        else:
            self.log.debug("Multiple depths")
            self.log.debug("Profile case variable")
            nb_obs, nb_profs = np.shape(obs_data.depth)
            ll_case2 = True
            ## Compte weight file
            tab_results['depth'] = obs_data.depth
            tab_results['observation'] =  obs_data.obs_value
            self.log.debug(f"Nb_profils {nb_obs} Nb levels {nb_profs}")
            maxdepth =  np.nanmax(obs_data.depth)
            self.log.debug(f'maxdepth {maxdepth}')
            ind_depth = self.getneardepth(depth_mod, maxdepth)
            ind_tot = len(depth_mod)
            ## Loop on levels
            if ind_depth < ind_tot-2:
                ind_max = ind_depth+2
            else:
                #ind_max = ind_depth + 1
                ind_max = ind_depth
            self.log.debug(f'maxdepth {maxdepth} maxind {ind_max} nearest depth model {depth_mod[ind_depth]}')
            ## Compute weight
            TreeObj = interp_KDtree(self.log)
            tab_weight = np.nan*np.zeros((nb_obs,nb_ptsinterp,ind_max),dtype=float)
            tab_indexes = np.ndarray(shape=(nb_obs, nb_ptsinterp,ind_max), dtype=int, order='F')
            start = timer()
            ## Parallelize Loop on levels
            self.log.debug(f'Dimensions {len(obs_data.lon)}')
            nb_compute = len(obs_data.lon)
            num_cores = -1
            # Compute weight
            # Implement lon/lat = 0 for all the values:
            with Parallel(n_jobs=num_cores, prefer="threads", verbose=1) as parallel:
                weight, indexes  = zip(*parallel(delayed(self.compute_weight)(ind,TreeObj,lon_mod,lat_mod,np.array(obs_data.lon),\
                                    np.array(obs_data.lat), nb_ptsinterp,\
                                    mask_tab[ind,:,:]) for ind in range(ind_max)))
            tab_weight = np.asarray(weight)
            tab_indexes = np.asarray(indexes)
            end = timer()
            self.log.debug('~~~~~~~~ Time elapsed compute weight on levels: %s %s' %
                            (str(end - start), " seconds ~~~~~~~~~~~~"))
            self.log.debug(f"{obs_data.depth=} {ind_max=} ")
            self.log.debug(f"{tab_indexes=}")
            array_results = np.full(
                                  shape=(nb_obs, len(variable), nb_profs),
                                         #fill_value=netCDF4.default_fillvals['f4'],
                                         fill_value=np.nan,
                                         dtype=float,
                                         order='F'
                                         )

            for var in range(len(variable)):
                varname = variable[var]
                if param_dict['data_format'] in self.list_insitu:
                    obs_tab = None
                else:
                    obs_tab = obs_data.obs_value[:,var,:]
                array_results[:,var,:] = self.interp_3D(tab_weight,
                                               tab_indexes,
                                               ind_max,
                                               model_data, nb_ptsinterp,
                                               varname, obs_data.depth, ll_verif = param_dict['verif'],
                                               obs = obs_tab,
                                               dateval=datevalue, lon=obs_data.lon, lat=obs_data.lat, light=ll_light,
                                               clim=ll_clim, depth_mod=depth_mod,
                                               fillvalue=np.nan, data_file='',
                                               #fillvalue=self.missing_value, data_file='',
                                               ll_uniq=ll_uniq, ll_nan=ll_nan, ll_case2=ll_case2)
            tab_results[leadtime] = array_results
        
        return tab_results, is_2D
        
    def read_and_interp(self, data_file, param_dict, date=None, leadtime=None, lead_int=None):
        """
        Read and Interp data files
        Read input file, compute coloc value and interp with KDtree
        Arguments
        --------------------
        data_file : input data file
        list_input_file : list of input files to interp
        param_dict : dictionnary with all the parameters
        Returns
        --------------------
        tab_results : dictionnary with interpolated arrays
        file_out : name of output file
        """
        # Initialisation
        ll_clim = False
        ll_json = False
        ll_verif = False
        ## Todo write into a file
        nb_ptsinterp = param_dict['nb_points']
        # Read input obs value
        self.log.debug(f"Format of the data {param_dict['data_format']}")
        if param_dict['prefix_data'].startswith("BGC_ARGO"):
            variable_obs = param_dict['prefix_data'].split('BGC_ARGO_')[1]+'_ADJUSTED'
            lon_obs, lat_obs, obs_value, nb_obs, depth, _FillValue = \
                    inputdata_loader(self.log).factory(param_dict['data_format']).\
                    read_pos_and_value(data_file, param_dict, varname=variable_obs)
        else:
            self.log.debug(f"Read pos and value : {param_dict['data_format']}")
            lon_obs, lat_obs, obs_value, nb_obs, depth, tab_dims, param_dict,  _FillValue = \
                    inputdata_loader(self.log).factory(param_dict['data_format']).\
                    read_pos_and_value(data_file, param_dict)
        tuple_obs = (np.shape(lon_obs)[0], np.shape(lat_obs)[0])
        ## TODO : Find this in another part because it's not well design for variable that are not model or add the exception of bathymetry , clim, stokes etc inside this function
        self.read_input_modeldata(param_dict, date, leadtime, lead_int)
        ## Find input model files
        # Selection of the input model data
        self.log.debug(f"Selection of {lon_obs} {lat_obs}")
        self.log.debug(f"Dimensions obs {len(np.shape(obs_value))}")
        #self.log.info(f"Selection of input model {param_dict['data_type']}")
        #self.log.info('========================================================')
        #list_input_file, ll_miss = Selector(param_dict['data_format'], param_dict['data_type'], self.log).\
        #    factory("GLOBAL_INPUT_MODEL_DATA").run(param_dict, date, leadtime, lead_int)
        #self.log.debug(f'Select model ok {list_input_file}')
        #nb_best, nb_fcst, nb_pers, nb_clim, nb_nwct = \
        #            self.check_dimensions(list_input_file)
        self.log.info(f"Interp file {data_file}")
        #self.log.info(f"Dimensions best : {nb_best} nwct: {nb_nwct} "\
        #              f"fcst : {nb_fcst} pers : {nb_pers} clim : {nb_clim}")
        #nb_profs, nb_levels, nb_vars = np.shape(obs_value)
        ## TODO compare parameters in obs_value and input variables to remove variable if necessary
        ll_singleobs = False
        if nb_obs == 1:
            ll_singleobs = True
        if param_dict['coordinates_hgr']:
            try:
                self.log.debug("Read coordinates for the correction of UV drifters")
                mod_hgr = xr.open_dataset(param_dict['coordinates_hgr'])
                self.gsinU = mod_hgr['gsinu']
                self.gcosU = mod_hgr['gcosu']
                self.gsinV = mod_hgr['gsinv']
                self.gcosV = mod_hgr['gcosv']
                self.gsinT = mod_hgr['gsint']
                self.gcosT = mod_hgr['gcost']
                self.log.debug(f"Read ok")
            except IOError:
                self.log.error("Open Error coordinates_hgr")
            self.log.debug("Open hgr ok")

        if param_dict['corr']:
            self.compute_corr = True
            self.log.info(f"Add Correction {param_dict['corr']}")
        else:
            self.log.debug(f"No correction")
        self.log.debug(f"{lon_obs=}")
        ll_best = False
        nb_obs2, nb_profs = np.shape(depth)
        self.log.debug("Number of observations {} Profondeurs {}".format(nb_obs2, nb_profs))
        if 'varname_select' in param_dict.keys():
            variable = param_dict['varname_select']
        else:
            variable = param_dict['varname']
        self.log.debug(f"Variable : {variable}")
        if self.nb_best > 0:
            if param_dict['data_origin'] == "GLOBCOLOUR":
                hdct_tab = np.ndarray(shape=(nb_obs, 1, 1), dtype=float, order='F')
            else :
                hdct_tab = np.ndarray(shape=(nb_obs, len(variable), nb_profs), dtype=float, order='F')
            hdct_tab[:, :, :] = self.missing_value
            self.log.debug("Taille hdct_tab {}".format(np.shape(hdct_tab)))
            ll_best = True
        ll_fcst = False
        self.log.debug(f"Number of forecasts {self.nb_fcst}")
        self.log.debug("---------------------------------")
        if self.nb_fcst > 0:
            if 'nb_fcsts' not in tab_dims.keys():
                ## CHR add option for BIO CHL 2D files
                nb_fcst = len(param_dict['lead_int'])
                if param_dict['data_origin'] == "GLOBCOLOUR":
                    fcst_tab = np.ndarray(shape=(nb_obs, 1, nb_fcst, 1), dtype=float, order='F')
                else:
                    fcst_tab = np.ndarray(shape=(nb_obs, len(variable), nb_fcst, nb_profs),
                                          dtype=float, order='F')
            else:
                fcst_tab = np.ndarray(shape=(nb_obs, len(variable), tab_dims['nb_fcsts'], nb_profs),
                                      dtype=float, order='F')
            fcst_tab[:, :, :, :] = self.missing_value
            ll_fcst = True
        else:
            if param_dict['fcst_mode']:
                if param_dict['data_origin'] == "GLOBCOLOUR":
                    fcst_tab = np.ndarray(shape=(nb_obs, 1, 1, 1), dtype=float, order='F')
                else:
                    nb_fcst = len(param_dict['lead_int'])
                    #fcst_tab = np.ndarray(shape=(nb_obs, len(variable), tab_dims['nb_fcsts'], nb_profs),
                    fcst_tab = np.ndarray(shape=(nb_obs, len(variable), nb_fcst, nb_profs),
                                      dtype=float, order='F')
                fcst_tab[:, :, :, :] = self.missing_value
        ll_pers = False
        if self.nb_pers > 0 or self.nb_nwct >0:
            if 'nb_fcsts' not in tab_dims.keys():
                nb_fcst = len(param_dict['lead_int'])
                #pers_tab = np.ndarray(shape=(nb_obs, len(variable), nb_pers+1, nb_profs),
                #                      dtype=float, order='F')
                pers_tab = np.ndarray(shape=(nb_obs, len(variable), nb_fcst, nb_profs),
                                      dtype=float, order='F')
            else:
                nb_fcst = len(param_dict['lead_int'])
                #pers_tab = np.ndarray(shape=(nb_obs, len(variable), tab_dims['nb_fcsts'], nb_profs),
                pers_tab = np.ndarray(shape=(nb_obs, len(variable), nb_fcst, nb_profs),
                                      dtype=float, order='F')
            pers_tab[:, :, :, :] = self.missing_value
            ll_pers = True
        if self.nb_clim > 0:
            if param_dict['data_origin'] == "GLOBCOLOUR":
                clim_tab = np.ndarray(shape=(nb_obs, 1, 1),
                           dtype=float, order='F')
                clim_tab[:, :, :] = self.missing_value
            else:
                clim_tab = np.ndarray(shape=(nb_obs, len(variable), nb_profs),
                                      dtype=float, order='F')
                clim_tab[:, :, :] = self.missing_value
            ll_clim = True
            self.log.debug('Clim ok')
        self.log.debug('Number obs to coloc {} '.format(nb_obs2))
        file_coloc_tmp = os.path.basename(data_file)
        file_out = file_coloc_tmp
        list_2D = ["smoc", "tides", "clim", "bathy", "stokes"]
        list_2D_smoc = ["smoc", "tides", "stokes"]
        # Loop on lead times
        ll_light = False
        ll_smoc = False
        ll_smoc_fcst = False
        ll_tides = False
        ll_stokes = False
        ll_stokes_fcst = False
        ll_c_smoc =  False
        ll_c_stokes =  False
        ll_bathy = False
        tab_depth = {}
        tab_results = {}
        indice = 0

        for i, key in enumerate(self.list_input_file.keys()):
            self.log.debug(f"Lead time {key}")
            if 'varname_select' in param_dict.keys():
                variable = param_dict['varname_select']
            else:
                variable = param_dict['varname']
            if key.lower() == "smoc":
                variable = param_dict['varname_smoc']
                ll_smoc = True
                smoc_tab = np.ndarray(shape=(nb_obs,len(variable), nb_profs),
                           dtype=float, order='F')
                smoc_tab[:, :, :] = self.missing_value
            if key.lower().startswith("smoc_fcst"):
                variable = param_dict['varname_smoc']
                ll_smoc_fcst = True
                if not ll_c_smoc :
                    nb_fcst2 = len(param_dict['lead_int2'])
                    smoc_fcst_tab = np.ndarray(shape=(nb_obs,len(variable), nb_fcst2, nb_profs),
                           dtype=float, order='F')
                    smoc_fcst_tab[:, :, :, :] = self.missing_value
                    ll_c_smoc = True
            if key.lower() == "tides":
                variable = param_dict['varname_tides']
                ll_tides = True
                tides_tab = np.ndarray(shape=(nb_obs, len(variable), nb_profs),
                           dtype=float, order='F')
                tides_tab[:, :, :] = self.missing_value
            if key.lower() == "stokes":
                variable = param_dict['varname_stokes']
                ll_stokes = True
                if not ll_c_stokes :
                    stokes_tab = np.ndarray(shape=(nb_obs, len(variable), nb_profs),
                               dtype=float, order='F')
                    stokes_tab[:, :, :] = self.missing_value
                    ll_c_stokes = True
            if key.lower().startswith("stokes_fcst"):
                nb_fcst2 = len(param_dict['lead_int2'])
                variable = param_dict['varname_stokes']
                ll_stokes_fcst = True
                stokes_fcst_tab = np.ndarray(shape=(nb_obs, len(variable), nb_fcst2, nb_profs),
                           dtype=float, order='F')
                stokes_fcst_tab[:, :, :, :] = self.missing_value
            if key.lower() == "bathy":
                ll_bathy = True
                variable = param_dict['varname_bathy']
                bathy_tab = np.ndarray(shape=(nb_obs,nb_profs), dtype=float, order='F')
                bathy_tab[:, :] = self.missing_value
            # Create weight file
            if key.lower() == "clim": ll_clim = True
            if key.lower() == "clim" and param_dict['data_type'].lower() == 'chloro':
                lon_mod, lat_mod = inputdata_loader(self.log).factory(
                    'NC_CLIM').read_pos(param_dict['coord_clim'])
                file_coloc_tmp = re.sub('.nc', '_Kdtree_CLIM_' +
                                        param_dict['cl_typemod'].upper()+'_idw' +
                                        str(nb_ptsinterp) + 'pt.p', file_out)
                variable = []
                #variable.append(param_dict['varname_select'][0].upper()+'_mean')
                variable.append(param_dict['varname_select'][0].upper()+'_mean')
                nlon, nlat = self.search_dims(param_dict['coord_clim'])
                tuple_coord = (nlon, nlat)
            elif key.lower() == "clim" and param_dict['data_type'].lower() == 'drifter_filtr' \
                or key.lower() == "clim" and param_dict['data_type'].lower() == 'drifter' \
                or key.lower() == "clim" and param_dict['data_type'].lower() == 'current':
                self.log.debug("CLIM coordinates")
                lon_mod, lat_mod = inputdata_loader(self.log).factory(
                    'NC_CLIM').read_pos(param_dict['coord_clim'])
                variable = ['U', 'V']
                nlon, nlat = self.search_dims(param_dict['coord_clim'])
                tuple_coord = (nlon, nlat)
            elif key.lower() == "bathy":
                self.log.debug("bathy coordinates")
                lon_mod, lat_mod = inputdata_loader(self.log).factory(
                    'bathymetrie').read_pos(param_dict['bathy_file'])
                nlon, nlat = self.search_dims(param_dict['bathy_file'])
                tuple_coord = (nlon, nlat)
            else:
                if param_dict['coordinates']:
                    if param_dict['cl_typemod'].lower() == "pgn":
                        if key.lower() in list_2D_smoc or key.lower().startswith("stokes_fcst") or key.lower().startswith("smoc_fcst"):
                            self.log.debug("SMOC coordinates")
                            lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('SMOC_coords').\
                                read_pos(param_dict['coordinates_SMOC'])
                        else:
                            self.log.debug("PGN coordinates")
                            lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('PGN_coords').\
                                read_pos(param_dict['coordinates'])

                    elif param_dict['cl_typemod'].lower() == "pgncg":
                        self.log.debug("PGNCG coordinates")
                        lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('PGN_coords').\
                                read_pos(param_dict['coordinates'])

                    elif param_dict['cl_typemod'].lower() == "allege":
                        lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('alleges_coords').\
                            read_pos(param_dict['coordinates'])
                        self.log.debug('Read coords Alleges OK ')
                        ll_light = True
                        if key.lower() == "clim" : ll_light = False
                    elif param_dict['cl_typemod'].lower() == "2ease":
                        lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('2ease_coords').\
                            read_pos(param_dict['coordinates'])
                    elif param_dict['cl_typemod'].lower() == "pgs" or param_dict['cl_typemod'].lower() == "pgs_ai":
                            self.log.debug("PGS coordinates")
                            lon_mod, lat_mod, mask_tab, depth_mod = inputdata_loader(self.log).factory('PGS_coords').\
                                read_pos(param_dict['coordinates'])
                    else:
                        self.log.error("Type mod not known")
                    nlon, nlat = self.search_dims(param_dict['coordinates'])
                    tuple_coord = (nlon, nlat)
                else:
                    self.log.error("No coordinates files")
                    # TODO read lon/lat from first file
                    raise
                file_coloc_tmp = re.sub('.nc', '_Kdtree_' +
                                        param_dict['cl_typemod'].upper()+'_idw' +
                                        str(nb_ptsinterp) + 'pt.p', file_out)
            weight_file = param_dict['coloc_rep'] + file_coloc_tmp
            self.log.debug("Weight_file: {} ".format(weight_file))
            same_grid = False
            if tuple_obs == tuple_coord:
                same_grid = True
                weights =  None
                indexes = None
            self.log.debug("Dimension tuples {} {} same_grid {}".format(tuple_obs,tuple_coord,same_grid))
            ll_uniq = False
            ll_nan = False
            ll_case2 = False
            # Coloc file
            if len(np.unique(depth)) == 1 or \
                key.lower() == "clim" and param_dict['data_type'].lower() == 'current':
                # Specifitc case for lon and lat == 0 
                ll_uniq = True
                self.log.debug("Uniq depth")
                if np.isnan(lon_obs).all() and np.isnan(lat_obs).all():
                    ll_nan = True
                    nb_obs, nb_profs = np.shape(depth)
                    length = len(lon_obs)  # Assuming lon_obs and lat_obs are the same length
                    weights = np.full((nb_obs, nb_profs), np.nan)
                    indexes = np.full((nb_obs, nb_profs), np.nan)
                else:
                    TreeObj = interp_KDtree(self.log) #level=self.log.level)
                    ## No mask with ice dataset create issues in the ANT
                    d1, indexes = TreeObj.tree_and_query(lon_mod, lat_mod,
                                                         lon_obs, lat_obs, nb_ptsinterp,\
                                                          weight_file=weight_file,save=False)
                    weights = TreeObj.get_weight(d1)
            elif len(np.unique(depth)) > 1:
                ll_case2 = True
                ## Compte weight file
                nb_obs, nb_profs = np.shape(depth)
                self.log.debug(f"Nb_profils {nb_obs} Nb levels {nb_profs}")
                maxdepth =  np.nanmax(depth)
                self.log.debug(f'maxdepth {maxdepth}')
                ind_depth = self.getneardepth(depth_mod, maxdepth)
                ind_tot = len(depth_mod)
                ## Loop on levels
                if ind_depth < ind_tot-2:
                   ind_max = ind_depth+2
                else:
                    #ind_max = ind_depth + 1
                    ind_max = ind_depth
                self.log.debug(f'maxdepth {maxdepth} maxind {ind_max} nearest depth model {depth_mod[ind_depth]}')
                ## Compute weight
                TreeObj = interp_KDtree(self.log)
                tab_weight = np.nan*np.zeros((nb_obs,nb_ptsinterp,ind_max),dtype=float)
                tab_indexes = np.ndarray(shape=(nb_obs, nb_ptsinterp,ind_max), dtype=int, order='F')
                start = timer()
                ## Parallelize Loop on levels
                self.log.debug(f'Dimensions {len(lon_obs)}')
                nb_compute = len(lon_obs)
                #num_cores = int(ind_max/10)
                #if num_cores < 1:
                #    num_cores = 1
                num_cores = 10
                ## Compute weight
                ## Implement lon/lat = 0 for all the values:
                with Parallel(n_jobs=num_cores, prefer="threads", verbose=0) as parallel:
                    weight, indexes  = zip(*parallel(delayed(self.compute_weight)(ind,TreeObj,lon_mod,lat_mod,np.array(lon_obs),\
                                                                    np.array(lat_obs), nb_ptsinterp,\
                                                                    mask_tab[ind,:,:]) for ind in range(ind_max)))
                tab_weight = np.asarray(weight)
                tab_indexes = np.asarray(indexes)
                end = timer()
                self.log.debug('~~~~~~~~ Time elapsed compute weight on levels: %s %s' %
                              (str(end - start), " seconds ~~~~~~~~~~~~"))

            # Interp model file for different lead times
            self.log.debug(f'Variable to interpolate {variable}')
            tab_results = {}
            list_file = []
            list_file = self.list_input_file[key]
            self.log.debug(f"Work on {len(variable)} variables")
            ## loop on variable
            for var in range(len(variable)):
               ## Get the correct file
               varname = variable[var]
               tab_results['varname'] = varname
               ll_miss = False
               try:
                   file_input = self.list_input_file[key][var]
               except IndexError as error:
                   self.log.error(f"Error {error} Missing file for key {key} var {var}")
                   self.log.error("Varname {}".format(varname))
                   ll_miss = True
                   sys.exit(1)
               ll_unique = False
               if not ll_miss:
                  if len(np.unique(depth)) == 1:
                      index_time=0
                      #if key.lower() == "clim": index_time=find_index_month(date)-1
                      if key.lower() == "clim" and param_dict['data_type'].lower() != 'chloro': index_time=find_index_month(date)-1
                      self.log.debug("Unique depth for all the observations")
                      if len(np.shape(depth)) == 1 :
                          depthval = depth[0]
                      elif len(np.shape(depth)) == 2 :
                          depthval = depth[0,0]
                      else:
                          self.log.error("Case not known")
                      ## Selection of the depth value
                      if key.lower() in list_2D or key.lower().startswith("stokes_fcst") or key.lower().startswith("smoc_fcst"):
                          tab_depth[key.lower()] = 0 
                      else:
                          tab_depth[key.lower()] = depthval 
                      self.log.debug("Depth of interpolation {}".format(tab_depth[key.lower()]))
                      self.log.debug("Index time {}".format(index_time))
                      self.log.debug(f"Interp 2D {varname=}")
                      self.log.debug(f"{indexes=}")
                      interp_values, ll_nan = self.interp_2D(weights, indexes,
                                                     file_input,
                                                     nb_ptsinterp,
                                                     varname, param_dict, key=key,
                                                     depth=tab_depth[key.lower()], fillvalue=_FillValue,
                                                     time=index_time, nblon=nlon, nblat=nlat,
                                                     lon_mod=lon_mod, lat_mod=lat_mod,
                                                     missing_value=self.missing_value,
                                                     same_grid=same_grid)
                      self.log.debug(f"Interp value {interp_values}")
                      self.compute = False
                      if varname in self.model_var and self.compute_corr and not self.compute: 
                          self.log.debug('Interp grid coordinates for the correction')
                          #self.tab_interp['gsinU'] = ma.array(self.gsinU).flatten()[indexes]
                          #self.tab_interp['gsinV'] = ma.array(self.gsinV).flatten()[indexes]
                          #self.tab_interp['gcosU'] = ma.array(self.gcosU).flatten()[indexes]
                          #self.tab_interp['gcosV'] = ma.array(self.gcosV).flatten()[indexes]
                          self.tab_interp['gsint'] = np.sum(weights * np.array(self.gsinT).flatten()
                                       [indexes], axis=1) / np.sum(weights, axis=1)
                          #ma.array(self.gsinT).flatten()[indexes]
                          self.tab_interp['gcost'] = np.sum(weights * np.array(self.gcosT).flatten()
                                        [indexes], axis=1) / np.sum(weights, axis=1)
                          #ma.array(self.gcosT).flatten()[indexes]
                          self.compute = True
                          # ucorr0 = values_u0*gcost_interp - values_v0*gsint_interp
                          # vcorr0 = values_u0*gsint_interp + values_v0*gcost_interp
                          # ucorr15 = values_u15_interp*gcost_interp - values_v15_interp*gsint_interp
                          # vcorr15 = values_u15_interp*gsint_interp + values_v15_interp*gcost_interp
                      if len(np.shape(obs_value)) ==  3:
                          ## Case for surface drifters and profiles with the dimensions 1 depth 1 profile
                          try:
                              if param_dict['data_type'].lower() == 'drifter_filtr' or param_dict['data_type'].lower() == 'drifter':
                                 obs_value2 = np.array(obs_value)[var, :, :]
                              else:
                                  #obs_value2 = np.array(obs_value)[:,:,var]
                                  obs_value2 = np.array(obs_value)[:, var, :]
                                  #obs_value2 = np.array(obs_value)[:, :, :].flatten()
                                  self.log.debug(f'Case 1 depth 1 profile : {obs_value2}')
                                  self.log.debug(f'{np.shape(obs_value2)}')
                                  self.log.debug(f'{np.shape(interp_values)}')

                              if not ll_nan:
                                interp_values = np.where(np.array(obs_value2).flatten() == _FillValue,
                                                         self.missing_value, np.array(interp_values))
                          except BroadcastError as e:
                              raise(f'PB file modele {file_input} obs {data_file} Error {e}')
                          self.log.debug(f'Interp : {interp_values}')

                      elif len(np.shape(obs_value)) ==  2:
                          interp_values = np.where(np.array(obs_value).flatten() == _FillValue,
                                                            self.missing_value, np.array(interp_values))
                      ll_unique = True
                  elif key.lower() == "clim" and param_dict['data_type'].lower() == 'current' :
                      depthval = 0
                      index_time=find_index_month(date)-1
                      interp_values = self.interp_2D(weights, indexes,
                                                     file_input,
                                                     nb_ptsinterp,
                                                     varname, param_dict, key=key,
                                                     depth=depthval, fillvalue=_FillValue,
                                                     time=index_time, nblon=nlon, nblat=nlat,
                                                     lon_mod=lon_mod, lat_mod=lat_mod,
                                                     missing_value=self.missing_value,
                                                     same_grid=same_grid)
                      self.log.debug("Max value for the interpolation {}".format(np.nanmax(interp_values)))
                      ll_unique = True
                      ##
                  else :
                     self.log.debug("Profile case variable : {}".format(variable[var]))
                     self.log.debug(f"{depth=} {ind_max=} ")
                     self.log.debug(f"{tab_indexes=}")
                     self.log.debug(f"{file_input=}")
                     #try:
                     if param_dict['data_format'] in self.list_insitu:
                         obs_tab = None
                     else:
                         obs_tab = obs_value[:,var,:]
                     interp_values = self.interp_3D(tab_weight,
                                                     tab_indexes,
                                                     ind_max,
                                                     file_input, nb_ptsinterp,
                                                     varname, depth, ll_verif = param_dict['verif'],
                                                     obs = obs_tab,
                                                     dateval=date, lon=lon_obs, lat=lat_obs, light=ll_light,
                                                     clim=ll_clim, depth_mod=depth_mod,
                                                     fillvalue=self.missing_value, data_file=data_file,
                                                     ll_uniq=ll_uniq, ll_nan=ll_nan, ll_case2=ll_case2)
                     #except Exception as e:
                     #    self.log.error(f"{param_dict['data_format']} {self.list_insitu} Pb interp 3D, Size obs {np.shape(obs_value[:,var,:])}")
                     #    self.log.error(f"Pb with file {e} {depth} {ind_max} {file_input} Obs {data_file}")
                     #    sys.exit(1)

                     ll_json = param_dict['json']
                     if ll_json:
                         if len(np.shape(obs_value)) ==  3: obs_value2 = np.array(obs_value)[:,var,:]
                         for ind_prf in range(nb_obs2):
                            index_ok = np.where(np.array(obs_value2[ind_prf,:]) != _FillValue)
                            dir_out2 = './json/'
                            filename_json = dir_out2+str(date)+"_"+varname+"_profile_"+str(ind_prf)+".json"
                            tab_profile = {}
                            tab_profile['longitude'] = np.array(lon_obs[ind_prf]).tolist()
                            tab_profile['latitude'] = np.array(lat_obs[ind_prf]).tolist()
                            tab_profile['Insitu'] = list(zip(np.array(depth[ind_prf,index_ok]).tolist()[0], np.array(obs_value2[ind_prf,index_ok]).tolist()[0]))
                            tab_profile['Model'] = list(zip(np.array(depth[ind_prf,index_ok]).tolist()[0], np.array(interp_values[ind_prf,index_ok]).tolist()[0]))
                            Writer(self.log).write_json_profile(filename_json, tab_profile)
               else:
                   interp_values = np.ndarray(shape=(nb_obs, nb_profs),
                                         dtype=float, order='F')
                   interp_values[:, :] = self.missing_value

               if key[0:4] == 'fcst':
                   if len(key) > 4:
                       indice = int(key[4])
                   else:
                       indice = 0
                   self.log.debug(f"Fead forecast {var=} {indice=} {ll_uniq=}")
                   if ll_unique :
                       try:
                          fcst_tab[:, var, indice, 0] = interp_values
                          #self.log.debug(f"Ok!!")
                          #threshold = 1e35
                          ## Filter out values greater than the threshold
                          #filtered_data = interp_values[interp_values < threshold]
                          #self.log.debug(f"Ok2!!")
                          ## Find the maximum value in the filtered array
                          #if filtered_data.size > 0:
                          #     max_value = np.max(filtered_data)
                          #     self.log.info(f"Max value Fead {var=} {indice=} {max_value}")
                          #else:
                          #    self.log.info(f"Max value  Fead nan")
                          #self.log.info('-------------------------------')
                          ###
                          if self.compute and var == len(variable)-1:
                             self.log.debug(f'Compute correction fcst {indice}')
                             fcst_tab[:, 0, indice, 0] = fcst_tab[:, 0, indice, 0] * \
                                self.tab_interp['gcost'] - fcst_tab[:, 1, indice, 0] * \
                                self.tab_interp['gsint']
                             fcst_tab[:, 1, indice, 0] = fcst_tab[:, 0, indice, 0]* \
                                 self.tab_interp['gsint'] + fcst_tab[:, 1, indice, 0]* \
                                 self.tab_interp['gcost']
                       except Exception as e:
                           self.log.debug(f"Error for the forecast variable {var} {indice} {data_file}")
                   else:
                       fcst_tab[:, var, indice, :] = interp_values
                        ##
               elif key[0:4] == 'nwct':
                   indice = 0
                   self.log.debug("%s Persistence nwct indice  %i " %(key,indice))
                   if ll_unique :
                       pers_tab[:, var, indice, 0] = interp_values
                       if self.compute and var == len(variable)-1:
                          self.log.debug(f'Compute correction nwct {var}')
                          pers_tab[:, 0, indice, 0] = pers_tab[:, 0, indice, 0] * \
                             self.tab_interp['gcost'] - pers_tab[:, 1, indice, 0] * \
                             self.tab_interp['gsint']
                          pers_tab[:, 1, indice, 0] = pers_tab[:, 0, indice, 0]* \
                              self.tab_interp['gsint'] + pers_tab[:, 1, indice, 0]* \
                              self.tab_interp['gcost']
                   else:
                       pers_tab[:, var, indice, :] = interp_values
               elif key[0:4] == 'pers':
                   if len(key) > 4:
                       indice = int(key[4])
                   else:
                       indice = 0
                   self.log.debug("Persistence indice %i " % (indice))
                   if ll_unique :
                        pers_tab[:, var, indice, 0] = interp_values
                        if self.compute and var == len(variable)-1:
                           self.log.debug(f'Compute correction pers {indice}')
                           pers_tab[:, 0, indice, 0] = pers_tab[:, 0, indice, 0] * \
                              self.tab_interp['gcost'] - pers_tab[:, 1, indice, 0] * \
                              self.tab_interp['gsint']
                           pers_tab[:, 1, indice, 0] = pers_tab[:, 0, indice, 0]* \
                              self.tab_interp['gsint'] + pers_tab[:, 1, indice, 0]* \
                              self.tab_interp['gcost']
                   else:
                       pers_tab[:, var, indice, :] = interp_values
               elif key[0:4] == 'hdct':
                   self.log.debug(f'{var=}{varname=}')
                   if ll_unique :
                       self.log.debug(f'Case uniq {var} {np.shape(hdct_tab)}')
                       try:
                           if ll_singleobs:
                               hdct_tab[0, var, :] = interp_values
                           else:
                               hdct_tab[:, var, 0] = interp_values
                       except ValueError as e:
                           raise BroadcastError(f"{file_input} {data_file} Custom BroadcastError: {e}")
                       if self.compute and var == len(variable)-1:
                           self.log.debug('Compute correction hindcast')
                           hdct_tab[:, 0, 0] = hdct_tab[:, 0, 0]* \
                              self.tab_interp['gcost'] - hdct_tab[:, 1, 0]* \
                              self.tab_interp['gsint']
                           hdct_tab[:, 1, 0] = hdct_tab[:, 0, 0]* \
                              self.tab_interp['gsint'] + hdct_tab[:, 1, 0]* \
                              self.tab_interp['gcost']
                   else:
                       hdct_tab[:, var, :] = interp_values

               elif key[0:4] == 'clim':
                   if ll_unique:
                       clim_tab[:, var, 0] = interp_values
                       index_nan = np.isnan(interp_values)
                       self.log.debug('% of bad values for clim {}'.format((np.shape(interp_values[index_nan])[0]/np.shape(interp_values)[0]) * 100))
                       if ll_json:
                           lon_nan = lon_obs[index_nan]
                           lat_nan = lat_obs[index_nan]
                           zip_val = list(zip(lon_nan.tolist(),lat_nan.tolist()))
                           filename = "missing_clim_"+str(date)+".json"
                           with open(filename, 'w') as outfile:
                               json.dump(zip_val, outfile,indent=4)
                   else:
                       clim_tab[:, var, :] = interp_values
               elif key == 'smoc':
                  self.log.debug("Fead with smoc values")
                  if ll_unique :
                      smoc_tab[:, var, 0] = interp_values
                  else:
                      smoc_tab[:, var, :] = interp_values
               elif key.startswith('smoc_fcst'):
                  self.log.debug("Fead with smoc fcst values")
                  indice = int(key[9])
                  if ll_unique :
                      smoc_fcst_tab[:, var, indice, 0] = interp_values
                      self.log.debug("indice {}".format(indice))
                      self.log.debug(" Var {}".format(var))
                      self.log.debug("Values {}".format(smoc_fcst_tab[:, var, indice, 0]))
                  else:
                      smoc_fcst_tab[:, var, indice, :] = interp_values
               elif key.startswith('stokes_fcst'):
                  indice = int(key[11])
                  self.log.debug("Fead Stokes forecast indice %i " % (indice))
                  if ll_unique :
                      stokes_fcst_tab[:, var, indice, 0] = interp_values
                  else:
                      stockes_fcst_tab[:, var, indice, :] = interp_values
               elif key == 'stokes':
                  if ll_unique :
                      stokes_tab[:, var, 0] = interp_values
                  else:
                      stockes_tab[:, var, :] = interp_values
               elif key[0:4] == 'tide':
                  if ll_unique :
                      tides_tab[:, var, 0] = interp_values
                  else:
                       tides_tab[:, var, :] = interp_values
               elif key[0:5] == 'bathy':
                  self.log.debug('Key bathy {} :  {}'.format(key, interp_values))
                  bathy_tab[:, 0] = interp_values
               else:
                   self.log.debug(f'Key not found {key}')
                   sys.exit(1)
        ####
        if ll_best:
            self.log.debug(f"Value hindcast {hdct_tab}")
            self.log.debug(f"Value max hindcast {np.shape(hdct_tab)}")
            self.log.debug("Inside best data origin {}".format(param_dict['data_origin']))
            if param_dict['data_origin'] == "GLOBCOLOUR":
                hdct_tab = hdct_tab[:, 0, 0].reshape(
                    lat_obs.shape[0], lon_obs.shape[0])
            tab_results['best_estimate'] = hdct_tab
        if ll_fcst:
            self.log.debug(f"Forecast leadtime {indice} {ll_clim}")
            if param_dict['data_origin'] == "GLOBCOLOUR" and not ll_clim:
                fcst_tab = fcst_tab[:, 0, indice, 0].reshape(
                    lat_obs.shape[0], lon_obs.shape[0])
            tab_results['forecast'] = fcst_tab
        else:
            if param_dict['fcst_mode']:
                if param_dict['data_origin'] == "GLOBCOLOUR" and not ll_clim:
                    fcst_tab = fcst_tab[:, 0, indice, 0].reshape(
                        lat_obs.shape[0], lon_obs.shape[0])
                elif param_dict['data_origin'] == "GLOBCOLOUR" and ll_clim:
                    fcst_tab = fcst_tab[:, 0, 0, 0].reshape(
                        lat_obs.shape[0], lon_obs.shape[0])
                tab_results['forecast'] = fcst_tab
        if ll_pers:
            tab_results['persistence'] = pers_tab
        else:
            tab_results['persistence'] = None

        if ll_clim:
            if param_dict['data_origin'] == "GLOBCOLOUR":
                clim_tab = clim_tab[:, 0, 0].reshape(
                    lat_obs.shape[0], lon_obs.shape[0])
            tab_results['climatology'] = clim_tab
        if ll_smoc: 
            tab_results['smoc'] = smoc_tab
        if ll_smoc_fcst: tab_results['smoc_fcst'] = smoc_fcst_tab
        if ll_stokes: tab_results['stokes'] = stokes_tab
        if ll_stokes_fcst: tab_results['stokes_fcst'] = stokes_fcst_tab
        if ll_tides: tab_results['tides'] = tides_tab
        if ll_bathy: tab_results['bathymetrie'] = bathy_tab
        if param_dict['Design_GODAE_file']:
            tab_results['observation'] = obs_value
            if 'varname_select' in param_dict.keys():
                tab_results['varname_select'] = param_dict['varname_select']
            tab_results['longitude'] = lon_obs
            tab_results['latitude'] = lat_obs
            tab_results['depth'] = depth
            if 'qc' in param_dict.keys():
                tab_results['qc'] = param_dict['qc']
            if 'LEVITUS_clim' in param_dict.keys():
                tab_results['LEVITUS_clim'] = param_dict['LEVITUS_clim']
        return tab_results, file_out

    def coloc_2D(self, lon_in, lon_out, lat_in, lat_out,
                 weight_file, nbpts_interp, log):
        """
        -------------------
        Coloc 2D
        Compute weight file with KDTree Method and store it in
        a pickle file
        Arguments
        --------------------
        lon_in : input longitude array
        lon_out : output longitude array
        lat_in : input latitude array
        lat_out : output latitude array
        weight_file : weight file
        nbpts_interp : number points for interpolation
            =>  1 = nearest
            =>  >1 IDW method
        log : logger object
        Returns
        --------------------
        WEIGHT : weight values
        inds_idw1 : index values in the grid
        """
        # Compute or not weight file
        if os.path.exists(weight_file):
            log.debug("load existing tree :%s " % (weight_file))
            if sys.version_info[0] >= 3:
                with open(weight_file, 'rb') as f:
                    zip_var = pickle.load(f)
                    d, inds_idw = list(zip(*zip_var))
            else:
                f = file(str(weight_file), 'r')
                zip_var = cPickle.load(f)
                d, inds_idw = zip(*zip_var)
            DIST = np.array(d)
            inds_idw1 = np.array(inds_idw)
            f.close
        else:
            log.debug("Create CKDTREE object")
            # Read input grid
            # Test size
            if len(lon_in.shape) > 1:
                X1, Y1 = lon_in, lat_in
            else:
                X1, Y1 = np.meshgrid(lon_in, lat_in)

            # Convert to cartesian
            xs, ys, zs = lon_lat_to_cartesian(X1.flatten(), Y1.flatten())
            # Read output grid
            X2, Y2 = lon_out, lat_out

            if isinstance(X2, np.float64) or isinstance(X2, np.float32):
                log.debug('X2 is a float')
            else:
                if len(X2)> 1:
                    global_step = float("{0:.2f}".format((X2[len(X2)-1]-X2[0])/(len(X2)-1)))
                    first_step = float("{0:.2f}".format(X2[1]-X2[0]))
                    log.debug('Step sizes %f %f ' % (global_step, first_step))
                else:
                    log.debug('Only one step')

            # 1D cases
            if  lon_out.ndim  == 0:
                log.debug('1D not regular')
                X2, Y2 = lon_out, lat_out
            elif lon_out.ndim < 2:
                if global_step == first_step:
                    # Regular
                    log.debug('1D regular => convert to meshgrid')
                    X2, Y2 = np.meshgrid(lon_out, lat_out)
                else:
                    # Irregular
                    log.debug('1D not regular')
                    X2, Y2 = lon_out, lat_out

            # Convert to cartesian
            xt, yt, zt = lon_lat_to_cartesian(X2.flatten(), Y2.flatten())
            if sys.version_info[0] >= 3:
                tree = cKDTree(list(zip(xs, ys, zs)))
                DIST, inds_idw1 = tree.query(
                    list(zip(xt, yt, zt)), k=nbpts_interp)
                with open(weight_file, 'wb') as f:
                    pickle.dump(list(zip(DIST, inds_idw1)),
                                f, pickle.HIGHEST_PROTOCOL)
            else:
                tree = cKDTree(zip(xs, ys, zs))
                DIST, inds_idw1 = tree.query(zip(xt, yt, zt), k=nbpts_interp)
                with open(weight_file, 'wb') as f:
                    cPickle.dump(zip(DIST, inds_idw1), f,
                                 protocol=cPickle.HIGHEST_PROTOCOL)
            f.close
        WEIGHT = 1.0 / DIST**2

        return WEIGHT, inds_idw1

    def test_lonlat(self, var_in, nblon, nblat):
        """
           Test if the matrix is in lon/lat instead of lat/lon
        """
        nsize1, nsize2 = np.shape(var_in)
        if nsize1 == nblon:
           ll_reshape = True
        else:
           ll_reshape = False
        return ll_reshape

    def is_netcdf_xarray(self, filename):
        try:
            # Open dataset without loading into memory
            with xr.open_dataset(filename, engine="netcdf4"):
                return True
        except Exception:
            return False

    def interp_2D(self, weight, inds, file, nbpts_interp, variable, param_dict,\
            depth=0, time=0, key=None, fillvalue=None, nblon=None, nblat=None,
            lon_mod=None, lat_mod=None, missing_value=np.nan, same_grid=False):
        """
        -------------------
        2D interpolator
        Arguments
        --------------------
        weight_file : weight values
        inds : index values
        nbpts_interp : number points for interpolation
            =>  1 = nearest
            =>  >1 IDW method
        variable :  input_variable
        Returns
        --------------------
        interp tab : interpolated array
        """
        self.log.debug('Interp2D Read input file : {}'.format(file))
        self.log.debug(f'Variable in interp_2D : {variable}')
        self.log.debug("-------------------------------------------------------")
        # Load variable
        if isinstance(file, (xr.Dataset, xr.DataArray)) or self.is_netcdf_xarray(file):
            try:
                dataset_in = file if isinstance(file, (xr.Dataset, xr.DataArray)) else xr.open_dataset(file, decode_times=False)
                var_in = dataset_in[variable]
            except Exception:
                self.log.error(f"Missing variable {variable} in {file}")
                raise
        else:
            var_in, lon, lat = ReadModel().read_lightout(file)
        ll_single = False
        ll_nan = False
        if depth > 0.5 :
            ## Find the 2 points that enclose the depth value
            nlons, nlats, ndepths, depth_mod = self.read_dims(dataset_in)
            ind_depth1, ind_depth2 = self.getnearpos(depth, depth_mod)
            w1, w2 = self.getweight(depth_mod[ind_depth1], depth_mod[ind_depth2], depth)
            self.log.debug(f"Index of depth value {ind_depth1} {ind_depth2} {depth_mod[ind_depth1]} {depth_mod[ind_depth2]}")
            self.log.debug(f"weight {w1} {w2}")
        else:
            # Surface case
            ll_single = True
            
        # Apply MSSH correction if needed
        if variable in self.list_ssh : #or param_dict['data_type'] == "SLA":
            cl_msshFile = param_dict['cl_msshFile']
            rp_mssh_shift = param_dict['rp_mssh_shift']
            self.log.info(f'Correction mssh {cl_msshFile=}')
            self.log.info(f'Shift {rp_mssh_shift}')
            dataset_mssh = xr.open_dataset(cl_msshFile)
            tab_mssh = dataset_mssh['mssh'].values
            self.log.info(f"Compute sla equiv with shift {rp_mssh_shift} {cl_msshFile}")

        if isinstance(var_in, xr.Dataset):
            if len(var_in.data_vars) == 1:
                var_in = list(var_in.data_vars.values())[0]
            else:
                raise ValueError(f"Expected a single variable but got multiple: {list(var_in.data_vars)}")
        if isinstance(var_in, xr.DataArray):
            var_data = var_in.values
        else:
            raise TypeError(f"Unsupported variable type: {type(var_in)}")
        if var_data.ndim == 3:
            var_data = var_data[time] if ll_single else (var_data[ind_depth1], var_data[ind_depth2])
        elif var_data.ndim == 4:
            if ll_single:
                var_data = var_data[time, 0]
            else:
                var_data = (var_data[time, ind_depth1], var_data[time, ind_depth2])
        # Apply SSH correction
        if variable in self.list_ssh and ll_single:
            var_data = np.where(var_data != fillvalue,  var_data - tab_mssh - rp_mssh_shift, var_data)
        # Handle same grid case
        if same_grid:
            return np.array(var_data).flatten(), False
        # Interpolation
        if np.isnan(inds).all():
            tab_interp = np.full(len(inds), np.nan)
            self.log.info("All missing values for indexes in interp_2D")
            return tab_interp, True

        flat_func = lambda x: np.array(x).flatten()[inds]
        if nbpts_interp > 1:
            if ll_single:
                if self.test_lonlat(var_data, nblon, nblat):
                    var_data = np.transpose(var_data)
                tab_interp = np.sum(weight * flat_func(var_data), axis=1) / np.sum(weight, axis=1)
            else:
                interp1 = np.sum(weight * flat_func(var_data[0]), axis=1) / np.sum(weight, axis=1)
                interp2 = np.sum(weight * flat_func(var_data[1]), axis=1) / np.sum(weight, axis=1)
                tab_interp = w1 * interp1 + w2 * interp2
        else:
            tab_interp = flat_func(var_data) if ll_single else w1 * flat_func(var_data[0]) + w2 * flat_func(var_data[1])

        tab_interp = np.where(np.isnan(tab_interp), missing_value, tab_interp)
        return tab_interp

    def compute_3D_weight(self, ind, var_in, inds,\
                       weight, ind_max, nbpts_interp):
        tab_ind = inds[ind,:,:]
        if np.isnan(inds).all():
            self.log.info("All nan values for inds")
            nlon, nlat = np.shape(weight[ind,:,:])
            tab_interp_tmp = np.full((nlon, nlat), np.nan)
        else:
            if nbpts_interp > 1:
                tab_interp_tmp = np.sum(weight[ind,:,:] * np.array(var_in[ind, :, :]).flatten()
                                           [tab_ind], axis=1) / np.sum(weight[ind,:,:], axis=1)
            else:
                ## Nearest neighbour case
                tab_interp_tmp = np.array(var_in[ind, :, :].values).flatten()[tab_ind]
        #self.log.debug('Depth loop : {}'.format(ind))
        return tab_interp_tmp

    def read_dims(self, dataset):
        dimensions = dataset.dims
        variable_names = list(dataset.variables.keys())
        variable_names_set = set(variable_names)
        list_lon = ['x', 'X', 'lon', 'LON', 'lons', 'longitude', 'LONGITUDE']
        list_lat = ['y', 'Y', 'lat', 'LAT', 'LATS', 'latitude', 'LATITUDE']
        list_depth = ['deptht', 'depth', 'depthu', 'depthv', 'z', 'Z', 'DEPTH', 'd']
        try:
            tab_lon = { k: dimensions[k] for k in dimensions.keys() & set(list_lon)}
            tab_lat = { k: dimensions[k] for k in dimensions.keys() & set(list_lat)}
            tab_depth = { k:dimensions[k] for k in dimensions.keys() & set(list_depth)}

        except:
            self.log.error('lon lat or depth not defined')
            raise

        if tab_lon: nlons = list(tab_lon.values())[0]
        if tab_lat: nlats = list(tab_lat.values())[0]
        if tab_depth: ndepths = list(tab_depth.values())[0]
        intersection = variable_names_set.intersection(set(list_depth))
        if intersection:
            depth_var_name = list(intersection)[0]
            depth_mod = dataset[depth_var_name][:].values
        else:
            self.log.error('No matching variable for depth')
            raise
        #depth_mod = dataset[list(tab_depth.keys())[0]][:].values

        return nlons, nlats, ndepths, depth_mod

    def interp_3D(self, weight, inds, ind_max, file, nbpts_interp, variable, depth,
                    time=0, ll_verif=False, obs=None, dateval=None, lon=None, lat=None,
                    depth_mod=None, light=False, clim=False, fillvalue=np.nan, data_file=None,
                    ll_uniq=False, ll_nan=False, ll_case2=False):
        """
           Interpolate profiles model values onto insitu vertical profiles
        """
        start = timer()
        ## Code allégé case
        if light:
            self.log.debug("Open lighout file")
            var_in, lon, lat = ReadModel.read_lightout(file)
        elif clim:
            tab_var_clim = {}
            tab_var_clim['votemper'] = 'temperature'
            tab_var_clim['vosaline'] = 'salinity'
            self.log.debug("Open clim file {}".format(file))
            #dataset_in = xr.open_dataset(file, decode_times=False)
            try:
                dataset_in = xr.open_mfdataset(file,chunks={'time_counter':1,'deptht':'auto','y':'auto','x':'auto'},parallel=True,combine='by_coords')
                var_in = dataset_in[variable][:, :, :].values
                nc_file = Dataset(file,'r')
                var_in = nc_file.variables[tab_var_clim[variable]][:, :, :]
            except:
                self.log.error("Missing variable variable in file {file}")
                raise
        else:
            try:
                # Load variable
                if isinstance(file, (xr.Dataset, xr.DataArray)) or self.is_netcdf_xarray(file):
                    try:
                        dataset_in = file if isinstance(file, (xr.Dataset, xr.DataArray)) else xr.open_dataset(file, decode_times=False)
                        var_in = dataset_in[variable]
                    except Exception:
                        self.log.error(f"Missing variable {variable} in {file}")
                        raise
                ##
                nlons, nlats, ndepths, depth_mod = self.read_dims(dataset_in)
                self.log.debug(f"{nlons, nlats, ndepths, depth_mod}")
                if len(np.shape(dataset_in[variable])) == 3:
                    var_in = dataset_in[variable][:, :, :].values
                elif len(np.shape(dataset_in[variable])) == 4:
                    var_in = dataset_in[variable][time, :, :, :].values
                else:
                    self.log.error(f"Dimension not known {len(np.shape(dataset_in[variable]))}")
            except:
                self.log.error(f"Problem with variable {variable} and file {file}")
                raise
        end = timer()
        self.log.debug('~~~~~~~~ Time elapsed read files : {} {}'.format
                        (str(end - start), " seconds ~~~~~~~~~~~~"))
        var_in = np.array(var_in)
        nb_obs, nb_profs = np.shape(depth)
        self.log.debug(f'Before computing {nb_obs=} {nb_profs=}{ind_max=}')
        start = timer()
        missing_value = 99999
        if np.isnan(inds).all():
            #values_interp = fillvalue*np.zeros((nb_obs, nb_profs))
            values_interp = np.full((nb_obs, nb_profs), np.nan)
            self.log.info(f'All missing values for indexes interp3D {data_file}')
        else:
            num_cores=10
            # Parallelize loop on levels
            if np.isnan(lon).all() and np.isnan(lat).all():
                values_interp = np.full((nb_obs, nb_profs), np.nan)
            else:
                try:
                    self.log.debug(f'Size indexes {np.shape(inds)}')
                    self.log.debug(f'Size weight {np.shape(weight)}')
                    self.log.debug(f'Size var_in {np.shape(var_in)}')
                    self.log.debug(f'Index max {ind_max}')
                    self.log.debug(f'nbpts_interp {nbpts_interp}')
                    with Parallel(n_jobs=num_cores, prefer="threads", verbose=0) as parallel:
                        tab_interp_tmp  = parallel(delayed(self.compute_3D_weight)(ind, var_in,\
                                                                                   inds, weight, ind_max, nbpts_interp) \
                                                                                   for ind in range(ind_max))
                except Exception as e:
                    self.log.error(f"Error {e} in compute_3D_weight for modele {file} and data {data_file} {ll_uniq} {ll_nan} {ll_case2}")
                    self.log.error(f"{lon} {lat}")
                    raise SystemExit(e)
                tab_interp_tmp = np.array(tab_interp_tmp)
                end = timer()
                self.log.debug('~~~~~~~~ Time elapsed loops on levels : %s %s' %
                                (str(end - start), " seconds ~~~~~~~~~~~~"))
                #if nb_profs > 1:
                values_interp = fillvalue*np.zeros((nb_obs, nb_profs))
                self.log.debug(f'~~~~ Start loop {nb_obs}~~~~')
                start = timer()
                self.log.debug(f'{depth}')
                tab_interp_tmp = np.array(tab_interp_tmp)
                end = timer()
                self.log.debug('~~~~~~~~ Time elapsed loops on levels : %s %s' %
                                (str(end - start), " seconds ~~~~~~~~~~~~"))
                values_interp = fillvalue*np.zeros((nb_obs, nb_profs))
                self.log.debug(f'~~~~ Start loop {nb_obs}~~~~')
                start = timer()
                self.log.debug(f'{depth}')
                self.log.debug(f'{np.shape(values_interp)}')
                nprof, nlevel = np.shape(values_interp)
                for ind_prof in range (nb_obs):
                    self.log.debug(f'{ind_prof}/{nb_obs}')
                    profondeur = depth[ind_prof, :]
                    self.log.debug(f'Depth {profondeur}')
                    tab_mod_ma = tab_interp_tmp[:,ind_prof]
                    self.log.debug(f'Model Value {tab_mod_ma=}')
                    self.log.debug(f"{len(depth_mod[0:ind_max])}")
                    self.log.debug(f"{depth_mod[0:ind_max]}")
                    self.log.debug(f"Ind min O Ind max {ind_max}")
                    # Identify non-missing values
                    non_missing = profondeur != missing_value
                    # Count non-missing values
                    non_missing_count = np.sum(non_missing)
                    if len(profondeur) == 1 and profondeur < depth_mod[0]:
                        values_interp[ind_prof,:] =  tab_mod_ma[0]
                    else:
                        fixture = vertical.Column(tab_mod_ma, depth_mod[0:ind_max])
                        self.log.debug('Profondeurs profiles {}'.format(profondeur[0:100]))
                        self.log.debug('Profondeurs modele {}'.format(depth_mod[0:ind_max]))
                        self.log.debug(f'non_missing_count: {non_missing_count}')

                        if np.any(~np.isnan(tab_mod_ma[:])):
                            val_interp = fixture.interpolate(profondeur)
                            values_interp[ind_prof,:] = val_interp[:]
                            if profondeur[0]  < depth_mod[0]:
                                values_interp[ind_prof, 0]= tab_mod_ma[0]
                            if np.ma.isMaskedArray(val_interp):
                                values_interp[ind_prof,:] = np.where(val_interp.mask, fillvalue, values_interp[ind_prof,:])
                        self.log.debug(f'{len(profondeur)=}')
                        if non_missing_count <= 2:
                            self.log.debug(f'missing_count {non_missing_count}')
                            self.log.debug(f'missing_count {data_file}')
                            non_missing_values = profondeur[(profondeur != missing_value) & (profondeur < depth_mod[0])]
                            non_missing_indices = np.where((profondeur != missing_value) & (profondeur < depth_mod[0]))[0]
                            values_interp[ind_prof, non_missing_indices] = tab_mod_ma[0]
                    self.log.debug(f'--------------------------------')
                    self.log.debug(f'Valeurs interp :{values_interp[ind_prof,:]}=')
                    self.log.debug(f'--------------------------------')
            self.log.debug('Valeurs profile {}'.format(obs[:]))
            self.log.debug('--------------------------------------------')
            end = timer()
            self.log.debug('~~~~~~~~ Time elapsed interp levels on profiles: %s %s' %
                            (str(end - start), " seconds ~~~~~~~~~~~~"))
        self.log.debug(f'{values_interp=}')
        return values_interp
