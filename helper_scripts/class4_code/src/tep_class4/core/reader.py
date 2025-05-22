import sys
import numpy as np
import netCDF4
from netCDF4 import Dataset
from . import utilities
import xarray as xr
import os
import datetime as dt
from tep_class4.core.utils import echeance

__version__ = 0.1
__date__ = "February 2019"
__authors__ = "C.REGNIER & C.SZCZYPTA"

def read_qc_CLASS4(cvar, nc):
    rla_qc_init2 = None
    if cvar in ['SST', 'SLA', 'PSAL', 'TEMP']:
        cl_qcvar = 'qc'
    elif cvar ==  'aice':
        ll_ice = True
        cl_qcvar = 'QC02'
        cl_qcvar2 = 'QC07'
        rla_qc_init2 = nc[cl_qcvar2][:, 0, :].values.ravel()
    elif cvar in ['VVEL', 'UVEL', 'UVEL_filtr', 'VVEL_filtr']:
        list_var = [*nc.variables.keys()]
        ## New case for filtered data
        liste1 = ['current_test_qc', 'position_qc', 'obs_qc', 'current_test']
        liste2 = ['position_qc', 'observation_qc', 'time_qc','depth_qc' ]
        if len(set(list_var).intersection(liste1)) == 4:
            ll_currents = True
            cl_qcvar = 'current_test_qc'
            cl_qcvar2 = 'position_qc'
            cl_qcvar3 = 'obs_qc'
            cl_qcvar4 = 'current_test'
            rla_qc_init2 = nc[cl_qcvar2][:]
            rla_qc_init3 = nc[cl_qcvar3][:]
            rla_qc_init4 = nc[cl_qcvar4][:]
        elif len(set(list_var).intersection(liste2)) == 4:
            ll_currents2 = True
            cl_qcvar = 'position_qc'
            cl_qcvar2 = 'observation_qc'
            cl_qcvar3 = 'time_qc'
            cl_qcvar4 = 'depth_qc'
            rla_qc_init2 = nc[cl_qcvar2][:]
            rla_qc_init3 = nc[cl_qcvar3][:]
            rla_qc_init4 = nc[cl_qcvar4][:]
    else:
        cl_qcvar = 'qc'
    rla_qc_init = nc[cl_qcvar][:].values
    return rla_qc_init[:].ravel(), rla_qc_init2

class GenericReader(object):

    def __init__(self, loglevel=2):
        self.log = Logger("Reader").run(loglevel)

class ReadModel(GenericReader):

    def read_lightout(self, lightout_file, typemod=None, maskfile=None):
        """
        M.Hamon 03/2016
        Lecture des fichiers alleges :
            - Lecture de l'entete du fichier binaire sur les 54 premiers bytes.
            - Lecture de la variable (Entier en 16 bits) à partir du 62eme byte jusqu'a la fin du fichier moins 4 bytes.
            - Conversion des valeurs.
        ALLEGE = read_lightout(lightout_file)
        C.REGNIER 06/2016
        Ajout des lon lat en fonction du type en entree
        """ 
        import os, sys, re
        from math import tan, atan, pi
        import numpy as np
        import struct

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        self.log.debug("Inside lightout file {}".format(lightout_file))
        if not os.path.isfile(str(lightout_file)):
           liberr.LibPytReadError("No such file or directory {} Please Check the path/name of the file".format(lightout_file))
        try:
            f = open(lightout_file, mode='rb')
            fileContent = f.read()
            f.close()
            DIM = struct.unpack(">13fh", fileContent[:54])
            offsetbyte = 62  # nombre de bytes du header
            # le header est étrangement range!
            LON = int(DIM[1])
            LAT = int(DIM[4])
            LEVEL = int(DIM[7])
            AUXMIN = DIM[10]
            AUXMAX = DIM[11]
            UNDEFO = DIM[12]
            MAXINT = DIM[13]
            nbbytes = len(fileContent)
            dimensions = LON * LAT * LEVEL
            var = np.zeros(dimensions)
            encodagevar = '>' + str(dimensions) + 'h'
            var = struct.unpack(encodagevar, fileContent[offsetbyte:nbbytes - 4])
            self.log.debug("AUXMIN: {} AUXMAX: {} undef:{}".format(AUXMIN, AUXMAX, UNDEFO))
            # ind_neg=np.where(np.array(var) != MAXINT)
            # if AUXMIN >= 0 :
            #    ## Cas unsign short
            #    print ("cas unsign short")
            #    minval=np.min(var)
            #    maxval=np.max(var)
            #    var_tmp=np.array(var)
            #    ind_ocean=np.where(np.array(var) != MAXINT)
            #    var_tmp[ind_ocean]=(np.array(var)[ind_ocean])*-1
            #    totuplevar=totuple(var_tmp)
            #    var=totuplevar
            # else :
            #    print ("cas sign short")
            AUXMID = (AUXMIN + AUXMAX) / 2.
            scale = AUXMAX - AUXMID
            if(scale != 0.):
               scale = 4. / scale
            else:
               scale = 1.
            self.log.debug("scale :{}".format(scale))
            self.log.debug("auxmid : {}".format(AUXMID))
            ATANMAX = atan((AUXMAX - AUXMID) * scale) / pi
            ATANMIN = atan((AUXMIN - AUXMID) * scale) / pi
            RMAXINT = MAXINT - 1
            self.log.debug("Maxint :{}".format(MAXINT))
            var_ok = np.nan * np.zeros(dimensions)
            for ii in np.arange(dimensions):
                    if var[ii] != MAXINT:
                            var_ok[ii] = tan(var[ii] / (RMAXINT / ATANMAX) * pi) / scale + AUXMID
            ALLEGE = var_ok.reshape((LEVEL, LAT, LON))
        except:
            self.log.error("Pb during reading lightened file")
            raise
        if int(LAT) == 1019. and int(LON) == 1440.:
            typemod = "PSY4"
        elif typemod:
            typemod = typemod
        else:
            typemod = "other"
        # # Read lon lat value an compute decimation
        self.log.debug('Typemod {}'.format(typemod))
        try:
            if typemod in ['PSY4', 'PSY3', 'PSY2']:
                if typemod == "PSY4":
                    maskfile = Statics().get_resource('coord_PSY4.nc')
                    il_sampling = 3
                elif typemod == "PSY3":
                    maskfile = Statics().get_resource('coord_PSY3.nc')
                    il_sampling = 2
                elif typemod == "PSY2":
                    maskfile = Statics().get_resource('coord_PSY2.nc')
                    il_sampling = 3
        except:
            self.log.error("Mask file not defined")
            raise
        # # Read maskfile
        nc_file = netCDF4.Dataset(maskfile, 'r')
        nav_lat = nc_file.variables['nav_lat'][:,:]
        nav_lon = nc_file.variables['nav_lon'][:,:]
        nbiraw = nav_lon.shape[1]
        nbjraw = nav_lon.shape[0]
        nav_lon_ligth = nav_lon[2:nbjraw - 1:il_sampling, 1:nbiraw - 1:il_sampling]
        nav_lat_ligth = nav_lat[2:nbjraw - 1:il_sampling, 1:nbiraw - 1:il_sampling]
        self.log.debug('min out : {}'.format(np.nanmin(ALLEGE)))
        self.log.debug('max out : {} '.format(np.nanmax(ALLEGE)))
        self.log.debug(' Shape : {} '.format(nav_lon_ligth.shape))
        return ALLEGE, nav_lon_ligth, nav_lat_ligth

def read_outputnp(file_out):
    if os.path.exists(file_out) and os.path.getsize(file_out) != 0:
        return True
    else:
        return False

def read_stats_class4_timeseries(filename, leadtime,  prod, varname,
               level, list_prof, dict_tmp, dict, window):
    dateCNESref = dt.datetime.strptime('19500101','%Y%m%d').toordinal()
    try:
        f1 = Dataset(filename, 'r', format='NETCDF4')
    except Exception as e:
        raise RuntimeError(f'Failed to open file {filename}') from e

    if 'stats_'+varname in f1.variables:
        stat_var = f1.variables['stats_'+varname][:]
        stat_var = np.where(stat_var.mask, np.nan, stat_var)
        ll_stat = True
    else:
        ll_stat = False

    rla_name = f1.variables['area_names'][:]
    area_name = pull_names(rla_name)
    #if GLO_AREA:
    #    area_name = ['Full Domain']
    rla_metric = f1.variables['metric_names'][:]
    metric_name = pull_names(rla_metric)
    vtime = (f1.variables['time'][:])/24 + dateCNESref
    fcst = f1.variables['forecasts'][:]

    f1.close()
    print(f'Search {leadtime=}')
    num_ech, name_ech = echeance(fcst, leadtime)
    print(num_ech, name_ech)

    rmsd = np.where(np.array(metric_name) == 'mean squared error')[0][0]
    mean_obs = np.where(np.array(metric_name) == 'mean of reference')[0][0]
    mean_prod = np.where(np.array(metric_name) == 'mean of product')[0][0]
    nb_obs = np.where(np.array(metric_name) == 'number of data values')[0][0]
    cor = np.where(np.array(metric_name) == 'anomaly correlation')[0][0]
    ini = False
    level_name = list_prof[level]
    print('^^^^^^^^^^^^^^^^^^^^^^')
    print(f'Create dictionnary {varname} {prod} {level_name}')
    print('^^^^^^^^^^^^^^^^^^^^^^')
    for a, area in enumerate(area_name):
        dict[leadtime][varname][prod][level_name][area] = {}
        dict_tmp[leadtime][varname][prod][level_name][area] = {}
        if ll_stat:
            dict_tmp[leadtime][varname][prod][level_name][area]['rmsd'] = \
                stat_var[:, num_ech, level, rmsd, a]
            dict_tmp[leadtime][varname][prod][level_name][area]['bias'] = \
                stat_var[:, num_ech, level, mean_prod, a] - \
                        stat_var[:, num_ech, level, mean_obs, a]
                #stat_var[:, num_ech, level, mean_prod, a] - \
                #        stat_var[:,0, level, mean_obs, a]
            dict_tmp[leadtime][varname][prod][level_name][area]['nb_obs'] = \
                stat_var[:, num_ech, level, nb_obs, a]
            dict_tmp[leadtime][varname][prod][level_name][area]['cor_ano'] = \
                stat_var[:, num_ech, level, cor, a]
        TIME = []
        for t in range(len(vtime)):
            TIME.append(dt.date.fromordinal(int(vtime[t])))
        if ll_stat:
            dict[leadtime][varname][prod][level_name][area]['time'] = \
                TIME[int(window/2):-int(window/2)]

    return dict_tmp, dict, area_name, name_ech


def read_area_class4(filename):
    try:
        f1 = Dataset(filename, 'r', format='NETCDF4')
    except Exception as e:
        raise RuntimeError(f'Failed to open file {filename}') from e
    rla_name = f1.variables['area_names'][:]
    rla_depths = f1.variables['depths'][:]
    fcst = f1.variables['forecasts'][:]
    area_name = pull_names(rla_name)
    formatted_list = []
    previous_depth = 0
    for depth in rla_depths:
        formatted_list.append(f"{int(previous_depth)}-{int(depth)}m")
        previous_depth = depth

    return area_name, rla_depths, formatted_list, fcst


def read_stats_class4(filename, level, varname,
                      dict_tmp, dict, filtered, list_ech,
                      window=5):
    try:
        f1 = Dataset(filename, 'r', format='NETCDF4')
    except Exception as e:
        raise RuntimeError(f'Failed to open file {filename}') from e

    if 'stats_'+varname in f1.variables:
        stat_var = f1.variables['stats_'+varname][:]
        stat_var = np.where(stat_var.mask, np.nan, stat_var)
        ll_stat = True
    else:
        ll_stat = False
    dateCNESref = dt.datetime.strptime('19500101','%Y%m%d').toordinal()
    rla_name = f1.variables['area_names'][:]
    area_name = pull_names(rla_name)
    #if GLO_AREA:
    #    area_name = ['Full Domain']
    rla_metric = f1.variables['metric_names'][:]
    metric_name = pull_names(rla_metric)
    vtime = (f1.variables['time'][:])/24 + dateCNESref
    fcst = f1.variables['forecasts'][:]
    f1.close()
    rmsd = np.where(np.array(metric_name) == 'mean squared error')[0][0]
    mean_obs = np.where(np.array(metric_name) == 'mean of reference')[0][0]
    mean_prod = np.where(np.array(metric_name) == 'mean of product')[0][0]
    nb_obs = np.where(np.array(metric_name) == 'number of data values')[0][0]
    cor = np.where(np.array(metric_name) == 'anomaly correlation')[0][0]
    ini = False
    for nf in list_ech:
        dict[varname][nf] = {}
        dict_tmp[varname][nf] = {}
        num_ech, name_ech = echeance(fcst, int(nf))
        for a, area in enumerate(area_name):
          #if area == "Full Domain":
            dict[varname][nf][area] = {}
            dict_tmp[varname][nf][area] = {}
            for lvls, levels in enumerate(level):
                dict[varname][nf][area][levels] = {}
                dict_tmp[varname][nf][area][levels] = {}
                if ll_stat:
                    dict_tmp[varname][nf][area][levels]['rmsd'] = \
                        stat_var[:, num_ech, lvls, rmsd, a]
                    dict_tmp[varname][nf][area][levels]['bias'] = \
                        stat_var[:, num_ech, lvls, mean_prod, a] - \
                                stat_var[:, 0, lvls, mean_obs, a]
                    dict_tmp[varname][nf][area][levels]['nb_obs'] = \
                        stat_var[:, num_ech, lvls, nb_obs, a]
                    dict_tmp[varname][nf][area][levels]['cor_ano'] = \
                        stat_var[:, num_ech, lvls, cor, a]
                TIME = []
                for t in range(len(vtime)):
                    TIME.append(dt.date.fromordinal(int(vtime[t])))
                if filtered:
                    dict[varname][nf][area][levels]['time'] = \
                        TIME[int(window/2):-int(window/2)]
                else:
                    dict[varname][nf][area][levels]['time'] = TIME

    return dict_tmp, dict, area_name

def read_mask_polar(params_conf):

    basin_mask_arc = params_conf['db'].get('NETCDF', 'file_basin_arc')
    basin_mask_ant = params_conf['db'].get('NETCDF', 'file_basin_ant')
    file_areamask = [basin_mask_arc, basin_mask_ant]
    file_areamask_arc, file_areamask_ant = file_areamask
    nc = netCDF4.Dataset(file_areamask_arc, 'r')
    rla_mask_arc = nc.variables['mask'][:, :]
    rla_lonmask_arc = nc.variables['longitude'][:, :]
    rla_lonmask_nan_arc = np.ma.filled(rla_lonmask_arc, np.nan)
    rla_latmask_arc = nc.variables['latitude'][:, :]
    nc.close()
    nc = netCDF4.Dataset(file_areamask_ant, 'r')
    rla_mask_ant = nc.variables['mask'][:, :]
    rla_lonmask_ant = nc.variables['longitude'][:, :]
    rla_lonmask_nan_ant = np.ma.filled(rla_lonmask_ant, np.nan)
    rla_latmask_ant = nc.variables['latitude'][:, :]
    nc.close()
    CMEMS_area = True
    rla_maskvalues = []

    area_names_arc = ['North Pole', 'Queen Elis Is', 'Beaufort Sea', 'Chuckchi Sea',
                      'Siberian Sea', 'Laptev Sea', 'Kara Sea', 'Barents Sea', 'Greenland Sea', 'Stheast Greenland', 'Bafin Bay',
                      'Hudson Bay', 'Labrador Sea', 'Bering Sea', 'Okhotsk Sea', ' Baltic Sea']
    area_names_ant = ['Weddel Sea', 'Southern Atlantic Ocean', 'Southern Indian Ocean', 'Southern West Pacific Ocean',
                      'Southern East Pacific Ocean', 'Ross Sea', 'Admundsen Sea', 'Bellingshausen Sea']
    params_conf['area_name'] = area_names_arc + area_names_ant
    maskvalues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]
    for area_mask in range(len(params_conf['area_name'])):
        if area_mask < 16:
            #print ("Arctic region")
            rla_mask = rla_mask_arc
        else:
            #print ("Antarctic region")
            rla_mask = rla_mask_ant
        rla_maskvalues.append(maskvalues[area_mask])
    cla_areaname = np.chararray(len(params_conf['area_name']), itemsize=25)
    cla_areaini = np.chararray(len(params_conf['area_name']), itemsize=25)
    for indice, i in enumerate(params_conf['area_name']):
        cla_areaname[indice] = i
        cla_areaini[indice] = i
    cla_area = list(cla_areaini)
    params_conf['CMEMS_area'] = CMEMS_area
    params_conf['cla_area'] = cla_area
    params_conf['cla_area_names'] = cla_areaname
    params_conf['cla_areaini'] = cla_areaini
    params_conf['maskvalues'] = maskvalues

    return params_conf, rla_mask_arc, rla_mask_ant, rla_maskvalues, rla_lonmask_nan_arc,\
        rla_latmask_arc, rla_lonmask_nan_ant,  rla_latmask_ant


def read_mask(params_conf):

    cla_areaname = params.cla_areasattr
    cla_areaini = params.liste_zone
    file_areamask = params_conf['db'].get(
        'NETCDF', 'file_' + params_conf['fmt_mask'])
    nc = netCDF4.Dataset(file_areamask, 'r')
    rla_mask = nc.variables['mask'][:]
    rla_lonmask = nc.variables['longitude'][:]
    rla_latmask = nc.variables['latitude'][:]
    rla_lonmask_nan = np.ma.filled(rla_lonmask, np.nan)
    rla_lonmask_nan_shift = np.where(
        rla_lonmask_nan < 0, rla_lonmask_nan+360, rla_lonmask_nan)
    if 'area_names' in nc.variables:
        rla_name = nc.variables['area_names'][:]
        area_name = utilities.pull_names(rla_name)
        cla_area = list(cla_areaini)
        cla_area = [var.decode("utf-8") for var in cla_area]
        CMEMS_area = True
        rla_maskvalues = None
    else:
        area_name = [0]
        fmt_ascii = 0
        CMEMS_area = False
        rla_maskvalues = np.unique(rla_mask)
        rla_maskvalues = np.delete(
            rla_maskvalues, np.where(rla_maskvalues == 0))
        rla_maskvalues = np.delete(rla_maskvalues, np.where(
            np.isnan(rla_maskvalues) == True))
        cla_area = list(np.arange((len(np.array(rla_maskvalues)))))
    nc.close()
    params_conf['CMEMS_area'] = CMEMS_area
    params_conf['cla_area'] = cla_area
    params_conf['cla_area_names'] = cla_areaname
    params_conf['cla_areaini'] = cla_areaini
    params_conf['area_name'] = area_name

    return params_conf, rla_mask, rla_maskvalues, rla_lonmask_nan, rla_latmask


def mask2(params_conf):

    try:
        prefix = 'product_quality_stats_'
        rla_lonmask_arc = None
        rla_lonmask_ant = None
        rla_latmask_arc = None
        rla_latmask_ant = None
        rla_latmask = None
        rla_lonmask = None
        rla_mask = None
        rla_mask_arc = None
        rla_mask_ant = None
        if params_conf['fmt_mask'] == 'basin_pol':
            dirout = params_conf['dirNCout']
            prefix = 'product_quality_stats_polar_'
            params_conf, rla_mask_arc, rla_mask_ant, rla_maskvalues, \
                rla_lonmask_arc, rla_latmask_arc, rla_lonmask_ant,\
                rla_latmask_ant = read_mask_polar(params_conf)
        else:
            params_conf, rla_mask, rla_maskvalues,\
                rla_lonmask, rla_latmask = read_mask(params_conf)
            if params_conf['fmt_mask'] == 'basin':
                dirout = params_conf['dirNCout']
            elif params_conf['fmt_mask'] == 'bins2d_arc':
                prefix = 'bins2d_product_quality_stats_NH_polar_' + \
                    params_conf['fmt_mask']
                dirout = params_conf['dirBOXout']
            elif params_conf['fmt_mask'] == 'bins2d_ant':
                prefix = 'bins2d_product_quality_stats_SH_polar_' + \
                    params_conf['fmt_mask']
                dirout = params_conf['dirBOXout']
            else:
                dirout = params_conf['dirBOXout']
    except:
        print("Error input grid not find:", sys.exc_info()[0])
        raise
    params_conf['dirout'] = dirout
    params_conf['prefix_fileout'] = prefix

    return params_conf, rla_mask, rla_maskvalues, rla_mask_arc, rla_mask_ant, rla_lonmask,\
        rla_latmask, rla_lonmask_arc, rla_latmask_arc, rla_lonmask_ant, rla_latmask_ant


def mask(fmt_mask, basin_mask, box_mask, bins025d_mask, bins05d_mask, bins1d_mask,
         bins2d_mask, bins4d_mask, dirNCout, dirBOXout):

    if fmt_mask == 'basin':
        file_areamask = basin_mask
        dirout = dirNCout
        prefix = 'product_quality_stats_'
    elif fmt_mask == 'basin_pol':
        file_areamask = basin_mask
        dirout = dirNCout
        prefix = 'product_quality_stats_ice_'
    elif fmt_mask == 'box':
        file_areamask = box_mask
        dirout = dirBOXout
        prefix = 'box_quality_stats_'
    elif fmt_mask == 'bins05d':
        file_areamask = bins05d_mask
        dirout = dirBOXout
        prefix = 'bins05d_quality_stats_'
    elif fmt_mask == 'bins025d':
        file_areamask = bins025d_mask
        dirout = dirBOXout
        prefix = 'bins025d_quality_stats_'
    elif fmt_mask == 'bins1d':
        file_areamask = bins1d_mask
        dirout = dirBOXout
        prefix = 'bins1d_quality_stats_'
    elif fmt_mask == 'bins2d':
        file_areamask = bins2d_mask
        dirout = dirBOXout
        prefix = 'bins2d_quality_stats_'
    elif fmt_mask == 'bins4d':
        file_areamask = bins4d_mask
        dirout = dirBOXout
        prefix = 'bins4d_quality_stats_'

    return file_areamask, dirout, prefix

def read_pos(filename, log):
    try:
        nc = netCDF4.Dataset(filename,'r')
        variable = 'nav_lon'
        rla_lon = nc.variables[variable][:,:]
        variable = 'nav_lat'
        rla_lat = nc.variables[variable][:,:]
    except:
        log.info ('pb lecture coordinates')

    return rla_lon,rla_lat


def read_pos_reg(filename):
    nc = netCDF4.Dataset(filename,'r')
    variable = 'longitude'
    rla_lon = nc.variables[variable][:]
    variable = 'latitude'
    rla_lat = nc.variables[variable][:]
    return rla_lon,rla_lat

def read_pos_clim(filename):
    nc = netCDF4.Dataset(filename,'r')
    variable = 'Lon'
    rla_lon = nc.variables[variable][:]
    variable = 'Lat'
    rla_lat = nc.variables[variable][:]
    return rla_lon,rla_lat

def read_drifter_pos(file):
    ncdata = xr.open_dataset(file)
    lon = ncdata['LONGITUDE'].values
    lat = ncdata['LATITUDE'].values
    ncdata.close()
    return lon, lat

def read_drifter(file):
    ncdata = xr.open_dataset(file)
    #variables = ncdata.data_vars
    nc = netCDF4.Dataset(file, 'r')
    variables = nc.variables
    #variables = ncdata.data_vars
    ll_var = False
    if 'EWCT' and 'NSCT' in variables.keys():
        ll_var = True
        u = ncdata['EWCT'].values
        v = ncdata['NSCT'].values
        u_qc = ncdata['EWCT_QC'].values
        v_qc = ncdata['NSCT_QC'].values
    else:
        u = None ; v = None
        u_qc = None ; v_qc = None
    if 'WSTN_MODEL' and 'WSTE_MODEL' in variables.keys():
        wind_v = ncdata['WSTN_MODEL'].values
        wind_u = ncdata['WSTE_MODEL'].values
    else:
        wind_u = None ; wind_v = None
    if 'DEPH' in variables.keys():
        depth = ncdata['DEPH'].values
        depth_qc = ncdata['DEPH_QC'].values
    elif 'depth' in variables.keys():
        depth = ncdata['depth'].values
        depth_qc = ncdata['depth_qc'].values
    elif 'PRES' in variables.keys():
        depth = ncdata['PRES'].values
        depth_qc = ncdata['PRES_QC'].values
    else:
        print ('Missing Values DEPH or PRES or depth')
        sys.exit(1)
    if 'CURRENT_TEST' in variables.keys():
        current_test = ncdata['CURRENT_TEST'].values
        if 'CURRENT_TEST_QC' in variables.keys():
            current_test_qc = ncdata['CURRENT_TEST_QC'].values
        else:
            current_test_qc = ""
    else:
        current_test = None ; current_test_qc = None
    if 'POSITION_QC' in variables.keys():
        position_qc =  ncdata['POSITION_QC'].values
    else:
        position_qc = None
    if 'TIME_QC' in variables.keys():
        time_qc =  ncdata['TIME_QC'].values
    else:
        time_qc = None
    if 'DC_REFERENCE' in  variables.keys():
        dc_reference =  ncdata['DC_REFERENCE'].values
    else:
        dc_reference =  None
    if 'LONGITUDE' in variables.keys():
        lon = ncdata['LONGITUDE'].values
    elif 'longitude' in variables.keys():
        lon = ncdata['longitude'].values
    if 'LATITUDE' in variables.keys():
        lat = ncdata['LATITUDE'].values
    elif 'latitude' in variables.keys():
        lat = ncdata['latitude'].values
    if 'TIME' in variables.keys():
        time = ncdata['TIME'].values
    elif 'time' in variables.keys():
        time = ncdata['time'].values
    elif 'obs_time' in variables.keys():
        time = ncdata['obs_time'].values
    if 'observation' in variables.keys():
        observation = ncdata['observation'].values
        u = observation[:, 0, 0]
        v = observation[:, 1, 0]
    #lon_value = np.nanmean(lon)
    #lat_value = np.nanmean(lat)
    ncdata.close()
    return lon, lat, depth, depth_qc, time, u, v, ll_var,\
            u_qc, v_qc, wind_u, wind_v, current_test, current_test_qc,\
            position_qc, time_qc, dc_reference

def read_drifter_filt(file):
    lon, lat, depth, depth_qc, time, u, v, ll_var,\
            u_qc, v_qc, wind_u, wind_v, current_test, current_test_qc,\
            position_qc, time_qc, dc_reference = read_drifter(file)
    ncdata = xr.open_dataset(file)
    variables = ncdata.data_vars

    if 'LONGITUDE_FILTR' in variables.keys():
        lon = ncdata['LONGITUDE_FILTR'].values
    elif 'longitude_filtr' in variables.keys():
        lon = ncdata['longitude_filtr'].values
    if 'LATITUDE_FILTR' in variables.keys():
        lat = ncdata['LATITUDE_FILTR'].values
    elif 'latitude_filtr' in variables.keys():
        lat = ncdata['latitude_filtr'].values
    ll_var = False
    if 'EWCT_FILTR' and 'NSCT_FILTR' in variables.keys():
        ll_var = True
        u_filtr = ncdata['EWCT_FILTR'].values
        v_filtr = ncdata['NSCT_FILTR'].values
        u_filtr_qc = ncdata['EWCT_FILTR_QC'].values
        v_filtr_qc = ncdata['NSCT_FILTR_QC'].values
    elif 'observation_filtr':
        ll_var = True
        u_filtr = ncdata['observation_filtr'][:,0].values
        v_filtr = ncdata['observation_filtr'][:,1].values
        u_filtr_qc = ncdata['observation_filtr_qc'][:,0].values
        v_filtr_qc = ncdata['observation_filtr_qc'][:,1].values
    else:
        print ('Missing observations')
        sys.exit(1)
        u_filtr = None ; v_filtr = None
        u_filtr_qc = None ; v_filtr_qc = None
    return lon, lat, depth, depth_qc, time, u, v, u_filtr, v_filtr, ll_var,\
            u_qc, v_qc, u_filtr_qc, v_filtr_qc, wind_u, wind_v, current_test, current_test_qc,\
            position_qc, time_qc, dc_reference


def read_moor(file, varname, log):
    """
       Read Mooring files
    """
    try:
        ncdata = xr.open_dataset(file)
        variables = ncdata.data_vars

        ll_var = False
        if varname in list(variables.keys()):
            ll_var = True
            vardata = ncdata[varname]
            vardata_qc = ncdata[varname+'_QC']
        else:
            vardata = np.nan
            vardata_qc = np.nan
        if 'DEPH' in list(ncdata.variables.keys()):
            depth = ncdata['DEPH']
            #depth = ncdata['DEPH']
            var_qc=[ var for var in ncdata.variables.keys() if var.startswith('DEP') and var.endswith('QC')] 
            depth_qc = ncdata[var_qc].values
        elif 'PRES' in list(ncdata.variables.keys()):
            depth = ncdata['PRES']
         #   depth = ncdata['PRES']
            depth_qc = ncdata['PRES_QC']
          #  depth_qc = ncdata['PRES_QC']
        else:
            log.info (f'{file} Missing Values DEPH or PRES')
            sys.exit(1)
        lon = ncdata['LONGITUDE']
        lat = ncdata['LATITUDE']
        time = ncdata['TIME'].values
        lon_value = np.nanmean(lon)
        lat_value = np.nanmean(lat)
        return lon_value, lat_value, depth, depth_qc, time, vardata, ll_var, vardata_qc
    except Exception as e:
        print(f'Error: {e}')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, np.nan

def read_depth(filename):
    nc = netCDF4.Dataset(filename,'r')
    multidim = True
    if 'deptht' in nc.variables.keys(): rla_depth = nc.variables['deptht'][:]
    else:
        rla_depth = [0]
        multidim = False

    return rla_depth, multidim

def read_GODAE_profile(varname, filename):
    '''
       Read GODAE profile
    '''
    ncdata = xr.open_dataset(filename)
    variables = ncdata.data_vars
    depth = ncdata['depth'].values
    if varname == 'votemper':
        index = 0
    elif varname == 'vosaline':
        index = 1
    tab_hdct = ncdata['best_estimate'].values
    tab_hdct = tab_hdct[:,index,:]
    tab_fcst = ncdata['forecast'].values
    tab_fcst = tab_fcst[:,index,:,:]
    tab_pers = ncdata['persistence'].values
    tab_pers = tab_pers[:,index,:,:]
    observations = ncdata['observation'][:,index,:].values

    return depth, tab_hdct, tab_fcst, tab_pers, observations
