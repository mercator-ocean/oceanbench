from __future__ import generators
import os
import sys
import string
import shutil
from glob import glob
import re
import numpy as np
import shelve
import json
from .Logger import Logger
from datetime import date
from netCDF4 import Dataset
import netCDF4
import gzip
import pickle
import fcntl
import time
from .FileRenamer import FileRenamer
##############################################################
# C.REGNIER Juin 2014
# Class Writer to write output file in ASCII or Geojson
##############################################################

class BroadcastError(Exception):
    pass

class NetCDFFileError(Exception):
    pass

class Writer():

    """ Class to write different type of file """

    def __init__(self, log):
        self.log = log
        self.fillvalue = netCDF4.default_fillvals['f4']
        self.fillvalue_corio = 99999
        self.type_data = {
                'profile_GODAE': {'code': 'INSITU_profile', 'varname':True, 'variable':'profile', 'defaults': {'name': 'In Situ profiles from CMEMS'}},
                'profile_NC_CORIO': {'code': 'INSITU_profile', 'varname':True, 'variable':'MERC', 'defaults': {'name': 'In Situ profiles from CMEMS product 013_047'}},
                'CORA5.2_NC_CORIO': {'code': 'INSITU_profile', 'varname':True, 'variable':'', 'defaults': {'name': 'In Situ profiles from CORA product INSITU_GLO_PHY_TS_DISCRETE_MY_013_001'}},
                'SST_GODAE': {'code': 'SST_DB', 'varname':True, 'variable':'SST', 'defaults': {'name': 'SST Drifting buoy'}},
                'SLA_GODAE': {'code': 'SLA_L3S', 'varname':True, 'variable':'SLA', 'defaults': {'name': 'SLA CMEMS L3S'}},
                'SLA-L3_CMS': {'code': 'SLA-L3_CMS', 'varname':True, 'variable':'phy_l3_1hz', 'defaults': {'name': 'SLA CMEMS L3S_1hz'}},
                'aice_GODAE': {'code': 'SEA_CONC', 'varname':False, 'variable':'aice', 'defaults': {'name': 'Sea ice concentration from AMSR2 brightness temperature'}},
                'chloro_GLOBCOLOUR': {'code': 'CHLORO-A_L3S', 'variable':'chl', 'defaults': {'name': 'log10-chlorophyll A Satellite'}},
                'SIT-legos': {'code': 'SEA_SIT', 'defaults': {'name': 'Sea ice thickness at 20 Hz from LEGOS'}},
                'SNOW-legos': {'code': 'SEA_SIT', 'defaults': {'name': 'Snow thickness at 20 Hz from LEGOS'}},
                'SIVOLU': {'code': 'SEA_SIT', 'defaults': {'name': 'Sea ice volume at 20 Hz from LEGOS'}},
                'SNVOLU': {'code': 'SEA_SIT', 'defaults': {'name': 'Snow volume at 20 Hz from LEGOS'}},
                'SICONC': {'code': 'SEA_SIT', 'defaults': {'name': 'Sea ice concentration at 20 Hz from LEGOS'}},
                'KaKuYearSH': {'code': 'SEA_SIT', 'defaults': {'name': 'Sea ice climaatology'}},
                'current': {'code': 'CURRENT_ZONAL_DB', 'defaults': {'name': 'Currents from CMEMS drifting buoys 001_048'}},
                'DRIFTER_filtr_CMEMS_013_048': {'code': 'CURRENT_ZONAL_DB', 'varname':False, 'variable':'currents-filtr', 'defaults': {'name': 'Filtered Currents from CMEMS drifting buoys 001_048'}},
                'DRIFTER_CMEMS_013_048': {'code': 'CURRENT_ZONAL_DB', 'varname':False, 'variable':'currents', 'defaults': {'name': 'Currents from CMEMS drifting buoys 001_048'}},
                'default': {'code': 'Default_code', 'varname':False, 'defaults': {'name': 'Data for validation'}},
       }
        self.dailylist = ['1dAV', '1d-m']

    def mkdir(self, path):
        "Create a directory, and parents if needed"
        if not os.path.exists(path):
            os.makedirs(path)

    def write_json_profile(self, filename, tab):
        with open(filename, 'w') as outfile:
                json.dump(tab, outfile,indent=4)

    def write_geojson(self, filename, position):
        try:
            #print 'Write Geojson file'
            liste_position = []
            for il_ind in range(len(position)):
                pos = position[il_ind]
                liste_position.append(tuple(pos))
            #print MultiPoint(liste_position)
            with open(filename, 'w') as outfile:
                json.dump(MultiPoint(liste_position), outfile)
        except:
            print ('Probleme d ecriture geojson')
            sys.exit(1)

    def write_ASCII(self, filename, position):
        try:
            #print 'Write ASCII file'
            # Creation fichier
            length = position.size
            # if length < 2 :
            #     print "Empty Position array"
            # elif length == 2 :
            #     print "Only 1 position"
            #     np.savetxt(filename, position[None,:],fmt='%12.5f',delimiter='    ')
            # else :
            #     np.savetxt(filename, position,fmt='%12.5f',delimiter='    ')
            np.savetxt(filename, position, fmt='%12.5f', delimiter='    ')
        # print len(position)
        #print 'Write ASCII file OK'
        except:
            print ('Probleme d ecriture ASCII')
            sys.exit(1)

    def write_geojson_fromdb(self, type_f, file_out, file_db):
        # Open a shelve file if not exist
        if not(os.path.isfile(file_db)):
            print ("Pb persistent file doesn't exist")
            sys.exit(1)
        else:
            shelf = shelve.open(file_db)
            try:
                tab_fin = shelf[type_f]
                #tab_fin[0,:][tab_fin[0,:] <0] += 360.
                #print "Type exist %s " %(type_f)
                self.write_geojson(file_out, tab_fin)
            except:
                print ("Type doesn't exist %s " % (type_f))
            shelf.close()
#
    def write_GODAE_files_best(self, filename, hdct_tab, dateval, model, variable):
        self.log("write_GODAE_files_best")
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3')
        nc_best = nc_fid.createVariable(
            'best_estimate', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
        nc_best.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_fid.variables['best_estimate'][:, :, :] = hdct_tab[:, :, :]
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.time_interp = "daily average fields"
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.best_estimate_description = "analysis produced between 7 and 14 days behind real time"
        nc_fid.close()  # close the new file

    def write_GODAE_files_clim(self, filename,clim_tab, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3')
        nc_clim = nc_fid.createVariable(
            'climatology', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
        nc_clim.setncatts({'long_name': u"Climatology Lumpkin V3.05",'units': u"m s-1"})
        nc_clim.setncatts({'comment': u"Monthly Means of Drifter Data"})
        nc_fid.variables['climatology'][:, :, :] = clim_tab[:, :, :]
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.close()  # close the new file


    def write_BIOARGO_files(self, filename, hdct_tab, dateval, model, variable, fcst = None, pers = None):
        nc_fid = Dataset(filename, 'r+')
        nc_best = nc_fid.createVariable(
            'best_estimate', 'f4', ('time', 'N_PROF', 'N_LEVELS'), fill_value=self.fillvalue)
        nc_best.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_fid.variables['best_estimate'][:, :, :] = hdct_tab[:, :, :]
        if fcst:
            nc_fcst = nc_fid.createVariable(
                'forecast', 'f4', ('time', 'N_PROF', 'N_LEVELS'), fill_value=self.fillvalue)
            nc_fcst.setncatts(
                {'long_name': u"Model forecast counterpart of obs. value"})
            nc_fid.variables['forecast'][:, :, :] = fcst
        if pers:
            nc_pers = nc_fid.createVariable(
                'pers', 'f4', ('time', 'N_PROF', 'N_LEVELS'), fill_value=self.fillvalue)
            nc_pers.setncatts(
                {'long_name': u"Model persistence counterpart of obs. value"})
            nc_fid.variables['persistence'][:, :, :] = pers

        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        nc_fid.obs_type = 'BIOARGO'
        nc_fid.time_interp = "daily average fields"
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.best_estimate_description = "analysis produced between 7 and 14 days behind real time"
        nc_fid.close()  # close the new file

    def write_LEGOS_generic_files(self, filename, param_dict, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+')
        dims = nc_fid.dimensions
        if 'forecast' in param_dict.keys():
            fcst_tab = param_dict['forecast']
            nb_obs, nb_dephs, nb_fcsts, nb_vars = np.shape(fcst_tab)
            if 'numfcsts' not in dims.keys(): nc_fid.createDimension('numfcsts', nb_fcsts)
            if 'numobs' not in dims.keys(): nc_fid.createDimension('numobs', nb_obs)
            nc_fcst = nc_fid.createVariable(
                'forecast', 'f4', ('numobs', 'numfcsts'), fill_value=self.fillvalue)
            nc_fcst.setncatts(
                {'long_name': u"Model forecast counterpart of obs. value"})
            nc_fid.variables['forecast'][:, :] = fcst_tab[:, 0, :, 0]
            ll_defined = True
        elif 'best' in param_dict.keys():
            hdct_tab = param_dict['best']
            nb_obs, nb_dephs, nb_vars = np.shape(hdct_tab)
            if 'numobs' not in dims.keys(): nc_fid.createDimension('numobs', nb_obs)
            varname = 'best_'+param_dict['varname']
            nc_best = nc_fid.createVariable(
                varname, 'f4', ('numobs'), fill_value=self.fillvalue)
            nc_best.setncatts(
                {'long_name': u"Model best estimate counterpart of obs. value"})
            nc_fid.variables[varname][:] = hdct_tab[:, :, :]
            ll_defined = True

        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN INTERNATIONAL'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.time_interp = "daily average fields"
        #nc_fid.description = "Drifting buoys from CMEMS product INSITU_GLO_NRT_OBSERVATIONS_"+cmems_prod
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.best_estimate_description = "analysis produced between 7 and 14 days behind real time"

    def write_CMEMS_generic_files(self, filename, param_dict, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+')
        dims = nc_fid.dimensions
        ll_defined = False
        if 'numobs' not in dims.keys():
            nc_fid.createDimension('numobs', nb_obs)
        if 'numfcsts' not in dims.keys():
            nc_fid.createDimension('numfcsts', nb_vars)
        if 'numdeps' not in dims.keys():
            nc_fid.createDimension('numdeps', nb_vars)
        if 'forecast' in param_dict.keys() and param_dict['forecast'] is not None:
            fcst_tab = param_dict['forecast']
            #nb_obs, nb_vars, nb_fcst, nb_depths = np.shape(fcst_tab)
            nc_fcst = nc_fid.createVariable(
                'forecast', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
            nc_fcst.setncatts(
                {'long_name': u"Model forecast counterpart of obs. value"})
            nc_fid.variables['forecast'][:, :, :, :] = fcst_tab[:, :, :, :]
            ll_defined = True
        if 'persistence' in param_dict.keys() and param_dict['persistence'] is not None:
            pers_tab = param_dict['persistence']
            #nb_obs, nb_vars, nb_fcst, nb_depths = np.shape(pers_tab)
            nc_pers = nc_fid.createVariable(
                'persistence', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
            nc_pers.setncatts(
                {'long_name': u"Model persistence counterpart of obs. value"})
            nc_fid.variables['persistence'][:, :, :, :] = pers_tab[:, :, :, :]
            ll_defined = True
        if 'smoc' in param_dict.keys() and param_dict['smoc'] is not None:
            smoc_tab = param_dict['smoc']
            #nb_obs, nb_vars, nb_fcst = np.shape(smoc_tab)
            nb_depths = 1
            nc_smoc = nc_fid.createVariable(
                'smoc_drift_best', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
            nc_smoc.setncatts({'long_name': u"Eastward and Northward total velocity (Eulerian + Waves + Tide) (smoc drift)",'units': u"m s-1"})
            nc_smoc.description = "GLOBAL_ANALYSIS_FORECAST_PHY_001_024 hourly mean merged surface currents from oceanic circulation, tides and waves"
            nc_fid.variables['smoc_drift_best'][:, :, :] = smoc_tab[:, :, :]
        if 'smoc_fcst' in param_dict.keys() and param_dict['smoc_fcst'] is not None:
            smoc_tab = param_dict['smoc_fcst']
            #nb_obs, nb_vars, nb_fcst, nb_depths = np.shape(smoc_tab)
            nb_depths = 1
            nc_smoc = nc_fid.createVariable(
                'smoc_drift_fcst', 'f4', ('numobs', 'numvars', 'numfcsts2', 'numdeps'), fill_value=self.fillvalue)
            nc_smoc.setncatts({'long_name': u"Forecasts Eastward and Northward total velocity (Eulerian + Waves + Tide) (smoc drift)",'units': u"m s-1"})
            nc_smoc.description = "GLOBAL_ANALYSIS_FORECAST_PHY_001_024 hourly mean merged surface currents from oceanic circulation, tides and waves"
            nc_fid.variables['smoc_drift_fcst'][:, :, :] = smoc_tab[:, :, :, :]
        if 'best' in param_dict.keys() and param_dict['best'] is not None:
            best_tab = param_dict['best']
            #nb_obs, nb_vars, nb_fcst = np.shape(best_tab)
            nb_depths = 1
            nc_best = nc_fid.createVariable(
                'best_estimate', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
            nc_best.setncatts(
                            {'long_name': u"Model best_estimate counterpart of obs. value"})
            nc_fid.variables['best_estimate'][:, :, :] = best_tab[:, :, :]

        if 'tides' in param_dict.keys() and param_dict['tides'] is not None:
            tides_tab = param_dict['tides']
            #nb_obs, nb_vars, nb_fcst = np.shape(tides_tab)
            nb_depths = 1
            nc_tides = nc_fid.createVariable(
                'tides', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
            nc_tides.setncatts({'long_name': u"Eastward and tide-induced velocity (Tide current)",'units': u"m s-1"})
            nc_tides.description = "FES2014 model at 1/12 for tides"
            nc_fid.variables['tides'][:, :, :] = tides_tab[:, :, :]
        if 'stokes' in param_dict.keys() and param_dict['stokes'] is not None:
            stokes_tab = param_dict['stokes']
            #nb_obs, nb_vars, nb_fcst = np.shape(stokes_tab)
            nb_depths = 1
            nc_stokes = nc_fid.createVariable(
                'stokes_best', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
            nc_stokes.setncatts({'long_name': u"Eastward and Northward wave-induced velocity (Stokes drift)",'units': u"m s-1"})
            nc_stokes.description = "MFWAM waves model for Stokes drift - GLOBAL_ANALYSIS_FORECAST_WAV_001_027 CMEMS dataset"
            nc_fid.variables['stokes_best'][:, :, :] = stokes_tab[:, :, :]
        if 'stokes_fcst' in param_dict.keys() and param_dict['stokes_fcst'] is not None:
            stokes_tab = param_dict['stokes_fcst']
            #nb_obs, nb_vars, nb_fcst, nb_depths = np.shape(stokes_tab)
            nb_depths = 1
            nc_stokes = nc_fid.createVariable(
                'stokes_fcst', 'f4', ('numobs', 'numvars', 'numfcsts2', 'numdeps'), fill_value=self.fillvalue)
            nc_stokes.setncatts({'long_name': u"Eastward and Northward wave-induced velocity (Stokes drift)",'units': u"m s-1"})
            nc_stokes.description = "MFWAM waves model for Stokes drift - GLOBAL_ANALYSIS_FORECAST_WAV_001_027 CMEMS dataset"
            nc_fid.variables['stokes_fcst'][:, :, :, :] = stokes_tab[:, :, :, :]
        if 'clim' in param_dict.keys() and param_dict['clim'] is not None:
            clim_tab = param_dict['clim']
            nb_depths = 1
            #nb_obs, nb_vars, nb_fcst = np.shape(stokes_tab)
            nc_clim = nc_fid.createVariable('climatology','f4',('numobs','numvars','numdeps'),fill_value=self.fillvalue)
            nc_clim.setncatts({'long_name': u"Climatology Lumpkin V3.05",'units': u"m s-1"})
            nc_clim.setncatts({'comment': u"Monthly Means of Drifter Data"})
            nc_fid.variables['climatology'][:, :, :] = clim_tab[:, :, :]
        if 'bathy' in param_dict.keys() and param_dict['bathy'] is not None:
            bathy_tab = param_dict['bathy']
            #nb_obs, nb_vars, nb_fcst = np.shape(stokes_tab)
            nc_clim = nc_fid.createVariable('bathymetry','f4',('numobs','numdeps'),fill_value=self.fillvalue)
            nc_clim.setncatts({'long_name': u"Bathymetry from ORCA12 model",'units': u"m"})
            nc_clim.setncatts({'comment': u"bathymetry_INDESO_V1.0_mskbdy"})
            nc_fid.variables['bathymetry'][:, :] = bathy_tab[:, :]

        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN INTERNATIONAL'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.time_interp = "daily average fields"
        #nc_fid.description = "Drifting buoys from CMEMS product INSITU_GLO_NRT_OBSERVATIONS_"+cmems_prod
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.best_estimate_description = "analysis produced between 7 and 14 days behind real time"
        nc_fid.close()  # close the new file

    def write_CMEMS_files(self, filename, hdct_tab, fcst_tab, pers_tab, clim_tab, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+')
        nb_obs, nb_vars, nb_fcst, nb_depths = np.shape(fcst_tab)
        dims = nc_fid.dimensions
        #print ('Dimensions forecast {}'.format(np.shape(fcst_tab)))
        #print ('Dimensions  {} {} {} {}'.format(nb_obs, nb_vars, nb_fcst, nb_depths))
        if 'numobs' not in dims.keys(): nc_fid.createDimension('numobs', nb_obs)
        if 'numfcsts' not in dims.keys(): nc_fid.createDimension('numfcsts', nb_vars)
        if 'numdeps' not in dims.keys(): nc_fid.createDimension('numdeps', nb_vars)
        nc_best = nc_fid.createVariable(
            'best_estimate', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
        nc_best.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_fcst = nc_fid.createVariable(
            'forecast', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
        nc_fcst.setncatts(
            {'long_name': u"Model forecast counterpart of obs. value"})
        nc_pers = nc_fid.createVariable(
            'persistence', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
        nc_pers.setncatts(
            {'long_name': u"Model persistence counterpart of obs. value"})
        nc_clim = nc_fid.createVariable(
            'climatology', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
        nc_clim.setncatts(
            {'long_name': u"Surface climatology from Lumpkin counterpart of obs. value"})
        nc_clim.setncatts(
            {'version': u"V3.05"}) 
        nc_fid.variables['best_estimate'][:, :, :] = hdct_tab[:, :, :]
        nc_fid.variables['forecast'][:, :, :, :] = fcst_tab[:, :, :, :]
        nc_fid.variables['persistence'][:, :, :, :] = pers_tab[:, :, :, :]
        nc_fid.variables['climatology'][:, :, :] = clim_tab[:, :, :]
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.time_interp = "daily average fields"
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.best_estimate_description = "analysis produced between 7 and 14 days behind real time"
        nc_fid.close()  # close the new file

    def write_GODAE_files(self, filename, hdct_tab, fcst_tab, pers_tab, dateval, model, variable):
        nc_fid = Dataset(filename,'r+')
        if np.any(hdct_tab):
            self.log.info('Hindcast not empty OK2')    
            nc_best = nc_fid.createVariable(
                'best_estimate', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
            nc_best.setncatts(
                {'long_name': u"Model best_estimate counterpart of obs. value"})
            nc_fid.variables['best_estimate'][:, :, :] = hdct_tab[:, :, :]
        if np.any(fcst_tab):
            self.log.info('Forecast not empty')    
            nc_fcst = nc_fid.createVariable(
                'forecast', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
            nc_fcst.setncatts(
                {'long_name': u"Model forecast counterpart of obs. value"})
            nc_fid.variables['forecast'][:, :, :, :] = fcst_tab[:, :, :, :]
        else:
            self.log.info('Forecast is empty')    
        if np.any(pers_tab):
            self.log.info('Persistence not empty')    
            nc_pers = nc_fid.createVariable(
                'persistence', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
            nc_pers.setncatts(
                {'long_name': u"Model persistence counterpart of obs. value"})
            nc_fid.variables['persistence'][:, :, :, :] = pers_tab[:, :, :, :]
        else:
            self.log.info('Persistence is empty')    
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.time_interp = "daily average fields"
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.best_estimate_description = "analysis produced between 7 and 14 days behind real time"
        nc_fid.close()  # close the new file

    def write_GLOBCOLOR_files(self, filename, tab1, tab2, tab3, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3_CLASSIC')
        nc_file = nc_fid.createVariable(
            'best_estimate', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file2 = nc_fid.createVariable(
            'forecast', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file3 = nc_fid.createVariable(
            'climatology', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_file2.setncatts(
            {'long_name': u"Model forecast counterpart of obs. value"})
        nc_file3.setncatts(
            {'long_name': u"Satellite L4 climatology counterpart of obs. value",
             'product': u"OCEANCOLOUR_GLO_CHL_L4_REP_OBSERVATIONS_009_082",
             'dataset': u"dataset-oc-glo-chl-multi-l4-gsm_4km_daily-climatology-v02"}
            )
        nc_fid.variables['best_estimate'][:] = tab1
        nc_fid.variables['forecast'][:] = tab2
        nc_fid.variables['climatology'][:] = tab3
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.contact = 'qualif@mercator-ocean.fr'
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 00:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.close()  # close the new file

    def write_GLOBCOLOR_files2(self, filename, tab, tab2, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+', format='NETCDF3_CLASSIC')
        nc_file = nc_fid.createVariable(
            'best_estimate', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file2 = nc_fid.createVariable(
            'climatology', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_file2.setncatts(
            {'long_name': u"Daily Satellite L4 climatology counterpart of obs. value",
             'product': u"OCEANCOLOUR_GLO_CHL_L4_REP_OBSERVATIONS_009_082",
             'dataset': u"dataset-oc-glo-chl-multi-l4-gsm_4km_daily-climatology-v02"})
        nc_fid.variables['best_estimate'][:] = tab
        nc_fid.variables['climatology'][:] = tab2
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.contact = 'qualif@mercator-ocean.fr'
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 00:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.close()  # close the new file

    def write_best_fcst(self, filename, tab1, tab2, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3_CLASSIC')
        nc_file = nc_fid.createVariable(
            'best_estimate', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file2 = nc_fid.createVariable(
            'forecast', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_file2.setncatts(
            {'long_name': u"Model forecast counterpart of obs. value"})
        nc_fid.variables['best_estimate'][:] = tab1[:, :]
        nc_fid.variables['forecast'][:] = tab2[:, :]
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.contact = 'qualif@mercator-ocean.fr'
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.close()  # close the new file

    def write_best(self, filename, tab1, dateval, model, dimension, variable, desc, variable2):
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3_CLASSIC')
        nc_file = nc_fid.createVariable(
            variable, 'f4', (dimension), fill_value=np.nan)
        nc_file.setncatts({'long_name': desc+" counterpart of obs. value"})
        nc_file.setncatts({'minvalue':  np.nanmin(tab1)})
        nc_file.setncatts({'maxvalue': np.nanmax(tab1)})
        nc_fid.variables[variable][:] = tab1[:]
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.obs_type = self.type_data.get(variable2, {}).get('defaults').get('name')
        nc_fid.contact = 'qualif@mercator-ocean.fr'
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        nc_fid.close()  # close the new file

    def write_class4_out(self, param_dict, list_keys):
        unique_dates = sorted({int(key.split('-')[0]) for key in list_keys['keyvalues']})
        unique_leadtime = sorted({key.split('-')[1] for key in list_keys['keyvalues']})
        unique_leadtime_unit = sorted({int(key.split('-')[2]) for key in list_keys['keyvalues']})
        tab_out = {}
        for dateval in unique_dates:
            fcst_tab = None
            pers_ab = None
            ll_fcst = False
            ll_pers = False
            variable = self.type_data.get(f"{param_dict['data_type']}_{param_dict['cl_src']}", {}).get('variable')
            frame = f"*{dateval}*{variable}.nc"
            self.log.debug(f"{param_dict['data_type']}")
            self.log.debug(f"{param_dict['cl_src']}")
            self.log.debug(f"{variable=}")
            list_file = glob(f"{param_dict['dirwork']}{frame}")
            if not list_file:
                frame = f"*{variable}*{dateval}*.nc"
                list_file = glob(f"{param_dict['dirwork']}{frame}")
            if not list_file:
                self.log.error(f"Files are missing {dateval} {variable}")
                sys.exit(1)
                continue
            self.log.debug(f"{frame=}")
            self.log.debug(f"{param_dict['dirwork']}")
            self.log.debug(f"{list_file=}")
            for file_out in list_file:
                type_file = os.path.basename(file_out).split('.')[0]
                substrings_to_remove = [f"{type_file}_", f"{dateval}_"]
                list_pfile = glob(f"{param_dict['dirwork']}{type_file}*.p")
                list_pfile_test = sorted([os.path.basename(file) for file in list_pfile])
                self.log.debug(f"{type_file=}")
                self.log.debug(f"List pickle files {list_pfile_test=}")
                # Remove specified substrings from each filename
                modified_list_pfile_test = []
                for filename in list_pfile_test:
                    modified_filename = filename
                    for substring in substrings_to_remove:
                        modified_filename = modified_filename.replace(substring, "")
                    modified_list_pfile_test.append(modified_filename)
                modified_list_pfile_test = [ name.split('.')[0] for name in modified_list_pfile_test ]
                self.log.debug(f"Modified list {modified_list_pfile_test=}")
                if 'observation' in modified_list_pfile_test:
                    # Read observation and varname
                    frame = f"{type_file}*{dateval}_observation.p"
                    list_file_obs = glob(f"{param_dict['dirwork']}{frame}")[0]
                    self.log.debug(f"File obs {list_file_obs}")
                    with open(list_file_obs, 'rb') as handle:
                        stat_obs = pickle.load(handle)
                    tab_out['observation'] = stat_obs
                if 'varname' in modified_list_pfile_test:
                    frame = f"{type_file}*{dateval}_varname.p"
                    list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                    self.log.debug(f"File param {list_file_var}")
                    with open(list_file_var, 'rb') as handle:
                        stat_var = pickle.load(handle)
                    tab_out['varname'] = stat_var
                if 'longitude' in modified_list_pfile_test:
                    frame = f"{type_file}*{dateval}_longitude.p"
                    list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                    with open(list_file_var, 'rb') as handle:
                        stat_var = pickle.load(handle)
                    tab_out['longitude'] = stat_var
                if 'latitude' in modified_list_pfile_test:
                    frame = f"{type_file}*{dateval}_latitude.p"
                    list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                    with open(list_file_var, 'rb') as handle:
                        stat_var = pickle.load(handle)
                    tab_out['latitude'] = stat_var
                if 'depth' in modified_list_pfile_test:
                    frame = f"{type_file}*{dateval}_depth.p"
                    list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                    with open(list_file_var, 'rb') as handle:
                        stat_var = pickle.load(handle)
                    tab_out['depth'] = stat_var
                if 'qc' in modified_list_pfile_test:
                    frame = f"{type_file}*{dateval}_qc.p"
                    list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                    with open(list_file_var, 'rb') as handle:
                        stat_var = pickle.load(handle)
                    tab_out['qc'] = stat_var
                if 'clim' in modified_list_pfile_test:
                    frame = f"{type_file}*{dateval}_clim.p"
                    list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                    with open(list_file_var, 'rb') as handle:
                        stat_var = pickle.load(handle)
                    tab_out['clim'] = stat_var
                frame = f"{type_file}*.nc"
                file_out = glob(f"{param_dict['dirwork']}{frame}")[0]
                for leadtime in unique_leadtime:
                    self.log.debug(f'Leadtime {leadtime}')
                    if "best_estimate" in leadtime:
                        lead_int = 0
                        frame = f"{type_file}*{dateval}_{leadtime}_{lead_int}.p"
                        self.log.info(f"{type_file=} {dateval=} {leadtime=} {lead_int=}")
                        self.log.info(f"HINDCAST {param_dict['dirwork']}{frame}")
                        pickle_file = glob(f"{param_dict['dirwork']}{frame}")[0]
                        with open(pickle_file, 'rb') as handle:
                            tab_out['best_estimate'] = pickle.load(handle)
                        nb_dims = len(np.shape(tab_out['best_estimate']))
                        if nb_dims == 2:
                            nlons, nlats = np.shape(tab_out['best_estimate'])
                        elif nb_dims == 3:
                            nb_obs, variable, nb_profs = np.shape(tab_out['best_estimate'])
                            self.log.debug(f" Best estimate {nb_obs=} {variable=} {nb_profs=}")
                    if leadtime == "forecast"  or leadtime == "persistence":
                        self.log.debug("Inside Forecast or persistence")
                        ll_create = False
                        for int_val, lead_rang in enumerate(unique_leadtime_unit):
                            frame = f"{type_file}*{dateval}_{leadtime}_{lead_rang}.p"
                            list_pickle_file = glob(f"{param_dict['dirwork']}{frame}")
                            self.log.debug(f"Leadtime {int_val=}")
                            #pickle_file = glob(f"{param_dict['dirwork']}{frame}")[0]
                            if  list_pickle_file:
                                pickle_file = list_pickle_file[0]
                                self.log.debug(f"{pickle_file=}")
                                with open(pickle_file, 'rb') as handle:
                                    stats_tmp = pickle.load(handle)
                                if int_val == 0 or not ll_create:
                                    nb_dims = len(np.shape(stats_tmp))
                                    self.log.info(f"number of dimensions {nb_dims=}")
                                    if nb_dims == 2:
                                        nlons, nlats = np.shape(stats_tmp)
                                    elif nb_dims == 3:
                                        nb_obs, variable, nb_profs = np.shape(stats_tmp)
                                        nb_fcst = len(unique_leadtime_unit)
                                    else:
                                        nb_obs, variable, nb_fcst, nb_profs = np.shape(stats_tmp)
                                    if leadtime == "forecast":
                                        if nb_dims == 2:
                                            nb_profs = 1
                                            nb_fcst = len(unique_leadtime_unit)
                                            fcst_tab = np.full(
                                                shape=(nlons, nlats, nb_fcst),
                                                fill_value=netCDF4.default_fillvals['f4'],
                                                dtype=float,
                                                order='F'
                                            )
                                            fcst_tab[:, : , int(lead_rang)] = stats_tmp[:, :]
                                            self.log.debug(f"Create array Forecast {nlons=} {nlats=} {variable=} {nb_fcst=} ")
                                        elif nb_dims == 3:
                                            print("Case 3")
                                            fcst_tab = np.full(
                                                shape=(nb_obs, variable, nb_fcst, nb_profs),
                                                fill_value=netCDF4.default_fillvals['f4'],
                                                dtype=float,
                                                order='F'
                                                )
                                            fcst_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , :]
                                            self.log.debug(f"Create array Forecast {nb_obs=} {variable=} {nb_fcst=} {nb_profs=}")
                                            #print('case1 create table')
                                            #print(fcst_tab[0:10, 0, 0, :])
                                            #print(fcst_tab[0:10, 1, 0, :])
                                        else:
                                            fcst_tab = np.full(
                                                shape=(nb_obs, variable, nb_fcst, nb_profs),
                                                fill_value=netCDF4.default_fillvals['f4'],
                                                dtype=float,
                                                order='F'
                                                )
                                            fcst_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , int(lead_rang), :]
                                            self.log.debug(f"Create array Forecast {nb_obs=} {variable=} {nb_fcst=} {nb_profs=}")
                                        ll_fcst = True
                                        ll_create = True
                                    if leadtime == "persistence":
                                        pers_tab = np.full(
                                            shape=(nb_obs, variable, nb_fcst, nb_profs),
                                            fill_value=netCDF4.default_fillvals['f4'],
                                            dtype=float,
                                            order='F'
                                            )
                                        ll_pers = True
                                        if nb_dims == 3:
                                            pers_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , int(lead_rang)]
                                        else:
                                            pers_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , int(lead_rang), :]
                                        self.log.debug(f"Create array Persistence leadtime O {nb_obs=} {variable=} {nb_fcst=} {nb_profs=}")
                                        ll_create = True
                                else:
                                    if leadtime == "forecast":
                                        ll_fcst = True
                                        self.log.debug(f"Dim fcst_tab {np.shape(fcst_tab)}")
                                        self.log.debug(f"Dim stats_tmp {np.shape(stats_tmp)}")
                                        self.log.debug(f"Leadtime {int(lead_rang)}")
                                        nb_dims = len(np.shape(stats_tmp))
                                        if nb_dims == 2:
                                            nlons, nlats = np.shape(stats_tmp)
                                            self.log.debug(f"fead {int(lead_rang)}")
                                            self.log.debug(f"Min {np.max(stats_tmp)}")
                                            self.log.debug(f"Max {np.max(stats_tmp)}")
                                            fcst_tab[:, : , int(lead_rang)] = stats_tmp[:, :]
                                        elif nb_dims == 3:
                                            fcst_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , :]
                                            #print('case2 {lead_rang}')
                                            #print(fcst_tab[0:10, 0, 1, :])
                                            #print(fcst_tab[0:10, 1, 1, :])
                                        else:
                                            fcst_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , int(lead_rang), :]
                                    if leadtime == "persistence":
                                        ll_pers = True
                                        if nb_dims == 3:
                                            pers_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , :]
                                        elif nb_dims == 4:
                                            pers_tab[:, : , int(lead_rang), :] = stats_tmp[:, : , int(lead_rang), :]
                                        self.log.debug(f"{pers_tab[0:10, 0 , int(lead_rang), 0:10]}")
                    if leadtime == 'climatology':
                        frame = f"{type_file}*{dateval}_climatology_0.p"
                        list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                        with open(list_file_var, 'rb') as handle:
                            stat_var = pickle.load(handle)
                        tab_out['climatology'] = stat_var
                    if leadtime == 'bathymetrie':
                        frame = f"{type_file}*{dateval}_bathymetrie_0.p"
                        list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                        with open(list_file_var, 'rb') as handle:
                            stat_var = pickle.load(handle)
                        tab_out['bathymetrie'] = stat_var
                    if leadtime == 'smoc':
                        #print(f"{type_file}*{dateval}_smoc_0.p")
                        frame = f"{type_file}*{dateval}_smoc_0.p"
                        list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                        with open(list_file_var, 'rb') as handle:
                            stat_var = pickle.load(handle)
                        tab_out['smoc'] = stat_var
                    if leadtime == 'stokes':
                        frame = f"{type_file}*{dateval}_stokes_0.p"
                        list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                        with open(list_file_var, 'rb') as handle:
                            stat_var = pickle.load(handle)
                        tab_out['stokes'] = stat_var
                    if leadtime == 'tides':
                        frame = f"{type_file}*{dateval}_tides_0.p"
                        list_file_var = glob(f"{param_dict['dirwork']}{frame}")[0]
                        with open(list_file_var, 'rb') as handle:
                            stat_var = pickle.load(handle)
                        tab_out['tides'] = stat_var

                if ll_fcst:
                    tab_out['forecast'] = fcst_tab
                if ll_pers:
                    tab_out['persistence'] = pers_tab
                ## Write output file
                self.log.debug(f"Write output file in Writer")
                self.write_output_frompickle(file_out,
                                             tab_out,
                                             str(dateval),
                                             param_dict['cl_config'],
                                             f"{param_dict['data_type']}_{param_dict['cl_src']}",
                                             param_dict['Design_GODAE_file'])
                                             #param_dict['data_type'])
                dest = param_dict['diroutput']
                filename = os.path.basename(file_out)
                #name_sys = "_".join([param_dict['cl_config_h'],param_dict['model_type'].lower()])
                name_sys = "_".join([param_dict['cl_conf'],param_dict['model_type'].lower()])
                fileout = FileRenamer(dest, name_sys, self.log).create_name(filename,
                                                                            param_dict['data_type'],
                                                                            param_dict['cl_src'])
                #fileout = f"{dest}{filename}"
                self.log.info(f"Move file {fileout}")
                if not os.path.exists(fileout):
                    shutil.move(file_out, fileout)
                else:
                    self.log.debug(f"Clean and move file {fileout}")
                    os.remove(fileout)
                    shutil.move(file_out, fileout)
                    #shutil.move(file_out, dest)
                tab_out = {}
                # remove p file
                for file_out in list_pfile:
                    os.remove(file_out)
                    self.log.debug(f"Remove {file_out}")


    def create_dims(self, list_dimensions, nc_fid,
                    nprofs=None, nlevels=None, nparams=None, nfcsts=None):
        if 'numobs' not in list_dimensions:
            nc_fid.createDimension('numobs', nprofs)
        else:
            self.log.debug('numobs ok')
        if 'string_length8' not in list_dimensions:
            nc_fid.createDimension('string_length8', 8)
        if 'string_length20' not in list_dimensions:
            nc_fid.createDimension('string_length20', 20)
        if 'numdeps' not in list_dimensions:
            nc_fid.createDimension('numdeps', nlevels)
        else:
            self.log.debug('numlevels ok')
        if 'numvars' not in list_dimensions:
            nc_fid.createDimension('numvars', nparams)
        else:
            self.log.debug('numvars ok')
        #if 'numfcsts' not in list_dimensions:
        if nfcsts is not None and 'numfcsts' not in list_dimensions:
            self.log.debug(f"Creation numfcsts {nfcsts}")
            nc_fid.createDimension('numfcsts', nfcsts)

    def write_output_frompickle(self, filename, tab_out, dateval, system, variable, ll_GODAE):
        # Check if the file exists
        if not os.path.exists(filename):
            raise NetCDFFileError(f"The NetCDF file '{filename}' does not exist.")
        # Check if the file is readable
        if not os.access(filename, os.R_OK):
            raise NetCDFFileError(f"The NetCDF file '{filename}' is not readable.")
        # Check if the file is writable
        if not os.access(filename, os.W_OK):
            raise NetCDFFileError(f"The NetCDF file '{filename}' is not writable.")
        try:
            #with Dataset(filename, 'r+') as nc_fid:
            #nc_fid = Dataset(filename, 'r+')
            try:
                nc_fid = Dataset(filename, 'r+', nc_bufsize=0, format='NETCDF4')
                self.log.info("Creation netcdf4 classic file")
                time.sleep(1.0)
            except OSError as e:
                raise(f"Failed to open copied NetCDF file: {e} {filename}")
            list_keys = tab_out.keys()
            list_dimensions = nc_fid.dimensions
            list_variables = list(nc_fid.variables)
            self.log.debug("open ok")
            self.log.info(list_dimensions)
            if 'varname' in tab_out.keys():
                varname_data = tab_out['varname']
            else:
                self.log.debug(f"varname_data not in tab_out")
                #sys.exit(1)
            self.log.debug(f"Write write_output_frompickle {filename}")
            ll_varname = self.type_data.get(variable, {}).get('varname')
            if 'best_estimate' in list_keys : #and np.any(tab_out['best_estimate']):
                nb_dims = len(np.shape((tab_out['best_estimate'])))
                if ll_GODAE is True:
                    nprofs, nparams, nlevels = np.shape(tab_out['best_estimate'])
                    self.log.info(f"Create dimensions {list_dimensions} {nprofs} {nlevels} {nparams}")
                    self.create_dims(list_dimensions,
                                     nc_fid,
                                     nprofs=nprofs,
                                     nlevels=nlevels,
                                     nparams=nparams)
                if ll_varname:
                    ndim_int = 20
                    #ndim_int = 8
                    if 'varname' not in list_variables:
                        self.log.debug("Create varname")
                        varname = nc_fid.createVariable('varname', 'S1', ('numvars', 'string_length20'))
                        #varname = nc_fid.createVariable('varname', 'S1', ('numvars', 'string_length8'), zlib=True, complevel=4)
                        varname.setncatts({'long_name': u"Variable name"})
                        self.log.debug(f'{varname=}')
                        for i, var_name in enumerate(varname_data):
                            varname[i, :] = list(var_name.ljust(ndim_int))
                        self.log.debug(f'Add varname ok {filename}')
                    elif 'varname' in tab_out.keys():
                        for i, var_name in enumerate(varname_data):
                            self.log.debug(f'Add varname{var_name} {ndim_int}')
                            nc_fid.variables['varname'][i, :] = list(var_name.ljust(ndim_int))
                        self.log.debug(f'Add varname ok {filename}')

                if ll_GODAE is True:
                    hdct_tab = tab_out['best_estimate'][:, :, :]
                    self.log.debug(f"Dimensions best {np.shape(tab_out['best_estimate'][:, :, :])}")
                else:
                    hdct_tab = tab_out['best_estimate'][:, :]
                self.log.info(f'Hindcast not empty {ll_GODAE}')
                if 'best_estimate' not in list_variables:
                    if ll_GODAE is True:
                        self.log.info('Create Var GODAE')
                        nc_best = nc_fid.createVariable(
                            'best_estimate', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                        nc_best.setncatts(
                            {'long_name': u"Model best_estimate counterpart of obs. value"})
                    else:
                        nc_best = nc_fid.createVariable(
                                            'best_estimate', 'f4', ('lat', 'lon'),
                                            fill_value=self.fillvalue,
                                            zlib=True, complevel=4)
                        nc_best.setncatts(
                                          {'long_name': u"Model best_estimate counterpart of obs. value"})
                try:
                    if ll_GODAE is True:
                        nc_fid.variables['best_estimate'][:, :, :] = hdct_tab[:, :, :]
                    else:
                        nc_fid.variables['best_estimate'][:, :] = hdct_tab[:, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with best_estimate {e}")
            if 'forecast' in list_keys and np.any(tab_out['forecast']):
                fcst_tab = tab_out['forecast']
                if ll_GODAE is True:
                    nprofs, nparams, nfcsts, nlevels = np.shape(tab_out['forecast'])
                    self.create_dims(list_dimensions,
                                     nc_fid,
                                     nprofs=nprofs,
                                     nlevels=nlevels,
                                     nparams=nparams,
                                     nfcsts=nfcsts)
                else:
                    nlat, nlon, nfcsts = np.shape(fcst_tab)
                    if 'numfcsts' not in list_dimensions:
                        self.log.debug(f"Creation numfcsts {nfcsts}")
                        nc_fid.createDimension('numfcsts', nfcsts)
                self.log.info(f'Forecast not empty {ll_GODAE}')
                if 'forecast' not in list_variables:
                    if ll_GODAE is True:
                        nc_fcst = nc_fid.createVariable(
                            'forecast', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                        nc_fcst.setncatts(
                            {'long_name': u"Model forecast counterpart of obs. value"})
                        print('create var')
                    else:
                        nc_best = nc_fid.createVariable(
                              'forecast', 'f4', ('lat', 'lon', 'numfcsts'),
                              fill_value=self.fillvalue,
                              zlib=True, complevel=4)
                        nc_best.setncatts(
                            {'long_name': u"Model forecast counterpart of obs. value"})
                try:
                    if ll_GODAE is True:
                        nc_fid.variables['forecast'][:, :, :, :] = fcst_tab[:, : ,:, :]
                    else:
                        nc_fid.variables['forecast'][:, :, :] = fcst_tab[:, : ,:]
                except ValueError as e:
                    raise BroadcastError(f"Error with forecast: {e}")
            else:
                self.log.debug('Forecast is empty')
            if 'persistence' in list_keys and np.any(tab_out['persistence']):
                pers_tab = tab_out['persistence']
                if ll_GODAE is True:
                    nprofs, nparams, nfcsts, nlevels = np.shape(tab_out['persistence'])
                    self.create_dims(list_dimensions,
                                     nc_fid,
                                     nprofs=nprofs,
                                     nlevels=nlevels,
                                     nparams=nparams,
                                     nfcsts=nfcsts)
                self.log.info('Persistence not empty')
                if 'persistence' not in list_variables:
                    if ll_GODAE is True:
                        nc_pers = nc_fid.createVariable(
                            'persistence', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                        nc_pers.setncatts(
                            {'long_name': u"Model persistence counterpart of obs. value"})
                try:
                    if ll_GODAE is True:
                        nc_fid.variables['persistence'][:, : , :, :] = pers_tab[:, :, :, :]
                    else:
                        nc_fid.variables['persistence'][:, : , :] = pers_tab[:, :, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with persistence: {e}")
            else:
                self.log.debug('Persistence is empty')
            ## Create observation variable
            if 'observation' in list_keys and np.any(tab_out['observation']):
                obs_tab = tab_out['observation']
                self.log.debug(f"Dimensions Observations np.shape{obs_tab}")
                if len(np.shape(obs_tab)) == 3:
                    nprofs, nparams, nlevels = np.shape(tab_out['observation'])
                    self.log.debug(f'Dimensions Observations {nprofs=} {nparams=} {nlevels=}')
                self.log.debug('Observation not empty')
                if 'observation' not in list_variables:
                    nc_obs = nc_fid.createVariable(
                            'observation', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_obs.setncatts(
                            {'long_name': u"Observation value"})
                try:
                    if len(np.shape(obs_tab)) == 1:
                        nc_fid.variables['observation'][:, 0, 0] = obs_tab[:]
                    else:
                        print(len(np.shape(obs_tab)))
                        if len(np.shape(obs_tab)) == 3:
                            nc_fid.variables['observation'][:, :, :] = obs_tab[:, :, :]
                        elif len(np.shape(obs_tab)) == 4:
                            nc_fid.variables['observation'][:, :, :] = obs_tab[:, :, :, 0]
                except ValueError as e:
                    raise BroadcastError(f"Error with observations: {e}")

            if 'longitude' in list_keys and np.any(tab_out['longitude']):
                lon_tab = tab_out['longitude']
                self.log.debug('Lon not empty')
                if 'longitude' not in list_variables:
                    nc_obs = nc_fid.createVariable(
                            'longitude', 'f4', ('numobs'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_obs.setncatts(
                            {'long_name': u"Longitude"})
                try:
                    nc_fid.variables['longitude'][:] = lon_tab[:]
                except ValueError as e:
                    raise BroadcastError(f"Error with longitudes: {e}")
            if 'latitude' in list_keys and np.any(tab_out['latitude']):
                lat_tab = tab_out['latitude']
                self.log.debug('Lat not empty')
                if 'latitude' not in list_variables:
                    nc_obs = nc_fid.createVariable(
                            'latitude', 'f4', ('numobs'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_obs.setncatts(
                            {'long_name': u"Latitude"})
                try:
                    nc_fid.variables['latitude'][:] = lat_tab[:]
                except ValueError as e:
                    raise BroadcastError(f"Error with latitudes: {e}")
            if 'depth' in list_keys and np.any(tab_out['depth']):
                depth_tab = tab_out['depth']
                self.log.info('depth not empty')
                if 'depth' not in list_variables:
                    nc_obs = nc_fid.createVariable(
                            'depth', 'f4', ('numobs', 'numdeps'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_obs.setncatts(
                            {'long_name': u"Depths of observations"})
                    try:
                        nc_fid.variables['depth'][:, :] = depth_tab[:, :]
                    except ValueError as e:
                        raise BroadcastError(f"Error with depth: {e}")
            if 'qc' in list_keys: # and np.isnan(tab_out['qc']).any():
            #if 'qc' in list_keys and np.any(tab_out['qc']):
                qc_tab = tab_out['qc']
                self.log.debug('QC not empty')
                if 'qc' not in list_variables:
                    nc_qc = nc_fid.createVariable(
                            'qc', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_qc.setncatts(
                            {'long_name': u"Depths of observations",
                             'flag_value' : u"0, 9",
                             'flag_meaning': "0 - good data. 9 - bad data."
                                })
                try:
                    nc_fid.variables['qc'][:, :, :] = qc_tab[:, :, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with qc: {e}")
            if 'climatology' in list_keys and np.any(tab_out['climatology']):
                clim_tab = tab_out['climatology']
                self.log.debug('Climatology not empty')
                if 'climatology' not in list_variables:
                    if ll_GODAE is True:
                        nc_qc = nc_fid.createVariable(
                                'climatology', 'f4', ('numobs', 'numvars', 'numdeps'),
                                fill_value=self.fillvalue_corio,
                                zlib=True, complevel=4)
                        nc_qc.setncatts(
                                {'long_name': u"Climatology",
                                 'comment' : u"Monthly fields interpolated to the correct day"
                                    })
                    else:
                        nc_clim = nc_fid.createVariable(
                            'climatology', 'f4', ('lat', 'lon'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                        nc_clim.setncatts(
                            {'long_name': u"Satellite L4 climatology counterpart of obs. value",
                             'product': u"OCEANCOLOUR_GLO_CHL_L4_REP_OBSERVATIONS_009_082",
                             'dataset': u"dataset-oc-glo-chl-multi-l4-gsm_4km_daily-climatology-v02"}
                            )
                try:
                    if ll_GODAE is True:
                        nc_fid.variables['climatology'][:, :, :] = clim_tab[:, :, :]
                    else:
                        nc_fid.variables['climatology'][:, :] = clim_tab[:, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with clim: {e}")

            if 'clim' in list_keys and np.any(tab_out['clim']):
                clim_tab = tab_out['clim']
                self.log.debug('Levitus clim not empty')
                if 'climatology' not in list_variables:
                    nc_qc = nc_fid.createVariable(
                            'climatology', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_qc.setncatts(
                            {'long_name': u"Climatology ",
                             'comment' : u"Monthly fields interpolated to the correct day"
                                })
                try:
                    nc_fid.variables['climatology'][:, :, :] = clim_tab[:, :, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with clim: {e}")

            if 'bathymetrie' in list_keys and np.any(tab_out['bathymetrie']):
                bathy_tab = tab_out['bathymetrie']
                self.log.debug('bathymetrie not empty')
                if 'bathymetrie' not in list_variables:
                    nc_qc = nc_fid.createVariable(
                            'bathymetrie', 'f4', ('numobs', 'numdeps'),
                            fill_value=self.fillvalue_corio,
                            zlib=True, complevel=4)
                    nc_qc.setncatts(
                            {'long_name': u"Bathymetrie",
                             'comment' : u"bathymetry_orca36.nc"
                                })
                try:
                    nc_fid.variables['bathymetrie'][:, :] = bathy_tab[:, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with bathymetrie: {e}")
            if 'stokes' in list_keys and np.any(tab_out['stokes']):
                stokes_tab = tab_out['stokes']
                self.log.debug('stokes not empty')
                if 'stokes' not in list_variables:
                    nc_qc = nc_fid.createVariable(
                            'stokes', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                    nc_qc.setncatts(
                            {'long_name': u"Eastward and Northward wave-induced velocity (Stokes drift)",
                             'units': u"m s-1",
                             'description': "MFWAM waves model for Stokes drift - GLOBAL_ANALYSIS_FORECAST_WAV_001_027 CMEMS dataset"
                            })
                try:
                    nc_fid.varniables['stokes'][:, :, :] = stokes_tab[:, :, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with stokes: {e}")
            if 'tides' in list_keys and np.any(tab_out['tides']):
                tides_tab = tab_out['tides']
                self.log.debug('tides not empty')
                if 'tides' not in list_variables:
                    nc_qc = nc_fid.createVariable(
                            'tides', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                    nc_qc.setncatts(
                            {'long_name': u"Eastward and tide-induced velocity (Tide current)",
                             'units': u"m s-1",
                             'description': "FES2014 model at 1/12 for tides"
                            })
                try:
                    nc_fid.variables['tides'][:, :, :] = tides_tab[:, :, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with tides: {e}")
            # Add global attributs
            nc_fid.institution = 'MERCATOR OCEAN INTERNATIONAL'
            if 'smoc' in list_keys and np.any(tab_out['smoc']):
                smoc_tab = tab_out['smoc']
                self.log.debug('smoc not empty')
                if 'smoc' not in list_variables:
                    nc_qc = nc_fid.createVariable(
                            'smoc', 'f4', ('numobs', 'numvars', 'numdeps'),
                            fill_value=self.fillvalue,
                            zlib=True, complevel=4)
                    nc_qc.setncatts(
                            {'long_name': u"Eastward and Northward total velocity (Eulerian + Waves + Tide) (smoc drift)",
                             'units': u"m s-1",
                             'description' : u"GLOBAL_ANALYSIS_FORECAST_PHY_001_024 hourly mean merged surface currents from oceanic circulation, tides and waves"
                            })
                try:
                    nc_fid.variables['smoc'][:, :, :] = smoc_tab[:, :, :]
                except ValueError as e:
                    raise BroadcastError(f"Error with smoc: {e}")
            # Add global attributs
            nc_fid.institution = 'MERCATOR OCEAN INTERNATIONAL'
            #nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
            nc_fid.contact = 'qualif@mercator-ocean.fr'
            today = date.today()
            yyyy = str(today)[0:4]
            yyyy2 = dateval[0:4]
            month = str(today)[5:7]
            month2 = dateval[4:6]
            day = str(today)[8:10]
            day2 = dateval[6:8]
            nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
            nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
            nc_fid.system = system
            nc_fid.close()  # Close the file
            self.log.debug(f'Write {filename} OK')
        except OSError as e:
            error_message = f"OSError NetCDF file: {e.errno} {e.strerror} {e.filename}"
            raise NetCDFFileError(error_message) from e
        except Exception as e:
            raise NetCDFFileError("Exception Error: {e}")
    
    def write_coriolis_file_raw(self, filename, tab_res, dateval, model, variable, varname_data):
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3_CLASSIC')
        list_keys = tab_res.keys()
        #nc_fid.createDimension('string_length8', 8)
        nc_fid.createDimension('string_length8', 20)

        if 'best' in list_keys and np.any(tab_res['best']):
            nprofs, nparams, nlevels = np.shape(tab_res['best'])
            nparams = len(varname_data)
            self.log.debug(f'Dimensions {nparams=}')
            if 'numobs' not in list_keys: nc_fid.createDimension('numobs', nprofs)
            if 'numdeps' not in list_keys: nc_fid.createDimension('numdeps', nlevels)
            if 'numvars' not in list_keys: nc_fid.createDimension('numvars', nparams)
            #varname = nc_fid.createVariable('varname', 'c', ('numvars', 'string_length7'))
            #varname = nc_fid.createVariable('varname', 'S1', ('numvars', 'string_length8'))
            varname = nc_fid.createVariable('varname', 'S1', ('numvars', 'string_length20'))
            varname.setncatts(
                              {'long_name': u"Variable name"})
            for i, var_name in enumerate(varname_data):
                #varname[i, :] = list(var_name.ljust(8))
                varname[i, :] = list(var_name.ljust(20))
            hdct_tab = tab_res['best'][:, :, :]
            #self.log.debug(f"Dimensions best {np.shape(tab_res['best'][:, :, :])}")
            self.log.info('Hindcast not empty OK1')
            nc_best = nc_fid.createVariable(
                'best_estimate', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue)
            self.log.info('Create var ok')
            nc_best.setncatts(
                {'long_name': u"Model best_estimate counterpart of obs. value"})
            self.log.debug('Create hindcast ok')
            nc_fid.variables['best_estimate'][:, :, :] = hdct_tab[:, :, :]
            self.log.debug('Push data hind ok')
        if 'forecast' in list_keys and np.any(tab_res['forecast']):
            fcst_tab = tab_res['forecast']
            nprofs, nparams, nfcsts, nlevels = np.shape(tab_res['forecast'])
            #if 'numobs' not in list_keys: nc_fid.createDimension('numobs', nprofs)
            #if 'nlevels' not in list_keys: nc_fid.createDimension('nlevels', nlevels)
            #if 'numvars' not in list_keys: nc_fid.createDimension('numvars', nparams)
            if 'numfcsts' not in list_keys: nc_fid.createDimension('numfcst', nfcsts)
            self.log.info('Forecast not empty')
            nc_fcst = nc_fid.createVariable(
                'forecast', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
            nc_fcst.setncatts(
                {'long_name': u"Model forecast counterpart of obs. value"})
            nc_fid.variables['forecast'][:, :, :, :] = fcst_tab[:, : ,:, :]
        else:
            self.log.info('Forecast is empty')
        if 'persistence' in list_keys and np.any(tab_res['persistence']):
            pers_tab = tab_res['persistence']
            nprofs, nparams, nfcsts, nlevels = np.shape(tab_res['persistence'])
            #if 'numobs' not in list_keys: nc_fid.createDimension('numobs', nprofs)
            #if 'nlevels' not in list_keys: nc_fid.createDimension('nlevels', nlevels)
            #if 'numvars' not in list_keys: nc_fid.createDimension('numvars', nparams)
            #if 'numfcsts' not in list_keys: nc_fid.createDimension('numfcst', nfcsts)
            self.log.info('Persistence not empty')
            nc_pers = nc_fid.createVariable(
                'persistence', 'f4', ('numobs', 'numvars', 'numfcsts', 'numdeps'), fill_value=self.fillvalue)
            nc_pers.setncatts(
                {'long_name': u"Model persistence counterpart of obs. value"})
            nc_fid.variables['persistence'][:, : , :, :] = pers_tab[:, :, :, :]
        else:
            self.log.info('Persistence is empty')
        self.log.debug('Add observation')
        # Create observation variable
        nc_obs = nc_fid.createVariable(
                'observation', 'f4', ('numobs', 'numvars', 'numdeps'), fill_value=self.fillvalue_corio)
        nc_obs.setncatts(
                {'long_name': u"Observation value"})
        self.log.debug(f"Dimension obs {np.shape(tab_res['observation'])}")
        nc_fid.variables['observation'][:, : , :] = tab_res['observation'][:, :, :]
        self.log.debug('Add observation ok')
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.contact = 'qualif@mercator-ocean.fr'
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model

    def write_coriolis_file(self, filename, tab1, tab2, tab3, dateval, model, variable):
        nc_fid = Dataset(filename, 'r+') #, format='NETCDF3_CLASSIC')
        nc_file = nc_fid.createVariable(
            'best_estimate', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file2 = nc_fid.createVariable(
            'climatology', 'f4', ('lat', 'lon'), fill_value=self.fillvalue)
        nc_file.setncatts(
            {'long_name': u"Model best_estimate counterpart of obs. value"})
        nc_file2.setncatts(
            {'long_name': u"Climatology counterpart of obs. value"})
        nc_fid.variables['best_estimate'][:] = tab1
        nc_fid.variables['climatology'][:] = tab3
        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.contact = 'qualif@mercator-ocean.fr'
        today = date.today()
        yyyy = str(today)[0:4]
        yyyy2 = dateval[0:4]
        month = str(today)[5:7]
        month2 = dateval[4:6]
        day = str(today)[8:10]
        day2 = dateval[6:8]
        nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 12:00:00"
        nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        nc_fid.system = model
        #print ('-------------------------------')

    def test_output(self, filename, param_dict, dateval, leadtime, lead):
        ## Test pickle files
        filename_tmp = re.sub(r'\.nc$', f'_{dateval}_{leadtime}_{lead}.p', filename)
        self.log.debug(f'{filename_tmp}')
        #filename_tmp = f"{param_dict['dirwork']}{filename_tmp}"
        return os.path.exists(filename_tmp)

    def write_outputnp(self, filename,
                       tab_results, param_dict,
                       dateval, leadtime, lead):
        self.mkdir(param_dict['diroutput'])
        filename_tmp = re.sub(r'\.nc$', f'_{dateval}_{leadtime}_{lead}.p', filename)
        filename_tmp = f"{param_dict['dirwork']}{filename_tmp}"
        filename_obs = re.sub(r'\.nc$',f'_{dateval}_observation.p', filename)
        filename_obs = f"{param_dict['dirwork']}{filename_obs}"
        filename_varname = re.sub(r'\.nc$',f'_{dateval}_varname.p', filename)
        filename_varname = f"{param_dict['dirwork']}{filename_varname}"
        filename_lon = re.sub(r'\.nc$',f'_{dateval}_longitude.p', filename)
        filename_lon = f"{param_dict['dirwork']}{filename_lon}"
        filename_lat = re.sub(r'\.nc$',f'_{dateval}_latitude.p', filename)
        filename_lat = f"{param_dict['dirwork']}{filename_lat}"
        filename_depth = re.sub(r'\.nc$',f'_{dateval}_depth.p', filename)
        filename_depth = f"{param_dict['dirwork']}{filename_depth}"
        filename_qc = re.sub(r'\.nc$',f'_{dateval}_qc.p', filename)
        filename_qc = f"{param_dict['dirwork']}{filename_qc}"
        filename_clim = re.sub(r'\.nc$',f'_{dateval}_clim.p', filename)
        filename_clim = f"{param_dict['dirwork']}{filename_clim}"
        filename_bathy = re.sub(r'\.nc$',f'_{dateval}_bathy.p', filename)
        filename_bathy = f"{param_dict['dirwork']}{filename_bathy}"
        dataorigin = param_dict['data_origin']
        # Specify the file name
        # Writing of the dictionary into the pickle file
        self.log.debug(f"{tab_results.keys()=}")
        if not os.path.isfile(filename_tmp):
            with open(filename_tmp, 'wb') as file:
                self.log.debug(f"Write Pickle file {filename_tmp} {np.shape(tab_results[leadtime])}")
                #threshold = 1e35
                ## Filter out values greater than the threshold
                #filtered_data = tab_results[leadtime][tab_results[leadtime] < threshold]
                ## Find the maximum value in the filtered array
                #if filtered_data.size > 0:
                #    max_value = np.max(filtered_data)
                #    self.log.error(f"Max value {max_value}")
                #else:
                #    self.log.error(f"Max value nan")
                #self.log.error('-------------------------------')
                pickle.dump(tab_results[leadtime], file)

        if 'observation' in tab_results.keys():
            if not os.path.isfile(filename_obs):
                self.log.debug(f"Write Pickle file obs {np.shape(tab_results[leadtime])} {filename_obs}")
                with open(filename_obs, 'wb') as file:
                    pickle.dump(tab_results['observation'], file)
        if 'varname_select' in tab_results.keys():
            if not os.path.isfile(filename_varname):
                self.log.debug(f"Write Pickle file varname {filename_varname}")
                with open(filename_varname, 'wb') as file:
                    pickle.dump(tab_results['varname_select'], file)
        if 'longitude' in tab_results.keys():
            if not os.path.isfile(filename_lon):
                with open(filename_lon, 'wb') as file:
                    self.log.debug(f"No longitude write {filename_lon} {np.shape(tab_results['longitude'])}")
                    pickle.dump(tab_results['longitude'], file)
        if 'latitude' in tab_results.keys():
            if not os.path.isfile(filename_lat):
                with open(filename_lat, 'wb') as file:
                    pickle.dump(tab_results['latitude'], file)
        if 'depth' in tab_results.keys():
            if not os.path.isfile(filename_depth):
                with open(filename_depth, 'wb') as file:
                    pickle.dump(tab_results['depth'], file)
        if 'qc' in tab_results.keys():
            if not os.path.isfile(filename_qc):
                with open(filename_qc, 'wb') as file:
                    pickle.dump(tab_results['qc'], file)
        if 'LEVITUS_clim' in tab_results.keys():
            if not os.path.isfile(filename_clim):
                with open(filename_clim, 'wb') as file:
                    pickle.dump(tab_results['LEVITUS_clim'], file)
        #dest = param_dict['diroutput']
        #shutil.move(filename_tmp, dest)


    def write_output(self, filename, tab_results, param_dict, dateval, ll_zip=True):
        dataorigin = param_dict['data_origin']
        self.mkdir(param_dict['diroutput'])
        if dataorigin == "GODAE":
            self.log.debug("Write inside GODAE file")
            self.log.debug(f"{tab_results.keys()}")
            if 'forecast'  not in tab_results.keys() and 'persistence' not in tab_results.keys() \
                    and 'best' in tab_results.keys():
                self.write_GODAE_files_best(
                    filename, tab_results['best'], dateval, param_dict['modele'],\
                    f"{param_dict['data_type']}_{param_dict['cl_src']}")
            elif 'forecast'  not in tab_results.keys() and 'persistence' not in tab_results.keys() \
                    and 'best' not in tab_results.keys() and 'clim' in tab_results.keys():
                self.write_GODAE_files_clim(
                    filename, tab_results['clim'], dateval, param_dict['modele'],\
                    f"{param_dict['data_type']}_{param_dict['cl_src']}")
            else:
                self.write_GODAE_files(
                    filename, tab_results['best'], tab_results['forecast'],
                    tab_results['persistence'], dateval, param_dict['modele'],\
                    f"{param_dict['data_type']}_{param_dict['cl_src']}")
        elif dataorigin == "BIOARGO":
                self.write_BIOARGO_files(
                    filename, tab_results['best'], dateval, param_dict['modele'],\
                    f"{param_dict['data_type']}_{param_dict['cl_src']}")
        elif dataorigin == "LEGOS":
            self.write_LEGOS_generic_files(filename, tab_results,dateval,\
                    param_dict['modele'],f"{param_dict['data_type']}_{param_dict['cl_src']}")

        elif dataorigin == "CMEMS" :
            self.write_CMEMS_generic_files(filename, tab_results,dateval, \
                    param_dict['modele'],f"{param_dict['data_type']}_{param_dict['cl_src']}")
            #self.write_CMEMS_files(
            #     filename, tab_results['best'], tab_results['forecast'],
            #     tab_results['persistence'], tab_results['clim'], dateval, param_dict['modele'], param_dict['data_type'])

        elif dataorigin == "GLOBCOLOUR" and param_dict['daily_pref'] == '7dAV':
            self.log.debug("Write inside GLOBCOLOUR files")
            self.write_GLOBCOLOR_files(filename, tab_results['best'],
                                       tab_results['forecast'], tab_results['clim'],
                                       dateval, param_dict['cl_config'],f"{param_dict['data_type']}_{param_dict['cl_src']}")
        elif dataorigin == "GLOBCOLOUR" and param_dict['daily_pref'] in self.dailylist:
            self.log.info("Write inside GLOBCOLOUR files")
            self.write_GLOBCOLOR_files2(filename, tab_results['best'],
                                        tab_results['clim'],
                                        dateval, param_dict['cl_config'],f"{param_dict['data_type']}_{param_dict['cl_src']}")
        elif dataorigin == "Microwat" or dataorigin == "SSS_ship":
            # Add missing value
            hdct_tab = np.where(
                np.array(tab_results['best']) > 9.e36, np.nan, np.array(tab_results['best']))
            self.write_best(filename, hdct_tab, dateval, param_dict['cl_config'],
                            param_dict['dimension'], 'best_estimate', param_dict['desc'],
                            f"{param_dict['data_type']}_{param_dict['cl_src']}")
            self.write_best(filename, tab_results['clim'], dateval, param_dict['cl_config'],
                            param_dict['dimension'], 'clim', param_dict['desc'],
                            f"{param_dict['data_type']}_{param_dict['cl_src']}")
        elif dataorigin == "CORIOLIS":
            self.write_coriolis_file(filename, tab_results['best'],
                                       tab_results['clim'],
                                       dateval, param_dict['cl_config'],
                                       f"{param_dict['data_type']}_{param_dict['cl_src']}")
        elif dataorigin == "CORIOLIS_RAW":
            self.write_coriolis_file_raw(filename, tab_results, dateval, param_dict['cl_config'],
                                         param_dict['data_type'], param_dict['varname_select'])
        else:
            self.log.error("Origin Not Known : %s :" % (dataorigin))

        # Remove corresponding colloc file
        for file_coloc in glob(param_dict['coloc_rep'] + '*_'+dateval+'*.p'):
            if os.path.exists(file_coloc):
                os.remove(file_coloc)

        self.log.debug('Creation file %s' % (filename))
        if ll_zip:
            # Zipped files and move in output directory
            self.log.debug(param_dict['dirwork'])
            #self.log.info(param_dict['dirwork']+"*"+param_dict['cl_conf'] +"*"+param_dict['varname']+"*.nc")
            #for filename in glob(param_dict['dirwork']+"*"+param_dict['cl_conf'] +
            #                     "*"+param_dict['varname']+"*.nc"):
            for filename in glob(param_dict['dirwork']+"*.nc"):
                self.log.debug('ZIP and move file %s' % (filename))
                f_in = open(filename, 'rb')
                f_out = gzip.open(filename+'.gz', 'wb')
                f_out.writelines(f_in)
                f_out.close()
                f_in.close()
                if param_dict['diroutput'] != param_dict['dirwork']:
                    if os.path.exists(param_dict['diroutput'] +
                                      os.path.basename(filename)+'.gz'):
                        os.remove(param_dict['diroutput'] +
                                  os.path.basename(filename)+'.gz')
                    shutil.move(filename+'.gz', param_dict['diroutput'])
                os.remove(filename)
                self.log.debug('File ok %s' % (filename))
        dest = os.path.join(param_dict['diroutput'],os.path.basename(filename))
        shutil.move(filename, dest)
