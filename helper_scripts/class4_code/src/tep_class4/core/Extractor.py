from __future__ import generators
import os
import sys
import string
import shutil
from glob import glob
import re
import numpy as np
import logging
import tarfile
import netCDF4
from os.path import dirname
sys.path.append(dirname(__file__))
from .sydate import SyDate
from .Logger import Logger
from .nco.nco import Nco
import xarray as xr
from netCDF4 import Dataset
import subprocess
from datetime import datetime

##############################################################
# C.REGNIER Juin 2014
# Class Extractor to extract profile : the selection depends on date
##############################################################


class Extractor(object):

    def __init__(self, log):

        self.log = log


    """ Factory extract file depending on type """

    def factory(self, type):
        """
         Create an interface for different type of data
         Factory
        """
        if type == "CORA3": return Extract_cora3(self.log)
        if type == "cora4.0_merc": return Extract_cora4('.tgz',self.log)
        if type == "cora4.1_raw": return Extract_cora4_1(self.log)
        if type == "CORA4.1_MERC": return Extract_cora4('.tgz',self.log)
        if type == "cora5.0_raw": return Extract_cora4('.tar.gz',self.log)
        if type == "CORA5.0": return Extract_cora4('.tgz', self.log)
        if type == "CORA5.2": return Extract_cora5_2(self.log)
        if type == "CORIO_ASSIM": return Extract_corio_cmems(self.log)
        if type == "CORIOLIS": return Extract_corio(self.log)
        if type == "CORIOLIS_RAW": return Extract_corio_raw(self.log)
        if type == "ENS": return ExtractEns(self.log)
        if type == "ARMOR_CO" or type == "ARMOR_AR" or type == "AR_": return Extract_ARMOR(self.log)
        if type == "GODAE_UV": return Extract_UVDrifters(self.log)
        if type == "GODAE_UV_filtr": return Extract_UVDrifters_filtr(self.log)
        if type == "GODAE": return Extract_GODAE(self.log)
        if type == "ASCII": return Extract_ASCII(self.log)
        if type == "L3m": return Extract_genericData(self.log)
        if type.startswith("BGC_ARGO"): return Extract_BGC_ARGO(self.log)
        if type == "L3mCHL": return Extract_CHLORO(self.log)
        if type == "Microwat_Lband_Orbite_1day_Lband" or type == "Microwat_6.9GHz_Orbite_1day": return Extract_genericData(self.log)
        if type == "001_048" : return Extract_CMEMS_drifters(self.log)
        if type == "sssdata_netcdf_good_FNAV" or type == "sssdata_netcdf_all": return Extract_SSSData(self.log)
        if type == "Legos_SIT" : return Extract_Legos_SIT(self.log)
        if type == "MDS_L3_SLA" : return Extract_MDS_L3_SLA(self.log)

    def timelinespace(self, date1, date2):
        
        """
         Create a list for a defined time range
         Arguments
         -----------------------------
         date1 : init date
         date2 : end date
         Returns
         -----------------------------
         list_time : list of time values
        """
        liste_time = []
        valeurj = date1
        while valeurj.__ge__(date1) and valeurj.__le__(date2):
            dateval = SyDate.__str__(valeurj)
            liste_time.append(dateval)
            valeurj = valeurj.goforward(1)
        return liste_time

    def run_list(self, param_dict, dateval):
        """
         Create a list of data corresponding to a specific date
         Arguments
         -----------------------------
         param_dict : dictionnary with all the parameters
         dateval : date value
         Returns
         -----------------------------
         liste_tot : list of input data corresponding to a date
        """
        liste_tot = []
        YEAR = dateval[0:4]
        liste_file = glob(param_dict['dirdata']+'/'+YEAR+'/*'+dateval+'*.nc')
        self.log.debug(f"Search file {param_dict['dirdata']}/{YEAR}/{dateval}*nc")
        for file in liste_file:
            try:
                filename = os.path.basename(file)
                if not os.path.exists(param_dict['dirwork']+filename):
                    shutil.copy(file, param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
        return liste_tot

class Extract_Legos_SIT(Extractor):

    """ Class to extract Legos SIT l3 files """
    def __init_(self, log):
        self.log = log

    def run_list(self, param_dict, daterun):
        liste_tot=[]
        liste_file=glob(param_dict['dirdata']+'/*'+daterun+'*.nc')
        self.log.info("Liste file {} {} ".format(liste_file, daterun))
        if not liste_file:
            self.log.error("Missing Legos_SIT {} {}".format(param_dict['dirdata'], daterun))
            raise
        self.log.debug('Liste file Legos_SIT {}'.format(liste_file))
        for file in liste_file:
            try:
                filename = os.path.basename(file)
                shutil.copy(file, param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                raise ("I/O error({0}): {1}".format(e.errno, e.strerror))

        return liste_file

    def run_over(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        valeurj_tmp = SyDate(str(date1))
        valeurj = valeurj_tmp
        valeurj2 = SyDate(str(date2))
        liste_tot=[]
        while  valeurj_tmp >= valeurj and valeurj_tmp <= valeurj2 :
            dateval = str(valeurj_tmp)
            self.log.info('Day 5.2 %s ' % (dateval))
            month=dateval[0:6]
            year=dateval[0:4]
            frame=dirdata+'/*'+dateval.strip()+'*.nc'
            liste_file=glob(frame)
            self.log.debug(f'{Frame=}')
            self.log.info(f'Liste file CORA {liste_file}')
            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    shutil.copy(new_file,dirtmp)
                    self.log.info('copy file {}'.format(dirtmp+filename))
                    liste_tot.append(dirtmp+filename)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj_tmp=valeurj_tmp.goforward(1)
        ## TODO Finish the run_over for SIT files

class Extract_CMEMS_drifters(Extractor):

    """ Class to extract CMEMS drifters """

    def run_list(self, param_dict, dateval):
        """
         Create a list of data corresponding to a specific date
         Arguments
         -----------------------------
         param_dict : dictionnary with all the parameters
         dateval : date value
         Returns
         -----------------------------
         liste_tot : list of input data corresponding to a date
        """
        liste_tot = []
        year = SyDate(dateval).year
        month = SyDate(dateval).month
        if month < 10:
            month = '0'+str(month)
        else:
            month = str(month)
        liste_file = glob(param_dict['dirdata']+str(year)+'/'+str(month)+'/*'+dateval+'*.nc')
        for file in liste_file:
            try:
                filename = os.path.basename(file)
                shutil.copy(file, param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
        return liste_tot

class Extract_SSSData(Extractor):

    """ Extract SSS Data files """

    def __init__(self):
        # ===============
        # LOGGER
        # ===============
        logging.basicConfig(format='%(asctime)s %(message)s')
        self.log = logging.getLogger("Extract_SSS_data")
        self.log.setLevel(logging.INFO)
        self.date = SyDate('19500101')

        self.dimension = 'TIME'
    def run(self, date1, date2, dirdata, dirtmp, typedata, template, prefix):

        logging.basicConfig(format='%(asctime)s %(message)s')
        self.log = logging.getLogger("Extract Generic files")
        self.log.setLevel(logging.INFO)
        liste_tot = []
        valeurj = date1
        options = []
        options.extend(['-h'])
        # Nco constructor
        nco = Nco()
        # Create a timeline
        timelinespace = self.timelinespace(date1,date2)
        liste_file = glob(dirdata+'/*nc')
        length = len(liste_file)
        for indice, file in enumerate(liste_file):
            try:
                filename = os.path.basename(file)
                data_type = filename.split(prefix)[1].split('_')[1]
                nc = netCDF4.Dataset(file, 'r')
                # Extract time variable
                variable = self.dimension
                v = nc.variables[variable][:]
                # Convert to date
                valeurtmp = valeurj
                date_all = [SyDate.__str__(self.date.setfromjulian(
                    int(valeurtmp))) for valeurtmp in v]
                unique_date = reduce(
                    lambda l, x: l+[x] if x not in l else l, date_all, [])
                ## Find the correspondance with 2 lists
                result = []
                list1 = set(timelinespace)
                list2 = set(unique_date)
                date_values = list(list1.intersection(list2))
                if date_values:
                    self.log.info("Not empty ")
                    for date in date_values:
                        ## find indexes in date_all
                        indexes = [i for i, x in enumerate(
                                   date_all) if x == date]
                        file_out = prefix+'_'+data_type+'_'+str(date)+'.nc'
                        self.log.info('Creation fileout : %s '%(file_out))
                        ## Create output extracted file
                        nco.ncks(input=file, output=dirtmp+file_out, dimension=self.dimension+','+str(
                            indexes[0])+','+str(indexes[-1]), options=options)
                        if dirtmp+file_out not in liste_tot:
                            liste_tot.append(dirtmp+file_out)
            except IOError as e:
                self.log.error ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
            # Search index 
            self.log.info("File ok : %s %i %i" %(file,indice,length))
        return liste_tot


class Extract_genericData(Extractor):

    """ Extract files """

    def run_list(self, param_dict, dateval):
        logging.basicConfig(format='%(asctime)s %(message)s')
        self.log =logging.getLogger("Extract Generic files")
        self.log.setLevel(logging.INFO)
        liste_tot = []
        liste_file = glob(param_dict['dirdata']+'/*'+dateval+'*.nc')
        for file in liste_file:
            try:
                filename = os.path.basename(file)
                shutil.copy(file, param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)

        return liste_tot

class Extract_BGC_ARGO(Extractor):

    """ Extract files """

    def __init__(self, log):
        self.log = log

    def run_list(self, param_dict, dateval):
        liste_tot = []
        self.log.info(f"Inside Extract_BGC_ARGO {dateval}")
        liste_file = glob(f"{param_dict['dirdata']}/*{dateval[0:6]}*.nc")
        for file in liste_file:
            try:
                filename = os.path.basename(file)
                shutil.copy(file, param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                self.log ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
        self.log.info(f"liste {liste_tot}")
        return liste_tot


class Extract_CHLORO(Extractor):

    """ Extract Chloro Globcolour files """
    
    def __init__(self, log):
        self.log = log

    def mkdir(self, path):
        "Create a directory, and parents if needed"
        if not os.path.exists(path):
            os.makedirs(path)

    def run_list(self, param_dict, dateval):
        liste_tot = []
        #liste_file = glob(param_dict['dirdata']+'/'+dateval+'*REP'+'*.nc')
        self.log.info('Extract_CHLORO')
        liste_file = glob(param_dict['dirdata']+'/'+dateval+'*.nc')
        self.log.info(f"{param_dict['dirdata']}/{dateval}'*.nc")
        if np.size(liste_file) > 0:
            liste_file = liste_file
        else:
            liste_file = glob(param_dict['dirdata']+'/'+dateval+'*DT'+'*.nc')
            if np.size(liste_file) > 0:
                liste_file = liste_file
            else:
                liste_file = glob(param_dict['dirdata']+'/'+dateval+'*NRT'+'*.nc')
        for file in liste_file:
            try:
                filename = 'Class4_'+dateval+'_'+param_dict['data_origin']+'_'+param_dict['cl_conf']+'_'+param_dict['varname'][0]+'.nc'
                self.mkdir(param_dict['dirwork'])
                shutil.copy(file,param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)

        return liste_tot

class Extract_corio_raw(Extractor):

    """ Extract coriolis raw common file """

    def __init__(self, log): 

        # ===============
        # LOGGER
        # ===============
        self.log = log
        self.date=SyDate('19500101')

    def search(self,dirdata,**kwargs):
        list_dateval=[]
        ind=0
        for file in glob(dirdata+"*.nc"):
            nc   = netCDF4.Dataset(file,'r')
            variable="JULD"
            v = nc.variables[variable][:]
            # Convert to date
            dateval=[SyDate.__str__(self.date.setfromjulian(int(valeurj))) for valeurj in v]
            ind=0
            for val in dateval :
                if SyDate(val).tojulian() < 16189 :
                   self.log.error("Probleme  : %s %i  "%(val,ind))
                else :
                    list_dateval.append(val)
                ind+=1
        dateval = sorted(set(list_dateval))
        return SyDate(dateval[0]),SyDate(dateval[-1])

    def run(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        liste_tot=[]
        ind=0
        dimprof='N_PROF'
        options = []
        options.extend(['-h'])
        # Nco constructor
        nco = Nco()
        for file in glob(dirdata+"*.nc"):
            frame=os.path.basename(file).split('.')[0]
            nc   = netCDF4.Dataset(file,'r')
            variable="JULD"
            v = nc.variables[variable][:]
            # Convert to date
            dateval=[SyDate.__str__(self.date.setfromjulian(int(valeurj))) for valeurj in v]
            ind=0
            for val in dateval :
                if SyDate(val).tojulian() < 16189 :
                    self.log.error("Probleme  : %s %i  "%(val,ind))
                else :
                    file_out=str(val)+'_'+frame+'_profile_'+str(ind)+'.nc'
                    nco.ncks(input=file,output=dirtmp+file_out, fortran=True ,dimension=str(dimprof)+','+str(ind+1),options=options)
                    liste_tot.append(dirtmp+file_out)
                    # self.log.info('Day %s ' % (dateval))
                ind+=1
        return liste_tot

class Extract_corio(Extractor):

    """ Extract coriolis common file """
    def __init__(self, log):
        self.log = log

    def run(self, param_dict, dateval):
        # ===============
        # LOGGER
        # ===============
        liste_tot=[]
        self.log.info('Day %s ' % (dateval))
        frame=param_dict['dir_data']+'/*'+dateval.strip()+'*.nc'
        liste_file=glob(frame)
        self.log.info('Liste file %s ' % (liste_file))
        for new_file in liste_file :
            try :
                filename=os.path.basename(new_file)
                shutil.copy(new_file,param_dict['dirtmp'])
                liste_tot.append(param_dict['dirtmp']+filename)
            except IOError as e:
                self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)

        return liste_tot


class Extract_cora5_2(Extractor):

    """ Extract cora 5.2 file """

    def __init__(self, log):
        self.date = SyDate('19500101')
        self.log = log

    def run(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        valeurj_tmp = SyDate(str(date1))
        valeurj = valeurj_tmp
        valeurj2 = SyDate(str(date2))
        liste_tot=[]
        self.log.info(f'Extract CORA 5.2 {valeurj_tmp} {valeurj} {valeurj2}')
        while  valeurj_tmp >= valeurj and valeurj_tmp <= valeurj2 :
            dateval = str(valeurj_tmp)
            #dateval=str(SyDate(str(self.date.setfromjulian(int(valeurj)))))
            self.log.info(f'Day 5.2 {dateval}')
            month=dateval[4:6]
            year=dateval[0:4]
            type_obs = "PR_PF"
            frame=dirdata+year+'/*'+dateval.strip()+'*{type_obs}_*.nc'
            #frame=dirdata+year+'/*'+dateval.strip()+'*.nc'
            self.log.info(f"{frame=}")
            liste_file=glob(frame)
            self.log.debug(f'{frame=}')
            self.log.info(f'Liste file CORA {liste_file}')

            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    shutil.copy(new_file,dirtmp)
                    self.log.debug('copy file ok {}'.format(dirtmp+filename))
                    liste_tot.append(dirtmp+filename)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj_tmp=valeurj_tmp.goforward(1)

        return liste_tot

    def run_over(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        valeurj_tmp = SyDate(str(date1))
        valeurj = valeurj_tmp
        valeurj2 = SyDate(str(date2))
        liste_tot=[]
        while  valeurj_tmp >= valeurj and valeurj_tmp <= valeurj2 :
            dateval = str(valeurj_tmp)
            self.log.info('Day 5.2 %s ' % (dateval))
            month=dateval[0:6]
            year=dateval[0:4]
            frame=dirdata+'/*'+dateval.strip()+'*.nc'
            liste_file=glob(frame)
            self.log.info('Liste file CORA %s ' % (liste_file))
            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    shutil.copy(new_file,dirtmp)
                    self.log.info('copy file {}'.format(dirtmp+filename))
                    liste_tot.append(dirtmp+filename)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj_tmp=valeurj_tmp.goforward(1)

        return liste_tot

    def run_list(self, param_dict, dateval):
        liste_tot = []
        YEAR = dateval[0:4]
        #type_obs = "PR_PF"
        #type_obs = "TS_MO"
       # type_obs = "TS_DB"
      #  type_obs = "XX"
        #type_obs = "PR_GL"
        #type_obs = "PR_CT"
        #type_obs = "TS_FB"
        type_obs = "*"
        #type_obs = "PR_BO"
        #type_obs = "PR_XB"
        liste_file = glob(f"{param_dict['dirdata']}/{YEAR}/*{dateval}*{type_obs}*.nc")
        #liste_file = glob(param_dict['dirdata']+'/'+YEAR+'/*'+dateval+'*.nc')
        self.log.debug(f"Search file  {liste_file}")
        for file in liste_file:
            try:
                filename = os.path.basename(file)
                if not os.path.exists(param_dict['dirwork']+filename):
                    self.log.debug(f"Copy file {file}")
                    if os.path.exists(file) and os.access(file, os.R_OK):
                        shutil.copy(file, param_dict['dirwork']+filename)
                    else:
                        print(f"Source file {file} is not accessible.")
                        sys.exit(1)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
        return liste_tot

class Extract_Legos_SIT(Extractor):

    """ Class to extract Legos SIT l3 files """
    def __init_(self, log):
        self.log = log

    def run_list(self, param_dict, daterun):
        liste_tot=[]
        liste_file=glob(param_dict['dirdata']+'/*'+daterun+'*.nc')
        self.log.info("Liste file {} {} ".format(liste_file, daterun))
        if not liste_file:
            self.log.error("Missing Legos_SIT {} {}".format(param_dict['dirdata'], daterun))

class Extract_cora4_1(Extractor):

    """ Extract cora 4.1 file """

    def __init__(self, log):
        self.date = SyDate('19500101')
        self.log = log

    def run(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        # ===============
        # LOGGER
        # ===============
        log = logging.getLogger("Extract_ARMOR")
        log.setLevel(logging.INFO)
        valeurj_tmp = SyDate(str(date1))
        # Level of logging output
        valeurj = valeurj_tmp
        valeurj2 = SyDate(str(date2))
        liste_tot=[]
        while  valeurj_tmp >= valeurj and valeurj_tmp <= valeurj2 :
            dateval = str(valeurj_tmp)
            #dateval=str(SyDate(str(self.date.setfromjulian(int(valeurj)))))
            self.log.info('Day %s ' % (dateval))
            month=dateval[0:6] 
            #frame=dirdata+month+'/*'+dateval.strip()+'*.nc'
            frame=dirdata+'/*'+dateval.strip()+'*.nc'
            liste_file=glob(frame)
            self.log.info('Liste file CORA %s ' % (liste_file))
            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    shutil.copy(new_file,dirtmp)
                    self.log.info('copy file {}'.format(dirtmp+filename))
                    liste_tot.append(dirtmp+filename)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj_tmp=valeurj_tmp.goforward(1)

        return liste_tot

class Extract_corio_cmems(Extractor):

    """ Extract CMEMS coriolis profiles files """

    def __init__(self, log):
        self.date = SyDate('19500101')
        self.log = log

    def run(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        # ===============
        # LOGGER
        # ===============
        valeurj_tmp = SyDate(str(date1))
        # Level of logging output
        valeurj = valeurj_tmp
        valeurj2 = SyDate(str(date2))
        liste_tot=[]
        while  valeurj_tmp >= valeurj and valeurj_tmp <= valeurj2 :
            dateval = str(valeurj_tmp)
            self.log.info('Day %s ' % (dateval))
            year=dateval[0:4]
            frame=dirdata+year+'/*'+dateval.strip()+'*.nc'
            liste_file=glob(frame)
            self.log.info('Liste file CMEMS profiles %s ' % (liste_file))
            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    shutil.copy(new_file,dirtmp)
                    self.log.info('copy file {}'.format(dirtmp+filename))
                    liste_tot.append(dirtmp+filename)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj_tmp=valeurj_tmp.goforward(1)

        return liste_tot

class Extract_ASCII(Extractor): 

    """ Extract ASCII file """

    def __init__(self):
        self.PREFIX='Ascii file extraction'
    def run_list(self, param_dict, dateval):
        liste_tot=[]
        liste_file=glob(param_dict['dirdata']+'/*_'+dateval+'*.txt')
        for file in liste_file :
            try :
                liste_tot.append(file)
                options = []
                options.extend(['-h','-O','-a','numfcsts'])
                nco.ncwa(input=new_file, output=dirtmp+filename, fortran=True ,options=options)
                nc = netCDF4.Dataset(dirtmp+filename, 'r')
                variables = nc.variables.keys()

            except IOError as e:
                print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
        return liste_tot

class Extract_UVDrifters_filtr(Extractor):

    """ Extract and reconstruct UV drifters """

    def __init__(self, log):
        self.log = log
        # Nco constructor
        self.nco = Nco()
        self.nb_vars = 2
        self.nb_depths = 1
        self.fillvalue_b = -128
    
    def rewrite(self, file, nb_fcst, nb_fcst2, file_new):
        """
            Rewrite netcdf file to the right GODAE format
        """
        # Create File
        self.log.info(f"Write new filtered file {file_new}")
        self.nb_fcst = nb_fcst
        self.nb_fcst2 = nb_fcst2
        #nc_fid = Dataset(file_new, 'w', format='NETCDF4', locking=True)
        ## Filtered unrealistic values with longitude > 180°
        nc_file = xr.open_dataset(file, decode_times=False)
        variables = nc_file.data_vars
        lon_obs  = nc_file['LONGITUDE_FILTR'].values
        lat_obs = nc_file['LATITUDE_FILTR'].values
        time_obs =  nc_file['TIME'].values
        time_obs_qc = nc_file['TIME_QC'].values
        position_qc = nc_file['POSITION_QC'].values
        depth_obs = nc_file['DEPH'].values
        depth_obs_qc = nc_file['DEPH_QC'].values
        code_values_obs = nc_file['PLATFORM_CODE'].values
        u_obs_filt = nc_file['EWCT_FILTR'].values
        v_obs_filt = nc_file['NSCT_FILTR'].values
        u_obs_qc = nc_file['EWCT_FILTR_QC'].values
        v_obs_qc = nc_file['NSCT_FILTR_QC'].values
        nc_file.close()
        nc = Dataset(file, 'r')
        missing_value =  nc.variables['EWCT_FILTR']._FillValue
        nc.close()
        #with Dataset(file_new, 'w', format='NETCDF4', locking=True) as nc_fid_rewrite:
        nc_fid_rewrite = Dataset(file_new, 'w', format='NETCDF4', locking=True)
        nc_fid_rewrite.createDimension('numobs', None)
        nc_fid_rewrite.createDimension('numvars', self.nb_vars)
        nc_fid_rewrite.createDimension('numdeps', self.nb_depths)
        nc_fid_rewrite.createDimension('numfcsts', self.nb_fcst)
        nc_fid_rewrite.createDimension('numfcsts2', self.nb_fcst2)
        nc_fid_rewrite.createDimension('string_length', 18)
        nc_fid_rewrite.createDimension('string64', 64)
        #nc_fid_rewrite.close()
        nc_lon = nc_fid_rewrite.createVariable('longitude','f4',('numobs'))
        nc_lon.setncatts({'long_name': u"Longitude of each location filtered over 3 days",\
                                    'units': u"degrees_east",\
                                    'comment': u"3-day Lanczos filter" })
        nc_lat = nc_fid_rewrite.createVariable('latitude', 'f4',('numobs'))
        nc_lat.setncatts({'long_name': u"Latitude of each location filtered over 3 days",\
                                'units': u"degrees_north",\
                                'comment': u"3-day Lanczos filter" })
        nc_depth = nc_fid_rewrite.createVariable('depth','f4',('numobs', 'numdeps'))
        nc_depth.setncatts({'long_name': u"Depth",\
                            'standard_name': u"depth",\
                            'valid_min':  -12000.0, \
                            'valid_max': 12000.0,\
                            'axis': u"Z", \
                            'positive': u"down"})
        nc_depth_qc  = nc_fid_rewrite.createVariable('depth_qc', 'i1',('numobs'),fill_value=self.fillvalue_b)
        nc_depth_qc.long_name =  "Depth quality flag"
        nc_depth_qc.valid_min = 0
        nc_depth_qc.valid_max = 9
        nc_depth_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_depth_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        nc_position_qc  = nc_fid_rewrite.createVariable('position_qc', 'i1',('numobs'),fill_value=self.fillvalue_b)
        nc_position_qc.long_name =  "Position quality flag"
        nc_position_qc.valid_min = 0
        nc_position_qc.valid_max = 9
        nc_position_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_position_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        nc_obsfiltr_qc  = nc_fid_rewrite.createVariable('observation_qc', 'i1',('numobs','numvars'),fill_value=self.fillvalue_b)
        nc_obsfiltr_qc.long_name =  "Filtered observations quality flag"
        nc_obsfiltr_qc.valid_min = 0
        nc_obsfiltr_qc.valid_max = 9
        nc_obsfiltr_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_obsfiltr_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        nc_plateform = nc_fid_rewrite.createVariable('plateform_code','c',('numobs','string64'))
        nc_plateform.long_name = "Platform code"
        nc_plateform.standard_name = "platform"

        nc_ws_qc  = nc_fid_rewrite.createVariable('windage_qc', 'i1',('numobs','numvars'),fill_value=self.fillvalue_b)
        nc_ws_qc.long_name =  "Windage quality flag"
        nc_ws_qc.valid_min = 0
        nc_ws_qc.valid_max = 9
        nc_ws_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_ws_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"

        nc_time = nc_fid_rewrite.createVariable('obs_time', 'f8',('numobs'))
        nc_time.setncatts({'long_name': u"Time",\
                    'standard_name': u"time",\
                    'units': u"days since 1950-01-01T00:00:00Z"})
        nc_time.standard_name = 'time'
        nc_time.calendar = 'gregorian'
        nc_time.axis = 'T'
        nc_time_qc  = nc_fid_rewrite.createVariable('time_qc', 'i1',('numobs'),fill_value=self.fillvalue_b)
        nc_time_qc.long_name =  "Time quality flag"
        nc_time_qc.valid_min = 0
        nc_time_qc.valid_max = 9
        nc_time_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_time_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        nc_obs = nc_fid_rewrite.createVariable('observation', 'f4',('numobs', 'numvars', 'numdeps'),fill_value=missing_value)
        nc_obs.setncatts({'long_name' : u"3-day filtered currents from in situ drifters",\
                                        "units" : u"m s-1",\
                                        'comment': u"3-day Lanczos filter" })
        nc_ws = nc_fid_rewrite.createVariable('windage', 'f4',('numobs', 'numvars', 'numdeps'),fill_value=missing_value)
        nc_ws.setncatts({'long_name' : u"wind slippage correction at the drog depth",\
                                        "units" : u"m s-1",\
                                        'comment': u"3-day Lanczos filter" })
        #variables = nc_file.data_vars
        nc_var = nc_fid_rewrite.createVariable('varname', 'S1',('numvars','string_length'))
        variables_vel = ['eastward_velocity ','northward_velocity']
        nc_var.setncatts({'long_name': u"name of variables"})
        nc_var[:] = netCDF4.stringtochar(np.array(variables_vel, 'S'))
        mask = lon_obs <= 180
        lon_obs = lon_obs[mask]
        lat_obs = lat_obs[mask]
        time_obs = time_obs[mask]
        time_obs_qc = time_obs_qc[mask]
        position_qc = position_qc[mask]
        depth_obs = depth_obs[mask]
        depth_obs_qc = depth_obs_qc[mask]
        code_values_obs = code_values_obs[mask]
        u_obs_filt = u_obs_filt[mask]
        v_obs_filt = v_obs_filt[mask]
        u_obs_qc = u_obs_qc[mask]
        v_obs_qc = v_obs_qc[mask]
        if 'EWCT_WS_FILTR' and 'NSCT_WS_FILTR' in variables.keys():
            ll_var_ws = True
            u_filtr_ws = nc_file['EWCT_WS_FILTR'].values
            v_filtr_ws = nc_file['NSCT_WS_FILTR'].values
            u_filtr_ws_qc = nc_file['EWCT_WS_FILTR_QC'].values
            v_filtr_ws_qc = nc_file['NSCT_WS_FILTR_QC'].values
            u_filtr_ws = u_filtr_ws[mask]
            v_filtr_ws = v_filtr_ws[mask]
            u_filtr_ws_qc = u_filtr_ws_qc[mask]
            v_filtr_ws_qc = v_filtr_ws_qc[mask]
        else:
            u_filtr_ws = np.full(u_obs_filt.shape, np.nan)
            v_filtr_ws = np.full(v_obs_filt.shape, np.nan)
            u_filtr_ws_qc = np.copy(u_obs_qc)
            v_filtr_ws_qc = np.copy(v_obs_qc)
            u_filtr_ws_qc[:, :] = 9
            v_filtr_ws_qc[:, :] = 9

        nc_fid_rewrite.variables['latitude'][:] = lat_obs
        nc_fid_rewrite.variables['longitude'][:] = lon_obs
        nc_fid_rewrite.variables['obs_time'][:] = time_obs
        nc_fid_rewrite.variables['time_qc'][:] = time_obs_qc
        nc_fid_rewrite.variables['position_qc'][:] = position_qc
        nc_fid_rewrite.variables['depth'][:,:] = depth_obs
        nc_fid_rewrite.variables['depth_qc'][:] = depth_obs_qc
        code_values2 = [var.decode("utf-8") for var in code_values_obs]
        nc_fid_rewrite.variables['plateform_code'][:] = code_values2
        nc_fid_rewrite.variables['observation'][:,0,:] = u_obs_filt[:,:]
        nc_fid_rewrite.variables['observation'][:,1,:] = v_obs_filt[:,:]
        nc_fid_rewrite.variables['observation_qc'][:,0] = u_obs_qc[:,:]
        nc_fid_rewrite.variables['observation_qc'][:,1] = v_obs_qc[:,:]
        nc_fid_rewrite.variables['windage'][:,0,:] = u_filtr_ws[:,:]
        nc_fid_rewrite.variables['windage'][:,1,:] = v_filtr_ws[:,:]
        nc_fid_rewrite.variables['windage_qc'][:,0] = u_filtr_ws_qc
        nc_fid_rewrite.variables['windage_qc'][:,1] = v_filtr_ws_qc

        # Add global attributs
        nc_fid_rewrite.institution = 'MERCATOR OCEAN'
        nc_fid_rewrite.contact = 'cregnier@mercator-ocean.fr'
        #nc_fid_rewrite.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid_rewrite.time_interp = "daily average fields"
        nc_fid_rewrite.Conventions = 'CF-1.6'
        nc_fid_rewrite.close()

    def list(self, daterun, dirdata, typedata):
        liste_tot=[]
        year = daterun[0:4]
        month = daterun[4:6]
        liste_file = glob(dirdata+ '/' + year + '/' + month + '/*' + daterun+'*.nc')
        if not liste_file:
            self.log.error(f"Missing Drifters files {dirdata}/{year}/{month}/*{daterun}*nc")
            return [], True
        else:
            self.log.info('Liste file UV drifters {}'.format(liste_file))
            return liste_file, False

    def run_list(self, param_dict, daterun):
        liste_tot=[]
        year = daterun[0:4]
        month = daterun[4:6]
        #liste_file=glob(param_dict['dirdata']+'/'+daterun[0:4]+'/*'+daterun+'*.nc')
        liste_file=glob(param_dict['dirdata'] + '/' + year + '/' + month + '/*' + daterun+'*.nc')
        if not liste_file:
            self.log.error(f"Missing Drifters files {param_dict['dirdata']}/{year}/{month}/*{daterun}*nc")
            return []
        self.log.info('Liste file UV drifters {}'.format(liste_file))
        nb_fcst = len(param_dict['lead_int'])
        nb_fcst2 = len(param_dict['lead_int2'])
        self.log.debug('Number of forecasts {}'.format(nb_fcst))
        for new_file in liste_file :
            try :
                file_new = 'class4_' + daterun + '_' + param_dict['cl_conf']\
                           +  '_' + param_dict['model_type'].lower() + \
                           '_currents-filtr.nc'
                ## If missing Read and Rewrite file
                if not  os.path.exists( param_dict['dirwork']+file_new):
                    self.log.debug(f"Write Filename {param_dict['dirwork']}{file_new}")
                    self.rewrite(liste_file[0], nb_fcst, nb_fcst2, param_dict['dirwork']+file_new)
                else:
                    self.log.debug(f"Filename exist {param_dict['dirwork']}{file_new}")
            except IOError as e:
                self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                raise
        liste_tot.append(param_dict['dirwork']+file_new)
        return liste_tot


class Extract_UVDrifters(Extractor):

    """ Extract and reconstruct UV drifters """

    def __init__(self, log):
        self.log = log
        # Nco constructor
        self.nco = Nco()
        self.nb_vars = 2
        self.nb_depths = 1
        self.fillvalue_b = -128

    def rewrite(self, file, nb_fcst, nb_fcst2, file_new):
        """
            Rewrite netcdf file to the right GODAE format
        """
        # Open Dataset
        self.log.info("Open file {}".format(file))
        nc_file = xr.open_dataset(file, decode_times=False)
        variables = nc_file.data_vars
        # Create File
        self.log.info("Write new file {}".format(file_new))
        self.nb_fcst = nb_fcst
        self.nb_fcst2 = nb_fcst2
        nc_fid = Dataset(file_new, 'w', format='NETCDF4', locking=True)
        nc_fid.createDimension('numobs', None)
        nc_fid.createDimension('numvars', self.nb_vars)
        nc_fid.createDimension('numdeps', self.nb_depths)
        nc_fid.createDimension('numfcsts', self.nb_fcst)
        nc_fid.createDimension('numfcsts2', self.nb_fcst2)
        nc_fid.createDimension('string_length', 18)
        nc_fid.createDimension('string64', 64)
        nc_lon = nc_fid.createVariable('longitude','f4',('numobs'))
        nc_lon.setncatts({'long_name': u"Longitudes",\
                                    'units': u"degrees_east"})
        nc_lat = nc_fid.createVariable('latitude', 'f4',('numobs'))
        nc_lat.setncatts({'long_name': u"Latitudes",\
                                'units': u"degrees_north"})
        nc_depth = nc_fid.createVariable('depth','f4',('numobs', 'numdeps'))
        nc_depth.setncatts({'long_name': u"Depth",\
                            'standard_name': u"depth",\
                            'valid_min':  -12000.0, \
                            'valid_max': 12000.0,\
                            'axis': u"Z", \
                            'positive': u"down"})
        nc_depth_qc  = nc_fid.createVariable('depth_qc', 'i1',('numobs'),fill_value=self.fillvalue_b)
        nc_depth_qc.long_name =  "Depth quality flag"
        nc_depth_qc.valid_min = 0
        nc_depth_qc.valid_max = 9
        nc_depth_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_depth_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        nc_position_qc  = nc_fid.createVariable('position_qc', 'i1',('numobs'),fill_value=self.fillvalue_b)
        nc_position_qc.long_name =  "Position quality flag"
        nc_position_qc.valid_min = 0
        nc_position_qc.valid_max = 9
        nc_position_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_position_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        
        nc_obs_qc  = nc_fid.createVariable('observation_qc', 'i1',('numobs','numvars'),fill_value=self.fillvalue_b)
        nc_obs_qc.long_name =  "Observations quality flag"
        nc_obs_qc.valid_min = 0
        nc_obs_qc.valid_max = 9
        nc_obs_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_obs_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"
        
        nc_plateform = nc_fid.createVariable('plateform_code','c',('numobs','string64'))
        nc_plateform.long_name = "Platform code"
        nc_plateform.standard_name = "platform"

        nc_ws_qc  = nc_fid.createVariable('windage_qc', 'i1',('numobs','numvars'),fill_value=self.fillvalue_b)
        nc_ws_qc.long_name =  "Windage quality flag"
        nc_ws_qc.valid_min = 0
        nc_ws_qc.valid_max = 9
        nc_ws_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_ws_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"

        nc_time = nc_fid.createVariable('obs_time', 'f8',('numobs'))
        nc_time.setncatts({'long_name': u"Time",\
                    'standard_name': u"time",\
                    'units': u"days since 1950-01-01T00:00:00Z"})
        nc_time.standard_name = 'time'
        nc_time.calendar = 'gregorian'
        nc_time.axis = 'T'
        nc_time_qc  = nc_fid.createVariable('time_qc', 'i1',('numobs'),fill_value=self.fillvalue_b)
        nc_time_qc.long_name =  "Time quality flag"
        nc_time_qc.valid_min = 0
        nc_time_qc.valid_max = 9
        nc_time_qc.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        nc_time_qc.flag_meanings = "no_qc_performed good_data probably_good_data bad_data_that_are_potentially_correctable bad_data value_changed not_used nominal_value interpolated_value missing_value"

        nc = Dataset(file, 'r')
        missing_value =  nc.variables['EWCT_FILTR']._FillValue
        nc_obs = nc_fid.createVariable('observation', 'f4',('numobs', 'numvars', 'numdeps'),fill_value=missing_value)
        nc_obs.setncatts({'long_name' : u"currents from in situ drifters",\
                                        "units" : u"m s-1" })
        nc_ws = nc_fid.createVariable('windage', 'f4',('numobs', 'numvars', 'numdeps'),fill_value=missing_value)
        nc_ws.setncatts({'long_name' : u"wind slippage correction at the drog depth",\
                                        "units" : u"m s-1",\
                                        'comment': u"raw data" })
        nc_var = nc_fid.createVariable('varname', 'S1',('numvars','string_length'))
        variables_vel = ['eastward_velocity ','northward_velocity']
        nc_var.setncatts({'long_name': u"name of variables"})
        nc_var[:] = netCDF4.stringtochar(np.array(variables_vel, 'S'))
        lon_obs  = nc_file['LONGITUDE'].values
        lat_obs = nc_file['LATITUDE'].values
        time_obs =  nc_file['TIME'].values
        time_obs_qc = nc_file['TIME_QC'].values
        position_qc = nc_file['POSITION_QC'].values
        depth_obs = nc_file['DEPH'].values
        depth_obs_qc = nc_file['DEPH_QC'].values
        code_values_obs = nc_file['PLATFORM_CODE'].values
        u_obs = nc_file['EWCT'].values
        v_obs = nc_file['NSCT'].values
        u_obs_qc = nc_file['EWCT_QC'].values
        v_obs_qc = nc_file['NSCT_QC'].values 
        code_values_obs = nc_file['PLATFORM_CODE'].values
        #variables = ncdata.data_vars
        mask = lon_obs <= 180
        lon_obs = lon_obs[mask]
        lat_obs = lat_obs[mask]
        time_obs = time_obs[mask]
        time_obs_qc = time_obs_qc[mask]
        position_qc = position_qc[mask]
        depth_obs = depth_obs[mask]
        depth_obs_qc = depth_obs_qc[mask]
        code_values_obs = code_values_obs[mask]
        u_obs = u_obs[mask]
        v_obs = v_obs[mask]
        u_obs_qc = u_obs_qc[mask]
        v_obs_qc = v_obs_qc[mask]
        ll_var_ws = False
        if 'EWCT_WS' and 'NSCT_WS' in variables.keys():
            ll_var_ws = True
            print('Windage ok')
            u_ws = nc_file['EWCT_WS'].values
            v_ws = nc_file['NSCT_WS'].values
            u_ws_qc = nc_file['EWCT_WS_QC'].values
            v_ws_qc = nc_file['NSCT_WS_QC'].values
            u_ws = u_ws[mask]
            v_ws = v_ws[mask]
            u_ws_qc = u_ws_qc[mask] 
            v_ws_qc = v_ws_qc[mask]
        else:
            print('No Windage')
            u_ws = np.full(u_obs.shape, np.nan)
            v_ws = np.full(v_obs.shape, np.nan)
            u_ws_qc = np.copy(u_obs_qc)
            v_ws_qc = np.copy(v_obs_qc)
            u_ws_qc[:, :] = 9
            v_ws_qc[:, :] = 9
        nc_fid.variables['longitude'][:] = lon_obs 
        nc_fid.variables['latitude'][:] = lat_obs
        nc_fid.variables['obs_time'][:] = time_obs
        nc_fid.variables['time_qc'][:] = time_obs_qc
        nc_fid.variables['position_qc'][:] = position_qc
        nc_fid.variables['depth'][:,:] = depth_obs
        nc_fid.variables['depth_qc'][:] = depth_obs_qc 
        #nc_fid.variables['longitude'][:] = nc_file['LONGITUDE'].values
        #nc_fid.variables['latitude'][:] = nc_file['LATITUDE'].values
        #nc_fid.variables['obs_time'][:] = nc_file['TIME'].values
        #nc_fid.variables['time_qc'][:] = nc_file['TIME_QC'].values
        #nc_fid.variables['position_qc'][:] = nc_file['POSITION_QC'].values
        #nc_fid.variables['depth'][:,:] = nc_file['DEPH'].values
        #nc_fid.variables['depth_qc'][:] = nc_file['DEPH_QC'].values
        #code_values = nc_file['PLATFORM_CODE'].values
        code_values2 = [var.decode("utf-8") for var in code_values_obs]
        nc_fid.variables['plateform_code'][:] = code_values2
        #u_cur = nc_file['EWCT'].values
        #v_cur = nc_file['NSCT'].values
        #nc_fid.variables['observation'][:,0,:] = u_cur[:,:]
        #nc_fid.variables['observation'][:,1,:] = v_cur[:,:]
        #nc_fid.variables['observation_qc'][:,0] = nc_file['EWCT_QC'].values
        #nc_fid.variables['observation_qc'][:,1] = nc_file['NSCT_QC'].values
        nc_fid.variables['observation'][:,0,:] = u_obs[:,:]
        nc_fid.variables['observation'][:,1,:] = v_obs[:,:]
        nc_fid.variables['observation_qc'][:,0] = u_obs_qc
        nc_fid.variables['observation_qc'][:,1] = v_obs_qc
        nc_fid.variables['windage'][:,0,:] = u_ws[:,:]
        nc_fid.variables['windage'][:,1,:] = v_ws[:,:]
        nc_fid.variables['windage_qc'][:,0] = u_ws_qc
        nc_fid.variables['windage_qc'][:,1] = v_ws_qc

        # Add global attributs
        nc_fid.institution = 'MERCATOR OCEAN'
        nc_fid.contact = 'cregnier@mercator-ocean.fr'
        #nc_fid.obs_type = self.type_data.get(variable, {}).get('defaults').get('name')
        nc_fid.time_interp = "daily average fields"
        #today = date.today()
        #yyyy = str(today)[0:4]
        #yyyy2 = dateval[0:4]
        #month = str(today)[5:7]
        #month2 = dateval[4:6]
        #day = str(today)[8:10]
        #day2 = dateval[6:8]
        #nc_fid.validity_time = yyyy2+"-"+month2+"-"+day2+" 00:00:00"
        #nc_fid.creation_date = yyyy+"-"+month+"-"+day+" 00:00:00"
        #nc_fid.system = model
        nc_fid.Conventions = 'CF-1.6'
        nc_fid.close()
        nc_file.close()

    def run_list(self,param_dict, daterun):
        liste_tot=[]
        #liste_file=glob(param_dict['dirdata']+'/'+daterun[0:4]+'/*'+daterun+'*.nc')
        year = daterun[0:4]
        month = daterun[4:6]
        liste_file=glob(param_dict['dirdata'] + '/' + year + '/' + month + '/*' + daterun+'*.nc')
        if not liste_file:
            self.log.error("Missing Drifters files {} {}".format(param_dict['dirdata'], daterun))
            return []
        self.log.info('Liste file UV drifters {}'.format(liste_file))
        nb_fcst = len(param_dict['lead_int'])
        nb_fcst2 = len(param_dict['lead_int2'])
        self.log.debug('Number of forecasts {}'.format(nb_fcst))
        for new_file in liste_file :
            try :
                #file_new = 'class4_'+daterun+'_'+param_dict['cl_config_h']+'_'+param_dict['model_type'].lower()+'_currents.nc'
                file_new = 'class4_' + daterun + '_' + param_dict['cl_conf']\
                           +  '_' + param_dict['model_type'].lower() + \
                           '_currents.nc'
                ## Read and Rewrite file
                self.rewrite(liste_file[0], nb_fcst, nb_fcst2, param_dict['dirwork']+file_new)
                self.log.debug('Filename {}'.format(param_dict['dirwork']+file_new))
            except IOError as e:
                self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                raise
        liste_tot.append(param_dict['dirwork']+file_new)
        return liste_tot

class Extract_MDS_L3_SLA(Extractor):
    """ Extract MDS L3 SLA file """

    def __init__(self, log):
        self.log = log

    # Function to extract dates and calculate differences
    def find_largest_date_difference(self, files):
        date_diffs = []
        date_pattern = re.compile(r"_(\d{8})_(\d{8})\.nc")

        for file in files:
            match = date_pattern.search(file)
            if match:
                start_date = datetime.strptime(match.group(1), "%Y%m%d")
                end_date = datetime.strptime(match.group(2), "%Y%m%d")
                day_difference = (end_date - start_date).days
                date_diffs.append((file, day_difference))
                # Find the file with the largest day difference
        largest_diff_file = max(date_diffs, key=lambda x: x[1])
        return largest_diff_file[0]

    def run_list(self, param_dict, dateval):
        liste_tot=[]
        year = dateval[0:4]
        dirdata = param_dict['dirdata']
        list_files = glob(f'{dirdata}{year}/nrt_global_*{dateval}*.nc')
        self.log.debug(f'{list_files=} {dateval}')
        if len(list_files) == 0:
            list_files = glob(f'{dirdata}{year- 1}/nrt_global_*{dateval}*.nc')
            year = str(int(year) - 1)
        list_sat = {os.path.basename(filename).split("_")[2] for filename in list_files}
        self.log.debug(f'{list_sat=}')
        nb_sat = 0
        for sat in list_sat:
            frame = "nrt_global_" + sat + "_phy_l3_1hz_"
            list_files = glob(f'{dirdata}{year}/{frame}{dateval}*.nc')
            if len(list_files) == 0:
                continue
            nb_sat += 1
            choose_file = self.find_largest_date_difference(list_files)
            #latest_file = max(list_files, key=os.path.getctime)
            #self.log.info(f"Satellite {sat} and latest file {latest_file}")
            self.log.info(f"Satellite {sat} and choose file {choose_file}")
            #date_sat = os.path.basename(latest_file).split("_")[6].split(".")[0]
            try:
                filename = os.path.basename(choose_file)
                if not os.path.exists(param_dict['dirwork']+filename):
                    shutil.copy(choose_file, param_dict['dirwork']+filename)
                liste_tot.append(param_dict['dirwork']+filename)
            except IOError as e:
                print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                sys.exit(1)
        self.log.debug(f"Nb Sats {nb_sat}")

        return liste_tot

class Extract_GODAE(Extractor):

    """ Extract GODAE file """

    def __init__(self, log):
        self.log = log

    def is_netcdf4_classic(self, file_path):
        try:
            output = subprocess.check_output(['ncdump', '-k', file_path], text=True)
            return 'classic' in output
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while checking the file type: {e}")
            return False
    def list(self, daterun, dirdata, typedata):
        if typedata != "SLA" and  typedata != "SST" and typedata != "profile" \
            and typedata !=  "aice" and typedata !=  "AMSR" and  typedata !=  "current" :
            self.log.error("Data %s not known !!! " %(typedata))
            sys.exit(1)
        #valeurj=date1
        liste_tot=[]
        if typedata ==  "AMSR" :
            liste_file=glob(dirdata+'/*'+daterun+'*.nc')
        else :
            try:
                liste_file = [glob(dirdata+'/class4*'+daterun+'*'+typedata+'*.nc')[0]]
            except IndexError:
               self.log.error(f'Missing CLASS4 {daterun} {typedata} file')
               liste_file = []
        if liste_file:
            return liste_file, False
        else:
            self.log.error(f'file is missing for {daterun}')
            return liste_file, True
    
    def run(self,daterun,dirdata,dirtmp,typedata,template,nb_fcst):

        if typedata != "SLA" and  typedata != "SST" and typedata != "profile" \
            and typedata !=  "aice" and typedata !=  "AMSR" and  typedata !=  "current" :
            self.log.error("Data %s not known !!! " %(typedata))
            sys.exit(1)
        #valeurj=date1
        liste_tot=[]
        # Nco constructor
        nco = Nco()
        #while  valeurj.__ge__(date1) and valeurj.__le__(date2) :
        #    dateval=SyDate.__str__(valeurj)
        self.log.info('Day  %s ' % (daterun))
        if typedata ==  "AMSR" :
            liste_file=glob(dirdata+'/*'+daterun+'*.nc')
        else :
            try:
                liste_file = [glob(dirdata+'/class4*'+daterun+'*'+typedata+'*.nc')[0]]
            except IndexError:
               self.log.error(f'Missing CLASS4 {daterun} {typedata} file')
               liste_file = []
        if liste_file:
            self.log.info(f'Liste file {liste_file}')
            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    if typedata ==  "AMSR" :
                        filename=re.sub('GL_h5',template,filename)
                    elif typedata ==  "current":
                        filename=filename
                    else:
                        template1 = filename.split('_')[0:]
                        if len(template1) > 5:
                            template_tmp = template1[2]+'_'+template1[3]+'_'+template1[4]
                        else:
                            template_tmp = template1[2]+'_'+template1[3]
                        filename=re.sub(template_tmp, template, filename)
                    # Copy extracted value of GODAE file with nco module
                    if typedata != "aice" and typedata != 'current':
                        # If it doesn't exist, create it
                        if not os.path.exists(dirtmp):
                            os.makedirs(dirtmp)
                        uncompressed_filename = re.sub(r'(\.nc)$', '_uncompressed.nc', filename)
                        uncompressed_file_path = os.path.join(dirtmp, uncompressed_filename)
                        # Define nco options
                        options = []
                        options.extend(['-h', '-O', '-a', 'numfcsts'])
                        # Change numfcts dim
                        # Test input file
                        if self.is_netcdf4_classic(new_file):
                            self.log.info(f"Write output file netcdf4 classical {dirtmp}{filename}")
                            try:
                                subprocess.run([
                                    'nccopy', '-d0',
                                    new_file,
                                    uncompressed_file_path
                                ], check=True)
                            except subprocess.CalledProcessError as e:
                                self.log.error(f"An error occurred during decompression: {e}")
                                raise
                            try:
                                nco.ncwa(input=uncompressed_file_path, output=os.path.join(dirtmp, filename), options=options)
                            except Exception as e:
                                self.log.error(f"An error occurred: {e}")
                                raise
                            finally:
                                # Remove the uncompressed file
                                if os.path.exists(uncompressed_file_path):
                                    os.remove(uncompressed_file_path)
                        else:
                            self.log.info(f"Write output file netcdf4 {dirtmp}{filename}")
                            nco.ncwa(input=new_file, output=dirtmp+filename, fortran=True ,options=options)
                        nc = netCDF4.Dataset(dirtmp+filename, 'r')
                        variables = nc.variables.keys()
                        if 'modeljuld' in variables and nb_fcst>0:
                           command ="ncap2 -h -O -s "
                           command2 = "\'defdim(\"numfcsts\","+str(nb_fcst)+");leadtime_new[$numfcsts]=leadtime;modeljuld_new[$numfcsts]=modeljuld\' "
                           os.system(command+command2+dirtmp+filename+" "+dirtmp+filename)
                        else:
                            self.log.info('no modeljuld !!')
                        options = []
                        if typedata == "SLA" :
                            #options.extend(['-h','-O','-xv','best_estimate,forecast,persistence,altimeter_bias,mdt_reference,leadtime,modeljuld'])
                            options.extend(['-h','-O','-xv','best_estimate,forecast,persistence,altimeter_bias,mdt_reference,leadtime,modeljuld,varname'])
                        else :
                            options.extend(['-h','-O','-xv','best_estimate,forecast,persistence,leadtime,modeljuld'])
                        nco.ncks(input=dirtmp+filename,output=dirtmp+filename, fortran=True ,options=options)
                        # Rename
                        options = []
                        if 'modeljuld' in variables and nb_fcst>0:
                            options.extend(['-h','-O','-v','modeljuld_new,modeljuld'])
                            nco.ncrename(input=dirtmp+filename,output=dirtmp+filename, options=options)
                            options = []
                            options.extend(['-h','-O','-v','leadtime_new,leadtime'])
                            nco.ncrename(input=dirtmp+filename,output=dirtmp+filename,options=options)
                        self.log.debug("add new dim ok {}".format(dirtmp+filename))
                    elif typedata == "current":
                        options = []
                        shutil.copy(dirdata+filename,dirtmp)
                    else:
                        #typedata == "aice" :
                        options = []
                        options.extend(['-h','-O','-xv','best_estimate,forecast,persistence'])
                        nco.ncks(input=new_file,output=dirtmp+filename,options=options)
                    if typedata == "AMSR" :
                        # add missing varname
                        # Solution2 : add dimensions and variable varname
                        cl_fileIn=dirtmp+filename
                        nc   = netCDF4.Dataset(cl_fileIn,'a')
                        nc.createDimension('string_length8',8)
                        # for AMSR
                        nc.createDimension('numvars',1)
                        nc.createVariable('varname','c',('numvars','string_length8'))
                        nc.variables['varname'][:]='ileadfra'
                        nc.close
                    self.log.debug('Fichier extracted %s ' % (dirtmp+filename)) 
                    liste_tot.append(dirtmp+filename)
                    # Design output File
                    # Design_file('GODAE').run(dirtmp+filename)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    raise
            #valeurj=valeurj.goforward(1)
        else:
            liste_tot = []
        return liste_tot



class Extract_ARMOR(Extractor):

    """ Extract ARMOR file """

    def __init__(self, log):
        self.log = log

    def run(self,date1,date2,dirdata,dirtmp,typedata,template,prefix):
        valeurj=date1
        liste_tot=[]
        while  valeurj.__ge__(date1) and valeurj.__le__(date2) :
            dateval=SyDate.__str__(valeurj)
            self.log.info('Day %s ' % (dateval)) 
            frame=dirdata+prefix+'*'+dateval.strip()+'*.nc*'
            liste_file=glob(frame)
            # liste_file=glob(dirdata+'/*[TS]*'+dateval+'*.nc')
            self.log.debug(f'Frame {dirdata}{prefix}*{dateval.strip()}*.nc*')
            self.log.info(f'Liste file ARMOR {liste_file}') 
            for new_file in liste_file :
                try :
                    filename=os.path.basename(new_file)
                    shutil.copy(new_file,dirtmp)
                    if filename[-2:] == 'gz' :
                        self.log.info('Dezip')
                        filename2=filename
                        dezip=ZipManipulator(dirtmp)
                        fichier=dirtmp+filename2
                        dezip.gunzip(fichier)
                        filename=fichier.split('.gz')[:][0]
                        # tar = tarfile.open(dirtmp+filename2, "r:gz")
                        # tar.extractall()
                        # tar.close()
                        liste_tot.append(filename)
                    else :
                        liste_tot.append(dirtmp+filename)
                    self.log.info("append : %s " %(filename))
                except IOError as e:
                    print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj=valeurj.goforward(1)

        return liste_tot

    def run_list(self,date1,date2,dirdata):
        liste_tot=[]
        valeurj=date1
        while  valeurj.__ge__(date1) and valeurj.__le__(date2) :
            dateval=SyDate.__str__(valeurj)
            #print dirdata+'/*'+dateval+'*.nc'
            liste_file=glob(dirdata+'/*'+dateval+'*.nc')
            for file in liste_file :
                try :
                    liste_tot.append(file)
                except IOError as e:
                    print ("I/O error({0}): {1}".format(e.errno, e.strerror))
                    sys.exit(1)
            valeurj=valeurj.goforward(1)
        return liste_tot


class Extract_cora3(Extractor) :

    """ Extract Cora3 file """

    def __init__(self, log):
        self.PREFIX='cora3.4_merc'
        self.log = log

    def run(self,date1,date2,dirdata,dirtmp,typedata,template):
        self.log.info('Extraction of Cora 3 dataset')
        presmonth=SyDate.__str__(date2)[:6]
        backmonth=SyDate.__str__(date1)[:6]
        liste_month=[]
        liste_file=[]
        if presmonth == backmonth : 
            liste_file.append(file)
            liste_month.append(presmonth)
        else :
            file1=self.PREFIX+"_"+presmonth+".tgz"
            liste_file.append(file1)
            file2=self.PREFIX+"_"+backmonth+".tgz"
            liste_file.append(file2)
            liste_month.append(presmonth)
            liste_month.append(backmonth)
        # Extract and copy file
        for file in liste_file:
            file_path=dirdata+file
            if os.path.isfile(file_path)  :
                try :
                    shutil.copy(file_path,dirtmp)
                    os.chdir(dirtmp)
                    tar = tarfile.open(file)
                    tar.extractall()
                    tar.close()
                    os.remove(file)
                except  IOError as e :
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    raise e
            else :
                self.log.error('file not known %s ' %(file_path))
                sys.exit(1)
        liste_tot=[]
        for month in liste_month : 
            valeurj=date1
            year=SyDate.__str__(valeurj)[:4]
            while  valeurj.__ge__(date1) and valeurj.__le__(date2) :
                dateval=SyDate.__str__(valeurj)
                self.log.info('Day  %s ' % (dateval)) 
                liste_file=glob(dirtmp+'/'+year+'/*'+dateval+'*.nc')
                self.log.info('Day list %s ' % (liste_file)) 
                for new_file in liste_file :
                    try :
                        # os.rename(new_file,dirtmp)
                        # shutil.move(dirtmp,new_file)
                        shutil.copy(new_file,dirtmp)
                        liste_tot.append(new_file)
                    except IOError as e:
                        self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                        raise e
                valeurj=valeurj.goforward(1)
            # Remove directory 
            shutil.rmtree(dirtmp+year)
        # Write log

        return liste_tot


class Extract_cora4(Extractor) :

    """ Extract Cora4 file """

    def __init__(self,compressi,log):
        """

        Constructor cora 4

        """
        self.PREFIX='cora4.0_merc'
        self.compress=compress
        self.log = log

    def run(self,date1,date2,dirdata,dirtmp,typedata,templat,prefix):
        self.log.info('Extractor cora4 : '+self.compress)
        presyear=SyDate.__str__(date2)[:4]
        backyear=SyDate.__str__(date1)[:4]
        liste_file=[]
        liste_year=[]
        if presyear == backyear : 
            file=prefix+"_"+presyear+self.compress
            liste_file.append(file)
            liste_year.append(presyear)
        else :
            file1=prefix+"_"+presyear+self.compress
            liste_file.append(file1)
            file2=prefix+"_"+backyear+self.compress
            liste_file.append(file2)
            liste_year.append(presyear)
            liste_year.append(backyear)
        # Extract and copy file
        for file in liste_file :
            file_path=dirdata+file
            # Test the existence of the year
            if not os.path.isdir(dirtmp+liste_year[0]):
                print ("Directory doesn t exist extract")
                if os.path.isfile(file_path)  :
                    try :
                        shutil.copy(file_path,dirtmp)
                        os.chdir(dirtmp)
                        tar = tarfile.open(file)
                        tar.extractall()
                        tar.close()
                        os.remove(file)
                    except  IOError as e :
                        self.log.error ("I/O error({0}): {1}".format(e.errno, e.strerror))
                        raise e
                        # print 'Problem with copy or extraction'
                else :
                    self.log.error('file not known %s ' %(file_path))
                    sys.exit(1)
            else :
                print ("Directory already exist")
        # Remove file
        liste_tot=[]
        for year in liste_year : 
            valeurj=date1
            while  valeurj.__ge__(date1) and valeurj.__le__(date2) :
                dateval=SyDate.__str__(valeurj)
                self.log.info('Day  %s ' % (dateval)) 
                liste_file=glob(dirtmp+'/'+year+'/*'+dateval+'*.nc')
                for file in liste_file :
                    try :
                        self.log.info('Fichier  %s ' % (file))
                        shutil.move(file,dirtmp)
                        filename=os.path.basename(file)
                        liste_tot.append(dirtmp+filename)
                    except IOError as e:
                        self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                        raise e
                valeurj=valeurj.goforward(1)
            # Remove directory 
            # shutil.rmtree(dirtmp+year)
        # Write log
        ok_file=dirtmp+"Extract_cora4_"+str(presyear)+"_"+str(backyear)+"_OK"
        return liste_tot

    def run_list(self,date1,date2,dirdata):
        liste_tot=[]
        valeurj=date1
        while  valeurj.__ge__(date1) and valeurj.__le__(date2) :
            dateval=SyDate.__str__(valeurj)
            liste_file=glob(dirdata+'/*'+dateval+'*.nc')
            for file in liste_file :
                try :
                    liste_tot.append(file)
                except IOError as e:
                    self.log.error("I/O error({0}): {1}".format(e.errno, e.strerror))
                    raise e
            valeurj=valeurj.goforward(1)
        return liste_tot
