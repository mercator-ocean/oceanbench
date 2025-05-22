from __future__ import generators
import os, sys, string, shutil, re
from glob import glob
import numpy as np
import json
from .Loader import Loader
from .nco.nco import Nco
from .QCcontrol import QCcontroller
import logging
import shelve
from .Writer import Writer
import itertools
from .Extractor import Extractor
from .sydate import SyDate
from collections import OrderedDict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .FileHandler import FileHandler
import yaml

#######################################################################################
## C.REGNIER Juillet 2014
## Class Selector to select profile : the selection depends on QC value and zoom
##              possibility to store the positions in a file
#######################################################################################

class Selector(object):

    """ Factory Class to run different objets depending on input type """

    def __init__(self, data_format, data_type, log,
                 lon_min=-180, lon_max=180, lat_min=-90,
                 lat_max=90, **kwargs):
        self.data_format = data_format
        self.data_type = data_type
        self.log = log
        self.lonmin = lon_min
        self.lonmax = lon_max
        self.latmin = lat_min
        self.latmax = lat_max

    def factory(self, type):
        #return eval(type + "()")
        if type == "CORIOLIS" or type == "GODAE" or \
           type == "CORIOLIS_RAW" or type == "GLOBCOLOUR":
            return SelectCorio(self.data_format, self.data_type, self.lonmin,\
                               self.lonmax, self.latmin, self.latmax)
        elif type == "Microwat":
            return Select(self.data_format, self.data_type)
        elif type == "GLOBAL_INPUT_DATA":
            return SelectListData(self.log)
        elif type == "GLOBAL_INPUT_MODEL_DATA":
            return SelectListModelData(self.log)
        else:
            print("Type not known %s " %(type))

    def get_template_alleges_new(self, dateval, param_dict):

        """ Get the template for new model inputs  """

        jdatemin = SyDate(dateval)
        type_run = param_dict['type_run']


    def find_next_month_year(self, dateval):
        # Convert the input string to a datetime object
        date_object = datetime.strptime(dateval, '%Y%m%d')
        year = date_object.year
        month = date_object.month
        # Calculate the next month and year using relativedelta
        next_month_date = date_object +  relativedelta(months=1)
        next_month_year = next_month_date.year
        next_month = next_month_date.month
        return year, f"{month:02d}", next_month_year, f"{next_month:02d}"


    def get_template_alleges(self, dateval, param_dict):

        """ Get the template for model inputs  """

        jdatemin = SyDate(dateval)
        type_run = param_dict['type_run']
        ## Find next daterun on Wednesday
        daytogo = 2-jdatemin.weekday()
        if daytogo > 0:
            first_date = jdatemin.goforward(daytogo)
        else:
            if type_run == 'RTR' or type_run == 'REA':
                nbdate = 7
            elif type_run == 'HINDCAST':
                nbdate = 14
            daytogo2 = nbdate+daytogo
            first_date = jdatemin.goforward(daytogo2)
        daterun = first_date
        if type_run == 'RTR' or type_run == 'REA':
            date1 = daterun.gobackward(7)
            date2 = daterun
        elif type_run == 'HINDCAST':
            date1 = daterun.gobackward(14)
            date2 = daterun.gobackward(7)
        if param_dict['data_type'] == 'profile':
            depthval='50.0'
            typeval='.end.'
            type_out='END'
            dateval=daterun
        elif param_dict['data_type'] == 'SLA':
            depthval='01.0'
            typeval='.run.'
            type_out='RUN'
        elif param_dict['data_type'] == 'SST':
            depthval='01.0'
            typeval='.run.'
            type_out='RUN'
        elif param_dict['data_type'] == 'aice':
            depthval='01.0'
            typeval='.run.'
            type_out='RUN'
        else:
            depthval='01.0'
            type_out='RUN'

        template = []
        for i, val in enumerate(param_dict['gridtype_light']):
            template.append(param_dict['dirmodel_light']+'ALLEGES_'+type_run.lower()+\
                            '_'+str(date1)+'_'+str(date2)+'_R'+str(daterun)+ \
                            '/'+type_out+'/'+val+'.d'+\
                            str(dateval)+typeval+str(depthval))
        self.log.debug("Template allege : {} ".format(template))
        return template

class SelectListModelData(Selector):

    """
        Class to select the list of input Model files
    """
    def __init__(self, log):
        super(Selector)
        self.log = log
        self.list_ice=['sit-legos','snow-legos','sivolu','snvolu','siconc','kaku',
                       'kaku_unc','kakuyear','kakuyear_unc','kakuyearsh','kakuyearsh_unc',
                       'kakush','kakush_unc']
        self.list_daily = ['1dAV', '1d-m']
        module_dir = os.path.dirname(__file__)
        self.config_path = "directory_patterns.yaml" 
        path = os.path.join(module_dir, self.config_path)
        with open(path, "r") as file:
            self.config = yaml.safe_load(file)

    def set_gridtype(self, param_dict):
        self.data_type = param_dict['data_type']
        if self.data_type.lower() == "sst" or self.data_type.lower() == "ssthf":
            param_dict['gridtype'] =  [param_dict['gridtemp']]
            param_dict['gridtype_clim'] = ["TEMP"]
            param_dict['gridtype_light'] = ["TN.B"]
            param_dict['varname_clim'] = ["temperature"]
        elif self.data_type.lower() == "sss":
            param_dict['gridtype'] =  [param_dict['gridsal']]
            param_dict['gridtype_clim'] = ["SAL"]
            param_dict['gridtype_light'] = ["SN.B"]
            param_dict['varname_clim'] = ["salinity"]
        elif self.data_type.lower() == "sla" or self.data_type.lower() == "sla-l3":
            param_dict['gridtype'] = [param_dict['grid2DT']]
            param_dict['gridtype_light'] = ["SLA.B"]
        elif self.data_type.lower() == "chloro":
            param_dict['gridtype'] = ["chl"]
        elif self.data_type.lower() == "nitrate":
            param_dict['gridtype'] = ["no3"]
        elif self.data_type.lower() == "ph":
            param_dict['gridtype'] = ["ph"]
        elif self.data_type.lower() == "aice":
            param_dict['gridtype'] = [param_dict['gridice']]
        elif self.data_type.lower() == "drifter_filtr" or self.data_type.lower() == "drifter":
            param_dict['gridtype'] = [param_dict['gridu'], param_dict['gridv']]
            param_dict['gridtype_clim'] = ["current"]
        elif self.data_type.lower() == "profile" or self.data_type.lower() == "cora5.2":
            list_grid = []
            list_grid_clim = []
            list_grid_light = []
            self.log.debug(f"Inside Profile {param_dict['ll_TEMP']}")
            if param_dict['ll_TEMP']:
                list_grid.append(param_dict['gridtemp'])
                list_grid_clim.append("TEMP")
                list_grid_light.append("TN.B")
            if param_dict['ll_PSAL']:
                list_grid.append(param_dict['gridsal'])
                list_grid_clim.append("PSAL")
                list_grid_light.append("SN.B")
            param_dict['gridtype'] = list_grid
            param_dict['gridtype_clim'] = list_grid_clim
            param_dict['gridtype_light'] = list_grid_light
        elif self.data_type.lower() == "current":
            #param_dict['gridtype'] = ["gridU", "gridV"]
            param_dict['gridtype'] = ["3DU-uo", "3DV-vo"]
        elif self.data_type.lower() in self.list_ice:
            param_dict['gridtype'] = param_dict['gridice']
        self.typemod = param_dict['cl_typemod'].lower()
        if self.typemod == "pgn":
            self.dirmodel = param_dict['dirmodel_pgn']
        elif self.typemod == "pgs":
            self.dirmodel = param_dict['dirmodel_pgs']
            param_dict['suffix'] = "0.25deg_P1D-m"
        elif self.typemod == "pgs_ai":
            self.dirmodel = param_dict['dirmodel_pgs']
        elif self.typemod == "pgncg":
            self.dirmodel = param_dict['dirmodel_pgncg']
        elif self.typemod == "allege" or "allege_new" \
             or self.typemod == "allege_now":
            self.dirmodel = param_dict['dirmodel_light']
        elif self.typemod == "2ease":
            self.dirmodel = param_dict['dirmodel_pgn']
        else:
            raise ('typemod not known')
            sys.exit(1)
        return param_dict

    def set_leadtime(self, time, param_dict, lead_new):
        lead_fin = []
        #lead_new = param_dict['lead_int']
        lead_new2 = param_dict['lead_int2']
        cl_config_use = param_dict['cl_config']
        ll_single = False
        self.log.debug(f"Time {time=}")
        self.log.debug(f"-----------------------")
        if  time == 'forecast':
            lead_new = [time[0]+time[4]+time[6:8]+element for element in lead_new ]
            lead_fin = lead_new
        elif time == 'persistence':
            self.log.info(param_dict['lead_time'])
            #if filter(lambda x: 'best_weekly' in x, param_dict['lead_time']):
            if 'best_weekly' in param_dict['lead_time']:
                lead_fin = ['pers']
            else:
                if len(lead_new) > 1 :
                    lead_tmp = ['nwct']
                    lead_new = [time[:4]+element for element in lead_new[1:]]
                    lead_tmp.extend(lead_new)
                    lead_fin = lead_tmp
                else:
                    if lead_new == '0':
                        lead_fin = ['nwct']
                    else:
                        lead_fin = [time[:4]+lead_new]
        elif time == 'best_estimate':
            if param_dict['cl_config'][0:6] == 'BIOMER':
                #lead_fin = [time[0]+time[4]+time[6:8]]
                lead_fin = ['hdct']
                self.log.debug("Config biomer")
            else:
                cl_config_use = re.sub('Q', '', param_dict['cl_config'])
                lead_fin = ['hdct']
                self.log.debug('Lead fin %s ' % (lead_fin))
        elif time == 'best':
            cl_config_use = re.sub('Q', '', param_dict['cl_config'])
            lead_fin = ['hdct_all']
            self.log.debug('Lead fin %s ' % (lead_fin))
            ll_single = True
        elif time == 'best_weekly':
            lead_fin = ['hdct', 'nwct']
        elif time == 'rtr':
            lead_fin = ['rtr']
        elif time == 'climatology':
            lead_fin = ['clim']
        elif time == 'bathymetrie':
            lead_fin = ['bathy']
        elif time == 'clim_lev13_A5B2':
            lead_fin = ['clim']
            type_clim = "WOA13_A5B2"
            dirclim = param_dict.get('climato','DIRLEV')
            type_clim = "WOA13_A5B2"
        elif time == 'smoc':
            lead_fin = ['smoc']
        elif time == 'smoc_fcst':
            lead_fin = ['fcst'+element for element in lead_new2 ]
            self.dirmodel_new = self.dirmodel+'DAILY/SMOC/'
        elif time == 'tides':
            lead_fin = ['tides']
        elif time == 'stokes':
            lead_fin = ['stokes']
        elif time == 'stokes_fcst':
            lead_fin = ['fcst'+element for element in lead_new2 ]
            self.dirmodel_new = self.dirmodel+'DAILY/SMOC/'
        else:
            self.log.error('In sorter Lead time %s not expected ' %(time))
            sys.exit(1)
        self.config_d = re.sub("V", "QV", param_dict['cl_config'])
        self.config_d = re.sub("V", "QV", param_dict['cl_conf'])
        self.modele = cl_config_use
        param_dict['modele'] = self.modele
        self.leadtime = lead_fin
        return param_dict

  #  def find_file(self, directory, modele, dateval, gridname_list):
  #      # List of possible date formats
  #      date_formats = [
  #          dateval,  # Assuming it's already formatted correctly
  #          f"{dateval[:4]}-{dateval[4:6]}-{dateval[6:]}",  # Convert %Y%m%d to %Y-%m-%d
  #          f"{dateval[:2]}{dateval[2:4]}{dateval[4:]}",  # Convert %Y%m%d to %YYMMDD
  #          f"{dateval[:4]}_{dateval[4:6]}_{dateval[6:]}"  # Convert to another variant if needed
  #      ]

  #      # Results to store all matching files
  #      all_files = []
  #      # Iterate over gridtypes and apply the date format logic
  #      for i, gridname in enumerate(gridname_list):
  #          # Iterate over different date formats and try to find matching files
  #          for dateval_formatted in date_formats:
  #              file_pattern = f"{directory}/{modele}_{dateval_formatted}_{gridname}.nc"
  #              file_tmp = glob(file_pattern)
  #              # If files are found, store them in the list
  #              if file_tmp:
  #                  all_files.extend(file_tmp)  # Add found files to the results
  #      # Return all found files or an empty list if none are found
  #      return all_files if all_files else None


    def get_directory(self, directory, dateval, param_dict):
        datev = SyDate(str(dateval))
        year = str(datev.year)
        month = datev.month
        year, month, next_year, next_month = self.find_next_month_year(dateval)

        # Initialize dirmodel_new and dirmodel_new_next
        dirmodel_new = self.dirmodel
        dirmodel_new_next = dirmodel_new

        # Determine the directory structure using YAML
        config = param_dict['cl_conf']
        self.config_d = re.sub("V", "QV", config)
        if directory == 'hdct':
            daily_pref = param_dict.get('daily_pref', None)
            if daily_pref == '7dAV':
                dirmodel_new = self.config['directory_patterns']['hdct']['7dAV'].format(dirmodel=self.dirmodel)
            else:
                if self.data_type.lower() in self.config['directory_patterns']['hdct']['default']:
                    dirmodel_new = self.config['directory_patterns']['hdct']['default'][self.data_type.lower()].format(dirmodel=self.dirmodel)
                else:
                    run_type = param_dict['type_run']
                    print(run_type)
                    print(self.config['directory_patterns']['hdct']['default'])
                    if run_type in self.config['directory_patterns']['hdct']['default']:
                        if run_type == 'REA':
                            print(self.config['directory_patterns']['hdct']['default'][run_type])
                            dirmodel_new = self.config['directory_patterns']['hdct']['default'][run_type].format(dirmodel=self.dirmodel, modele=config)
                        else:
                            dirmodel_new = self.config['directory_patterns']['hdct']['default'][run_type].format(dirmodel=self.dirmodel, year=year, month=month)
                    else:
                        if param_dict.get('naming') == 'new':
                            config_slice = config[0:5]
                            dirmodel_new = self.config['directory_patterns']['hdct']['default']['config_d'].format(
                                    dirmodel=self.dirmodel,
                                    config=config_slice,
                                    config_lower=config.lower())
                        else:
                            config_slice = self.modele[0:4]
                            dirmodel_new = self.config['directory_patterns']['hdct']['default']['PSY4'].format(
                                    dirmodel=self.dirmodel,
                                    modele=config_slice,
                                    modele_lower=self.modele.lower())
        elif directory == 'hdct_all':
            dirmodel_new = self.config['directory_patterns']['hdct_all'].get(self.typemod, self.config['directory_patterns']['hdct_all']['default']).format(dirmodel=self.dirmodel)
        elif directory == 'fcst':
            daily_pref = param_dict.get('daily_pref', None)
            if daily_pref == '7dAV':
                dirmodel_new = self.config['directory_patterns']['fcst']['7dAV'].format(dirmodel=self.dirmodel)
            else:
                if param_dict.get('naming') == 'new':
                    config_slice = config[0:5]
                    dirmodel_new = self.config['directory_patterns']['fcst']['default']['GLO12'].format(
                                    dirmodel=self.dirmodel,
                                    config=config_slice,
                                    config_lower=config.lower())
                elif param_dict.get('naming') == 'old':
                    config_slice = self.modele[0:4]
                    dirmodel_new = self.config['directory_patterns']['fcst']['default']['PSY4'].format(
                                    dirmodel=self.dirmodel,
                                    modele=config_slice,
                                    modele_lower=self.modele.lower())
                elif param_dict.get('naming') == 'bio':
                    config_slice = self.modele[0:4]
                    dirmodel_new = self.config['directory_patterns']['fcst']['default']['BIO4'].format(
                                    dirmodel=self.dirmodel)
        elif directory == 'clim':
            dirmodel_new = self.config['directory_patterns']['clim'].format(dirclim=param_dict['dirclim'])

        elif directory in ["smoc", "tides", "stokes"]:
            dirmodel_new = self.config['directory_patterns']['smoc'].format(dirmodel=self.dirmodel)

        elif directory in ['smoc_fcst', 'stokes_fcst']:
            dirmodel_new = self.config['directory_patterns']['smoc_fcst'].format(dirmodel=self.dirmodel)

        elif directory == 'bathy':
            dirmodel_new = self.config['directory_patterns']['bathy'].format(bathy_file=param_dict['bathy_file'])

        else:
            if param_dict.get('naming') == 'new':
                config_slice = f"{config[0:5]}Q"
                dirmodel_new = self.config['directory_patterns']['fcst']['default']['GLO12'].format(
                                dirmodel=self.dirmodel,
                                config=config_slice,
                                config_lower=self.config_d.lower())
            elif param_dict.get('naming') == 'old':
                config_slice = f"{self.modele[0:4]}Q"
                dirmodel_new = self.config['directory_patterns']['fcst']['default']['PSY4'].format(
                                dirmodel=self.dirmodel,
                                modele=config_slice,
                                modele_lower=self.modele.lower())
            else:
                dirmodel_new = self.dirmodel

        # Continue with the logic to return or log the directory paths
        return dirmodel_new, dirmodel_new_next



    def run(self, param_dict, dateval, leadtime, lead_int):
        list = OrderedDict()
        self.log.debug(f'Param dict {param_dict}')
        param_dict = self.set_gridtype(param_dict)
        self.data_type = param_dict['data_type']
        self.type_run = param_dict['type_run']
        self.log.debug(f'Data type {self.data_type}')
        MONTH = str(dateval)[4:6]
        YEAR = dateval[:4]
        DAY = dateval[6:]
        # Format the date with dashes
        dateval_formatted = f'{YEAR}-{MONTH}-{DAY}'
        self.log.debug(f"Grid type {param_dict['gridtype']} {self.dirmodel=}")
        # loop on lead time
        new_naming = False
        if  param_dict['naming'] == 'new':
            new_naming = True
        #for time in param_dict['lead_time']:
        list_tmp = []
        self.log.debug(f'Lead Time {leadtime}')
        #lead_new = param_dict['lead_int']
        #lead_new2 = param_dict['lead_int2']
        cl_config_use = param_dict['cl_config']
        config = param_dict['cl_conf']
        ll_single = False
        param_dict = self.set_leadtime(leadtime, param_dict, lead_int)
        ## Find and Load the good file
        ind = 0
        ll_miss = False
        for directory in self.leadtime:
            self.log.debug("directory {}".format(directory))
            ll_miss_rep = False
            datev = SyDate(str(dateval))
            year = str(datev.year)
            month_up = datev.month + 1
            # Access mapping from YAML
            dirmodel_new, dirmodel_new_next = self.get_directory(directory, dateval, param_dict)
            self.log.debug(f"Get directory {dirmodel_new} {dirmodel_new_next}")
            self.log.debug(f"{self.config} {dirmodel_new} {self.modele} {self.type_run}")
            # Handle different type of file
            file_handler = FileHandler(self.config,
                                       dirmodel_new,
                                       self.modele,
                                       self.typemod,
                                       self.data_type,
                                       directory,
                                       self.type_run,
                                       dateval,
                                       param_dict,
                                       self.log)
            list_file = file_handler.load_files()
            self.log.debug('directory %s' % (dirmodel_new))

            if not list_file:
                self.log.error("Repertory %s" % (dirmodel_new))
                self.log.error(f"File missing for {directory} {dateval}")
                ll_miss_rep = True
                continue
                #sys.exit(1)
            self.log.debug('dirmodel  %s ' % (self.dirmodel))
            if not ll_miss_rep : self.log.debug('directory and file  %s %s ' % (directory, list_file[0]))
            list[directory] = list_file
        if not list:
            self.log.error("No repertories")
            ll_miss = True
        else:
            self.log.debug(f'List input files {list}')
        self.log.debug(f'List input files {list}')
        return list, ll_miss

class SelectListData(Selector):

    """
        Class to select the list of input files
    """

    def __init__(self, log):
        super(Selector)
        self.log = log

    def run(self, param_dict, daterun):
        """
         Create list of input data for a defined daterun
         Parameters
        -----------------------
        param_dict : dictionnary of parameters
        daterun : datevalue
        Returns
        -----------------------
        list_files : 
        """
        ll_miss = False
        if param_dict['zoom'] or param_dict['data_origin'] == 'GODAE':
            self.log.debug('========================================================')
            self.log.debug('              DATE Selection and extraction zoom or GODAE ')
            self.log.debug('========================================================')
            self.log.debug(f"Prefix {param_dict['prefix_data']}")
            #list_files = Extractor().factory(param_dict['prefix_data']).run(param_dict['date1'],\
            #param_dict['date2'],param_dict['dirdata'],param_dict['dirwork'],param_dict['data_type'],param_dict['template'])
            list_files = Extractor(self.log).factory(param_dict['prefix_data']).run(daterun,param_dict['dirdata'],\
                         param_dict['dirwork'],param_dict['data_type'],param_dict['template'], len(param_dict['lead_int']))
        elif param_dict['do_decomp']:
            self.log.debug('do decomp {} '.format(param_dict['prefix_data']))
            list_files = Extractor(self.log).factory(param_dict['prefix_data']).run(
                param_dict['date1'], param_dict['date2'], param_dict['dirdata'],\
                param_dict['dirwork'], param_dict['data_type'], param_dict['template'], param_dict['prefix_data'])
        elif param_dict['over']:
            self.log.debug('inside over {} '.format(param_dict['prefix_data']))
            list_files = Extractor(self.log).factory(param_dict['prefix_data']).run_over(
                param_dict['date1'], param_dict['date2'], param_dict['diroutput'],\
                param_dict['dirwork'], param_dict['data_type'], param_dict['template'], param_dict['prefix_data'])
        else:
            self.log.debug('========================================================')
            self.log.debug('              DATE Selection and extraction  %s         ' %(param_dict['prefix_data']))
            self.log.debug(f" Prefix Data {param_dict['prefix_data']}")
            self.log.debug('========================================================')
            list_files = Extractor(self.log).factory(param_dict['prefix_data']).run_list(param_dict, daterun)
        if not list_files:
            self.log.error(f"Missing Obs files {param_dict['prefix_data']}"
                           f" {param_dict['dirdata']} {param_dict['dirwork']}"
                           f"{param_dict['prefix_data']}")
            ll_miss = True
        else:
            self.log.debug('List Files  %s ' % (list_files))
        return list_files, ll_miss

class Select(Selector):
    def __init__(self, data_format, data_type, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.data_format = data_format
        self.data_type = data_type
    def store_pos(self, tab_pos, type_f, file_db):
        logging.basicConfig(format='%(asctime)s %(message)s')
        log = logging.getLogger("store_pos")
        ## Level of logging output
        log.setLevel(logging.INFO)
        # Open a shelve file if not exist
        if not os.path.isfile(file_db):
            log.debug("Create persistent file")
            shelf = shelve.open(file_db, 'c')
        else:
            log.debug("File exist %s " %(file_db))
            shelf = shelve.open(file_db)
        # Add position values in the serialized dictionnary
        try:
            ll_exist = shelf[type_f]
        except:
            log.debug("Create %s" %(type_f))
            shelf[type_f] = tab_pos
            ll_exist = []
        if len(ll_exist) > 0:
            try:
                log.debug("Tab_fin")
                tab_fin = shelf[type_f]
                log.debug("concatenation")
                tab_fin = np.concatenate((tab_fin,tab_pos))
                shelf[type_f] = tab_fin
            except:
                print("Pb with concatenation store_pos")
                sys.exit(1)
        shelf.close()

class SelectCorio(Selector):

    """ Class for Coriolis files """

    def __init__(self, data_format, data_type, lon_min, lon_max, lat_min, lat_max, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.data_format = data_format
        self.data_type = data_type
        self.lonmin = lon_min
        self.lonmax = lon_max
        self.latmin = lat_min
        self.latmax = lat_max
        if data_format == "NC_CORIO":
            self.dimprof = 'N_PROF'
        elif data_format == "NC_GODAE":
            self.dimprof = 'numobs'

    def run(self, fichier, zoom, do_qc, TYPEDATA, qc_level):
        logging.basicConfig(format='%(asctime)s %(message)s')
        log = logging.getLogger(" Run Selector")
        ## Level of logging output
        #log.setLevel(logging.INFO)
        log.setLevel(logging.ERROR)
        # Read position of profiles
        position = Loader.factory(self.data_format).read_pos(fichier)
        nb_profs = len(position[:, 0])
        log.info('Nb profiles : %i' %(nb_profs))
        if do_qc:
            log.info('Run with QC')
            log.info('Type of DATA %s ' %(self.data_type))
            # Compute QC flag
            #QCflag=QCcontroller(self.data_type).factory(TYPEDATA).QCrun(fichier,qc_level)
            #QCobj=QCcontroller(self.data_type,TYPEDATA).factory()
            QCflag=QCcontroller(self.data_type,TYPEDATA).factory().QCrun(fichier, qc_level)
            #QCflag=QCobj.QCrun(fichier,qc_level)
        else :
            QCflag=np.chararray(nb_profs, itemsize=1)
            # QCflag=np.array[nb_profs]
            QCflag[:]="1"
        #log.info('QC Flags %s ' %(QCflag)) 
        #  Index of valid profiles
        index= [i for i,x in enumerate(QCflag) if x == "1"]
        log.info('Profiles Totaux : %i / Valides : %i ' % (len(QCflag),len(index)))
        nb_prof_clean=len(index)
        ## Loop on valid profil to find if it
        il_ind=0
        il_ind2=0
        list_file=[]
        list_index=[]
        tab_new=np.ndarray(shape=(1,2),dtype=float, order='F')
        # init
        tab_new[:,:]=np.nan
        for il_it in index:
            if zoom :
                position_lon=position[il_it,0]
                position_lat=position[il_it,1]
                if (position_lon <=  self.lonmax) & (position_lon >= self.lonmin) & \
                    (position_lat <= self.latmax) & ( position_lat >= self.latmin) :
                    log.info("Profile inside region : %f %f %f %f  %f %f  " % (position_lon,position_lat,self.lonmin, \
                                                                                 self.lonmax,self.latmin,self.latmax)) 
                    list_index.append(il_it)
                    if il_ind == 0 :
                        #tab_pos[0,:]=position[il_it,:]
                        #tab_new=position[il_it,:]
                        tab_new[0,0]=position[il_it,0]
                        tab_new[0,1]=position[il_it,1]
                    else :
                        tab_new=np.vstack((tab_new,position[il_it,:]))
                    il_ind=il_ind+1
            else :
                list_index.append(il_it)
                if il_ind2 == 0 :
                    tab_new=position[il_it,:]
                else :
                    tab_new=np.vstack((tab_new,position[il_it,:]))
                il_ind2=il_ind2+1
        if np.isnan(tab_new).all() :
        #try
        #   tab_new
        #except :
            tab_new=[]
            tab_new=np.array(tab_new)
        return tab_new,list_index


    def write(self,fichier,list_index,output_dir):
        # Write extracted files
        logging.basicConfig(format='%(asctime)s %(message)s')
        log = logging.getLogger(" Run Selector")
        ## Level of logging output
        #log.setLevel(logging.INFO)
        log.setLevel(logging.ERROR)
        ## Nco constructor
        nco = Nco()
        il_ind=0
        file_out=os.path.basename(fichier)
        for il_it in list_index:
            file_out2=os.path.basename(fichier).split('.')[0]+'_pf_'+str(il_it)+'.nc'
            if il_ind == 0 :
                ## Create file with unlimited dimension
                options = []
                options.extend(['-h'])
                nco.ncks(input=fichier,output=output_dir+file_out, fortran=True ,dimension=str(self.dimprof)+','+str(il_it+1),options=options)
                nco.ncecat(input=output_dir+file_out,output=output_dir+file_out,fortran=True,options=options)
                options = []
                options.extend(['-h','-O','-a',self.dimprof+',record'])
                nco.ncpdq(input=output_dir+file_out,output=output_dir+file_out,fortran=True,options=options)
                ## Remove record dimension
                options = []
                options.extend(['-h','-O','-a','record'])
                nco.ncwa(input=output_dir+file_out,output=output_dir+file_out,fortran=True,options=options)
            else :
                options = []
                options.extend(['-h'])
                log.info("Do ncks")
                nco.ncks(input=fichier,output=output_dir+file_out2, fortran=True ,append=True,dimension=self.dimprof+','+str(il_it+1),options=options)
                log.info("Do ncks ok")
                infiles = [output_dir+file_out,output_dir+file_out2]
                log.info("Inputfiles :  %s " %(infiles) )
                ## Concatenation of valid profiles
                log.info("Do ncrcat")
                nco.ncrcat(input=infiles, output=output_dir+'test.nc',fortran=True,options=options) 
                log.info("Do ncrcat OK")
                ## remove profile
                os.remove(output_dir+file_out2) 
                log.info("Do remove OK")
                os.rename(output_dir+'test.nc',output_dir+file_out)
                log.info("Do rename OK")
            il_ind=il_ind+1

    def store_pos(self,tab_pos,type_f,file_db):
        logging.basicConfig(format='%(asctime)s %(message)s')
        log = logging.getLogger("store_pos")
        ## Level of logging output
        log.setLevel(logging.INFO)
        # Open a shelve file if not exist
        if not(os.path.isfile(file_db)) :
            log.debug("Create persistent file")
            shelf = shelve.open(file_db, 'c')
        else:
            log.debug("File exist %s " %(file_db))
            shelf = shelve.open(file_db)
        # Add position values in the serialized dictionnary
        try :
            ll_exist = shelf[type_f]
        except :
            log.debug("Create %s" %(type_f))
            shelf[type_f] = tab_pos
            ll_exist=[]
        if len(ll_exist) > 0 :
            try :
                log.debug("Tab_fin")
                tab_fin=shelf[type_f]
                log.debug("concatenation")
                tab_fin=np.concatenate((tab_fin,tab_pos))
                shelf[type_f] = tab_fin
            except:
                print("Pb with concatenation store_pos")
                sys.exit(1)
        ## Create global position
        type_var='GLO'
        # Clean tab_pos to remove bad value
        nb_prof=len(tab_pos[:,0])
        nb_prof_safe=(tab_pos[:,0] < 99999 ) & (tab_pos[:,0] < 99999 )
        tab_pos_new=np.ndarray(shape=(nb_prof_safe.sum(),2),dtype=float, order='F')
        il_val=0
        for il_ind in range(nb_prof) :
            value=nb_prof_safe[il_ind]
            if value :
                tab_pos_new[il_val,:]=tab_pos[il_ind,:]
                il_val=il_val+1
        try :
            ll_GLO = shelf[type_var]
        except :
            shelf[type_var] = tab_pos_new
            ll_GLO=[]
        if len(ll_GLO) > 0 :
            try :
                tab_fin=shelf[type_var]
                tab_fin=np.concatenate((tab_fin,tab_pos_new))
                shelf[type_var] = tab_fin
            except :
                print("Pb with concatenation in GLO")
                sys.exit(1)
        shelf.close()

    def store_pos_single(self,tab_pos,type_f,file_db):
        logging.basicConfig(format='%(asctime)s %(message)s')
        log = logging.getLogger("store_pos_single")
        ## Level of logging output
        #log.setLevel(logging.DEBUG)
        log.setLevel(logging.INFO)
        # Open a shelve file if not exist
        if not(os.path.isfile(file_db)) :
            log.debug('Create persistent file')
            shelf = shelve.open(file_db, 'c')
        else:
            log.debug("File exist %s " %(file_db))
            shelf = shelve.open(file_db)
        # Add position values in the serialized dictionnary
        try:
            ll_exist = shelf[type_f]
        except:
            shelf[type_f] = tab_pos
            ll_exist=[]
        if len(ll_exist) > 0 :
            try :
                log.debug("Concatenation")
                tab_fin=shelf[type_f]
                log.debug("==== TAB FIN ==============")
                log.debug(tab_fin)
                log.debug(tab_fin.size)
                log.debug("===== TAB POS =============")
                log.debug(tab_pos)
                log.debug(tab_pos.size)
                log.debug("==================")

               # Clean tab_pos to remove bad value
                if np.any( tab_pos  < 9999 ) :
                    if  tab_pos.size > 2 and tab_fin.size == 2 :
                        log.debug('Cas -1 concatenate')
                        new_position=np.concatenate(([tab_fin],tab_pos))
                    elif tab_pos.size == 2 and tab_fin.size == 2 :
                        log.debug('Cas 0 concatenate')
                        new_position=np.concatenate(([tab_fin],[tab_pos]))
                    elif tab_pos.size == 2 :
                        log.debug('Cas 1 concatenate')
                        new_position=np.concatenate((tab_fin,[tab_pos]))
                        log.debug('Cas 1 OK')
                    else :
                        log.debug('Cas 2 concatenate')
                        new_position=np.concatenate((tab_fin,tab_pos))
                        log.debug('Cas 2 ok')
                    log.debug("Concatenation ok")
                    shelf[type_f] = new_position
                else :
                    log.debug('Bad profiles')
            except:
                print("Pb with concatenation single")
                sys.exit(1)
        shelf.close()
