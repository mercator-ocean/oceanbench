from __future__ import generators
import os
import sys
import string
import shutil
from glob import glob
import re
import numpy as np
import netCDF4
import json

if sys.version_info[0] >= 3:
    import configparser as ConfigParser
else:
    import ConfigParser
import operator
import getopt
import datetime as dt
import errno
import pandas as pd
from datetime import date
from .Logger import Logger
from calendar import monthrange
from .reader import *
from ast import literal_eval
from os.path import dirname
from .sydate import SyDate
from tep_class4.core.utils import date_range
from .QCcontrol import QCcontroller
import ast

##############################################################
# C.REGNIER Juin 2014
# Class Loader to load different type of file with a pattern factory
##############################################################


class Loader(object):
    """Create factory for the loader"""

    @staticmethod
    def factory(type):
        """
        Factory for different type of loaders
        """
        # return eval(type + "()")
        if type == "NC_CORIO":
            return LoaderNCcorio()
        if type == "NC_CORIO_ORIGIN":
            return LoaderNCcorio()
        if type == "NC_CHLORO":
            return LoaderNCChloro()
        if type == "NC_BIOARGO":
            return LoaderNCBioArgo()
        if type == "NC_CLIM":
            return LoaderNCClim()
        if type == "NC_INSCLIM":
            return LoaderInsituClim()
        if type == "NC_GODAE":
            return LoaderNCgodae()
        if type == "Microwat":
            return LoaderSyntheticValue()
        if type == "SSS_ship":
            return LoaderSSSVessel()
        if type == "ASCII":
            return AsciiLoader()
        if type == "LIST":
            return ListWriter()
        if type == "NML":
            return NmlLoader()
        if type == "DAT":
            return ColocLoader()
        if type == "INPUT":
            return InputLoader()
        if type == "INPUTSTATS":
            return InputLoader_stats()
        if type == "INPUT_DATA":
            return InputDataLoader()
        if type == "PGN_coords":
            return PGN_coords_Loader()
        if type == "PGN_coords_clim":
            return PGN_coords_clim_Loader()
        if type == "SMOC_coords":
            return SMOC_coords_Loader()
        if type == "DRIFTER_CMEMS":
            return LoaderCMEMSDrifter()
        if type == "DRIFTER_CMEMS_filtr":
            return LoaderCMEMSDrifter_filtr()
        if type == "alleges_coords":
            return alleges_coords()
        if type == "bathymetrie":
            return bathy_coords()
        if type == "Legos_SIT":
            return read_Legos_SIT()
        if type == "2ease_coords":
            return ease_coords_Loader()
        if type == "L3_SEALEVEL":
            return L3_SEALEVEL_Loader()
        assert 0, "Format not known: " + type

    def generic_read_pos_and_value(self, filename, varname):
        rla_lon, rla_lat, nc = self.generic_read_pos(filename)
        _FillValue = nc.variables[varname]._FillValue

        return rla_lon, rla_lat, _FillValue, nc

    def generic_read_pos(self, filename):

        nc = netCDF4.Dataset(filename, "r")
        ll_ravel = False
        if "longitude" in nc.variables:
            var_lon = "longitude"
            var_lat = "latitude"
        elif "lon" in nc.variables and not "lon_20hz" in nc.variables and not "lon_20_ku" in nc.variables:
            var_lon = "lon"
            var_lat = "lat"
        elif "LON" in nc.variables:
            var_lon = "LON"
            var_lat = "LAT"
        elif "Lon" in nc.variables:
            var_lon = "Lon"
            var_lat = "Lat"
        elif "LONGITUDE" in nc.variables:
            var_lon = "LONGITUDE"
            var_lat = "LATITUDE"
        elif "Longitude" in nc.variables:
            var_lon = "Longitude"
            var_lat = "Latitude"
        elif "lon_20hz" in nc.variables:
            var_lon = "lon_20hz"
            var_lat = "lat_20hz"
            ll_ravel = True
        elif "lon_20_ku" in nc.variables:
            var_lon = "lon_20_ku"
            var_lat = "lat_20_ku"
            ll_ravel = True
        if ll_ravel:
            rla_lon = np.ravel(nc.variables[var_lon][:, :])
            rla_lat = np.ravel(nc.variables[var_lat][:, :])
        else:
            rla_lon = nc.variables[var_lon][:]
            rla_lat = nc.variables[var_lat][:]

        return rla_lon, rla_lat, nc


class InputDataLoader(Loader):
    """
    Function to read input data
    """

    def run(self, param_dict, fichier, log):
        # Extract if necessary and read position of input data
        ll_interp = True
        if param_dict["do_qc"] or param_dict["zoom"]:
            if param_dict["zoom"]:
                log.info("REGIONAL MODE => AREA SELECTION ")
                selector = Selector(
                    param_dict["data_format"],
                    param_dict["data_type"],
                    param_dict["lon_min"],
                    param_dict["lon_max"],
                    param_dict["lat_min"],
                    param_dict["lat_max"],
                ).factory(param_dict["data_origin"])
                [new_position, list_index] = selector.run(
                    fichier, param_dict["zoom"], param_dict["do_qc"], param_dict["data_origin"], param_dict["qc_level"]
                )
                if list_index:
                    selector.write(fichier, list_index, param_dict["dirtmp"])
                    file_new = "ext-" + os.path.basename(fichier)
                    os.remove(fichier)
                    fichier = param_dict["dirwork"] + file_new
                    lon, lat, obs_value = Loader.factory(param_dict["data_format"]).read_pos_and_value(fichier)
                else:
                    os.remove(fichier)
                    log.info("No position find in %s for the zoom " % (fichier))
                    ll_file_interp = False
        else:
            # Case Global without qc
            log.info("GLOBAL MODE => SELECT ALL ")
            # selector = Selector(param_dict['data_format'], param_dict['data_type']).factory(
            #    param_dict['data_origin'])
            if param_dict["data_origin"] == "Microwat":
                lon, lat, juld = Loader.factory(param_dict["data_format"]).read_pos(fichier)
                nb_obs = int(np.shape(lat)[0])
                position = np.ndarray(shape=(nb_obs, 2), dtype=float, order="F")
                position[:, 0] = lon
                position[:, 1] = lat
            elif param_dict["data_type"] == "AMSR":
                position = Loader.factory(param_dict["data_format"]).read_pos_amsr(fichier)
            else:
                position = Loader.factory(param_dict["data_format"]).read_pos(fichier)

        return position, ll_interp
        # return lon, lat, ll_interp


class LoaderNCChloro(Loader):
    def __init__(self):
        self.varname = ["CHL"]

    def read_pos_and_value(self, filename):
        rla_lon, rla_lat, _FillValue, nc = self.generic_read_pos_and_value(filename, self.varname)
        rla_value = nc.variables[self.varname][0, :, :]
        nb_obs = int(np.shape(rla_value.flatten())[0])
        if "depth" in nc.variables.keys():
            rla_depth = nc.variables["depth"][:]
        elif "DEPH" in nc.variables.keys():
            rla_depth = nc.variables["DEPH"][:]
        elif "pres" in nc.variables.keys():
            rla_depth = nc.variables["pres"][:]
        elif "PRES" in nc.variables.keys():
            rla_depth = nc.variables["PRES"][:]
        else:
            ## No depth: surface variable
            nb_lats, nb_lons = np.shape(rla_value)
            rla_depth = np.ndarray(shape=(nb_lats, nb_lons), dtype=float, order="F")
            rla_depth[:, :] = 0
        nc.close()
        return rla_lon, rla_lat, rla_value, nb_obs, rla_depth, _FillValue


class LoaderNCBioArgo(Loader):

    def read_pos_and_value(self, filename, varname="NITRATE_ADJUSTED"):
        rla_lon, rla_lat, _FillValue, nc = self.generic_read_pos_and_value(filename, varname)
        rla_value = nc.variables[varname][:, :, :]
        nb_obs, nb_vars, nb_deps = np.shape(rla_value)
        if "depth" in nc.variables.keys():
            rla_depth = nc.variables["depth"][:, 0, :]
        elif "DEPH" in nc.variables.keys():
            rla_depth = nc.variables["DEPH"][:, 0, :]
        elif "DEPH_ADJUSTED" in nc.variables.keys():
            rla_depth = nc.variables["DEPH_ADJUSTED"][:, 0, :]
        elif "pres" in nc.variables.keys():
            rla_depth = nc.variables["pres"][:, 0, :]
        elif "PRES" in nc.variables.keys():
            rla_depth = nc.variables["PRES"][:, 0, :]
        elif "PRES_ADJUSTED" in nc.variables.keys():
            rla_depth = nc.variables["PRES_ADJUSTED"][:, 0, :]
        else:
            ## No depth: surface variable
            nb_lats, nb_lons = np.shape(rla_value)
            rla_depth = np.ndarray(shape=(nb_lats, nb_lons), dtype=float, order="F")
            rla_depth[:, :] = 0
        nc.close()

        return (
            np.float32(rla_lon),
            np.float32(rla_lat),
            np.float32(rla_value),
            nb_obs,
            np.float32(rla_depth),
            _FillValue,
        )


class LoaderInsituClim(Loader):
    def __init__(self):
        self.varname = "Temperature"

    def read_pos_and_value(self, filename):
        rla_lon, rla_lat, nb_obs, _FillValue = self.generic_read_pos_and_value(filename, self.varname)
        rla_value = nc.variables[self.varname][0, :, :]
        nb_obs = int(np.shape(rla_value.flatten())[0])
        nc.close()
        return rla_lon, rla_lat, rla_value, nb_obs, _FillValue

    def read_pos(self, filename):
        rla_lon, rla_lat, nc = self.generic_read_pos(filename)
        return rla_lon, rla_lat


class LoaderNCClim(Loader):
    def __init__(self):
        self.varname = "CHL_MEAN"

    def read_pos_and_value(self, filename):

        rla_lon, rla_lat, nb_obs, _FillValue = self.generic_read_pos_and_value(filename, self.varname)
        rla_value = nc.variables[self.varname][0, :, :]
        nb_obs = int(np.shape(rla_value.flatten())[0])
        nc.close()
        return rla_lon, rla_lat, rla_value, nb_obs, _FillValue

    def read_pos(self, filename):
        rla_lon, rla_lat, nc = self.generic_read_pos(filename)
        return rla_lon, rla_lat


class InputLoader_stats(Loader):
    def __init__(self):
        self.lead_int = 0  # list(range(10))

    def load(self, options, level):
        """Loader for input options"""
        keywords = [
            "date1=",
            "date2=",
            "config=",
            "typemod=",
            "loglevel=",
            "best",
            "fcst",
            "pers",
            "all",
            "clim",
            "bathy",
            "listdata=",
            "typedata=",
            "zarr",
        ]
        list_inputtype = ["pgs", "pgs_ai", "pgn", "pgncg"]

        cla_opts = getopt.getopt(options, "d", keywords)
        list_LEAD_TIME = []
        ll_listdata = True
        ll_netcdf = False
        cl_data_id = ""
        for o, p in cla_opts[0]:
            if o in ["--date1"]:
                ll_date1 = True
                cl_start_date = p
            elif o in ["--date2"]:
                ll_date2 = True
                cl_end_date = p
            elif o in ["--typemod"]:
                cl_typemod = p
                if cl_typemod not in list_inputtype:
                    print("input data type not known : ")
                    print(f"must be {list_inputtype} not {cl_typemod}")
                    sys.exit(1)
            elif o in ["--fcst"]:
                list_LEAD_TIME.append("forecast")
                ll_forecast = True
            elif o in ["--daterun"]:
                ll_daterun = True
            elif o in ["--config"]:
                ll_config = True
                cl_config = p
            elif o in ["--loglevel"]:
                level = int(p)
            elif o in ["--listdata"]:
                ll_listdata = True
                list_data = p
            elif o in ["--all"]:
                list_LEAD_TIME = ["best_estimate", "forecast", "persistence"]
            elif o in ["--zarr"]:
                ll_zarr = True
            elif o in ["--netcdf"]:
                ll_netcdf = True
        # LOGGER
        log = Logger(f"TEP_CLASS4_{cl_config}_{cl_start_date}_{cl_end_date}").run(level)
        log.debug("Loglevel {}".format(log))
        confname = "config_" + str(cl_config) + "_allvars" + ".cfg"
        # Create dictionnary
        param_dict = {}
        # param_dict['listdata'] = list_data
        param_dict["date1"] = cl_start_date
        param_dict["date2"] = cl_end_date
        param_dict["cl_config"] = cl_config
        param_dict["lead_time"] = list_LEAD_TIME
        param_dict["lead_int"] = self.lead_int
        param_dict["fcst_mode"] = ll_forecast
        param_dict["is_zarr"] = ll_zarr
        param_dict["is_netcdf"] = ll_netcdf
        param_dict["cl_typemod"] = cl_typemod.upper()
        param_dict["confname"] = confname
        param_dict["cl_data_id"] = cl_data_id
        return param_dict, log


class InputLoader(Loader):
    def __init__(self):
        self.lead_int = 0  # list(range(10))

    def load(self, options, level):
        """Loader for input options"""
        keywords = [
            "date1=",
            "date2=",
            "delta=",
            "nbdate=",
            "daterun=",
            "config=",
            "data=",
            "src=",
            "nbproc=",
            "typemod=",
            "schedule=",
            "data_id=",
            "loglevel=",
            "daymax=",
            "dirinput=",
            "diroutput=",
            "dirlog=",
            "best",
            "fcst",
            "pers",
            "all",
            "clim",
            "bathy",
            "SMOC",
            "stokes",
            "listdata=",
            "typedata=",
            "tides",
            "zarr",
        ]
        ll_delta = None
        ll_date1 = None
        ll_date2 = None
        ll_daterun = None
        ll_nbdate = None
        ll_config = None
        ll_data = None
        ll_src = None
        cl_daterun = None
        ll_data_id = None
        ll_cycle = None
        daymax = None
        dask = False
        ll_input = False
        ll_output = False
        ll_dirlog = False
        ll_best = False
        ll_clim = False
        ll_bathy = False
        ll_forecast = False
        ll_persistence = False
        ll_listdata = False
        list_var = []
        nb_proc = 0
        ll_zarr = False
        ll_netcdf = True
        cl_schedule = "d"
        cl_data_id = ""
        list_LEAD_TIME = []
        list_inputtype = ["pgs", "pgs_ai", "pgn", "pgncg", "allege", "2ease", "allege_new", "allege_now"]
        # try:
        #    opts, args = getopt.getopt(options, "h", [keywords])
        # except getopt.GetoptError:
        #    print("Invalid option. Use --help for usage information.")
        #    sys.exit(2)
        # for opt, arg in opts:
        #    if opt in ("-h", "--help"):
        #        print("Usage: script.py [--best]")
        #        sys.exit()
        #    elif opt == "--best":
        #        print("Performing the best operation.")
        cla_opts = getopt.getopt(options, "d", keywords)

        for o, p in cla_opts[0]:
            if o in ["--delta"]:
                ll_delta = True
                cl_delta = p
            elif o in ["--date1"]:
                ll_date1 = True
                cl_start_date = p
            elif o in ["--date2"]:
                ll_date2 = True
                cl_end_date = p
            elif o in ["--typemod"]:
                cl_typemod = p
                if cl_typemod not in list_inputtype:
                    print("input data type not known : ")
                    print(f"must be {list_inputtype} not {cl_typemod}")
                    sys.exit(1)
            elif o in ["--daterun"]:
                ll_daterun = True
                cl_daterun = p
            elif o in ["--nbdate"]:
                ll_nbdate = True
                cl_nbdate = p
            elif o in ["--config"]:
                ll_config = True
                cl_config = p
            elif o in ["--data"]:
                ll_data = True
                cl_data = p
            elif o in ["--src"]:
                ll_src = True
                cl_src = p
            elif o in ["--schedule"]:
                cl_schedule = p
            elif o in ["--nbproc"]:
                ll_optim = True
                cl_nbproc = p
                nb_proc = int(cl_nbproc)
            elif o in ["--data_id"]:
                ll_data_id = True
                cl_data_id = p
            elif o in ["--loglevel"]:
                level = int(p)
            elif o in ["--daymax"]:
                daymax = p
            elif o in ["-d"]:
                dask = True
            elif o in ["--dirinput"]:
                dirinput = p
                ll_input = True
            elif o in ["--diroutput"]:
                diroutput = p
                ll_output = True
            elif o in ["--dirlog"]:
                dirlog = p
                ll_dirlog = True
            elif o in ["--best"]:
                list_LEAD_TIME.append("best_estimate")
            elif o in ["--fcst"]:
                list_LEAD_TIME.append("forecast")
                ll_forecast = True
            elif o in ["--pers"]:
                list_LEAD_TIME.append("persistence")
            elif o in ["--bathy"]:
                list_LEAD_TIME.append("bathymetrie")
            elif o in ["--tides"]:
                list_LEAD_TIME.append("tides")
            elif o in ["--stokes"]:
                list_LEAD_TIME.append("stokes")
            elif o in ["--SMOC"]:
                list_LEAD_TIME.append("smoc")
            elif o in ["--clim"]:
                list_LEAD_TIME.append("climatology")
            elif o in ["--listdata"]:
                ll_listdata = True
                ll_data = True
                ll_src = True
                cl_data = "all"
                cl_src = "all"
                list_data = p
            elif o in ["--all"]:
                list_LEAD_TIME = ["best_estimate", "forecast", "persistence"]
            elif o in ["--zarr"]:
                ll_zarr = True
                ll_netcdf = False
        # LOGGER
        if ll_dirlog:
            log = Logger(f"TEP_CLASS4_{cl_data}_{cl_config}", dirlog=dirlog).run(level)
        else:
            log = Logger(f"TEP_CLASS4_{cl_data}_{cl_config}").run(level)
        log.debug("Loglevel {}".format(log))
        list_var.append(ll_date1)
        list_var.append(ll_date2)
        list_var.append(ll_daterun)
        list_var.append(ll_nbdate)
        if not ll_config:
            log.error("Config is missing : add --config= in the launcher")
            log.error("Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --date1=XXXX --date2=XXXX or ")
            log.error("Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --delta=XXXX (delayed time before today)")
            log.error(
                "Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --daterun=XXXX --nbdate=X --delta=X  (for example 1 week before today -14 and today -7)"
            )
            sys.exit(1)
        if not ll_data or not ll_src:
            log.error(
                "Type is missing : add --data=SLA/SST/profile/profile_ARMOR and --src=GODAE/CORIO. in the launcher"
            )
            log.error("Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --date1=XXXX --date2=XXXX or ")
            log.error("Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --delta=XXXX (delayed time before today)")
            log.error(
                "Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --daterun=XXXX --nbdate=X --delta=X  (for example 1 week before today -14 and today -7)"
            )
            sys.exit(1)
        if ll_delta and list_var.count(True) == 0:
            log.info("Case 1 Delta only")
            datetoday = SyDate(dt.datetime.today().strftime("%Y%m%d"))
            date1 = datetoday.gobackward(int(cl_delta))
            date2 = date1
        elif ll_date1 and ll_date2 and not ll_delta and not ll_daterun:
            log.debug("Case 2 Date1 Date2")
            # Test LEN of input date
            datelen1 = len(cl_start_date)
            datelen2 = len(cl_end_date)
            log.debug("Lenght of date %i %i" % (datelen1, datelen2))
            if datelen1 == datelen2 and datelen1 == 6:
                log.debug("Monthly values")
                log.debug("Values %s %s :" % (cl_start_date[0:4], cl_start_date[4:6]))
                nb_day1 = monthrange(int(cl_start_date[0:4]), int(cl_start_date[4:6]))[1]
                cl_start_date2 = cl_start_date + "01"
                cl_end_date2 = cl_start_date + str(nb_day1)
                log.debug(cl_start_date2)
                log.debug(cl_end_date2)
                date1 = SyDate(cl_start_date2)
                date2 = SyDate(cl_end_date2)
                date1 = SyDate(cl_start_date2)
                date2 = SyDate(cl_end_date2)
                cl_schedule = "m"
            elif datelen1 == datelen2 and datelen1 == 8:
                log.debug(f"Daily values {cl_start_date} {cl_end_date}")
                date1 = SyDate(cl_start_date)
                date2 = SyDate(cl_end_date)
                cl_schedule = "d"
            elif datelen1 == datelen2 and datelen1 == 4:
                log.debug("Year values")
                date1 = SyDate(cl_start_date + "0101")
                date2 = SyDate(cl_end_date + "1231")
                cl_schedule = "y"
            else:
                log.error("Input date must be in year month or day")
                date1 = cl_start_date
                date2 = cl_end_date
        elif ll_daterun and ll_nbdate and ll_delta:
            log.debug("Case 3 Backward run")
            jrun = SyDate(cl_daterun)
            date1 = jrun.gobackward(int(cl_delta))
            date2 = jrun.gobackward(int(cl_delta) - int(cl_nbdate) + 1)
        elif ll_date1:
            log.debug("Only one date")
            if len(str(cl_start_date)) == 4:
                date1 = SyDate(cl_start_date + "0101")
                date2 = SyDate(cl_start_date + "1231")
                log.debug("Year values")
                cl_schedule = "y"
            else:
                log.debug("Date must be a year")
                sys.exit(1)
        elif not ll_date1 and not ll_date2:
            log.debug("No date: last month")
            date_now = dt.datetime.now()
            if date_now.month == 1:
                cl_start_date = str(date_now.year - 1) + "1201"
                cl_end_date = str(date_now.year) + "0101"
            elif date_now.month < 10:
                cl_start_date = str(date_now.year) + "0" + str(date_now.month - 1) + "01"
                cl_end_date = str(date_now.year) + "0" + str(date_now.month) + "01"
            elif date_now.month == 11:
                cl_start_date = str(date_now.year) + "0" + str(date_now.month - 1) + "01"
                cl_end_date = str(date_now.year) + str(date_now.month) + "01"
            else:
                cl_start_date = str(date_now.year) + str(date_now.month - 1) + "01"
                cl_end_date = str(date_now.year) + str(date_now.month) + "01"
            date1 = SyDate(cl_start_date)
            date2 = SyDate(cl_end_date)
        elif cl_daterun == "all":
            log.debug("Need to find dates in file")
            date1 = None
            date2 = None
        else:
            log.error("Case not valid")
            log.error("Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --date1=XXXX --date2=XXXX or ")
            log.error("Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --delta=XXXX (delayed time before today)")
            log.error(
                "Type TEP_CLASS4.py --data=XXX --src=XXX --config=XXX --daterun=XXXX --nbdate=X --delta=X  \
                            (for example 1 week before today -14 and today -7)"
            )
            sys.exit(1)
        if ll_date1 and ll_date2:
            log.debug("Date1 %s Date2 %s " % (date1, date2))
        confname = "config_" + str(cl_config) + "_" + str(cl_src) + "_" + str(cl_data) + ".cfg"

        # Logger
        log.debug("Nombre de proc %i" % (nb_proc))
        log.debug("Type des fichiers d'entree %s " % (cl_typemod))
        log.debug("Schedule %s " % (cl_schedule))
        log.debug("Dates %s %s " % (date1, date2))
        log.debug("Datatype %s " % (cl_data))
        log.debug("Daterun %s " % (cl_daterun))
        # Create dictionnary
        param_dict = {}
        if ll_listdata:
            param_dict["listdata"] = list_data
        param_dict["date1"] = date1
        param_dict["date2"] = date2
        param_dict["confname"] = confname
        param_dict["data_type"] = cl_data
        param_dict["cl_daterun"] = cl_daterun
        param_dict["cl_schedule"] = cl_schedule
        param_dict["cl_src"] = cl_src
        param_dict["nb_proc"] = nb_proc
        param_dict["cl_typemod"] = cl_typemod.upper()
        param_dict["cl_data_id"] = cl_data_id
        param_dict["daymax"] = daymax
        param_dict["dask"] = dask
        param_dict["lead_time"] = list_LEAD_TIME
        param_dict["lead_int"] = self.lead_int
        param_dict["fcst_mode"] = ll_forecast
        param_dict["is_zarr"] = ll_zarr
        param_dict["is_netcdf"] = ll_netcdf
        if ll_input:
            param_dict["dirdata"] = dirinput
        if ll_output:
            param_dict["diroutput"] = diroutput
        if ll_dirlog:
            param_dict["dirlog"] = dirlog
        # Create time vector between date1 and date2
        # param_dict['date1'] = pd.to_datetime(param_dict['date1']).strftime('%Y%m%d')
        # param_dict['date2'] = pd.to_datetime(param_dict['date2']).strftime('%Y%m%d')
        start_date = dt.datetime(
            year=int(str(param_dict["date1"])[0:4]),
            month=int(str(param_dict["date1"])[4:6]),
            day=int(str(param_dict["date1"])[6:8]),
        )

        end_date = dt.datetime(
            year=int(str(param_dict["date2"])[0:4]),
            month=int(str(param_dict["date2"])[4:6]),
            day=int(str(param_dict["date2"])[6:8]),
        )

        vect_time = date_range(start_date, end_date, 1, "days")
        param_dict["time_period"] = pd.to_datetime(vect_time).strftime("%Y%m%d")
        log.debug(f"Input leadtime {param_dict['lead_int']}")

        return param_dict, log


class NmlLoader(Loader):
    """Loader for namelists"""

    def load(self, filename, level, param_dict, log):
        # log = Logger("NmlLoader").run(level)
        db = ConfigParser.ConfigParser()
        for file in filename:
            db.read(os.path.expanduser(filename))
        if not db.sections():
            log.error("Configuration file not found {} ! ".format(filename))
            sys.exit(1)
        param_dict["db"] = db
        param_dict["dirhome"] = os.path.dirname(os.path.realpath(__file__)) + "/../"
        param_dict["nml_dir"] = param_dict["dirhome"] + "NAMELIST/"
        param_dict["dir_log"] = param_dict["dirhome"] + "log/"
        # Add new parameters in the dictionnary
        # if 'sla' in param_dict['data_type'].lower():
        # if param_dict['data_type'] == "SLA" :
        param_dict["cl_msshFile"] = db.get("static", "cl_msshFile")
        param_dict["rp_mssh_shift"] = float(db.get("static", "rp_mssh_shift"))
        if param_dict["cl_typemod"].lower() == "pgn":
            param_dict["dirmodel_pgn"] = db.get("input_dir", "DATA_IN_PGN")
            param_dict["dirmodel"] = param_dict["dirmodel_pgn"]
            param_dict["cl_config"] = db.get("GLOBAL", "CONFIGURATION")
            param_dict["varname"] = literal_eval(db.get("GLOBAL", "variable_ng"))
            param_dict["coordinates"] = db.get("static", "fic_coordinates")
        elif param_dict["cl_typemod"].lower() == "pgs" or param_dict["cl_typemod"].lower() == "pgs_ai":
            param_dict["dirmodel_pgs"] = db.get("input_dir", "DATA_IN_PGS")
            param_dict["cl_config"] = db.get("GLOBAL", "CONFIGURATION")
            param_dict["varname"] = literal_eval(db.get("GLOBAL", "variable_sg"))
            param_dict["coordinates"] = db.get("static", "fic_coordinates")
        elif param_dict["cl_typemod"].lower() == "pgncg":
            param_dict["dirmodel_pgncg"] = db.get("input_dir", "DATA_IN_PGN_CG")
            param_dict["cl_config"] = db.get("GLOBAL", "CONFIGURATION_CG")
            param_dict["varname"] = literal_eval(db.get("GLOBAL", "variable_cg"))
            param_dict["coordinates"] = db.get("static", "fic_coordinates_cg")
        else:
            self.log.error("Input not known /= pgn pgs pgs_ai pgn_cg")
            sys.exit(1)
        if "dirdata" not in param_dict.keys():
            param_dict["dirdata"] = db.get("input_dir", "INPUTDATA")
        try:
            param_dict["dirmodel_light"] = db.get("input_dir", "DATA_IN_LIGHT")
        except:
            param_dict["dirmodel_light"] = ""
        try:
            param_dict["insitu_corr"] = db.get("GLOBAL", "corr_pot")
        except:
            param_dict["insitu_corr"] = False
        try:
            param_dict["Design_GODAE_file"] = ast.literal_eval(db.get("GLOBAL", "Design_GODAE_file"))
        except:
            param_dict["Design_GODAE_file"] = False
        try:
            param_dict["EDITO"] = db.get("input_dir", "EDITO")
        except:
            param_dict["EDITO"] = ""
        try:
            param_dict["typedata"] = literal_eval(db.get("GLOBAL", "typedata"))
        except:
            param_dict["typedata"] = ""
        try:
            param_dict["list_variables"] = literal_eval(db.get("GLOBAL", "list_variables"))
        except:
            param_dict["list_variables"] = ""
        try:
            param_dict["good_qc"] = literal_eval(db.get("GLOBAL", "good_qc"))
        except:
            param_dict["good_qc"] = ""
        try:
            param_dict["DEPTHS"] = literal_eval(db.get("GLOBAL", "DEPTHS"))
        except:
            param_dict["DEPTHS"] = [0, 5, 100, 300, 600]
        try:
            param_dict["time_centered"] = literal_eval(db.get("GLOBAL", "time_centered"))
        except:
            param_dict["time_centered"] = False

        # param_dict['Design_GODAE_file'] = db.get('GLOBAL', 'Design_GODAE_file')
        param_dict["dirmodel"] = db.get("input_dir", "DATA_IN")
        param_dict["nb_level"] = int(db.get("GLOBAL", "LEVELS"))
        param_dict["zoom"] = int(db.get("GLOBAL", "zoom"))
        param_dict["write_geojson"] = int(db.get("GLOBAL", "write_geojson"))
        param_dict["cl_config_h"] = re.sub("QV", "V", param_dict["cl_config"])
        if param_dict["write_geojson"]:
            # Write Geojson file
            log.debug("Write Geojson file")
            # Db file for extracted positions
            file_db = "CLASS4_shelf.db"
            file_db2 = "Coloc_CLASS4_shelf.db"
            # Clean old db
            if os.path.isfile(file_db):
                os.remove(file_db)
            # Clean old db
            if os.path.isfile(file_db2):
                os.remove(file_db2)
        param_dict["qc_level"] = int(db.get("GLOBAL", "qc_level"))
        param_dict["do_qc"] = int(db.get("GLOBAL", "do_qc"))
        param_dict["do_sort"] = int(db.get("GLOBAL", "do_sort"))
        param_dict["verif"] = literal_eval(db.get("GLOBAL", "verif"))
        param_dict["json"] = literal_eval(db.get("GLOBAL", "ll_json"))
        param_dict["maskname"] = db.get("static", "maskname")
        try:
            param_dict["direxec"] = db.get("GLOBAL", "DIRBIN")
        except:
            param_dict["direxec"] = os.path.dirname(os.path.realpath(__file__)) + "/../bin/"
        try:
            param_dict["over"] = literal_eval(db.get("GLOBAL", "over"))
        except:
            param_dict["over"] = False
        # ==========================================================================
        # Sort model values if necessary
        # ==========================================================================
        if param_dict["do_sort"]:
            log.debug("              Sort Values for interp               ")
            param_dict["sort_rep"] = param_dict.get("input_dir", "DATA_SORT")
            file_template = param_dict["cl_config"] + "_" + param_dict["daily_pref"] + "_"
            if cl_schedule == "WEEKLY":
                Sorter().run_weekly(
                    param_dict["lead_time"],
                    param_dict["lead_int"],
                    date1,
                    date2,
                    daterun,
                    param_dict["file_template"],
                    param_dict["gridtemp"],
                    param_dict["gridsal"],
                    param_dict["dirmodel"],
                    param_dict["sort_rep"],
                )
            elif cl_schedule == "DAILY":
                Sorter().run_daily(
                    param_dict["lead_time"],
                    param_dict["lead_int"],
                    date1,
                    date2,
                    param_dict["file_template"],
                    param_dict["dirmodel"],
                    param_dict["sort_rep"],
                )
            else:
                log.error("Schedule not valid")
                sys.exit(1)
        else:
            log.debug("              Input data already sorted               ")
            param_dict["sort_rep"] = param_dict["dirmodel"]

        log.debug("Sort rep : %s" % (param_dict["sort_rep"]))
        param_dict["do_decomp"] = int(db.get("GLOBAL", "do_decomp"))
        param_dict["model_type"] = db.get("GLOBAL", "MODELTYPE")
        try:
            param_dict["data_format"] = literal_eval(db.get("input_dir", "DATA_FMT"))
        except:
            param_dict["data_format"] = []

        param_dict["dirwork"] = db.get("output_dir", "DIRTMP")
        if not os.path.exists(param_dict["dirwork"]):
            try:
                os.makedirs(param_dict["dirwork"])
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(param_dict["dirwork"]):
                    pass
        # param_dict['dirwork'] = db.get('output_dir', 'DIRWORK')
        if "diroutput" not in param_dict.keys():
            param_dict["diroutput"] = db.get("output_dir", "DIROUTPUT")
        try:
            param_dict["diroutput2"] = db.get("output_dir", "DIROUTPUT2")
        except:
            param_dict["diroutput2"] = ""
        param_dict["dirclim"] = db.get("climato", "DIRCLIM")
        try:
            param_dict["dirclim2"] = db.get("climato", "DIRCLIM2")
        except:
            param_dict["dirclim2"] = ""
        try:
            param_dict["coord_clim"] = glob(param_dict["dirclim"] + "*.nc")[0]
        except:
            param_dict["coord_clim"] = ""
        try:
            param_dict["coordinates_hgr"] = db.get("static", "coordinates_hgr")
        except:
            param_dict["coordinates_hgr"] = ""

        param_dict["dirtmp"] = param_dict["dirwork"] + "ext-"
        try:
            param_dict["prefix_data"] = literal_eval(db.get("input_dir", "PREFIX"))
        except:
            param_dict["prefix_data"] = []
        param_dict["daily_pref"] = db.get("GLOBAL", "DAILY_PREF")
        try:
            param_dict["data_origin"] = db.get("input_dir", "DATA_ORIGIN")
        except:
            param_dict["data_origin"] = []
        param_dict["cl_conf"] = db.get("GLOBAL", "CONF")
        # param_dict['template'] = param_dict['cl_config_h'] + \
        #    '_'+param_dict['model_type'].lower()
        param_dict["template"] = param_dict["cl_conf"] + "_" + param_dict["model_type"].lower()
        # param_dict['lead_time'] = db.get('input_dir', 'LEAD_TIME').split()
        # param_dict['lead_int'] = db.get('input_dir', 'LEAD_INT').split()
        param_dict["coloc_rep"] = db.get("output_dir", "DIRCOLOC")
        param_dict["gridtemp"] = db.get("GLOBAL", "gridtemp")
        param_dict["gridsal"] = db.get("GLOBAL", "gridsal")
        try:
            param_dict["gridu"] = db.get("GLOBAL", "gridu")
        except:
            param_dict["gridu"] = ""
        try:
            param_dict["gridv"] = db.get("GLOBAL", "gridv")
        except:
            param_dict["gridv"] = ""
        try:
            param_dict["gridice"] = db.get("GLOBAL", "gridice")
        except:
            param_dict["gridice"] = "icemod"
        try:
            param_dict["grid2DT"] = db.get("GLOBAL", "grid2DT")
        except:
            param_dict["grid2DT"] = "grid2D"
        try:
            param_dict["naming"] = db.get("GLOBAL", "name")
        except:
            param_dict["naming"] = "old"
        param_dict["type_run"] = db.get("GLOBAL", "TYPE_RUN")
        param_dict["type_archi"] = db.get("GLOBAL", "TYPE_ARCHI")
        param_dict["nb_points"] = int(db.get("GLOBAL", "nb_points_interp"))
        param_dict["schedule"] = db.get("GLOBAL", "SCHEDULE")
        param_dict["ll_zip"] = literal_eval(db.get("GLOBAL", "ll_zip"))
        param_dict["contact"] = db.get("GLOBAL", "CONTACT")
        param_dict["inst"] = db.get("GLOBAL", "INSTITUTION")
        param_dict["version"] = db.get("GLOBAL", "VERSION")
        param_dict["desc"] = db.get("GLOBAL", "DESCRIPTION")
        try:
            param_dict["varname_smoc"] = literal_eval(db.get("GLOBAL", "variable_SMOC"))
        except:
            param_dict["varname_smoc"] = ""
        try:
            param_dict["varname_stokes"] = literal_eval(db.get("GLOBAL", "variable_stokes"))
        except:
            param_dict["varname_stokes"] = ""
        try:
            param_dict["varname_tides"] = literal_eval(db.get("GLOBAL", "variable_tides"))
        except:
            param_dict["varname_tides"] = ""
        try:
            param_dict["lead_int"] = db.get("input_dir", "LEAD_INT").split()
            param_dict["lead_int2"] = param_dict["lead_int2"]
        except:
            param_dict["lead_int2"] = ""
        try:
            param_dict["varname_bathy"] = db.get("GLOBAL", "variable_bathy").split()
        except:
            param_dict["lead_int2"] = ""
        try:
            param_dict["corr"] = literal_eval(db.get("GLOBAL", "corr"))
        except:
            param_dict["corr"] = False

        if param_dict["zoom"] or param_dict["do_qc"]:
            param_dict["lon_min"] = int(param_dict.get("GLOBAL", "LONMIN"))
            param_dict["lon_max"] = int(param_dict.get("GLOBAL", "LONMAX"))
            param_dict["lat_min"] = int(param_dict.get("GLOBAL", "LATMIN"))
            param_dict["lat_max"] = int(param_dict.get("GLOBAL", "LATMAX"))
        try:
            param_dict["coordinates_SMOC"] = db.get("static", "fic_coordinates_SMOC")
        except:
            param_dict["coordinates_SMOC"] = ""
        try:
            param_dict["bathy_file"] = db.get("static", "fic_bathy")
        except:
            param_dict["bathy_file"] = ""

        if param_dict["cl_data_id"]:
            param_dict["dirwork"] = param_dict["dirwork"] + str(cl_data_id) + "/"
            param_dict["diroutput"] = param_dict["diroutput"] + str(cl_data_id) + "/"
            param_dict["dirdata"] = param_dict["dirdata"] + str(cl_data_id) + "/"
        elif param_dict["cl_conf"] == "BIOMER4" or param_dict["cl_conf"] == "BIOMER4V2":
            param_dict["dirwork"] = param_dict["dirwork"]
            param_dict["diroutput"] = param_dict["diroutput"]
            date_ini = SyDate(str(param_dict["date1"]))
            # if param_dict['prefix_data'] == "L3mCHL": param_dict['dirdata'] = param_dict['dirdata']+ '/'+str(date_ini.year)

        param_dict["dirwork"] = param_dict["dirwork"]
        param_dict["diroutput"] = param_dict["diroutput"]
        param_dict["dirdata"] = param_dict["dirdata"]

        return param_dict


class LoaderNCcorio(Loader):
    """Loader for coriolis dataset"""

    def __init__(self):
        self.varname_list = []
        self.list_temp = ["votemper", "thetao", "temperature"]
        self.list_psal = ["vosaline", "so", "salinity"]

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "LONGITUDE"
        rla_lon = nc.variables[variable][:]
        # rla_lon[rla_lon <0] += 360.
        variable = "LATITUDE"
        rla_lat = nc.variables[variable][:]
        n_profiles = len(rla_lon)
        # position=np.ndarray(shape=(n_profiles,2))
        # position[:,0]=rla_lon[:]
        # position[:,1]=rla_lat[:]
        # Solution 1
        position = np.ndarray(shape=(n_profiles, 2), dtype=float, order="F", buffer=np.array([rla_lon[:], rla_lat[:]]))
        # Solution 2
        # il_val=0
        # for il_ind in range(nb_tot):
        #  if  rla_lon[il_ind] < 9999. and rla_lat[il_ind] < 9999. :
        #    position[il_val,0]=rla_lon[il_ind]
        #    position[il_val,1]=rla_lat[il_ind]
        #    il_val=il_val+1
        # np.set_printoptions(suppress=True)
        nc.close()
        return position

    def read_pos_and_value(self, filename, param_dict, qc_level=0):

        ll_pres = 0
        ll_depth = 0
        ll_temp = 0
        ll_psal = 0
        nc = netCDF4.Dataset(filename)
        if "DEPH" in nc.variables.keys():
            ll_depth = 1
        if "PRES" in nc.variables.keys():
            ll_pres = 1
        if "TEMP" in nc.variables.keys():
            rla_temp = nc.variables["TEMP"][:, :]
            ll_temp = 1
            self.varname_list.append("TEMP")
            n_profiles, n_profs = np.shape(rla_temp)
        if "PSAL" in nc.variables.keys():
            rla_temp = nc.variables["PSAL"][:, :]
            ll_psal = 1
            self.varname_list.append("PSAL")
            n_profiles, n_profs = np.shape(rla_temp)
        if not ll_temp and not ll_psal:
            print(f"PROF not valid with no TEMP and no PSAL {filename}")
            sys.exit(1)
        nb_variables = len(self.varname_list)
        # print(self.varname_list)
        # print(f"Number of variables in {filename} {nb_variables}")
        # rla_obs = np.ndarray(shape=(n_profiles,n_profs, nb_variables))
        rla_obs = np.ndarray(shape=(n_profiles, nb_variables, n_profs))
        for ind in range(nb_variables):
            rla_obs[:, ind, :] = nc.variables[self.varname_list[ind]][:, :]
            _FillValue = nc.variables[self.varname_list[ind]]._FillValue

        tab_dims = {}
        dimension_names = nc.dimensions.keys()
        dimensions_dict = {dim_name: len(dim) for dim_name, dim in nc.dimensions.items()}

        for dim_name in dimension_names:
            dimension = nc.dimensions[dim_name]
            tab_dims[dim_name] = len(dimension)
        if ll_depth and ll_pres:
            rla_depth = nc.variables["PRES"][:, :]
        elif ll_depth and not ll_pres:
            rla_depth = nc.variables["DEPH"][:, :]
        elif ll_pres and not ll_depth:
            rla_depth = nc.variables["PRES"][:, :]
        else:
            print("Not pres and depth")
            sys.exit(1)
        # "if 'depth' in nc.variables.keys():
        # "    rla_depth = nc.variables['depth'][:,0,:]
        # "elif 'DEPH' in nc.variables.keys():
        # "    rla_depth = nc.variables['DEPH_ADJUSTED'][:,0,:]
        # "elif 'pres' in nc.variables.keys():
        # "    rla_depth = nc.variables['pres'][:,0,:]
        # "elif 'PRES' in nc.variables.keys():
        # "    rla_depth = nc.variables['PRES'][:,0,:]
        if "LONGITUDE" in nc.variables.keys():
            rla_lon = nc.variables["LONGITUDE"][:]
        if "LATITUDE" in nc.variables.keys():
            rla_lat = nc.variables["LATITUDE"][:]
        nc.close()
        list_mod = param_dict["varname"]
        if len(set(self.list_psal).intersection(param_dict["varname"])) == 1:
            list_psal = list(set.intersection(set(self.list_psal), set(param_dict["varname"])))
        if len(set(self.list_temp).intersection(param_dict["varname"])) == 1:
            list_temp = list(set.intersection(set(self.list_temp), set(param_dict["varname"])))
        # print(f"{list_psal=} {ll_psal}")
        # print(f"{list_temp=} {ll_temp}")
        param_dict["ll_PSAL"] = False
        param_dict["ll_TEMP"] = False
        if not ll_psal:
            # print('Remove psal')
            list_mod = list(set(list_mod) - set(list_psal))
        else:
            param_dict["ll_PSAL"] = True
        if not ll_temp:
            # print('Remove temp')
            list_mod = list(set(list_mod) - set(list_temp))
        else:
            param_dict["ll_TEMP"] = True
        param_dict["varname_select"] = list_mod
        ## Qc control
        QCflag = QCcontroller(param_dict["data_type"], param_dict["data_origin"]).factory().QCrun(filename, qc_level)
        param_dict["qc"] = QCflag
        return (
            np.float32(rla_lon),
            np.float32(rla_lat),
            np.float32(rla_obs),
            n_profiles,
            np.float32(rla_depth),
            tab_dims,
            param_dict,
            _FillValue,
        )


class LoaderSyntheticValue(Loader):
    """Loader for synthetic dataset"""

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "longitude"
        rla_lon = nc.variables[variable][:]
        variable = "latitude"
        rla_lat = nc.variables[variable][:]
        n_profiles = len(rla_lon[:])
        # Optional Read juld
        # variable='juld'
        # rla_juld= nc.variables[variable][:]
        nc.close()
        return rla_lon, rla_lat


class LoaderSSSVessel(Loader):

    def read_pos_and_value(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "LON"
        rla_lon = nc.variables[variable][:]
        variable = "LAT"
        rla_lat = nc.variables[variable][:]
        n_profiles = len(rla_lon[:])
        rla_value = nc.variables["SSS"][:]
        _FillValue = nc.variables["SSS"]._FillValue
        nb_obs = len(rla_lon)
        nc.close()
        return rla_lon, rla_lat, rla_value, nb_obs, _FillValue
        # Optional Read juld
        # variable='juld'
        # rla_juld= nc.variables[variable][:]


class L3_SEALEVEL_Loader(Loader):
    """Loader for L3_SEALEVEL dataset"""

    def __init__(self):
        self.varname = "observation"

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "longitude"
        rla_lon = nc.variables[variable][:]
        variable = "latitude"
        rla_lat = nc.variables[variable][:]
        n_profiles = len(rla_lon[:])
        # Optional Read juld
        variable = "time"
        rla_juld = nc.variables[variable][:]
        position = np.ndarray(shape=(n_profiles, 2), dtype=float, order="F")
        il_val = 0
        for il_ind in range(n_profiles):
            position[il_val, 0] = rla_lon[il_ind]
            position[il_val, 1] = rla_lat[il_ind]
            il_val = il_val + 1
        np.set_printoptions(suppress=True)
        nc.close()
        return position


class LoaderNCgodae(Loader):
    """Loader for GODAE dataset"""

    def __init__(self):
        self.varname = "observation"

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "longitude"
        rla_lon = nc.variables[variable][:]
        variable = "latitude"
        rla_lat = nc.variables[variable][:]
        n_profiles = len(rla_lon[:])
        # Optional Read juld
        # variable='juld'
        # rla_juld= nc.variables[variable][:]
        position = np.ndarray(shape=(n_profiles, 2), dtype=float, order="F")
        il_val = 0
        for il_ind in range(n_profiles):
            position[il_val, 0] = rla_lon[il_ind]
            position[il_val, 1] = rla_lat[il_ind]
            il_val = il_val + 1
        np.set_printoptions(suppress=True)
        nc.close()
        return position

    def read_pos_amsr(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "LON"
        rla_lon = nc.variables[variable][:]
        variable = "LAT"
        rla_lat = nc.variables[variable][:]
        n_profiles = len(rla_lon[:])
        # Optional Read juld
        # variable='juld'
        # rla_juld= nc.variables[variable][:]
        position = np.ndarray(shape=(n_profiles, 2), dtype=float, order="F")
        il_val = 0
        for il_ind in range(n_profiles):
            position[il_val, 0] = rla_lon[il_ind]
            position[il_val, 1] = rla_lat[il_ind]
            il_val = il_val + 1
        np.set_printoptions(suppress=True)
        nc.close()
        return position

    def read_pos_and_value(self, filename):
        rla_lon, rla_lat, _FillValue, nc = self.generic_read_pos_and_value(filename, self.varname)
        rla_value = nc.variables[self.varname][:]
        # ncdata = xr.open_dataset(filename)
        # variables = ncdata.data_vars
        dimensions = nc.dimensions
        nb_fcsts = None
        tab_dims = {}
        for dims in dimensions.keys():
            sizedim = dimensions[dims].size
            if dims == "numfcsts":
                tab_dims["nb_fcsts"] = sizedim
            elif dims == "numobs":
                tab_dims["nb_obs"] = sizedim
            elif dims == "numdeps":
                tab_dims["nb_depths"] = sizedim

        if "depth" in nc.variables.keys():
            rla_depth = nc.variables["depth"][:]
        elif "DEPH" in nc.variables.keys():
            rla_depth = nc.variables["DEPH"][:]
        elif "pres" in nc.variables.keys():
            rla_depth = nc.variables["pres"][:]
        elif "PRES" in nc.variables.keys():
            rla_depth = nc.variables["PRES"][:]
        else:
            ## No depth: surface variable
            nb_obs, nb_vars, nb_deps = np.shape(rla_value)
            rla_depth = np.ndarray(shape=(nb_obs, nb_deps), dtype=float, order="F")
            rla_depth[:, 0] = 0
        nb_obs = len(rla_value)
        ### TODO Return the dimensions of the forecast
        ## CREGNIER
        # ncdata.close()
        nc.close()
        return rla_lon, rla_lat, rla_value, nb_obs, rla_depth, tab_dims, _FillValue


class PGN_coords_Loader(Loader):
    """Loader for PGN dataset"""

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variables = nc.variables
        list_lon = ["x", "X", "lon", "LON", "lons", "longitude", "LONGITUDE", "nav_lon"]
        list_lat = ["y", "Y", "lat", "LAT", "LATS", "latitude", "LATITUDE", "nav_lat"]
        list_depth = ["deptht", "depth", "z", "Z", "DEPTH", "nav_lev"]
        list_mask = ["tmask", "mask"]
        try:
            lon_name = [k for k in variables.keys() & set(list_lon)][0]
            lat_name = [k for k in variables.keys() & set(list_lat)][0]
            depth_name = [k for k in variables.keys() & set(list_depth)][0]
            mask_name = [k for k in variables.keys() & set(list_mask)][0]
        except:
            print("lon lat or depth not defined")
            raise
        rla_lon = nc.variables[lon_name][:, :]
        rla_lat = nc.variables[lat_name][:, :]
        rla_mask = nc.variables[mask_name][:, :, :]
        depth_mod = nc.variables[depth_name][:]
        nlat, nlon, ndepth = np.shape(rla_mask)
        mask_tab = np.full((nlat, nlon, ndepth), True)
        mask_tab = np.where(rla_mask == 1, False, mask_tab)
        nc.close()
        return rla_lon, rla_lat, mask_tab, depth_mod


class ease_coords_Loader(Loader):
    """Loader for 2 ease stereopolar grid"""

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variables = nc.variables
        list_lon = ["x", "X", "lon", "LON", "lons", "longitude", "LONGITUDE", "nav_lon"]
        list_lat = ["y", "Y", "lat", "LAT", "LATS", "latitude", "LATITUDE", "nav_lat"]
        try:
            lon_name = [k for k in variables.keys() & set(list_lon)][0]
            lat_name = [k for k in variables.keys() & set(list_lat)][0]
        except:
            print("lon lat or depth not defined")
            raise
        rla_lon = nc.variables[lon_name][:, :]
        rla_lat = nc.variables[lat_name][:, :]
        depth_mod = 0
        mask_tab = np.nan
        nc.close()
        return rla_lon, rla_lat, mask_tab, depth_mod


class PGN_coords_clim_Loader(Loader):
    """Loader for PGN dataset"""

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "lon"
        rla_lon = nc.variables[variable][:, :]
        variable = "lat"
        rla_lat = nc.variables[variable][:, :]
        variable = "tmask"
        rla_mask = nc.variables[variable][:, :, :]
        variable = "nav_lev"
        depth_mod = nc.variables[variable][:]
        nlat, nlon, ndepth = np.shape(rla_mask)
        mask_tab = np.full((nlat, nlon, ndepth), True)
        mask_tab = np.where(rla_mask == 1, False, mask_tab)
        nc.close()
        return rla_lon, rla_lat, mask_tab, depth_mod


class alleges_coords(Loader):
    """Loader for Alleges Mask"""

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "longitude"
        rla_lon = nc.variables[variable][:, :]
        variable = "latitude"
        rla_lat = nc.variables[variable][:, :]
        variable = "depth"
        rla_depth = nc.variables[variable][:]
        variable = "mask"
        rla_mask = nc.variables[variable][:, :, :]
        ndepth, nlat, nlon = np.shape(rla_mask)
        print("depth {} Lat {} Lon {}".format(ndepth, nlat, nlon))
        mask_tab = np.full((ndepth, nlat, nlon), True)
        mask_tab = np.where(rla_mask == 1, False, mask_tab)
        nc.close()
        return rla_lon, rla_lat, mask_tab, rla_depth


class bathy_coords(Loader):
    """Loader for Alleges Mask"""

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "nav_lon"
        rla_lon = nc.variables[variable][:, :]
        variable = "nav_lat"
        rla_lat = nc.variables[variable][:, :]
        nc.close()
        return rla_lon, rla_lat


class read_Legos_SIT(Loader):
    """Loader LEGOS SIT files"""

    def __init__(self):
        self.varname_list = ["snow_depth_W99", "radar_freeboard_20hz"]

    def read_pos_and_value(self, filename):

        try:
            nc = netCDF4.Dataset(filename, "r")
            if len(set(nc.variables).intersection(self.varname_list)) == 1:
                self.varname = list(set.intersection(set(self.varname_list), set(nc.variables)))[0]
            nc.close()
        except:
            raise cl4err.Class4FatalError("Pb read_Legos_SIT varname missing {}".format(self.varname_list))
        rla_lon, rla_lat, _FillValue, nc = self.generic_read_pos_and_value(filename, self.varname)
        rla_value = nc.variables[self.varname][:]
        if "depth" in nc.variables.keys():
            rla_depth = nc.variables["depth"][:]
        elif "DEPH" in nc.variables.keys():
            rla_depth = nc.variables["DEPH"][:]
        elif "pres" in nc.variables.keys():
            rla_depth = nc.variables["pres"][:]
        elif "PRES" in nc.variables.keys():
            rla_depth = nc.variables["PRES"][:]
        else:
            ## No depth: surface variable
            nb_obs = np.shape(rla_lon)[0]
            nb_deps = 1
            rla_depth = np.ndarray(shape=(nb_obs, nb_deps), dtype=float, order="F")
            rla_depth[:, 0] = 0
        nc.close()
        return rla_lon, rla_lat, rla_value, nb_obs, rla_depth, _FillValue


class AsciiLoader(Loader):
    """Loader for ASCII dataset"""

    def read_pos(self, filename):
        # print 'read ASCII file : %s ' %(filename)
        # data=np.genfromtxt(filename,usecols = (2,3),skip_header=1)
        data = np.genfromtxt(filename, usecols=(1, 2), skip_header=1)
        position = np.array(data)
        return position


class ColocLoader(Loader):
    """Loader for coloc ASCII file"""

    def read_pos(self, filename):
        # print 'read ASCII file : %s ' %(filename)
        # data=np.genfromtxt(filename,usecols = (5,6))
        data = np.genfromtxt(filename, usecols=(15, 16))
        position = np.array(data)
        return position


class LoaderCMEMSDrifter_filtr(Loader):
    """
    Loader for CMEMS drifters filtered
    """

    def read_pos_and_value(self, filename):
        (
            lon,
            lat,
            depth,
            depth_qc,
            time,
            u,
            v,
            u_filtr,
            v_filtr,
            ll_var,
            u_qc,
            v_qc,
            u_filtr_qc,
            v_filtr_qc,
            wind_u,
            wind_v,
            current_test,
            current_test_qc,
            position_qc,
            time_qc,
            dc_reference,
        ) = read_drifter_filt(filename)
        nb_obs = len(lon)
        _FillValue = np.nan
        tab_dims = {}
        # for dims in dimensions.keys():
        #    sizedim = dimensions[dims].size
        #    if dims == 'numfcsts':
        #        tab_dims['nb_fcsts'] = sizedim
        #    elif dims == 'numobs':
        #        tab_dims['nb_obs'] = sizedim
        #    elif dims == 'numdeps':
        #        tab_dims['nb_depths'] = sizedim

        return lon, lat, [u_filtr, v_filtr], nb_obs, depth, tab_dims, _FillValue


class LoaderCMEMSDrifter(Loader):
    """
    Loader for CMEMS drifters
    """

    def read_pos_and_value(self, filename):
        (
            lon,
            lat,
            depth,
            depth_qc,
            time,
            u,
            v,
            ll_var,
            u_qc,
            v_qc,
            wind_u,
            wind_v,
            current_test,
            current_test_qc,
            position_qc,
            time_qc,
            dc_reference,
        ) = read_drifter(filename)
        nb_obs = len(lon)
        _FillValue = np.nan
        tab_dims = {}
        # for dims in dimensions.keys():
        #    sizedim = dimensions[dims].size
        #    if dims == 'numfcsts':
        #        tab_dims['nb_fcsts'] = sizedim
        #    elif dims == 'numobs':
        #        tab_dims['nb_obs'] = sizedim
        #    elif dims == 'numdeps':
        #        tab_dims['nb_depths'] = sizedim

        return lon, lat, [u, v], nb_obs, depth, tab_dims, _FillValue


class SMOC_coords_Loader(Loader):
    """
    SMOC coordinates
    """

    def read_pos(self, filename):
        nc = netCDF4.Dataset(filename, "r")
        variable = "longitude"
        rla_lon = nc.variables[variable][:]
        variable = "latitude"
        rla_lat = nc.variables[variable][:]
        variable = "depth"
        rla_depth = nc.variables[variable][:]
        variable = "mask"
        rla_mask = nc.variables[variable][:, :, :]
        ndepth, nlat, nlon = np.shape(rla_mask)
        # print ("depth {} Lat {} Lon {}".format(ndepth,nlat, nlon))
        mask_tab = np.full((ndepth, nlat, nlon), True)
        mask_tab = np.where(rla_mask == 1, False, mask_tab)
        nc.close()
        return rla_lon, rla_lat, mask_tab, rla_depth
