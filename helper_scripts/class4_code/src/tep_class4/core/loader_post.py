import os
import sys
from getopt import getopt
import configparser as ConfigParser
import numpy as np
import netCDF4
from netCDF4 import Dataset
import logging
from tep_class4.core.supobs import toolkitStats
from tep_class4.core.param import Def_param_bulletin as params
from ast import literal_eval

# Serialization
import pickle as cPickle
from .sydate import SyDate

__author__ = "C.REGNIER"
__version__ = 0.1
__Date__ = "February 2019"


def load_db(_DBFILE_):
    """
    Function to load env files
    inputs:
    _DBFILE_: cfg file
    return:
    db: configparser class
    """
    db = ConfigParser.ConfigParser()
    db.read(os.path.expanduser(_DBFILE_))
    if not db.sections():
        print("Error reading configuration file %s" % _DBFILE_)
        sys.exit(1)
    return db


def read_input_files(params_conf):
    """
    Function to read input layers file
    and serialized colocated positions
    inputs:
    params_conf: dictionnary with input parameters
    returns:
    rla_surface: output surface layer
    rla_layers: output layers
    table_hull: serialized colocated shapes
    """
    # *********************  read depth/level file  *********************
    rla_surface = [0.0, 0.0]
    rla_layers = []
    if params_conf["cexp"] == "PHYS":
        fileid = open(params_conf["cl_layerFile"])
        for line in fileid:
            lt = line.split()
            rla_layers.append(float(lt[0]))
        fileid.close()
    if params_conf["cexp"] == "BIO":
        rla_layers = [0.0, 0.0]
    # *********************  Read serialized file  *********************
    if params_conf["fmt_mask"] == "basin":
        file_pi = params_conf["dir_static"] + "mask_CMEMS_area_all_regions.p"
        # f = open(file_pi, 'rb')
        # table_hull = cPickle.load(f)
        with open(file_pi, "rb") as f:
            table_hull = cPickle.load(f, encoding="latin")

    elif params_conf["fmt_mask"] == "basin_pol":
        file_pi = params_conf["dir_static"] + "mask_CMEMS_area_all_polar_NS_regions.p"
        # f = open(file_pi, 'rb')
        # table_hull = cPickle.load(f)
        with open(file_pi, "rb") as f:
            table_hull = cPickle.load(f, encoding="latin")
    else:
        table_hull = None

    return rla_surface, rla_layers, table_hull


def input_params(arg, log):
    """
    Function to manage input parameters
    inputs:
    arg: stdout from inputs
    log: input logger
    outputs:
    """
    params_conf = {}
    letters = "s:e:c:t:d:b:f:m:p:l"
    keywords = [
        "day1=",
        "day2=",
        "cfg=",
        "type=",
        "fmt_mask=",
        "bilan=",
        "fmt_ascii=",
        "daily_off=",
        "percent=",
        "nblevel=",
        "debug=",
    ]
    cla_opts = getopt(arg, letters, keywords)

    cl_start_date = "YYYYMMDD"
    cl_end_date = "YYYYMMDD"
    ll_date1 = False
    ll_date2 = False
    params_conf["ll_debug"] = False
    params_conf["daily_off"] = "on"
    params_conf["bilan"] = "off"
    params_conf["fmt_ascii"] = "off"
    params_conf["fmt_mask"] = "basin"
    params_conf["percent"] = "off"
    params_conf["nblevel"] = ""
    _DBFILE_ = "./environment.cfg"
    params_conf["missing_value"] = netCDF4.default_fillvals["f4"]

    for o, p in cla_opts[0]:
        if o in ["-s", "--day1"]:
            cl_start_date = p
            ll_date1 = True
        elif o in ["-e", "--day2"]:
            cl_end_date = p
            ll_date2 = True
        elif o in ["-t", "--type"]:
            params_conf["cexp"] = p
        elif o in ["-c", "--cfg"]:
            _DBFILE_ = p.strip()
        elif o in ["-d", "--daily_off"]:
            if p == "None":
                params_conf["daily_off"] = "on"
            else:
                params_conf["daily_off"] = str(p)
        elif o in ["-b", "--bilan"]:
            if p == "None":
                params_conf["bilan"] = "off"
            else:
                params_conf["bilan"] = str(p)
        elif o in ["-f", "--fmt_ascii"]:
            if p == "None":
                params_conf["fmt_ascii"] = "off"
            else:
                params_conf["fmt_ascii"] = str(p)
        elif o in ["-m", "--fmt_mask"]:
            if p == "None":
                params_conf["fmt_mask"] = "basin"
            else:
                params_conf["fmt_mask"] = str(p)
        elif o in ["-p", "--percent"]:
            if p == "None":
                params_conf["percent"] = "off"
            else:
                params_conf["percent"] = str(p)
        elif o in ["-p", "--nblevel"]:
            if p == "None":
                params_conf["nblevel"] = ""
            else:
                params_conf["nblevel"] = str(p)
                print(f"Level :: {p}")
        elif o in ["-l", "--debug"]:
            debug = str(p)
            if p == "0":
                log.setLevel(20)
            elif p == "1":
                log.setLevel(10)
            elif p == "True":
                log.setLevel(logging.DEBUG)
            elif p == "False":
                log.setLevel(logging.INFO)
            else:
                log.error("Not known : debug option should be 0 or 1")

    log.debug(f"{params_conf}")

    # *********************  Define input date / fmt date  *********************

    if not ll_date1 and not ll_date2:
        cl_start_date, cl_end_date = toolkitStats.datenow()
    params_conf["ll_arc"] = False
    params_conf["ll_ant"] = False
    if "arc" in params_conf["fmt_mask"]:
        params_conf["ll_arc"] = True
    if "ant" in params_conf["fmt_mask"]:
        params_conf["ll_ant"] = True
    params_conf["cla_dates"] = toolkitStats.char_period(cl_start_date, cl_end_date)
    params_conf["cl_start_jdate"] = str(SyDate(cl_start_date).tojulian())
    params_conf["cl_end_jdate"] = str(SyDate(cl_end_date).tojulian())
    params_conf["cl_start_date"] = cl_start_date
    params_conf["cl_end_date"] = cl_end_date

    # *********************  Load config file  *********************

    params_conf["db"] = load_db(_DBFILE_)
    params_conf["config_model"] = params_conf["db"].get("PARAMS", "CONFIGMDL").upper()
    params_conf["config_obs"] = params_conf["db"].get("PARAMS", "CONFIGOBS").upper()
    try:
        params_conf["list_TAC_profiles"] = literal_eval(params_conf["db"].get("PARAMS", "list_TAC_profiles"))
    except Exception as e:
        print(f"Missing option list_TAC_profiles {e}")
    params_conf["cgrid"] = params_conf["db"].get("NETCDF", "grid")
    params_conf["dirNCin"] = params_conf["db"].get("input_dir", "DIR_IN")
    params_conf["dirNCout"] = params_conf["db"].get("output_dir", "DIR_OUT_NC")
    params_conf["dirBOXout"] = params_conf["db"].get("output_dir", "DIR_OUT_BOX")
    params_conf["path_out"] = params_conf["db"].get("output_dir", "DIR_OUT")
    params_conf["dirASCIIout"] = params_conf["db"].get("output_dir", "DIR_OUT_ASCII")
    params_conf["cl_layerFile"] = params_conf["db"].get("LAYER", "layerFile" + params_conf["nblevel"])
    params_conf["prefix"] = params_conf["db"].get("INPUT", "PREFIX1")
    params_conf["cl_confres"] = params_conf["db"].get("NETCDF", "MyOConfig")
    if params_conf["percent"] == "on":
        params_conf["cl_conf_percent"] = params_conf["db"].get("NETCDF", "MyConfigPercent")
    params_conf["cl_contact"] = params_conf["db"].get("NETCDF", "CONTACT")
    params_conf["cl_institution"] = params_conf["db"].get("NETCDF", "INSTITUTION")
    params_conf["cl_ref"] = params_conf["db"].get("NETCDF", "REF")
    params_conf["cl_fmt"] = params_conf["db"].get("NETCDF", "FMT")
    params_conf["dir_static"] = params_conf["db"].get("NETCDF", "DIR_MASK_IN")
    params_conf["cla_metricname"] = params.cla_metricname2
    params_conf["metric_percent"] = params.cla_metricname_Q
    params_conf["cmems_basin"] = params.liste_zone

    if not os.path.exists(params_conf["dirNCout"]):
        os.makedirs(params_conf["dirNCout"])

    return params_conf, log


def load_param(db_file):

    params_conf = {}
    params_conf["db"] = load_db(db_file)
    params_conf["ZONE"] = params_conf["db"].get("GLOBAL", "ZONE")
    params_conf["SUFFIX"] = params_conf["db"].get("GLOBAL", "SUFFIX")
    params_conf["var_u"] = params_conf["db"].get("GLOBAL", "var_u")
    params_conf["var_v"] = params_conf["db"].get("GLOBAL", "var_v")
    params_conf["nb_ptsinterp"] = params_conf["db"].get("GLOBAL", "nb_ptsinterp")
    params_conf["system"] = params_conf["db"].get("GLOBAL", "system")
    params_conf["systemh"] = params_conf["db"].get("GLOBAL", "systemh")
    params_conf["nb_fcst"] = params_conf["db"].get("GLOBAL", "nb_fcst")
    params_conf["nb_fcst_smoc"] = params_conf["db"].get("GLOBAL", "nb_fcst_smoc")
    params_conf["nb_pers"] = params_conf["db"].get("GLOBAL", "nb_pers")
    params_conf["model"] = params_conf["db"].get("GLOBAL", "model")
    params_conf["DIRDATA"] = params_conf["db"].get("input_dir", "DIRDATA")
    params_conf["DIRDATA_filtr"] = params_conf["db"].get("input_dir", "DIRDATA_filtr")
    params_conf["TYPE"] = params_conf["db"].get("input_dir", "TYPE")
    params_conf["dirdata_smoc"] = params_conf["db"].get("input_dir", "dirdata_smoc")
    params_conf["dirdata_mod"] = params_conf["db"].get("input_dir", "dirdata_mod")
    params_conf["dirdata_mod_weekly"] = params_conf["db"].get("input_dir", "dirdata_mod_weekly")
    params_conf["path_data"] = params_conf["db"].get("output_dir", "path_data")
    params_conf["DIR_INTERP"] = params_conf["db"].get("output_dir", "DIR_INTERP")
    params_conf["dirout"] = params_conf["db"].get("output_dir", "dirout")
    params_conf["coordinates"] = params_conf["db"].get("static", "coordinates")
    params_conf["coordinates_reg"] = params_conf["db"].get("static", "coordinates_reg")
    params_conf["coordinates_clim"] = params_conf["db"].get("static", "coordinates_clim")
    try:
        params_conf["coordinates_zgr"] = params_conf["db"].get("static", "coordinates_zgr")
    except:
        params_conf["coordinates_zgr"] = ""
    try:
        params_conf["coordinates_hgr"] = params_conf["db"].get("static", "coordinates_hgr")
    except:
        params_conf["coordinates_hgr"] = ""

    return params_conf


# def stat_mod_obs(stat, num_col, amask, nobs, mean_obs, rms_obs, mean_prod, rms_prod, rmse, cor, tab_prof, yout):
def stat_mod_obs(stat, num_col, amask, nobs, mean_obs, rms_obs, mean_prod, rms_prod, rmse, cor):  # , tab_prof, yout):

    # if num_col == 0:
    #    for ncol in range(yout):
    #        print(f'{num_col=}')
    #        print(f'{ncol=}')
    #        stat[ncol, :, 1, amask] = mean_obs
    #        stat[ncol, :, 2, amask] = np.square(rms_obs)
    stat[num_col, :, 1, amask] = mean_obs
    stat[num_col, :, 2, amask] = np.square(rms_obs)
    stat[num_col, :, 0, amask] = nobs
    stat[num_col, :, 3, amask] = mean_prod
    stat[num_col, :, 4, amask] = np.square(rms_prod)
    stat[num_col, :, 5, amask] = np.square(rmse)
    stat[num_col, :, 6, amask] = cor

    num_col += 1
    return stat, num_col


# -----------------------------------------------------------------------------------------------------------------


def stat_obs_glob(stat, yout, ilayer, mean, rms):
    for ncol in range(yout):
        stat[ncol, ilayer, 1] = mean
        stat[ncol, ilayer, 2] = np.square(rms)
    return stat


# -----------------------------------------------------------------------------------------------------------------


def stat_mod_glob(stat, num_col, ilayer, nobs, mean, rms, rmse, cor, mae, VE):
    stat[num_col, ilayer, 0] = nobs
    stat[num_col, ilayer, 3] = mean
    stat[num_col, ilayer, 4] = np.square(rms)
    stat[num_col, ilayer, 5] = np.square(rmse)
    stat[num_col, ilayer, 6] = cor
    stat[num_col, ilayer, 7] = mae
    stat[num_col, ilayer, 8] = VE
    num_col += 1

    return stat, num_col


# -----------------------------------------------------------------------------------------------------------------


def stat_mod_glob2(stat, num_col, ilayer, nobs, mean_obs, rms_obs, mean, rms, rmse, cor, mae, VE):
    stat[num_col, ilayer, 0] = nobs
    stat[num_col, ilayer, 1] = mean_obs
    stat[num_col, ilayer, 2] = np.square(rms_obs)
    stat[num_col, ilayer, 3] = mean
    stat[num_col, ilayer, 4] = np.square(rms)
    stat[num_col, ilayer, 5] = np.square(rmse)
    stat[num_col, ilayer, 6] = cor
    stat[num_col, ilayer, 7] = mae
    stat[num_col, ilayer, 8] = VE
    num_col += 1

    return stat, num_col


# -----------------------------------------------------------------------------------------------------------------
