# Module with utilities
import numpy as np
import sys
import pandas as pd
import datetime as dt
import os, sys
import getopt
import shlex, subprocess
import xarray as xr
from datetime import datetime
import traceback
import logging
from .loader_post import load_db
from datetime import datetime, timedelta
import yaml
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

__version__ = 0.2
__date__ = "Octobre 2021"
__author__ = "C.REGNIER"


class Parameters:
    def __init__(self, config_file):
        self.parameters = self.load_parameters(config_file)

    def load_parameters(self, config_file):
        with open(config_file, "r") as file:
            return yaml.safe_load(file)

    def get_parameter(self, category, key):
        return self.parameters.get(category, {}).get(key)

    def get_category(self, category):
        return self.parameters.get(category, {})

    def set_parameter(self, category, key, value):
        if category in self.parameters:
            self.parameters[category][key] = value
        else:
            self.parameters[category] = {key: value}


def save_stats_to_netcdf(results_stats, lead_times, depth_labels, output_path):

    ds_out = xr.Dataset()

    for var, stats in results_stats.items():
        tab_rmsd = stats["rmsd"]
        tab_cumul_sq_diff = stats["sq_diff_cumul"]
        tab_n_obs = stats["nb_obs"]

        if var in ["SST", "SLA", "aice", "DRIFTER_filtr", "DRIFTER"]:  # 2D vars
            dims = ["leadtime"]
            coords = {"leadtime": lead_times}
            ds_out[f"{var}_rmsd"] = xr.DataArray(tab_rmsd, dims=dims, coords=coords)
            ds_out[f"{var}_sq_diff_cumul"] = xr.DataArray(tab_cumul_sq_diff, dims=dims, coords=coords)
            ds_out[f"{var}_nb_obs"] = xr.DataArray(tab_n_obs, dims=dims, coords=coords)

        else:  # 3D vars like temperature, salinity (index 0, 1)
            dims = ["leadtime", "depth_bin"]
            variable_names = ["temperature", "salinity"]
            coords = {
                "leadtime": lead_times,
                "depth_bin": depth_labels,  # You could also use mid-depths or similar
            }
            for ind_var, variable in enumerate(variable_names):
                ds_out[f"{variable}_rmsd"] = xr.DataArray(tab_rmsd[:, :, ind_var], dims=dims, coords=coords)
                ds_out[f"{variable}_sq_diff_cumul"] = xr.DataArray(
                    tab_cumul_sq_diff[:, :, ind_var], dims=dims, coords=coords
                )
                ds_out[f"{variable}_nb_obs"] = xr.DataArray(tab_n_obs[:, :, ind_var], dims=dims, coords=coords)
    ds_out.to_netcdf(output_path)


def save_all_metrics(file_list, list_date, fileout):
    # Convert to datetime64
    time_counter = pd.to_datetime(list_date, format="%Y%m%d")
    # Convert to Julian days since 1950-01-01
    dates_dt = pd.to_datetime(list_date, format="%Y%m%d")
    ref_date = pd.Timestamp("1950-01-01")
    julian_days = (dates_dt - ref_date).days
    # Open and concatenate
    ds = xr.open_mfdataset(file_list, concat_dim="time", combine="nested", parallel=True)

    # Add time_counter to the dataset
    ds = ds.assign_coords(time=("time", time_counter))
    # Optionally save
    ds.to_netcdf(fileout)


def save_metrics_to_netcdf(tab_rmsd, tab_cumul_sq_diff, tab_n_obs, lead_times, output_path):
    """
    Save computed metrics to a NetCDF file with lead time as coordinate.

    Parameters
    ----------
    tab_rmsd : np.ndarray
        Array of RMSD values.
    tab_cumul_sq_diff : np.ndarray
        Array of cumulative squared differences.
    tab_n_obs : np.ndarray
        Array of number of valid observations.
    lead_times : list or np.ndarray
        The lead times (coordinate).
    output_path : str
        Path to save the output NetCDF file.
    """

    da_rmsd = xr.DataArray(data=tab_rmsd, coords={"leadtime": lead_times}, dims=["leadtime"], name="rmsd")

    da_cumul = xr.DataArray(
        data=tab_cumul_sq_diff, coords={"leadtime": lead_times}, dims=["leadtime"], name="cumul_squared_diff"
    )

    da_nobs = xr.DataArray(data=tab_n_obs, coords={"leadtime": lead_times}, dims=["leadtime"], name="n_obs")

    # Combine into one Dataset
    ds = xr.Dataset({"rmsd": da_rmsd, "cumul_squared_diff": da_cumul, "n_obs": da_nobs})

    # Save to NetCDF
    ds.to_netcdf(output_path)


def mkdir(path):
    """
    Create a directory, and parents if needed
    """
    if not os.path.exists(path):
        os.makedirs(path)


def yaml_loader(path):
    try:
        with open(path, "r") as file:
            patterns = yaml.safe_load(file)
        return patterns
    except Exception as e:
        raise (f"Failed to load patterns from {path}: {e}")
        sys.exit(1)


def loader(options):
    """
    Input parameters settings
    """

    tab_options = {}
    list_typemod = ["BIO", "PHYS"]
    tab_options["typemod"] = list_typemod
    list_config = [
        "FOAM",
        "GIOPS",
        "CMCC",
        "CGOFS",
        "BLK",
        "PSY3V3R3",
        "HYCOM_RTOFS",
        "PSY4V2R2",
        "PSY4V3R1",
        "BIOMER4",
        "FREEBIORYS2",
        "NERSC",
        "BIOMER4V2",
        "ENSMEAN",
        "GLO12",
        "GLO12V4",
        "GLO4",
        "FOAM12",
        "GLORYS12V1",
        "GLO12V5-130",
        "GREP-mean",
        "GLO12V5-140",
        "GLO12V5-150",
        "BIO4GLO12",
        "XIHE",
        "GLONET",
        "GLONET14",
        "GLO12V4_PGS",
    ]
    tab_options["config"] = list_config
    list_mask = ["basin", "basin_pol", "box", "bins2d", "bins2d_arc", "bins2d_ant", "bins4d"]
    tab_options["fmt_mask"] = list_mask
    list_bilan, list_ascii, list_dailyoff, list_percent = (["on", "off"],) * 4
    tab_options["bilan"], tab_options["fmt_ascii"], tab_options["daily_off"], tab_options["percent"] = (
        ["on", "off"],
    ) * 4
    tab_options["day1"] = []
    tab_options["day2"] = []
    list_nblevel = ["50", "75", "23"]
    tab_options["nb_levels"] = list_nblevel
    tab_conf = {}
    list_options = [
        "config",
        "day1",
        "day2",
        "typemod",
        "fmt_mask",
        "bilan",
        "fmt_ascii",
        "daily_off",
        "percent",
        "nb_levels",
    ]
    ## Initialize dictionnary with None values
    tab_conf = dict.fromkeys(list_options)
    tab_conf["debug"] = False
    try:
        # the keyword
        cla_opts = getopt.getopt(options, ["empid="])
        if len(cla_opts[1]) < 4 or len(cla_opts[1]) > 10:
            usage(len(cla_opts[1]))
            sys.exit()
        for ind, var in enumerate(range(len(cla_opts[1]))):
            if cla_opts[1][ind] == "-l":
                tab_conf["debug"] = True
            else:
                tab_conf[list_options[ind]] = cla_opts[1][ind]
                generic_test(tab_options[list_options[ind]], cla_opts[1][ind], list_options[ind])
        test_days(tab_conf["day1"], tab_conf["day2"])
        if len(cla_opts[1]) == 4:
            usage1()
        if len(cla_opts[1]) == 5:
            usage2()
        if len(cla_opts[1]) == 6:
            usage3()
        if len(cla_opts[1]) == 7:
            usage4()
        if len(cla_opts[1]) == 8:
            usage5()
        if len(cla_opts[1]) == 9:
            usage6()
        # if len(cla_opts[1]) == 10:
        #    usage7()

    except getopt.GetoptError as err:
        raise (f"Error {err} Missing options {keywords}")

    tab_conf["pbsworkdir"] = os.path.dirname(os.path.realpath(__file__))
    _DBFILE_ = f"{tab_conf['pbsworkdir']}/../cfg/global_params.cfg"
    _DBFILE2_ = f"{tab_conf['pbsworkdir']}/../cfg/environment_{tab_conf['config']}.cfg"
    db = load_db(_DBFILE_)
    db2 = load_db(_DBFILE2_)
    tab_conf["dirout"] = db2.get("output_dir", "DIR_OUT")
    tab_conf["dir_bin"] = db.get("PARAMS", "py_env") + "bin/"
    tab_conf["sub"] = f"{tab_conf['dirout']}/sub/"
    mkdir(tab_conf["sub"])

    return tab_conf


def generic_test(list_var, var, type_param):
    """
    Generic test for values
    """
    if type_param == "day1" or type_param == "day2":
        if not var:
            sys.exit(f"{type_param} is missing need {list_var}")
        isValiDate(var, type_param)
    else:
        if not var or var not in list_var:
            sys.exit(f"{type_param} is missing need {list_var}")


def convdatetime(var):
    """
    Convert date YYYYMMDD in a datetime value
    """
    try:
        year = var[0:4]
        month = var[4:6]
        day = var[6:8]
        datetime_val = datetime(int(year), int(month), int(day))
    except Exception as e:
        raise ValueError(f"Var {var} not a valid date for datetime , error {traceback.format_exc()}")
    return datetime_val


def test_days(day1, day2):
    """
    Test if a date2 > date 1
    """
    if convdatetime(day2) < convdatetime(day1):
        sys.exit(f"{day2} must be > at {day1}")


def isValiDate(var, type_param):
    """
    Test if a date is valid or not
    """
    isValiDate = True
    try:
        datevalue = convdatetime(var)
    except ValueError:
        isValiDate = False
        sys.exit(f"Value {var} for {type_param} not a valid date")

    return isValiDate


def generate_date_list(start, end):
    # Convert strings to datetime objects
    start_date = datetime.strptime(start, "%Y%m%d")
    end_date = datetime.strptime(end, "%Y%m%d")
    # Generate list of dates between start_date and end_date
    date_list = [(start_date + timedelta(days=x)).strftime("%Y%m%d") for x in range((end_date - start_date).days + 1)]
    return date_list


def generate_date_list_week(start, end, delta=7):
    # Convert strings to datetime objects
    start_date = datetime.strptime(start, "%Y%m%d")
    end_date = datetime.strptime(end, "%Y%m%d")
    # Generate list of weekly dates
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=delta)
    return date_list


def create_launcher_multi(tab_conf, nbtasks="30", timecounter="06:00:00", batch="#SBATCH", nb_procs="2"):
    ## Create qsub job
    if tab_conf["fmt_mask"] == "basin":
        typejob = "timeserie"
    elif tab_conf["bilan"] != None:
        typejob = "bilan"
    elif tab_conf["percent"] != None:
        typejob = "percent"
    else:
        typejob = "others"
    _DBFILE_ = f"{tab_conf['pbsworkdir']}/../cfg/environment_{tab_conf['config']}.cfg"
    EXE = os.path.join(tab_conf["pbsworkdir"], "../exe_compute_stats_CMEMS_areas_multiprocess.py")
    dir_bin = tab_conf["dir_bin"]
    day1 = tab_conf["day1"]
    day2 = tab_conf["day2"]
    datatype = tab_conf["typemod"]
    fmt_mask = tab_conf["fmt_mask"]
    bilan = tab_conf["bilan"]
    fmt_ascii = tab_conf["fmt_ascii"]
    daily_off = tab_conf["daily_off"]
    percent = tab_conf["percent"]
    nblevel = tab_conf["nb_levels"]
    debug = tab_conf["debug"]
    nb_process = int(nb_procs) * int(nbtasks)

    Jobfile = (
        tab_conf["sub"]
        + "Class4_post_"
        + typejob
        + "_"
        + tab_conf["config"]
        + "_"
        + tab_conf["day1"]
        + "_"
        + tab_conf["day2"]
        + ".sub"
    )
    obFichier = open(Jobfile, "w+")
    obFichier.write("#!/usr/bin/env python" + "\n")
    obFichier.write(batch + " -J " + "Class4_post_" + typejob + "_" + tab_conf["day1"] + "_" + tab_conf["day2"] + "\n")
    file_out = (
        f"{tab_conf['sub']}CLASS4_{typejob}_postjob_{tab_conf['config']}_{tab_conf['day1']}_{tab_conf['day2']}_%j.o"
    )
    file_err = (
        f"{tab_conf['sub']}CLASS4_{typejob}_postjob_{tab_conf['config']}_{tab_conf['day1']}_{tab_conf['day2']}_%j.e"
    )
    obFichier.write(f"{batch} -e {file_err} \n")
    obFichier.write(f"{batch} -o {file_out} \n")
    if nb_procs == "12":
        obFichier.write(batch + " -p multi12" + "\n")
        obFichier.write(batch + " -q qosmulti12" + "\n")
    else:
        obFichier.write(batch + " -p multi" + "\n")
        obFichier.write(batch + " -q qosmulti" + "\n")
    obFichier.write(batch + " -N " + nb_procs + "\n")
    obFichier.write(batch + " --exclusive" + "\n")
    obFichier.write(batch + " --ntasks-per-node=" + nbtasks + "\n")
    obFichier.write(batch + " --time=" + timecounter + "\n")
    obFichier.write("HDF5_USE_FILE_LOCKING = False" + "\n")
    obFichier.write("import os" + "\n")
    codexe = (
        tab_conf["dir_bin"]
        + "mpirun -np "
        + str(nb_process)
        + " "
        + tab_conf["dir_bin"]
        + f"python {EXE} --day1={day1} --day2={day2} --cfg={_DBFILE_} --type={datatype} --fmt_mask={fmt_mask} --bilan={bilan} --fmt_ascii={fmt_ascii} --daily_off={daily_off} --percent={percent} --nblevel={nblevel} --debug={debug}"
    )
    obFichier.write("codexe =" + "'" + codexe + "'" + "\n")
    obFichier.write("os.system(codexe)" + "\n")
    obFichier.close()

    return Jobfile


def launch_sbatch(Jobfile):

    SUBEXE = "/usr/bin/sbatch"
    code_launch = f"{SUBEXE} {Jobfile}"
    args = shlex.split(code_launch)
    try:
        subprocess.run(args)
    except:
        print(f"Launch {Jobfile} failed")
        sys.exit(1)


def usage1():
    print(" *************************** DEFAULT CASE **************************")
    print(" ")
    print("     Case 1 Save stats computed at the daily time step over CMEMS areas    ")
    print("                     on netCDF format file                          ")
    print(" ")
    print(" ****************************************************************** ")


def usage2():
    print(" ")
    print(" ------------------------------------------------------------------------------")
    print("     -->   Case 2 Define the type of mask (default case : CMEMS areas)  <--           ")
    print("                Available masks : basin/basin_pol/box/bins2d/bins4d            ")
    print(" ------------------------------------------------------------------------------")
    print(" ")


def usage3():
    print(" ")
    print(" ------------------------------------------------------------------------------")
    print("     -->   Case 3 Define the type of mask (default case : CMEMS areas)  <--           ")
    print("                Available masks : basin/box/bins2d/bins4d                      ")
    print("                                     &                                         ")
    print("        -->   Activate/Deactivate save of stats computed over  <--             ")
    print("                       the whole considered period                             ")
    print(" ------------------------------------------------------------------------------")
    print(" ")


def usage4():
    print(" ")
    print(" ------------------------------------------------------------------------------")
    print("     -->   Case 4 Define the type of mask (default case : CMEMS areas)  <--           ")
    print("                Available masks : basin/box/bins2d/bins4d                      ")
    print("                                     &                                         ")
    print("        -->   Activate/Deactivate save of stats computed over  <--             ")
    print("                       the whole considered period                             ")
    print("                                     &                                         ")
    print("            -->   Activate/Deactivate save of stats on   <--                   ")
    print("                          ASCII format file                                    ")
    print(" ------------------------------------------------------------------------------")
    print(" ")


def usage5():
    print(" ")
    print(" ------------------------------------------------------------------------------")
    print("     -->   Case 5 Define the type of mask (default case : CMEMS areas)  <--           ")
    print("                Available masks : basin/box/bins2d/bins4d                      ")
    print("                                     &                                         ")
    print("        -->   Activate/Deactivate save of stats computed over  <--             ")
    print("                       the whole considered period                             ")
    print("                                     &                                         ")
    print("            -->   Activate/Deactivate save of stats on   <--                   ")
    print("                          ASCII format file                                    ")
    print("                                     &                                         ")
    print("     -->   Activate/Deactivate save of stats at the daily time step  <--       ")
    print(" ------------------------------------------------------------------------------")
    print(" ")


def usage6():
    print(" ")
    print(" ------------------------------------------------------------------------------")
    print("     -->   Case 6 Define the type of mask (default case : CMEMS areas)  <--           ")
    print("                Available masks : basin/box/bins2d/bins4d                      ")
    print("                                     &                                         ")
    print("        -->   Activate/Deactivate save of stats computed over  <--             ")
    print("                       the whole considered period                             ")
    print("                                     &                                         ")
    print("            -->   Activate/Deactivate save of stats on   <--                   ")
    print("                          ASCII format file                                    ")
    print("                                     &                                         ")
    print("     -->   Activate/Deactivate save of stats at the daily time step  <--       ")
    print("                                     &                                         ")
    print("     -->   Activate bias and percentile computation for temperature  <--       ")
    print("                             and salinity                                      ")
    print(" ------------------------------------------------------------------------------")
    print(" ")


def usage7():
    print(" ")
    print(" ------------------------------------------------------------------------------")
    print("     -->   Case 7 Define the type of mask (default case : CMEMS areas)  <--           ")
    print("                Available masks : basin/box/bins2d/bins4d                      ")
    print("                                     &                                         ")
    print("        -->   Activate/Deactivate save of stats computed over  <--             ")
    print("                       the whole considered period                             ")
    print("                                     &                                         ")
    print("            -->   Activate/Deactivate save of stats on   <--                   ")
    print("                          ASCII format file                                    ")
    print("                                     &                                         ")
    print("     -->   Activate/Deactivate save of stats at the daily time step  <--       ")
    print("                                     &                                         ")
    print("     -->   Activate bias and percentile computation for temperature  <--       ")
    print("                             and salinity                                      ")
    print("                                     &                                         ")
    print("     -->   Change the number of considered levels to compute profile  <--      ")
    print("                             T and S statistics                                ")
    print(" ------------------------------------------------------------------------------")
    print(" ")


def usage(nb_args):
    print("              Incorrect number of Arguments")
    print(f"   nb of arguments {nb_args}                           <-- ")
    print("    arguments required                                  <-- ")
    print("    Value1 = Config                                         ")
    print("    Value2 = Day min                                        ")
    print("    Value3 = Day max                                        ")
    print("    Value4 = Type of data: BIO/PHYS")
    print("                                                            ")
    print(" -->              Optional argument                     <-- ")
    print("    Value5 = Type of mask: basin/box/bins2d/bins4d          ")
    print("    Value6 = Save global stats: on/off                      ")
    print("    Value7 = Save stats on ASCII fmt files: on/off          ")
    print("    Value8 = Save daily time step stats: on/off             ")
    print("    Value9 = Compute misfit and percentile (profile): on/off")
    print("    Value10 = Number of considered level profile            ")


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


def echeance(fcst, nf):
    """
    Find lead time
    """
    num_ech = np.where(fcst == nf)[0][0]
    if nf == 0:
        name_ech = "Hindcast"
    else:
        name_ech = "Forecast " + str(nf) + " hours"
    return num_ech, name_ech


def plot(iobs, rla_lonobs, rla_latobs, region, log):
    """
    Function to plot selected data
    """
    log.info("Plot region %s " % (region))
    area_names_arc = [
        "North Pole",
        "Queen Elis Is",
        "Beaufort Sea",
        "Chuckchi Sea",
        "Siberian Sea",
        "Laptev Sea",
        "Kara Sea",
        "Barents Sea",
        "Greenland Sea",
        "Stheast Greenland",
        "Bafin Bay",
        "Hudson Bay",
        "Labrador Sea",
        "Bering Sea",
        "Okhotsk Sea",
        " Baltic Sea",
    ]
    area_names_ant = [
        "Weddel Sea",
        "Southern Atlantic Ocean",
        "Southern Indian Ocean",
        "Southern West Pacific Ocean",
        "Southern East Pacific Ocean",
        "Ross Sea",
        "Admundsen Sea",
        "Bellingshausen Sea",
    ]

    rla_lonobs = np.where(rla_lonobs > 180, rla_lonobs - 360, rla_lonobs)
    value = rla_lonobs[:].copy()
    value[:] = 1.0
    lon_inter = 20
    lat_inter = 20
    zone = "GLO"
    proj = "cyl_pac"
    if region in area_names_arc:
        proj = "npstere"
        zone = "Arctic2"
    if region in area_names_ant:
        zone = "Antarctic"
        proj = "spstere"
    # colormap = plt.cm.coolwarm
    # titre = 'Points in '+region+" with matplotlib meth"
    # outputfile = "Selected_points_"+region+".png"
    # colormap = plt.cm.coolwarm
    # cmin = 0
    # cmax = 1
    # dpi = 75
    # font = 16
    # dot_size = 2.0
    # log.info("Plot Map")
    # Plot_map(rla_lonobs[iobs].tolist(), rla_latobs[iobs].tolist(), zone, lon_inter, lat_inter, proj,\
    #        dpi, font).scatter_val(value[iobs].tolist(), colormap, cmin, cmax, dot_size, titre, outputfile)


def compute_last_day_month(y, m):
    return (dt.date(y + int(m / 12), m % 12 + 1, 1) - dt.date(y, m, 1)).days


def find_index_month(date):
    dateval = pd.to_datetime(date)
    index = dateval.month
    return index


from datetime import datetime, timedelta


def generate_dateslist(daterun: str, start_date: str = "20250101") -> list:
    """
    Generate a list of dates from start_date to daterun (inclusive).

    Parameters:
    - daterun (str): End date in format %Y%m%d (e.g., '20250430').
    - start_date (str): Start date in format %Y%m%d. Default is '20250101'.

    Returns:
    - list of datetime.date objects from start_date to daterun.
    """
    try:
        date_start = datetime.strptime(start_date, "%Y%m%d").date()
        date_end = datetime.strptime(daterun, "%Y%m%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

    if date_end < date_start:
        raise ValueError("daterun must be greater than or equal to start_date")

    delta = date_end - date_start
    return [date_start + timedelta(days=i) for i in range(delta.days + 1)]


from datetime import datetime, timedelta
from typing import List


def generate_nbdateslist(daterun: str, nb_dates: int, direction: str = "forward") -> List[str]:
    """
    Generate a list of date strings from daterun in forward or backward direction.

    Parameters:
    - daterun (str): Starting date in format %Y%m%d (e.g., '20250430').
    - nb_dates (int): Number of dates to generate (including daterun).
    - direction (str): 'forward' (default) or 'backward'.

    Returns:
    - List[str]: List of date strings in format %Y%m%d.
    """
    try:
        start_date = datetime.strptime(daterun, "%Y%m%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid daterun format: {e}")

    if nb_dates <= 0:
        raise ValueError("nb_dates must be a positive integer")

    if direction == "forward":
        return [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range(nb_dates)]
    elif direction == "backward":
        return [(start_date - timedelta(days=i)).strftime("%Y%m%d") for i in reversed(range(nb_dates))]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")


def create_dict_list(date1, date2, lead_time, lead_int):
    list_keys = {}
    list_keys_values = []
    list_uniq = ["best_estimate", "climatology", "bathymetrie"]
    date_start = datetime.strptime(date1, "%Y%m%d").date()
    date_end = datetime.strptime(date2, "%Y%m%d").date()
    # date_start = datetime.strptime(date1, "%Y%M%D").date()
    # date_end = datetime.strptime(date2, "%Y%M%D").date()
    delta = date_end - date_start
    for i in range(delta.days + 1):
        dateval = date_start + timedelta(days=i)
        dateval = dateval.strftime("%Y%m%d")
        for lead in lead_time:
            if lead in list_uniq:
                list_int = [0]
            else:
                list_int = lead_int
            for int_val in list_int:
                keyvalue = f"{dateval}-{lead}-{int_val}"
                list_keys_values.append(keyvalue)
    list_keys["keyvalues"] = list_keys_values
    return list_keys


def create_dict(date1, date2, lead_time, lead_int):
    list_keys = {}
    valeurj = date1
    list_keys_values = []
    list_uniq = ["best_estimate", "climatology", "bathymetrie"]
    while valeurj >= date1 and valeurj <= date2:
        dateval = str(valeurj)
        for lead in lead_time:
            if lead in list_uniq:
                list_int = [0]
            else:
                list_int = lead_int
            for int_val in list_int:
                keyvalue = f"{dateval}-{lead}-{int_val}"
                list_keys_values.append(keyvalue)
        valeurj = valeurj.goforward(1)
    list_keys["keyvalues"] = list_keys_values
    return list_keys


def split_year_into_months(year):
    months = []
    for month in range(1, 13):
        first_day = datetime(year, month, 1)
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        months.append((first_day.strftime("%Y%m%d"), last_day.strftime("%Y%m%d")))
    return months


def date_range(start_date, end_date, increment, period):
    result = []
    nxt = start_date
    delta = relativedelta(**{period: increment})
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    return result


def get_quarters(base_year):
    quarters = {}
    for quarter in range(1, 5):
        # Determine the start and end dates for each quarter
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        print(start_month, end_month)
        # Handle the case where end_month exceeds 12
        if end_month > 12:
            end_month -= 12
            base_year += 1

        # Determine the last day of the last month in the quarter
        last_day_of_last_month = (datetime(base_year, end_month % 12 + 1, 1) - timedelta(days=1)).day
        start_date = datetime(base_year, start_month, 1)
        end_date = datetime(base_year, end_month, last_day_of_last_month)
        # Get the first letter of each month in uppercase
        start_month_name = start_date.strftime("%b").upper()[:1]
        mid_month_name = (end_date + timedelta(days=-46)).strftime("%b").upper()[:1]
        end_month_name = end_date.strftime("%b").upper()[:1]
        # Use the first letter of each month as keys in the dictionary
        key = f"{start_month_name}{mid_month_name}{end_month_name}"
        quarters[key] = (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

    return quarters


def process_netCDF_stats(system, fileName, config_file, resol):
    params = Parameters(config_file)
    ## Read grid
    if resol == "bins2d":
        input_grid = params.get_category("grid_input_2deg")
    elif resol == "bins4d":
        input_grid = params.get_category("grid_input_4deg")
    # input_grid = params.parameters['grid_input']
    # list_forecasts = params.parameters['list_forecasts']
    list_forecasts = params.get_category("list_forecasts")
    nc_file = netCDF4.Dataset(input_grid, "r")
    lon = nc_file.variables["longitude"][:, :]
    lat = nc_file.variables["latitude"][:, :]
    mask = nc_file.variables["mask"][:, :]
    nlat, nlon = np.shape(lon)
    nc_file.close()
    nc_file2 = netCDF4.Dataset(str(fileName), "r")
    depths = nc_file2.variables["depths"][:]
    forecasts = nc_file2.variables["forecasts"][:]
    list_variables = filter(lambda x: x.startswith("stats"), nc_file2.variables.keys())
    list_skill = filter(lambda x: x.startswith("skill"), nc_file2.variables.keys())
    list_stats = nc_file2.variables["metric_names"][:]
    area_names = pull_names(list_stats)
    # key_value_pairs = [(key, list_forecasts[key]) for key in list_forecasts if list_forecasts[key] in forecasts]
    key_value_pairs = [key for key in list_forecasts if list_forecasts[key] in forecasts]

    def process_area_stat(stat_name, variable_data):
        # print(f"{stat_name=}")
        # print(f"{frcst=}")
        # print(f"{prof=}")
        # print(f"{stat=}")
        stat_data = nc_file2.variables[variable_data][frcst, prof, stat, :]
        return np.ma.masked_invalid(stat_data)

    results = {}
    results[system] = {}
    for variable in list_variables:
        variable_tmp = variable.replace("stats_", "")
        results[system][variable_tmp] = {}
        # variable_results = {}
        # print(variable)
        # np.empty((len(depths), len(list_forecasts), len(area_names)), dtype=object)
        list_dimensions = nc_file2.variables[variable].dimensions
        if "depths" in list_dimensions:
            nb_depths = len(nc_file2.dimensions["depths"])
        else:
            nb_depths = 1

        for prof in range(nb_depths):
            profvalue = str(depths[prof])
            if nb_depths > 1:
                if prof > 0:
                    bprofvalue = f"{int(depths[prof-1])}-{int(depths[prof])}m"
                else:
                    bprofvalue = f"0-{int(depths[prof])}m"
            else:
                bprofvalue = "0m"
            ## Add 1 value case for the surface
            results[system][variable_tmp][bprofvalue] = {}

            for frcst, fcst_value in enumerate(key_value_pairs):
                results[system][variable_tmp][bprofvalue][fcst_value] = {}
                for stat in range(len(area_names)):
                    stat_name = str(area_names[stat])
                    stat_data = process_area_stat(stat_name, variable)
                    if stat_name.startswith("number of data values"):
                        nb_obs = stat_data
                    elif stat_name.startswith("mean of reference"):
                        mean_obs = process_area_stat("mean of reference", variable)
                    elif stat_name.startswith("mean of product"):
                        mean_mod = process_area_stat("mean of product", variable)
                    elif stat_name.startswith("mean squared error"):
                        msd = process_area_stat("mean squared error", variable)
                    elif stat_name.startswith("anomaly correlation"):
                        anomaly_corr = stat_data
                    elif stat_name.startswith("mean squared reference"):
                        MS_obs = stat_data
                    elif stat_name.startswith("variance of reference"):
                        VAR_obs = stat_data
                    elif stat_name.startswith("mean absolute error"):
                        MAE = stat_data
                    elif stat_name.startswith("scatter index"):
                        Scatter_index = stat_data
                    elif stat_name.startswith("explained variance"):
                        VE = stat_data
                forecast_results = {
                    "MD": np.full((nlat, nlon), np.nan),
                    "MD_norm": np.full((nlat, nlon), np.nan),
                    "RMSD": np.full((nlat, nlon), np.nan),
                    "VE": np.full((nlat, nlon), np.nan),
                    "MAE": np.full((nlat, nlon), np.nan),
                    "SI": np.full((nlat, nlon), np.nan),
                    "anomaly corr": np.full((nlat, nlon), np.nan),
                    "nb_obs": np.where(nb_obs.mask, np.nan, nb_obs),
                }
                # Create an index array to map valmask to 0-based indexing
                valmask_index = mask - 1
                misfit = mean_obs - mean_mod
                rmsd = np.sqrt(msd)
                # VE = 100 - np.multiply(100, np.divide(np.sqrt(msd), np.sqrt(MS_obs-mean_obs)**2))
                SI = np.multiply(100, np.divide(np.sqrt(msd), np.sqrt(MS_obs)))
                # SI = np.multiply(100, np.divide(np.sqrt(msd), np.sqrt(MS_obs)))
                MD_norm = np.multiply(100, np.divide(abs(misfit), np.sqrt(MS_obs)))
                # VE = np.multiply(stat_data)
                # SI = np.multiply(100, np.divide(np.sqrt(msd), np.sqrt(MS_obs)))
                # print(VE)
                # print(np.nanmax(VE))
                # print(np.nanmin(VE))
                # VE = ma.masked_where(VE > 200, VE)
                # VE = ma.masked_where(VE < -100, VE)
                # SI = ma.masked_where(SI < 0, VE)
                # Reshape the results to 2D
                forecast_results["MD"] = misfit[valmask_index].reshape((nlat, nlon))
                forecast_results["anomaly_corr"] = anomaly_corr[valmask_index].reshape((nlat, nlon))
                forecast_results["RMSD"] = rmsd[valmask_index].reshape((nlat, nlon))
                forecast_results["VE"] = VE[valmask_index].reshape((nlat, nlon))
                forecast_results["MD_norm"] = MD_norm[valmask_index].reshape((nlat, nlon))
                forecast_results["SI"] = SI[valmask_index].reshape((nlat, nlon))
                forecast_results["MAE"] = MAE[valmask_index].reshape((nlat, nlon))
                results[system][variable_tmp][bprofvalue][fcst_value] = forecast_results
        print(f"{variable=} ok")

    return lon, lat, results
