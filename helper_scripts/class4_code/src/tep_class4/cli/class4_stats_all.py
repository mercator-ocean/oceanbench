#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set ts=4 sw=4 expandtab:
"""
Description:
    New CLASS4 launcher to compute CLASS4 with CLASS4 coming from GODAE,
    ARMOR, CORIOLIS, different ASCII files

History:
    July 2014       : Creation - C. REGNIER
    November 2014   : Add multiple forecast lead times for interpolation,
                      clims, and run chaining
    December 2014   : Update for zoomed region case
    November 2016   : New version with KDTree method for coloc/interp
    Nov/Dec 2017    : Modular refactor for CLASS4 portability (SLA, SST, SSS)
    September 2018  : Add interpolator for 2D variables, refactoring
    July 2020       : Add 3D in-situ profiles
    April 2025      : Edito version

Developer:
    C. REGNIER
"""

import os
import sys
from os.path import dirname
import resource
from timeit import default_timer as timer
import argparse
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from collections import namedtuple
from datetime import datetime, timedelta
from glob import glob

# Import local modules
from tep_class4.core.Selector import Selector
from tep_class4.core.Extractor import Extractor
from tep_class4.core.Loader import Loader
from tep_class4.core.arithm import init_array_1D, init_array_3D, compute_rmsd_3d_by_depth_bins, compute_rmsd, nanrmse
from tep_class4.core.Interpolator import Interpolator
from tep_class4.core.Writer import Writer
from tep_class4.core.utils import create_dict_list, save_metrics_to_netcdf, save_stats_to_netcdf, save_all_metrics
from tep_class4.core.utils import generate_nbdateslist, generate_date_list_week
from tep_class4.core.inputdata_loader import inputdata_loader

resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def main(level=2):
    """
    Main TEP CLASS4 function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--date1", required=True)
    parser.add_argument("--date2", required=True)
    parser.add_argument("--typemod", required=True)
    parser.add_argument("--loglevel", type=int, default=2)
    parser.add_argument("--zarr", action="store_true")
    parser.add_argument("--fcst (only forecasts)", action="store_true")
    parser.add_argument("--hdct (only hindcast)", action="store_true")
    parser.add_argument("--all (hdct fcst and pers)", action="store_true")
    args = parser.parse_args()
    # ==================================================================
    #  PART-1 -  Load input parameters
    # ==================================================================
    param_dict, log = Loader().factory("INPUTSTATS").load(sys.argv[1:], level)
    log.debug("========================================================")
    log.debug("          Load input options and parameter file         ")
    log.debug("========================================================")
    # Load db configuration file
    conf_path = dirname(__file__)
    filename = conf_path + "/cfg/" + param_dict["confname"]
    ## Todo : create confname
    param_dict = Loader.factory("NML").load(filename, level, param_dict, log)
    nb_forecasts = len(param_dict["lead_int"])
    nb_leadtimes = len(param_dict["lead_time"])
    dict_variables = param_dict["varname"]
    dict_typedata = param_dict["typedata"]
    list_var = param_dict["list_variables"]
    data_format = param_dict["data_format"]
    prefix_data = param_dict["prefix_data"]
    day_centered = param_dict["time_centered"]
    good_qc = param_dict["good_qc"]
    config = param_dict["cl_config"]
    var2D = ["SST", "SLA", "aice", "DRIFTER_filtr", "DRIFTER"]
    start_date = datetime.strptime(param_dict["date1"], "%Y%m%d").date()
    end_date = datetime.strptime(param_dict["date2"], "%Y%m%d").date()
    current = start_date
    depth_profs = param_dict["DEPTHS"]
    depth_labels = [f"{depth_profs[i]}-{depth_profs[i+1]}" for i in range(len(depth_profs) - 1)]
    depth_labels.append(f"{depth_profs[0]}-{depth_profs[-1]}")
    N_Depths = len(depth_profs)
    N = len(param_dict["lead_int"])
    list_date = generate_date_list_week(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
    while current <= end_date:
        daterun = current.strftime("%Y%m%d")
        log.info(f"{nb_forecasts=} {nb_leadtimes=} {daterun}")
        list_dates = generate_nbdateslist(daterun, nb_forecasts)
        # READ Model file
        input_filename = f"{param_dict['EDITO']}{daterun}.zarr"
        log.info(f"Model file {input_filename}")
        input_model = xr.open_dataset(input_filename, engine="zarr")
        results_stats = {}
        for var in list_var:
            if var in var2D:
                tab_rmsd, tab_cumul_sq_diff, tab_n_obs = (init_array_1D(N) for _ in range(3))
            else:
                tab_rmsd, tab_cumul_sq_diff, tab_n_obs = (init_array_3D(N, N_Depths, 2) for _ in range(3))
            log.info(f"Work on variable {var=}")
            # Loop on input observations
            ObsData = namedtuple("ObsData", ["lon", "lat", "obs_value", "nb_obs", "depth", "qc", "dims", "FillValue"])
            variable = dict_variables[var]
            good_qc_var = good_qc[var]
            # Normalize to a list
            if isinstance(variable, str):
                variable = [variable]
            # Check if all variables exist
            missing = [var for var in variable if var not in input_model.variables]
            if missing:
                raise KeyError(f"Variable(s) not found in dataset: {missing}")
                sys.exit(1)
            selected_data = input_model[variable]
            # Selection of the input data
            # selector = Selector(
            #                    data_format[var],
            #                    var,
            #                    log
            #                   )
            # global_input = selector.factory("GLOBAL_INPUT_DATA")
            interpolator = Interpolator(log)
            # Create dictionnary
            # list_keys = create_dict_list(
            #                            list_dates[0],
            #                            list_dates[-1],
            #                            param_dict['lead_time'],
            #                            param_dict['lead_int']
            #                            )
            # Define depth bins and labels
            # combined_list = []
            leadtime = param_dict["lead_time"][0]
            for ind_time, lead_day in enumerate(param_dict["lead_int"]):
                dateval = list_dates[ind_time]
                if day_centered:
                    target_date = pd.to_datetime(dateval) + pd.Timedelta(hours=12)
                else:
                    target_date = pd.to_datetime(dateval)
                log.info(f"Selection of input data {data_format[var]} {target_date} ")
                log.info("=================================================")
                data_file, ll_miss = Extractor(log).factory(prefix_data[var]).list(dateval, param_dict["dirdata"], var)
                if ll_miss:
                    log.info(f"File is missing for the observation {dateval}")
                    continue
                log.info(f"{data_file=}")
                # Read input obs value
                lon_obs, lat_obs, obs_value, nb_obs, depth, qc, tab_dims, param_dict, _FillValue = (
                    inputdata_loader(log).factory(data_format[var]).read_pos_and_value(data_file[0], param_dict)
                )
                obs_data = ObsData(
                    lon_obs, lat_obs, obs_value.squeeze(), nb_obs, depth.squeeze(), qc.squeeze(), tab_dims, _FillValue
                )
                # Find the corresponding array in selected_data
                time_index = selected_data.time.to_index()
                # Find exact index
                if target_date in time_index:
                    time_idx = time_index.get_loc(target_date)
                    log.info(f"Index found {time_idx}")
                    data_at_date = selected_data.isel(time=time_idx)
                else:
                    raise ValueError(f"Date {target_date.strftime('%Y-%m-%d')} not found in dataset.")
                # Interp values
                tab_results, is_2D = interpolator.interp_class4(
                    obs_data,
                    data_at_date,
                    variable,
                    param_dict,
                    datevalue=dateval,
                    leadtime=leadtime,
                )
                tab_results["qc"] = obs_data.qc
                # Compute results
                if is_2D:
                    log.debug("Case 2D")
                    tab_rmsd[ind_time], tab_cumul_sq_diff[ind_time], tab_n_obs[ind_time] = compute_rmsd(
                        np.squeeze(tab_results["observation"]),
                        np.squeeze(tab_results["forecast"]),
                        tab_results["qc"],
                        good_qc=good_qc_var,
                    )
                    log.debug(tab_rmsd[ind_time], tab_cumul_sq_diff[ind_time], tab_n_obs[ind_time])
                else:
                    log.debug("Case 3D")
                    tab_rmsd[ind_time, :, 0], tab_cumul_sq_diff[ind_time, :, 0], tab_n_obs[ind_time, :, 0] = (
                        compute_rmsd_3d_by_depth_bins(
                            np.squeeze(tab_results["observation"][:, 0, :]),
                            np.squeeze(tab_results["forecast"][:, 0, :]),
                            tab_results["depth"][:, :],
                            tab_results["qc"][:, 0, :],
                            depth_profs,
                            good_qc_var,
                        )
                    )
                    tab_rmsd[ind_time, :, 1], tab_cumul_sq_diff[ind_time, :, 1], tab_n_obs[ind_time, :, 1] = (
                        compute_rmsd_3d_by_depth_bins(
                            np.squeeze(tab_results["observation"][:, 1, :]),
                            np.squeeze(tab_results["forecast"][:, 1, :]),
                            tab_results["depth"][:, :],
                            tab_results["qc"][:, 1, :],
                            depth_profs,
                            good_qc_var,
                        )
                    )
            results_stats[var] = {"rmsd": tab_rmsd, "sq_diff_cumul": tab_cumul_sq_diff, "nb_obs": tab_n_obs}
        outputname = f"{param_dict['diroutput']}results_class4_{config}_allvars_forecast_R{daterun}.nc"
        lead_times = param_dict["lead_int"]
        save_stats_to_netcdf(results_stats, lead_times, depth_labels, outputname)
        current += timedelta(days=7)
    file_list = sorted(glob(f"{param_dict['diroutput']}results_class4_{config}_allvars_forecast_*nc"))
    log.info(f"Liste file {file_list}")
    fileout = f"{param_dict['diroutput']}results_class4_{config}_{list_date[0]}_{list_date[-1]}.nc"
    # Concatenate them along the 'time' dimension
    log.info("Save metrics in a file")
    save_all_metrics(file_list, list_date, fileout)


# Run the main code
if __name__ == "__main__":
    main()
