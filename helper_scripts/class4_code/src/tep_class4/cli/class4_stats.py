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

# Import local modules
from tep_class4.core.Selector import Selector
from tep_class4.core.Extractor import Extractor
from tep_class4.core.Loader import Loader
from tep_class4.core.arithm import init_array_1D, compute_rmsd_3d_by_depth_bins, compute_rmsd, nanrmse
from tep_class4.core.Interpolator import Interpolator
from tep_class4.core.Writer import Writer
from tep_class4.core.utils import create_dict_list, save_metrics_to_netcdf
from tep_class4.core.utils import generate_nbdateslist
from tep_class4.core.inputdata_loader import inputdata_loader

resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def main(level=2):
    """
    Main TEP CLASS4 function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--daterun", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--data", required=True)
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
    param_dict, log = Loader().factory("INPUT").load(sys.argv[1:], level)
    log.debug("========================================================")
    log.debug("          Load input options and parameter file         ")
    log.debug("========================================================")
    # Load db configuration file
    conf_path = dirname(__file__)
    filename = conf_path + "/cfg/" + param_dict["confname"]
    param_dict = Loader.factory("NML").load(filename, level, param_dict, log)
    nb_forecasts = len(param_dict["lead_int"])
    nb_leadtimes = len(param_dict["lead_time"])
    daterun = param_dict["cl_daterun"]
    log.info(f"{nb_forecasts=} {nb_leadtimes=} {daterun}")
    list_dates = generate_nbdateslist(daterun, nb_forecasts)
    ObsData = namedtuple("ObsData", ["lon", "lat", "obs_value", "nb_obs", "depth", "qc", "dims", "FillValue"])
    # READ Model file
    variable = param_dict["varname"]
    input_filename = f"{param_dict['EDITO']}{daterun}.zarr"
    log.info(f"Model file {input_filename}")
    input_model = xr.open_dataset(input_filename, engine="zarr")
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
    selector = Selector(param_dict["data_format"], param_dict["data_type"], log)
    global_input = selector.factory("GLOBAL_INPUT_DATA")
    interpolator = Interpolator(log, dask=param_dict["dask"])
    # Create dictionnary
    list_keys = create_dict_list(list_dates[0], list_dates[-1], param_dict["lead_time"], param_dict["lead_int"])
    # Define depth bins and labels
    DEPTH_BINS = [(0, 5), (5, 100), (100, 300), (300, 600)]
    DEPTH_BIN_LABELS = ["0-5", "5-100", "100-300", "300-600"]
    combined_list = []
    leadtime = param_dict["lead_time"][0]
    N = len(param_dict["lead_int"])
    tab_rmsd, tab_cumul_sq_diff, tab_n_obs = (init_array_1D(N) for _ in range(3))
    for ind_time, lead_day in enumerate(param_dict["lead_int"]):
        dateval = list_dates[ind_time]
        target_date = pd.to_datetime(dateval)  # + pd.Timedelta(hours=12)
        log.info(f"Selection of input data {param_dict['data_format']} " f"{param_dict['data_type']}")
        log.info("=================================================")
        # data_file, ll_miss = global_input.run(param_dict, dateval)
        data_file, ll_miss = (
            Extractor(log)
            .factory(param_dict["prefix_data"])
            .list(dateval, param_dict["dirdata"], param_dict["data_type"])
        )
        if ll_miss:
            log.info(f"File is missing for the observation {dateval}")
            continue
        log.info(f"{data_file=}")
        # Read input obs value
        log.debug(f"Format of the data {param_dict['data_format']}")
        lon_obs, lat_obs, obs_value, nb_obs, depth, qc, tab_dims, param_dict, _FillValue = (
            inputdata_loader(log).factory(param_dict["data_format"]).read_pos_and_value(data_file[0], param_dict)
        )
        obs_data = ObsData(
            lon_obs, lat_obs, obs_value.squeeze(), nb_obs, depth.squeeze(), qc.squeeze(), tab_dims, _FillValue
        )
        # Find the corresponding array in selected_data
        # Convert time coordinate to pandas Index
        time_index = selected_data.time.to_index()
        # Find exact index
        if target_date in time_index:
            time_idx = time_index.get_loc(target_date)
            print(f"Index found {time_idx}")
            # leadtime = f'fcst{time_idx}'
            data_at_date = selected_data.isel(time=time_idx)
        else:
            raise ValueError(f"Date {target_date.strftime('%Y-%m-%d')} not found in dataset.")
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
        # print(np.nanmax(np.squeeze(tab_results['forecast'])[1800:1850]))
        # print(np.nanmax(np.squeeze(tab_results['observation'])[1800:1850]))
        # print(np.nanmin(np.squeeze(tab_results['forecast'])[1800:1850]))
        # print(np.nanmin(np.squeeze(tab_results['observation'])[1800:1850]))
        # print(np.squeeze(tab_results['forecast'])[1800:1850])
        # print(np.squeeze(tab_results['observation'])[1800:1850])
        # print(np.shape(tab_results['forecast']))
        # print(np.nanmax(tab_results['forecast']))
        # print(np.nanmin(tab_results['forecast']))
        # print('======')
        # print(np.nanmax(tab_results['observation']))
        # print(np.nanmin(tab_results['observation']))
        # print('======')
        # print(np.nanmin(np.squeeze(tab_results['forecast']) - np.squeeze(tab_results['observation'])))
        # print(np.nanmax(np.squeeze(tab_results['forecast']) - np.squeeze(tab_results['observation'])))
        # print(nanrmse(np.squeeze(tab_results['forecast']),
        #                     np.squeeze(tab_results['observation'])))
        if is_2D:
            print(np.shape(tab_results["forecast"]))
            tab_rmsd[ind_time], tab_cumul_sq_diff[ind_time], tab_n_obs[ind_time] = compute_rmsd(
                np.squeeze(tab_results["observation"]), np.squeeze(tab_results["forecast"])
            )
            print(tab_rmsd[ind_time], tab_cumul_sq_diff[ind_time], tab_n_obs[ind_time])
        else:
            print("Case 3D")
            print(np.shape(tab_results["forecast"]))
            print(np.shape(tab_results["depth"]))
            # rmsd_temp, cumul_temp, nobs_temp = compute_rmsd_3d_by_depth_bins(np.squeeze(tab_results['observation'][:, 0, :]),
            #                                                                 np.squeeze(tab_results['forecast'][:, 0, :]),
            #                                                                 tab_results['depth'][:, :], depth_bins=[0, 5, 100, 300, 600])
            # print(np.shape(rmsd_temp))
            # print(rmsd_temp)
            # print(tab_results.keys())
            # print(tab_results['observation'][80:85, 1, :])
            # print(tab_results['forecast'][80:85, 1, :])
            # print(tab_results['qc'][80:85, 1, :])
            rmsd_psal, cumul_psal, nobs_psal = compute_rmsd_3d_by_depth_bins(
                np.squeeze(tab_results["observation"][:, 1, :]),
                np.squeeze(tab_results["forecast"][:, 1, :]),
                tab_results["depth"][:, :],
                tab_results["qc"][:, 1, :],
                depth_bins=[0, 5, 100, 300, 600],
            )
            tab_rmsd[ind_time, :], tab_cumul_sq_diff[ind_time, :], tab_n_obs[ind_time, :] = compute_rmsd(
                np.squeeze(tab_results["observation"]), np.squeeze(tab_results["forecast"])
            )
            print(tab_rmsd[ind_time], tab_cumul_sq_diff[ind_time], tab_n_obs[ind_time])

    outputname = f"{param_dict['diroutput']}results_class4_{param_dict['data_type']}_{param_dict['cl_conf']}_forecast_R{daterun}.nc"
    lead_times = param_dict["lead_int"]
    save_metrics_to_netcdf(tab_rmsd, tab_cumul_sq_diff, tab_n_obs, lead_times, outputname)


# Set the level of logger (log_level = 2)
# Run the main code
if __name__ == "__main__":
    main()
