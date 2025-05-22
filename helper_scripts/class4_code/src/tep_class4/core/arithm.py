import numpy as np
import sys
import datetime
from copy import deepcopy
import numpy.ma as ma
import netCDF4
__author__      = "C.REGNIER"
__version__ = 0.1

import numpy as np


def init_array_1D(size):
    return np.full(size, netCDF4.default_fillvals['f4'], dtype=float, order='F')

def init_array_2D(size, size2):
    return np.full((size, size2), netCDF4.default_fillvals['f4'], dtype=float, order='F')

def init_array_3D(size, size2, size3):
    return np.full((size, size2, size3), netCDF4.default_fillvals['f4'], dtype=float, order='F')

def compute_rmsd_3d_by_depth_bins(data1, data2, depth_array, data_qc, depth_bins, good_qc):
    """
    Compute RMSD between data1 and data2 over specified depth bins,
    aggregated across all profiles (not per-profile).

    Parameters:
    - data1, data2: masked arrays of shape (num_obs, num_depths)
    - depth_array: array of shape (num_obs, num_depths), giving the actual depth of each point
    - data_qc : ndarray (num_obs, num_depths)
                QC flags: only values with QC == 0 are used in the statistics
    - depth_bins: list of bin edges (e.g., [0, 5, 100, 300, 600])

    Returns:
    - rmsd_bin: list of RMSD per depth bin
    - cumul_sq_diff_bin: list of total squared diff per bin
    - n_obs_bin: list of valid observation counts per bin
    """
    
    data1 = ma.masked_invalid(data1)
    data2 = ma.masked_invalid(data2)
    depth_array = ma.masked_invalid(depth_array)
    
    assert data1.shape == data2.shape == depth_array.shape, "Shapes of input arrays must match"

    # Create QC mask (True for valid values)
    qc_mask = (data_qc == good_qc)

    rmsd_bins = []
    cum_sq_diff_bins = []
    n_obs_bins = []

    for i in range(len(depth_bins) - 1):
        bin_mask = (depth_array >= depth_bins[i]) & (depth_array < depth_bins[i + 1])

        # Valid where bin_mask is True and both data1 and data2 are not masked
        #valid_mask = bin_mask & (~data1.mask) & (~data2.mask)
        
        bin_mask_2d = np.broadcast_to(bin_mask, data1.shape)
        valid_mask = bin_mask_2d & (~data1.mask) & (~data2.mask) & qc_mask


        sq_diff = ma.masked_where(~valid_mask, (data1 - data2)**2)
        cum_sq_diff = sq_diff.sum()
        n_obs = valid_mask.sum()
        rmsd = np.sqrt(cum_sq_diff / n_obs) if n_obs > 0 else np.nan

        rmsd_bins.append(rmsd)
        cum_sq_diff_bins.append(cum_sq_diff)
        n_obs_bins.append(n_obs)
        #diff = data1.data - data2.data
        #sq_diff = np.square(diff)
        #cumul_sq_diff_bin[i] = np.sum(sq_diff[valid_mask])
        #n_obs_bin[i] = np.sum(valid_mask)

    # Total RMSD over all bins (0 to max(depth_bins))
    dmin_total = depth_bins[0]
    dmax_total = depth_bins[-1]
    total_mask = (depth_array >= dmin_total) & (depth_array <= dmax_total)
    total_mask_2d = np.broadcast_to(total_mask, data1.shape)

    valid_mask_total = total_mask_2d & (~data1.mask) & (~data2.mask) & qc_mask

    sq_diff_total = ma.masked_where(~valid_mask_total, (data1 - data2)**2)

    cum_sq_diff_total = sq_diff_total.sum()
    n_obs_total = valid_mask_total.sum()
    rmsd_total = np.sqrt(cum_sq_diff_total / n_obs_total) if n_obs_total > 0 else np.nan
    
    rmsd_bins.append(rmsd_total)
    cum_sq_diff_bins.append(cum_sq_diff_total)
    n_obs_bins.append(n_obs_total)
    # Compute RMSD per bin
    #rmsd_bin = [np.sqrt(cumul / n) if n > 0 else np.nan
    #            for cumul, n in zip(cumul_sq_diff_bin, n_obs_bin)]

    return rmsd_bins, cum_sq_diff_bins, n_obs_bins

def compute_rmsd(data1, data2, data_qc=None, good_qc=0):
    """
    Compute RMSD between two arrays, skipping invalid/masked values.
    and optionally filtering by QC flags (QC == 0 only).
    Returns:
        rmsd: root mean square difference
        cumul_sq_diff: sum of squared differences
        n_obs: number of valid observations
    """
    # Convert to masked arrays if not already
    data1 = np.ma.masked_invalid(data1)
    data2 = np.ma.masked_invalid(data2)

    # Mask positions where either data1 or data2 is masked
    combined_mask = np.ma.mask_or(data1.mask, data2.mask)
    
    # If QC is provided, mask anything not QC == 0
    if data_qc is not None:
        qc_mask = (data_qc != good_qc)
        combined_mask = np.logical_or(combined_mask, qc_mask)

    # Apply combined mask
    diff = ma.masked_array(data1 - data2, mask=combined_mask)

    # Count valid observations
    n_obs = diff.count()
    if n_obs == 0:
        return np.nan, 0.0, 0  # Or use np.ma.masked for rmsd if you prefer

    cumul_sq_diff = np.sum(diff**2)
    rmsd = np.sqrt(cumul_sq_diff / n_obs)

    return rmsd, cumul_sq_diff, n_obs

def fillVal2nan(data,fillVal):
    return np.ma.filled(np.ma.masked_array(data,(data==fillVal)), fill_value=np.nan)

#--------------------------------------------------------------------------------------------------

def nanmean(data, **args):
    return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).mean(**args), fill_value=np.nan)

def weighted_nanmean(data, weights, **kwargs):
   # Mask invalid values (NaNs)
   masked_data = ma.masked_invalid(data)
   # Apply weights
   weighted_data = ma.masked_array(masked_data, mask=masked_data.mask, fill_value=np.nan) * weights
   # Calculate the area-weighted mean
   weighted_mean = np.ma.average(weighted_data, **kwargs)
   return weighted_mean

#--------------------------------------------------------------------------------------------------
def nansum(data, **args):
    return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).sum(**args), fill_value=np.nan)

#--------------------------------------------------------------------------------------------------
def nanrms(data, **args):
    out = np.ma.filled(np.ma.masked_array(np.square(data),np.isnan(data)).mean(**args), fill_value=np.nan)
    return np.sqrt(out)

#--------------------------------------------------------------------------------------------------
def nanms(data, **args):
    out = np.ma.filled(np.ma.masked_array(np.square(data),np.isnan(data)).mean(**args), fill_value=np.nan)
    return out

def weighted_nanms(data, weights, **kwargs):
    # Mask invalid values (NaNs)
    masked_data = ma.masked_invalid(data)
    # Apply weights to squared values
    weighted_squared_data = ma.masked_array(masked_data ** 2, mask=masked_data.mask, fill_value=np.nan) * weights
    # Calculate the area-weighted mean of squared values
    weighted_mean_squared = np.ma.average(weighted_squared_data, **kwargs)
    return weighted_mean_squared

#--------------------------------------------------------------------------------------------------
def nansums(data, **args):
    out = np.ma.filled(np.ma.masked_array(np.square(data),np.isnan(data)).sum(**args), fill_value=np.nan)
    return out

#--------------------------------------------------------------------------------------------------
def nanrmse(data1,data2, **args):
    data1 = np.array(data1)
    data2 = np.array(data2)
    out = np.ma.filled(np.ma.masked_array(np.square(data2-data1),np.isnan(data2-data1)).mean(**args), fill_value=np.nan)
    return np.sqrt(out)

def weighted_nanrmse(data1, data2, weights, **kwargs):
    data1 = np.array(data1)
    data2 = np.array(data2)
    # Mask invalid values (NaNs) in both datasets
    masked_diff = ma.masked_invalid(data2 - data1)
    # Apply weights to squared differences
    weighted_squared_diff = ma.masked_array(masked_diff ** 2, mask=masked_diff.mask, fill_value=np.nan) * weights
    # Calculate the area-weighted mean of squared differences
    weighted_mean_squared_diff = np.ma.average(weighted_squared_diff, **kwargs)
    # Calculate the RMSE and fill NaNs
    rmse = np.sqrt(np.ma.filled(weighted_mean_squared_diff, fill_value=np.nan))
    return rmse

#--------------------------------------------------------------------------------------------------
def nansummd(data1,data2, **args):
    data1 = np.array(data1)
    data2 = np.array(data2)
    out = np.ma.filled(np.ma.masked_array((data2-data1),np.isnan(data2-data1)).sum(**args), fill_value=np.nan)
    return out

#--------------------------------------------------------------------------------------------------
def nansummad(data1,data2, **args):
    data1 = np.array(data1)
    data2 = np.array(data2)
    out = np.ma.filled(np.ma.masked_array((abs(data2-data1)),np.isnan(abs(data2-data1))).sum(**args), fill_value=np.nan)
    return out

def nanmeanMAE(data1,data2, **args):
    data1 = np.array(data1)
    data2 = np.array(data2)
    out = np.ma.filled(np.ma.masked_array((abs(data2-data1)),np.isnan(abs(data2-data1))).mean(**args), fill_value=np.nan)
    return out
#--------------------------------------------------------------------------------------------------
def nanmse(data1,data2, **args):
    out = np.ma.filled(np.ma.masked_array(np.square(data2-data1),np.isnan(data2-data1)).mean(**args), fill_value=np.nan)
    return out

def weighted_nanmse(data1, data2, weights, **kwargs):
    # Mask invalid values (NaNs) in both datasets
    masked_diff = ma.masked_invalid(data2 - data1)
    # Apply weights to squared differences
    weighted_squared_diff = ma.masked_array(masked_diff ** 2, mask=masked_diff.mask, fill_value=np.nan) * weights
    # Calculate the area-weighted mean of squared differences
    weighted_mean_squared_diff = np.ma.average(weighted_squared_diff, **kwargs)
    # Fill NaNs in the result
    mse = np.ma.filled(weighted_mean_squared_diff, fill_value=np.nan)
    return mse



#--------------------------------------------------------------------------------------------------
def nansumse(data1,data2, **args):
    out = np.ma.filled(np.ma.masked_array(np.square(data2-data1),np.isnan(data2-data1)).sum(**args), fill_value=np.nan)
    return out

#--------------------------------------------------------------------------------------------------
def nanstd(data, **args):
    return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).std(**args), fill_value=np.nan)

def weighted_nanstd(data, weights, **kwargs):
    # Mask invalid values (NaNs)
    masked_data = ma.masked_invalid(data)
    # Apply weights to the masked data
    weighted_data = ma.masked_array(masked_data, mask=masked_data.mask, fill_value=np.nan) * weights
    # Calculate the area-weighted standard deviation
    weighted_std = np.ma.sqrt(np.ma.average((weighted_data - np.ma.average(weighted_data, **kwargs))**2, weights=weights, **kwargs))
    # Fill NaNs in the result
    std = np.ma.filled(weighted_std, fill_value=np.nan)

    return std




#--------------------------------------------------------------------------------------------------
def nanmin(data, **args):
    return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).min(**args), fill_value=np.nan)

#--------------------------------------------------------------------------------------------------
def nanmax(data, **args):
    return np.ma.filled(np.ma.masked_array(data,np.isnan(data)).max(**args), fill_value=np.nan)

def weighted_nanpearson(data1, data2, weights):
    # Mask invalid values (NaNs) in both datasets
    masked_data1 = ma.masked_invalid(data1)
    masked_data2 = ma.masked_invalid(data2)

    # Apply weights to the masked data
    weighted_data1 = ma.masked_array(masked_data1, mask=masked_data1.mask, fill_value=np.nan) * weights
    weighted_data2 = ma.masked_array(masked_data2, mask=masked_data2.mask, fill_value=np.nan) * weights

    # Check if shapes are compatible for broadcasting
    try:
        np.broadcast(weighted_data1, weighted_data2, weights)
    except ValueError:
        raise ValueError("Shapes of data and weights are not compatible for broadcasting.")

    # Calculate means
    mean_data1 = np.ma.average(weighted_data1)
    mean_data2 = np.ma.average(weighted_data2)

    # Calculate covariance
    #print(np.shape(weighted_data1))
    #print(np.shape(mean_data1))
    #print(np.shape(weighted_data2))
    #print(np.shape(mean_data2))
    #print(np.shape(weights))

    cov12 = np.ma.average((weighted_data1 - mean_data1) * (weighted_data2 - mean_data2), weights=weights)

    # Calculate weighted standard deviations
    std1 = np.ma.sqrt(np.ma.average((weighted_data1 - mean_data1)**2, weights=weights))
    std2 = np.ma.sqrt(np.ma.average((weighted_data2 - mean_data2)**2, weights=weights))

    # Calculate the area-weighted Pearson correlation coefficient
    if std1 * std2 == 0:
       return 1  # Return 1 for perfect correlation if one of the standard deviations is zero

    return cov12 / (std1 * std2)

#--------------------------------------------------------------------------------------------------
def nanpearson(data1,data2):
    std1   = nanstd(data1)
    std2   = nanstd(data2)
    cov12  = nanmean((data1-nanmean(data1))*(data2 - nanmean(data2)))

    if cov12 == 0 : return 0
    if std1*std2 == 0 : return 1

    return cov12/(std1*std2)
