from __future__ import generators
import os, sys, string, shutil, glob, re
import numpy as np
import netCDF4
from netCDF4 import Dataset, stringtochar, chartostring

##------------------------------------------------------
## C.REGNIER Juin 2014
## Class for Reading Netcdf files in 1-2-3-4 Dimensions
##------------------------------------------------------


class NetcdReader:
    """Class to read netcdf dataset"""

    def read_1D(self, filename, namevar):
        # print 'read netcdf file 1D : %s ' %(filename)
        nc = Dataset(filename, "r")
        variable = namevar
        v = nc.variables[variable][:]
        type_var = v.dtype
        # print trim(type_var)
        ##if str(type_var) == "|S1":
        ##    ## Bug correction for Char
        ##    v.set_auto_maskandscale(False)
        ##    #rla_var_1D = chartostring(nc.variables[variable][:])
        rla_var_1D = nc.variables[variable][:]
        return rla_var_1D

    def read_2D(self, filename, namevar):
        nc = netCDF4.Dataset(filename, "r")
        variable = namevar
        v = nc.variables[variable][:, :]
        type_var = v.dtype
        ##if str(type_var) == "|S1":
        ##    ## Bug correction for Char
        ##    v.set_auto_maskandscale(False)
        rla_var_2D = nc.variables[variable][:, :]
        return rla_var_2D

    def read_3D(self, filename, namevar):
        nc = netCDF4.Dataset(filename, "r")
        variable = namevar
        v = nc.variables[variable][:, :, :]
        type_var = v.dtype
        if str(type_var) == "|S1":
            ## Bug correction for Char
            v.set_auto_maskandscale(False)
        rla_var_3D = nc.variables[variable][:, :, :]
        return rla_var_3D

    def read_4D(self, filename, namevar):
        nc = netCDF4.Dataset(filename, "r")
        variable = namevar
        v = nc.variables[variable][:, :, :, :]
        type_var = v.dtype
        if str(type_var) == "|S1":
            ## Bug correction for Char
            v.set_auto_maskandscale(False)
        rla_var_4D = chartostring(nc.variables[variable][:, :, :, :])
        return rla_var_4D


##
