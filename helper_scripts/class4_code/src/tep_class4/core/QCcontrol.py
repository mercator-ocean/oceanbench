from __future__ import generators
import os,sys,string,shutil,glob,re
import numpy as np
import netCDF4
import xarray as xr
from .NetcdReader import *

#######################################################################################
## C.REGNIER Juin 2014 
## Class for applying QC control
#######################################################################################

class QCcontroller(object):

    """ Factory QCcontroller Class to run different objets depending on input type """

    def __init__(self,data_type,data_source,**kwargs):
        self.data_type = data_type
        self.data_src = data_source
    # Factory creation based on class name:
    def factory(self):
        if self.data_src == "ARGO" :
            return QCArgo()
        elif self.data_src == "CORIOLIS":
            return QCCorio(self.data_type,self.data_src)
        elif self.data_src == "CORIOLIS_RAW":
            return QCCorio_raw(self.data_type,self.data_src)
        elif self.data_src == "ENS":
            return QCEns()
        elif self.data_src == "GODAE":
            return QC_GODAE(self.data_type,self.data_src)
        else :
            print ("Type %s not known " %(type))
            sys.exit(1)

class QCArgo(QCcontroller):

    """ Class to compute qc on Argo files """

    def QCrun(self,filename):
        nc = netCDF4.Dataset(filename)
        # Read Position QC
        namevar='POSITION_QC'
        cla_posqc=NetcdReader().read_1D(filename,namevar)
        namevar='JULD_QC'
        cla_juldqc=NetcdReader().read_1D(filename,namevar)
        namevar='DEPH_QC'
        cla_depthqc=NetcdReader().read_2D(filename,namevar)
        namevar='PRES_QC'
        rla_presqc=NetcdReader().read_2D(filename,namevar)
        namevar='LONGITUDE'
        rla_lon=NetcdReader().read_2D(file,namevar)
        nc.close()

class QC_GODAE(QCcontroller):

    """ Class to compute qc on GODAE CLASS4 files"""

    def __init__(self,data_type,data_source,**kwargs):
        super(QCcontroller,self).__init__(**kwargs)
        self.data_type=data_type
        self.data_src = data_source
    def QCrun(self,filename,qc_level):
        """
          Class to compute au qc on GODAE CLASS4 files
        """
        nc = netCDF4.Dataset(filename)
        ll_depth=None
        ll_qc=None
        for var in nc.variables : 
            if var  == 'depth': 
                ll_depth=True
            if  var == 'qc' : 
                ll_qc=True

        #if not ll_depth and not ll_qc : 
        #   print 'Missing depth and qc values !!'
        #   sys.exit(1)
        #else :
        if  ll_depth and ll_qc : 
            n_prof = len(nc.dimensions["numobs"])
            # Read QC
            namevar='qc'
            if self.data_type  == 'profile' : 
                rla_qc=NetcdReader().read_1D(filename,namevar)
            else :
                rla_qc=NetcdReader().read_3D(filename,namevar)
            namevar='depth'
            rla_depth=NetcdReader().read_2D(filename,namevar)
            ## Read LATITUDE
            namevar='latitude'
            rla_lat=NetcdReader().read_1D(filename,namevar)
            ## Read LONGITUDE
            namevar='longitude'
            rla_lon=NetcdReader().read_1D(filename,namevar)
            flag_prof=[]
            fillvalue_dep=99999.
            fillvalue_err=9999.
            for n_prof in range(n_prof):
                val_depth=rla_depth[n_prof,0]
                val_lat=rla_lat[n_prof]
                val_lon=rla_lon[n_prof]
                if self.data_type  == 'profile' : 
                    val_qc=(rla_qc[n_prof,:,0] == 0).sum()
                    flag_qc=9
                    if np.any( val_qc >= 1) :
                        flag_qc=0
                else :
                    flag_qc=rla_qc[n_prof]
                flag_lonlat='9'
                if  val_lat < 91. and val_lat > -91  :  
                    if val_lon == 0 and val_lat == 0 : 
                        flag_lonlat='9'
                    else :
                        flag_lonlat='1' 
                ##====================
                ## Final QC flag
                ##====================

                if flag_lonlat == '1' and flag_qc == 0 :
                    flag_prof.append('1')
                else :
                    #print 'Profile Nok  %s %s : ' %(flag_lonlat,flag_qc) 
                    flag_prof.append('0')
            nc.close()

        return flag_prof

class QCCorio_raw(QCcontroller):

    """ Class to compute qc on Coriolis files"""

    def __init__(self,data_type,data_source,**kwargs):
        super(QCcontroller,self).__init__(**kwargs)
        self.data_type=data_type
        self.data_src = data_source

    def QCrun(self, filename, qc_level):
        #print "Class to compute au qc on Coriolis files"
        ds = xr.open_dataset(filename)
        # Get the list of variable names
        variable_names = list(ds.variables)
        ll_pres=0
        ll_depth=0
        ll_temp=0
        ll_psal=0
        if 'DEPTH' in variable_names:
            ll_depth=1
        if 'PRES' in variable_names:
             ll_pres=1
        if var == 'TEMP':
            ll_temp=1
        if var == 'PSAL':
            ll_psal=1
        if not ll_temp and ll_psal :
            print ('PROF not valid with no TEMP and no PSAL')


class QCCorio(QCcontroller):

    """ Class to compute qc on Coriolis files"""

    def __init__(self,data_type,data_source,**kwargs):
        super(QCcontroller,self).__init__(**kwargs)
        self.data_type=data_type
        self.data_src = data_source
    def QCrun(self,filename,qc_level):
        #print "Class to compute au qc on Coriolis files"
        nc = netCDF4.Dataset(filename)
        ll_pres=0
        ll_depth=0
        ll_temp=0
        ll_psal=0
        for var in nc.variables : 
            if var  == 'DEPH': 
                ll_depth=1
            elif var == 'PRES': 
                ll_pres=1
            elif var == 'TEMP':
                ll_temp=1 
            elif var == 'PSAL':
                ll_psal=1 
        if not ll_temp and ll_psal : 
            print ('PROF not valid with no TEMP and no PSAL')
        # Read N_PROFS
        n_prof = len(nc.dimensions["N_PROF"])
        # Read Position QC
        namevar='POSITION_QC'
        cla_posqc=NetcdReader().read_1D(filename,namevar)
        namevar='JULD_QC'
        cla_juldqc=NetcdReader().read_1D(filename,namevar)
        if ll_depth and ll_pres :
            #print 'Read PRES'
            namevar='PRES_QC'
            cla_depthqc=NetcdReader().read_2D(filename,namevar)
            namevar='PRES'
            rla_depth=NetcdReader().read_2D(filename,namevar)
        elif ll_depth and not ll_pres :
            #print "Read JULD_QC %s " %(cla_juldqc)
            #print 'Read DEPH'
            namevar='DEPH_QC'
            cla_depthqc=NetcdReader().read_2D(filename,namevar)
            namevar='DEPH'
            rla_depth=NetcdReader().read_2D(filename,namevar)
            #print "Read DEPTH_QC %s " %(cla_depthqc)
        elif ll_pres and not ll_depth : 
            #print 'Read PRES'
            namevar='PRES_QC'
            cla_depthqc=NetcdReader().read_2D(filename,namevar)
            namevar='PRES'
            rla_depth=NetcdReader().read_2D(filename,namevar)
        else :
            print ('Not pres and depth')
            sys.exit(1)
        ## Read LATITUDE
        namevar='LATITUDE'
        rla_lat=NetcdReader().read_1D(filename,namevar)
        ## Read LONGITUDE
        namevar='LONGITUDE'
        rla_lon=NetcdReader().read_1D(filename,namevar)
        ## Read TEMP	
        if ll_temp :
            namevar='TEMP'
            rla_temp=NetcdReader().read_2D(filename,namevar)
            namevar='TEMP_QC'
            cla_tempqc=NetcdReader().read_2D(filename,namevar)
        ## Read PSAL	
        if ll_psal :
            namevar='PSAL'
            rla_psal=NetcdReader().read_2D(filename,namevar)
            namevar='PSAL_QC'
            cla_psalqc=NetcdReader().read_2D(filename,namevar)

        flag_prof=[]
        fillvalue_dep=  99999.
        fillvalue_temp=  9999.
        fillvalue_sal=  9999.
        fillvalue_err=  9999.
        for n_prof in range(n_prof): 
            val_posqc=cla_posqc[n_prof]
            val_juldqc=cla_juldqc[n_prof]
            if qc_level == 0 :
                val_depthqc=((cla_depthqc[n_prof,:] == b'1') | (cla_depthqc[n_prof,:] == b'0') | \
                                                   (cla_depthqc[n_prof,:] == b'7')).sum()
                if ll_temp :
                    val_tempqc=((cla_tempqc[n_prof,:] == b'1') & (rla_temp[n_prof,:] != fillvalue_temp  ) \
                                & (rla_depth[n_prof,:] != fillvalue_dep) & \
                                ((cla_depthqc[n_prof,:] == b'1') | (cla_depthqc[n_prof,:] == b'0') | (cla_depthqc[n_prof,:] == b'7'))).sum()
                if ll_psal:
                    val_psalqc=((cla_psalqc[n_prof,:] == b'1') & (rla_psal[n_prof,:] != fillvalue_sal  ) \
                               &  (rla_depth[n_prof,:] != fillvalue_dep) & \
                                 ((cla_depthqc[n_prof,:] == b'1') | (cla_depthqc[n_prof,:] == b'0') | (cla_depthqc[n_prof,:] == b'7'))).sum()
            elif qc_level == 1 :
                val_depthqc=((cla_depthqc[n_prof,:] == b'1') | (cla_depthqc[n_prof,:] == b'0')).sum()
                if ll_temp :
                    val_tempqc=((cla_tempqc[n_prof,:] == b'1') & (rla_temp[n_prof,:] != fillvalue_temp  ) \
                            &  (rla_depth[n_prof,:] != fillvalue_dep) & \
                                ((cla_depthqc[n_prof,:] == b'1') | (cla_depthqc[n_prof,:] == b'0'))).sum()
                if ll_psal:
                    val_psalqc=((cla_psalqc[n_prof,:] == b'1') & (rla_psal[n_prof,:] != fillvalue_sal  ) \
                            &  (rla_depth[n_prof,:] != fillvalue_dep) & \
                                ((cla_depthqc[n_prof,:] == b'1') | (cla_depthqc[n_prof,:] == b'0'))).sum()
            val_depth=rla_depth[n_prof,0]
            val_lat=rla_lat[n_prof]
            val_lon=rla_lon[n_prof]
            #print val_posqc,val_juldqc,val_depthqc
            ## Test if PRES and DEPTH are presents if both are present take PRES 
            ## JULD_QC_OK = 0 1 5 8 
            ## POSITION QC_OK = 0 1 5 8
            ## TEMP_QC PSAL_QC =1 
            ## DEPH_QC et PRES_QC =1  or = 0 et PRES et DEPTH /= 9999
            ## Validation Position
            flag_lonlat='9'
            if  val_lat < 91. and val_lat > -91  :  
                if val_lon == 0 and val_lat == 0 : 
                    flag_lonlat='9'
                else :
                    flag_lonlat='1' 
            ## Validation Day 
            flag_qcday='9'
            if val_juldqc == '0' or val_juldqc == '1' or val_juldqc == '5' or val_juldqc =='8' :
                flag_qcday='1' 
            ## Validation position 
            flag_qcpos='9' 
            if val_posqc == '0' or val_posqc == '1' or val_posqc == '5' or val_posqc =='8' :
                flag_qcpos='1' 
            ## Validation QC DEPTH
            flag_depthqc='9'
            if np.any( val_depthqc >= 1) :
                flag_depthqc='1'
            ## Validation DEPTH 
            flag_depth='9'
            if val_depth != fillvalue_dep :
                flag_depth='1'
            ## Validation value
            flag_value='9'
            if ll_temp and ll_psal: 
                if np.any( val_tempqc >= 1 or val_psalqc >=1 ) :
                    flag_value='1'
            elif ll_temp :
                if np.any( val_tempqc >= 1) :
                    flag_value='1'
            elif ll_psal :
                if np.any( val_psalqc >= 1) :
                    flag_value='1'

            ##====================
            ## Final QC flag
            ##====================
            #print flag_qcday,flag_qcpos,flag_depth 
            if flag_qcday == '1' and flag_qcpos == '1' and flag_depthqc == '1' and flag_lonlat  == '1' and flag_value == '1' :
                flag_prof.append('1')
               #print 'Profile ok %i flag_qcday %s flag_qcpos %s flag_depthqc %s flag_depth %s flag_lonlat %s : ' %(n_prof,flag_qcday,flag_qcpos,flag_depthqc,flag_depth,flag_lonlat) 
            else :
               #print 'Profile Nok %i flag_qcday %s flag_qcpos %s flag_depthqc %s flag_depth %s flag_lonlat %s flag_value %s : ' \
                   #%(n_prof,flag_qcday,flag_qcpos,flag_depthqc,flag_depth,flag_lonlat,flag_value) 
                flag_prof.append('0')
        nc.close()
        return flag_prof

#    TODO  CLASSE for ENS dataset
#    class QCEns(filename)


