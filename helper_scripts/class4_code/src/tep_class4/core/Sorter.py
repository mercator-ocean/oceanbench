import sys
import os
import re
import glob
from .sydate import SyDate
from .sydate import SyDate as sy
from .Logger import Logger
from ModuleTime import Time
##############################################################
# C.REGNIER Juin 2014
# Class Sorter to sort data values
##############################################################


class Sorter():

    def symlink(self, src, dst):
        "Create a symbolic link, force if dst exists"
        if not os.path.islink(dst) and os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
# Not using filecmp increases speed with 15%
#      if os.path.isfile(dst) and filecmp.cmp(src, dst) == 0:
        if os.path.isfile(dst):
            os.unlink(dst)
        if os.path.islink(dst):
            os.unlink(dst)
        self.mkdir(os.path.dirname(dst))
        if not os.path.exists(dst):
            os.symlink(src, dst)

    def mkdir(self, path):
        "Create a directory, and parents if needed"
        if not os.path.exists(path):
            os.makedirs(path)

    def run_weekly(self, param_dict, date1, date2, daterun):
        """ Sort files by lead times
        Arguments
        --------------------
        param_dict : dictionnary of parameters
        date1 : date init
        date2 : date end
        daterun : run date
        """
        lead_time = param_dict['lead_time']
        lead_val = param_dict['lead_int'],
        config = param_dict['file_template']
        gridtemp = param_dict['gridtemp']
        gridsal = param_dict['gridsal']
        dirmodel = param_dict['dirmodel']
        dirsort = param_dict['sort_rep']
        log = Logger("Sorter weekly").run(2)
        for time in lead_time:
            lead_fin = []
            lead_new = lead_val
            if time == 'forecast':
                lead_new = [time[0]+time[4]+time[6:8] +
                            element for element in lead_new]
                lead_fin = lead_new
            elif time == 'persistence':
                if filter(lambda x: 'best_weekly' in x, lead_time):
                    lead_fin = ['pers']
                else:
                    lead_tmp = ['nwct']
                    lead_new = [time[:4]+element for element in lead_new[1:]]
                    lead_tmp.extend(lead_new)
                    lead_fin = lead_tmp
            elif time == 'best_estimate':
                lead_fin = ['hdct']
            elif time == 'best_weekly':
                lead_fin = ['hdct', 'nwct']
            else:
                log.error('Lead time %s not expected ' % (time))
            # Loop on directories
            valeurj = date1
            while valeurj.__ge__(date1) and valeurj.__le__(date2):
                for rep in lead_fin:
                    cl_rep = dirsort+rep
                    if os.path.isdir(cl_rep):
                        print ("rep existant %s" % (cl_rep))
                    else:
                        try:
                            os.mkdir(cl_rep)
                        except OSError as e:
                            log.info("Creation error : %s" % (e.strerror))
                    log.info('Interp day  %s ' % (sy.__str__(valeurj)))
                    dateval = sy.__str__(valeurj)
                    nametype = config+dateval
                    nametype0 = config+dateval
                    # Find Hindcast and nowcast
                    for fic in glob.glob(dirmodel+nametype+"*"+gridtemp+"*.nc"):
                        log.info("Fichier :: %s " % (fic))
                        fic_bis = os.path.basename(fic)
                        if rep == 'hdct' and Time().isHindcast(fic_bis):
                            fic_temp_hindcast = fic
                            fic_psal_hindcast = re.sub(gridtemp, gridsal, fic)
                            log.info("fichier hindcast T %s" %
                                     (fic_temp_hindcast))
                            log.info("fichier hindcast S %s" %
                                     (fic_psal_hindcast))
                            file_out_temp = cl_rep+'/'+fic_bis
                            file_out_psal = re.sub(
                                gridtemp, gridsal, file_out_temp)
                            try:
                                self.symlink(fic_temp_hindcast, file_out_temp)
                                self.symlink(fic_psal_hindcast, file_out_psal)
                            except OSError as e:
                                log.info("Link Error : %s" % (e.strerror))
                        elif rep == 'nwct' and Time().isNowcast(fic_bis):
                            fic_temp_nowcast = fic
                            fic_psal_nowcast = re.sub(gridtemp, gridsal, fic)
                            log.info("fichier nowcast T %s" %
                                     (fic_temp_nowcast))
                            log.info("fichier nowcast S %s" %
                                     (fic_psal_nowcast))
                            file_out_temp = cl_rep+'/'+fic_bis
                            file_out_psal = re.sub(
                                gridtemp, gridsal, file_out_temp)
                            try:
                                self.symlink(fic_temp_nowcast, file_out_temp)
                                self.symlink(fic_psal_nowcast, file_out_psal)
                            except OSError as e:
                                log.info("Link Error : %s" % (e.strerror))
                    # Find Forecast
                    JBUDATET0 = valeurj.goforward(1)
                    BUDATET0 = sy.__str__(JBUDATET0)
                    nametype = config+BUDATET0
                    for fic in glob.glob(dirmodel+nametype+"*"+gridtemp+"*.nc"):
                        fic_bis = os.path.basename(fic)
                        if rep == "fcst3" and Time().isForecast1(fic_bis):
                            RUNDATET0 = fic_bis.split('_')[5].split('.')[
                                0].split('R')[1]
                            # log.info("Date du run pour le  forecast : %s " % RUNDATET0)
                            RUNDATET0 = sy(RUNDATET0)
                            JRUNDATET1 = RUNDATET0.goforward(7)
                            JF3DATET0 = RUNDATET0.goforward(3)
                            nametypef3d = config+sy.__str__(JF3DATET0)
                            for fic2 in glob.glob(dirmodel+nametypef3d+"*"+gridtemp+"*.nc"):
                                fic3 = os.path.basename(fic2)
                                if Time().isForecast1(fic3):
                                    fic_temp_f3t0 = fic2
                                    fic_psal_f3t0 = re.sub(
                                        gridtemp, gridsal, fic2)
                                    filenameT = cl_rep+'/'+nametype0+'_'+gridtemp+'.nc'
                                    filenameS = re.sub(
                                        gridtemp, gridsal, filenameT)
                                    try:
                                        self.symlink(fic_temp_f3t0, filenameT)
                                        self.symlink(fic_psal_f3t0, filenameS)
                                    except OSError as e:
                                        log.error("Link Error : %s" %
                                                  (e.strerror))
                                    log.info(": %s " % fic_temp_f3t0)
                        if rep == "fcst6" and Time().isForecast1(fic_bis):
                            RUNDATET0 = fic_bis.split('_')[5].split('.')[
                                0].split('R')[1]
                            # log.info("Date du run pour le  forecast : %s " % RUNDATET0)
                            RUNDATET0 = sy(RUNDATET0)
                            JRUNDATET1 = RUNDATET0.goforward(7)
                            JF6DATET0 = RUNDATET0.goforward(6)
                            nametypef6d = config+sy.__str__(JF6DATET0)
                            for fic2 in glob.glob(dirmodel+nametypef6d+"*"+gridtemp+"*.nc"):
                                fic3 = os.path.basename(fic2)
                                if Time().isForecast1(fic3):
                                    fic_temp_f6t0 = fic2
                                    fic_psal_f6t0 = re.sub(
                                        gridtemp, gridsal, fic2)
                                    filenameT = cl_rep+'/'+nametype0+'_'+gridtemp+'.nc'
                                    filenameS = re.sub(
                                        gridtemp, gridsal, filenameT)
                                    try:
                                        self.symlink(fic_temp_f6t0, filenameT)
                                        self.symlink(fic_psal_f6t0, filenameS)
                                    except OSError as e:
                                        log.error("Link Error : %s" %
                                                  (e.strerror))
                                    log.info(": Forecast 6D %s %s" %
                                             (fic_temp_f6t0, nametype))
                    # find Persistence
                    RUNDATET0 = SyDate(daterun).gobackward(14)
                    BUDATET0 = sy.__str__(RUNDATET0)
                    nametype = config+BUDATET0
                    for fic in glob.glob(dirmodel+nametype+"*"+gridtemp+"*.nc"):
                        fic_bis = os.path.basename(fic)
                        if rep == "pers" and Time().isPersistence(fic_bis):
                            fic_temp_persistence = fic
                            log.info("Fichier  %s" % (fic))
                            fic_psal_persistence = re.sub(
                                gridtemp, gridsal, fic)
                            filenameT = cl_rep+'/'+nametype0+'_'+gridtemp+'.nc'
                            filenameS = re.sub(gridtemp, gridsal, filenameT)
                            try:
                                self.symlink(fic_temp_persistence, filenameT)
                                self.symlink(fic_psal_persistence, filenameS)
                            except OSError as e:
                                log.error("Link Error : %s" % (e.strerror))
                valeurj = valeurj.goforward(1)
    # TODO : add sort for weekly system => for the moment the work is done by the soft WIND

    def run_daily(self, param_dict, date1, date2, daterun):
        lead_time = param_dict['lead_time']
        lead_val = param_dict['lead_int'],
        config = param_dict['file_template']
        gridtemp = param_dict['gridtemp']
        gridsal = param_dict['gridsal']
        dirmodel = param_dict['dirmodel']
        dirsort = param_dict['sort_rep']
        print (lead_time, lead_val, date1, date2, config, dirmodel, dirsort)
