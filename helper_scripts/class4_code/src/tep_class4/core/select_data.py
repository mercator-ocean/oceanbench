class select_data():

    def select_drifters(log, params_dict, YEAR, MONTH, ts, depthvalue=15.0):

        """ Function to select drifters data depending on YEAR MONTH """

        TYPE = params_dict['TYPE']
        SUFFIX = params_dict['SUFFIX']
        ZONE = params_dict['ZONE']
        DIRDATA = params_dict['DIRDATA']
        log.info(f'{DIRDATA=}')
        if TYPE == "013_030":
            frame = ZONE + '_' + str(YEAR) + str(MONTH) + '_' + SUFFIX + '*.nc'
        elif TYPE == "013_048":
            frame = ZONE + '_' + SUFFIX + '*' + str(YEAR) + str(MONTH)+'.nc'
        log.info(f'{DIRDATA}{YEAR}{MONTH}')
        log.info(f'{frame=}')

        nb_files = len(glob(DIRDATA + str(YEAR) + str(MONTH) + '/' + frame))
        log.debug(f'Number of files {nb_files}')
        list_file = []
        list_lon = []
        list_lat = []
        list_drift = []
        list_time = []
        list_u15 = []
        list_v15 = []
        list_depth = []
        list_depth0 = []
        list_depthqc0 = []
        list_depthqc = []
        list_windu = []
        list_windv = []
        list_u0 = []
        list_v0 = []
        list_uqc = []
        list_vqc = []
        list_time = []
        list_time_ref = []
        list_current_test0 = []
        list_current_test15 = []
        list_current_test_qc15 = []
        list_current_test_qc0 = []
        list_position_qc = []
        list_time_qc = []
        list_dc_reference = []
        for f, file in enumerate(glob(params_dict['DIRDATA'] + str(YEAR) +
                                 str(MONTH) + '/' + frame)):
            list_file.append(file)
            if params_dict['TYPE'] == "013_030":
                drifter = os.path.basename(file).\
                        split('_')[4].split('.')[0]
            elif params_dict['TYPE'] == "013_048":
                drifter = os.path.basename(file).\
                        split('_')[3].split('.')[0]
            # list_drifters.append(drifter)
            # ind_dict[drifter] = {}
            try:
                lon_value, lat_value, depth, depth_qc, time, u_ewct, v_nsct,\
                        ll_var, u_qc, v_qc, wind_u, wind_v, current_test, \
                        current_test_qc, position_qc, time_qc, dc_reference = \
                        read_drifter(file)
            except cl4err:
                raise cl4err.Class4FatalError(f"Pb with drifter file {file}")
            date_val = pd.to_datetime(time).strftime('%Y-%m-%d')
            w_depth15 = np.where(np.array(depth) == depthvalue)
            ndim1, ndim2 = np.shape(depth)
            vtime = time.astype('M8[ms]').astype('O')
            vect_time = []
            if len(w_depth15[0]) > 0:
                ind_x15 = w_depth15[0]
                ind_y15 = w_depth15[1]
                u_ewct15 = np.array(u_ewct)[ind_x15, ind_y15]
                v_nsct15 = np.array(v_nsct)[ind_x15, ind_y15]
                lon_15 = np.array(lon_value)[ind_x15]
                lat_15 = np.array(lat_value)[ind_x15]
                u_ewct0 = np.array(u_ewct)[ind_x15, 0]
                v_nsct0 = np.array(v_nsct)[ind_x15, 0]
                depth0 = np.array(depth)[ind_x15, 0]
                depth15 = np.array(depth)[ind_x15, 1]
                u_qc0 = np.array(u_qc)[ind_x15, 0]
                v_qc0 = np.array(v_qc)[ind_x15, 0]
                time_qc_tmp = time_qc[ind_x15]
                depth0 = np.array(depth)[ind_x15, 0]
                depth_qc0 = np.array(depth_qc)[ind_x15, 0]
                current_test_qc0 = np.array(current_test_qc)[ind_x15, 0]
                current_test0 = np.array(current_test)[ind_x15, 0]
                position_qc_tmp = position_qc[ind_x15]
                dc_reference_tmp = dc_reference[ind_x15]
                for t in range(len((w_depth15[0]))):
                    vect_time.append(vtime[t].strftime('%Y%m%d%H'))
                ndays = len(vect_time)
                vector_time = np.nan*np.zeros((ndays))
                for il_ind, daytime in enumerate(np.array(vect_time)):
                    vector_time[il_ind] = (dt.datetime.strptime(daytime, '%Y%m%d%H').
                                           toordinal()-dateCNESref)*24+int(daytime[8:10])
                time_15 = np.array(date_val)[ind_x15]
                u_qc15 = np.array(u_qc)[ind_x15, ind_y15]
                v_qc15 = np.array(v_qc)[ind_x15, ind_y15]
                depth15 = np.array(depth)[ind_x15, ind_y15]
                depth_qc15 = np.array(depth_qc)[ind_x15, ind_y15]
                current_test_qc15 = np.array(current_test_qc)[ind_x15, ind_y15]
                current_test15 = np.array(current_test)[ind_x15, ind_y15]
                if len(np.shape(wind_u)) > 0 and len(np.shape(wind_v)) > 0:
                    u_wind = np.array(wind_u)[:, 0]
                    v_wind = np.array(wind_v)[:, 0]
                else:
                    print(f"Missing_wind {file}")
                    nb_times, nb_depths = np.shape(u_ewct)
                    u_wind = np.nan*np.zeros(nb_times)
                    v_wind = np.nan*np.zeros(nb_times)
                if len(w_depth15[0]) != ndim1:
                    log.error(f"Missing some depth at 15m depth {file}")
                ll_var15 = True
            else:
                ll_var15 = False
                log.error(f'Missing depth 15m {file=}')
            list_lon.extend(lon_15)
            list_lat.extend(lat_15)
            list_time_ref.extend(time_15)
            list_time.extend(vector_time.tolist())
            list_depth.extend(depth15)
            list_depth0.extend(depth0)
            list_u15.extend(u_ewct15)
            list_v15.extend(v_nsct15)
            list_windu.extend(u_wind)
            list_windv.extend(v_wind)
            list_u0.extend(u_ewct0)
            list_v0.extend(v_nsct0)
            list_uqc.extend(u_qc15)
            list_vqc.extend(v_qc15)
            list_depthqc0.extend(depth_qc0)
            list_depthqc.extend(depth_qc15)
            list_current_test0.extend(current_test0)
            list_current_test15.extend(current_test15)
            list_current_test_qc15.extend(current_test_qc15)
            list_current_test_qc0.extend(current_test_qc0)
            list_position_qc.extend(position_qc_tmp)
            list_time_qc.extend(time_qc_tmp)
            list_dc_reference.extend(dc_reference_tmp)
            for i in range(len(lon_15)):
                list_drift.append(str(drifter))
            #log.debug(f"Read drifter OK {f} / {nb_files}")
            #print(f"drifter={list_drift}")
        if not list_lon:
            raise BaseException('Missing input UV drifters')
        log.debug("Read all drifters OK")
        tab_value = {}
        for day in ts:
            log.debug(f"Dayvalue {day=}")
            tab_value[str(day)] = {}
            ind = np.where(np.array(list_time_ref) == str(day))
            tab_value[str(day)]['time'] = np.array(list_time)[ind]
            tab_value[str(day)]['lon'] = np.array(list_lon)[ind]
            tab_value[str(day)]['lat'] = np.array(list_lat)[ind]
            tab_value[str(day)]['depth'] = np.array(list_depth)[ind]
            tab_value[str(day)]['depth0'] = np.array(list_depth0)[ind]
            tab_value[str(day)]['u'] = np.array(list_u15)[ind]
            tab_value[str(day)]['v'] = np.array(list_v15)[ind]
            tab_value[str(day)]['windu'] = np.array(list_windu)[ind]
            tab_value[str(day)]['windv'] = np.array(list_windv)[ind]
            tab_value[str(day)]['u0'] = np.array(list_u0)[ind]
            tab_value[str(day)]['v0'] = np.array(list_v0)[ind]
            tab_value[str(day)]['uqc'] = np.array(list_uqc)[ind]
            tab_value[str(day)]['vqc'] = np.array(list_vqc)[ind]
            tab_value[str(day)]['depth_qc0'] = np.array(list_depthqc0)[ind]
            tab_value[str(day)]['depth_qc'] = np.array(list_depthqc)[ind]
            tab_value[str(day)]['current_test0'] = np.array(list_current_test0)[ind]
            tab_value[str(day)]['current_test15'] = np.array(list_current_test15)[ind]
            tab_value[str(day)]['current_test_qc15'] = np.array(list_current_test_qc15)[ind]
            tab_value[str(day)]['current_test_qc0'] = np.array(list_current_test_qc0)[ind]
            tab_value[str(day)]['position_qc'] = np.array(list_position_qc)[ind]
            tab_value[str(day)]['time_qc'] = np.array(list_time_qc)[ind]
            tab_value[str(day)]['dc_reference'] = np.array(list_dc_reference)[ind]
            tab_value[str(day)]['id_drift'] = np.array(list_drift)[ind]
            log.debug(f"Position {tab_value[str(day)]['lon']}")

        return tab_value, nb_files
