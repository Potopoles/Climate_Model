import numpy as np
import time


def diagnostics(GR, WIND, UWIND, VWIND, COLP, POTT):
    vmax = 0
    mean_wind = 0
    mean_temp = 0
    for k in range(0,GR.nz):
        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )
        vmax = max(vmax, np.max(WIND[:,:,k][GR.iijj]))
        mean_wind += np.sum( WIND[:,:,k][GR.iijj]*COLP[GR.iijj]*GR.A[GR.iijj] ) / \
                        np.sum( COLP[GR.iijj]*GR.A[GR.iijj] )
        mean_temp += np.sum(POTT[:,:,k][GR.iijj]*GR.A[GR.iijj]*COLP[GR.iijj])/ \
                np.sum(GR.A[GR.iijj]*COLP[GR.iijj])
    mean_wind = mean_wind/GR.nz
    mean_temp = mean_temp/GR.nz

    mean_colp = np.sum(COLP[GR.iijj]*GR.A[GR.iijj])/np.sum(GR.A[GR.iijj])

    return(WIND, vmax, mean_wind, mean_temp, mean_colp)


def print_ts_info(GR, WIND, UWIND, VWIND, COLP, POTT):

    WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
                                    WIND, UWIND, VWIND, COLP, POTT)

    #print('#### ' + str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,2)) + \
    #        '  days  vmax: ' + str(np.round(vmax,1)) + '  m/s vmean: ' + \
    #        str(np.round(mean_wind,3)) + ' m/s Tmean: ' + \
    #        str(np.round(mean_temp,7)) + \
    #        '  K  COLP: ' + str(np.round(mean_colp,2)) + ' Pa')


    print('################')
    print(str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,2)) + \
            '  days  vmax: ' + str(np.round(vmax,1)) + '  m/s vmean: ' + \
            str(np.round(mean_wind,3)) + ' m/s Tmean: ' + \
            str(np.round(mean_temp,7)) + \
            '  K  COLP: ' + str(np.round(mean_colp,2)) + ' Pa')

    real_time_sec = time.time() - GR.start_time
    faster_than_reality = int(GR.sim_time_sec/real_time_sec)
    percentage_done = np.round(GR.sim_time_sec/(GR.i_sim_n_days*36*24),2)
    try:
        to_go_sec = int((100/percentage_done - 1)*real_time_sec)
        tghr = int(np.floor(to_go_sec/3600))
        tgmin = int(np.floor(to_go_sec - tghr*3600)/60)
        tgsec = int(to_go_sec - tghr*3600 - tgmin*60)
        to_go_string = 'Finish in ' + str(tghr) + ' h '+ str(tgmin) \
                            + ' m ' + str(tgsec) + ' s.'
    except:
        to_go_string = 'unknown'

    print(str(percentage_done) + ' %   ' + str(faster_than_reality) + \
            ' faster than reality. ' + to_go_string)
    
