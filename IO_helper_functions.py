import numpy as np
import time
import copy
from geopotential import diag_pvt_factor
from constants import con_kappa, con_g 
from namelist import comp_mode, nth_ts_time_step_diag
from org_namelist import wp
from diagnostics import console_output_diagnostics
#from diagnostics_cython import console_output_diagnostics_c
#from diagnostics_cuda import console_output_diagnostics_gpu





def NC_output_diagnostics(GR, F, UWIND, VWIND, WWIND, POTT, COLP, PVTF, PVTFVB,
                    PHI, PHIVB, RHO, MIC):

    # VORTICITY
    VORT = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan, dtype=wp)
    for k in range(0,GR.nz):
        VORT[:,:,k][GR.iijj] = (   \
                                + ( VWIND[:,:,k][GR.iijj_ip1    ] + \
                                    VWIND[:,:,k][GR.iijj_ip1_jp1] ) / 2 \
                                - ( VWIND[:,:,k][GR.iijj_im1    ] + \
                                    VWIND[:,:,k][GR.iijj_im1_jp1] ) / 2 \
                               ) / (2*GR.dx[GR.iijj]) \
                               - ( \
                                + ( UWIND[:,:,k][GR.iijj_jp1    ] + \
                                    UWIND[:,:,k][GR.iijj_ip1_jp1] ) / 2 \
                                - ( UWIND[:,:,k][GR.iijj_jm1    ] + \
                                    UWIND[:,:,k][GR.iijj_ip1_jm1] ) / 2 \
                               ) / (2*GR.dy) \


    ## PRESSURE AND EFFECTIVE TEMPERATURE
    #PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)
    #PAIR = 100000*np.power(PVTF, 1/con_kappa)
    #TAIR = POTT*PVTF

    WWIND_ms = copy.deepcopy(WWIND)
    for ks in range(1,GR.nzs-1):
        WWIND_ms[:,:,ks][GR.iijj] = ( PHI[:,:,ks][GR.iijj] - PHI[:,:,ks-1][GR.iijj] ) / \
                                    ( con_g * 0.5 * (GR.dsigma[ks] + GR.dsigma[ks-1] ) ) * \
                                            WWIND[:,:,ks][GR.iijj]


    ALTVB = PHIVB / con_g
    dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]
    WVP = np.sum(F.QV[GR.iijj]*dz*RHO[GR.iijj],2)
    CWP = np.sum(F.QC[GR.iijj]*dz*RHO[GR.iijj],2)


    return(VORT, WWIND_ms, WVP, CWP)



####################################################################
####################################################################
####################################################################




def print_ts_info(GR, CF, GF):
    

    if GR.ts % nth_ts_time_step_diag == 0:
        t_start = time.time()
        if comp_mode == 2:
            GF.copy_stepDiag_fields_to_host(GR)
        vmax, mean_wind, mean_temp, mean_colp = console_output_diagnostics(GR, CF)
        t_end = time.time()
        GR.diag_comp_time += t_end - t_start

        print(str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,3))  + '\t day' + \
                '  days  vmax: ' + str(np.round(vmax,1)) + '  m/s vmean: ' + \
                str(np.round(mean_wind,3)) + ' m/s Tmean: ' + \
                str(np.round(mean_temp,7)) + \
                '  K  COLP: ' + str(np.round(mean_colp,2)) + ' Pa')

        # test for crash
        if (np.sum(np.isnan(CF.UWIND[GR.iisjj])) > 0) | (np.max(CF.UWIND[GR.iisjj]) > 500):
            raise ValueError('MODEL CRASH')

    if GR.ts % nth_ts_time_step_diag == 0:
        try:
            faster_than_reality = np.round(GR.sim_time_sec/GR.total_comp_time,2)
            percentage_done = np.round(GR.sim_time_sec/(GR.i_sim_n_days*36*24),2)
            to_go_sec = int((100/percentage_done - 1)*GR.total_comp_time)
            tghr = int(np.floor(to_go_sec/3600))
            tgmin = int(np.floor(to_go_sec - tghr*3600)/60)
            tgsec = int(to_go_sec - tghr*3600 - tgmin*60)
            to_go_string = 'Finish in ' + str(tghr) + ' h '+ str(tgmin) \
                                + ' m ' + str(tgsec) + ' s.'

            print(str(percentage_done) + ' %   ' + str(faster_than_reality) + \
                    ' faster than reality. ' + to_go_string)
            print('################')
        except:
            pass




####################################################################
####################################################################
####################################################################




    
def print_computation_time_info(GR):
    # FINNAL OUTPUT
    print('DONE')
    print('took ' + str(np.round(GR.total_comp_time/60,2)) + ' mins.')
    print('Relative amount of CPU time')
    print('#### gernal')
    print('IO         :  ' + str(int(100*GR.IO_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.IO_time)) + '\ts')
    print('#### dynamics')
    print('total      :  ' + str(int(100*GR.dyn_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.dyn_comp_time)) + '\ts')
    print('horAdv     :  ' + str(int(100*GR.wind_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.wind_comp_time)) + '\ts')
    print('temperature:  ' + str(int(100*GR.temp_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.temp_comp_time)) + '\ts')
    print('tracer     :  ' + str(int(100*GR.trac_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.trac_comp_time)) + '\ts')
    print('continuity :  ' + str(int(100*GR.cont_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.cont_comp_time)) + '\ts')
    print('diagnostics:  ' + str(int(100*GR.diag_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.diag_comp_time)) + '\ts')
    print('time steps :  ' + str(int(100*GR.step_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.step_comp_time)) + '\ts')
    print('#### other')
    print('copy       :  ' + str(int(100*GR.copy_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.copy_time)) + '\ts')
    print('radiation  :  ' + str(int(100*GR.rad_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.rad_comp_time)) + '\ts')
    print('microphyis :  ' + str(int(100*GR.mic_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.mic_comp_time)) + '\ts')
    print('soil       :  ' + str(int(100*GR.soil_comp_time/GR.total_comp_time)) + '  \t%\t' \
                           + str(int(GR.soil_comp_time)) + '\ts')



