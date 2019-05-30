#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          IO_helper_functions.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190530
License:            MIT

Helper functions for IO.
###############################################################################
"""
import copy
import numpy as np

from namelist import comp_mode, nth_ts_print_diag
from org_namelist import wp
from constants import con_kappa, con_g 
###############################################################################



def NC_output_diagnostics(GR, UWIND, VWIND, WWIND, POTT,
                        COLP, PVTF, PVTFVB, PHI, PHIVB, RHO, QV, QC):

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


    # vertical wind in m/s
    WWIND_ms = copy.deepcopy(WWIND)
    for ks in range(1,GR.nzs-1):
        WWIND_ms[:,:,ks][GR.iijj] = (( PHI[:,:,ks][GR.iijj] - 
                                        PHI[:,:,ks-1][GR.iijj] ) /
                                    ( con_g * 0.5 * (GR.dsigma[0,0,ks] + 
                                        GR.dsigma[0,0,ks-1] ) ) * 
                                            WWIND[:,:,ks][GR.iijj] )


    # water vapor path and liquid water path
    ALTVB = PHIVB / con_g
    dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]
    WVP = np.sum(QV[GR.iijj]*dz*RHO[GR.iijj],2)
    CWP = np.sum(QC[GR.iijj]*dz*RHO[GR.iijj],2)


    return(VORT, WWIND_ms, WVP, CWP)



####################################################################
####################################################################
####################################################################


def diagnose_print_diag_fields(GR, WIND, COLP, POTT):

    max_wind = np.max(WIND[GR.ii,GR.jj,:])
    
    mean_wind = 0
    mean_temp = 0
    for k in range(0,GR.nz):
        mean_wind += ( np.sum( WIND[GR.ii,GR.jj,k]*
                               COLP[GR.ii,GR.jj,0]*GR.A[GR.ii,GR.jj,0] ) /
                       np.sum( COLP[GR.ii,GR.jj,0]*GR.A[GR.ii,GR.jj,0] ) )
        mean_temp += ( np.sum( POTT[GR.ii,GR.jj,k]*
                               GR.A[GR.ii,GR.jj,0]*COLP[GR.ii,GR.jj,0] ) /
                       np.sum( GR.A[GR.ii,GR.jj,0]*COLP[GR.ii,GR.jj,0] ) )
    mean_wind = mean_wind/GR.nz
    mean_temp = mean_temp/GR.nz

    mean_colp = np.sum(COLP[GR.ii,GR.jj,0] * 
                       GR.A[GR.ii,GR.jj,0])/np.sum(GR.A[GR.ii,GR.jj,0])

    return(max_wind, mean_wind, mean_temp, mean_colp)



def print_ts_info(GR, F):
    

    if GR.ts % nth_ts_print_diag == 0:
        if comp_mode == 2:
            F.copy_device_to_host(F.PRINT_DIAG_FIELDS)
        GR.timer.start('diag')
        vmax, mean_wind, mean_temp, mean_colp = \
                    diagnose_print_diag_fields(GR, F.host['WIND'], 
                                        F.host['COLP'], F.host['POTT'])
        GR.timer.stop('diag')

        print(str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,3))  +
                '\t days' +
                ' vmax: ' + str(np.round(vmax,1)) + '  m/s vmean: ' +
                str(np.round(mean_wind,3)) + ' m/s Tmean: ' +
                str(np.round(mean_temp,7)) +
                '  K  COLP: ' + str(np.round(mean_colp,2)) + ' Pa')

        # test for crash
        if ((np.sum(np.isnan(F.host['UWIND'][GR.iisjj])) > 0) |
           (np.max(F.host['UWIND'][GR.iisjj]) > 500)):
            raise ValueError('MODEL CRASH')

    if GR.ts % nth_ts_print_diag == 0:
        try:
            faster_than_reality = np.round(GR.sim_time_sec/
                                    GR.timer.timings['total'],2)
            percentage_done = np.round(GR.sim_time_sec/
                                    (GR.i_sim_n_days*36*24),2)
            to_go_sec = int((100/percentage_done - 1)*
                                    GR.timer.timings['total'])
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


