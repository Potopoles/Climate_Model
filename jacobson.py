#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          jacobson.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190528
License:            MIT

###############################################################################
"""
import copy
#import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
from namelist import pTop, njobs, comp_mode
from namelist import i_run_new_style
from org_namelist import HOST, DEVICE
from constants import con_Rd

from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from bin.continuity_cython import colp_tendency_jacobson_c, vertical_wind_jacobson_c
if not i_run_new_style:
    from continuity_cuda import (colp_tendency_jacobson_gpu,
                                    vertical_wind_jacobson_gpu)
    from wind_cuda import wind_tendency_jacobson_gpu

from wind import wind_tendency_jacobson
from bin.wind_cython import wind_tendency_jacobson_c

from bin.temperature_cython import temperature_tendency_jacobson_c
###### NEW
from dyn_org_discretizations import (TendencyFactory,
                                    DiagnosticsFactory) 
Tendencies = TendencyFactory()
Diagnostics = DiagnosticsFactory()
###### NEW


#from moisture import water_vapor_tendency, cloud_water_tendency
#from bin.moisture_cython import water_vapor_tendency_c, cloud_water_tendency_c
#from moisture_cuda import water_vapor_tendency_gpu, cloud_water_tendency_gpu

from geopotential import diag_geopotential_jacobson
from bin.geopotential_cython import diag_geopotential_jacobson_c
from geopotential_cuda import diag_geopotential_jacobson_gpu

from diagnostics import diagnose_POTTVB_jacobson, interp_COLPA
from bin.diagnostics_cython import diagnose_POTTVB_jacobson_c
from diagnostics_cuda import diagnose_POTTVB_jacobson_gpu

from boundaries import exchange_BC
from boundaries_cuda import exchange_BC_gpu

from jacobson_cuda import time_step_2D

from numba import cuda
import numba


def tendencies_jacobson(GR, GR_NEW, F, subgrids, NF):

    ##############################
    ##############################
    t_start = time.time()
    if i_run_new_style == 1:

        if comp_mode == 1:

            NF.old_to_new(F, host=True)
            Tendencies.continuity(HOST, GR_NEW,
                        **NF.get(Tendencies.fields_continuity, target=HOST))
            NF.new_to_old(F, host=True)

        elif comp_mode == 2:

            NF.old_to_new(F, host=False)
            Tendencies.continuity(DEVICE, GR_NEW,
                    **NF.get(Tendencies.fields_continuity, target=DEVICE))
            NF.new_to_old(F, host=False)

    else:
        # PROGNOSE COLP
        if comp_mode == 1:
            F.dCOLPdt, F.UFLX, F.VFLX, F.FLXDIV = colp_tendency_jacobson_c(GR,
                                            F.COLP, F.UWIND, F.VWIND,
                                            F.dCOLPdt, F.UFLX, F.VFLX, F.FLXDIV)
            F.dCOLPdt = np.asarray(F.dCOLPdt)
            F.UFLX = np.asarray(F.UFLX)
            F.VFLX = np.asarray(F.VFLX)
            F.FLXDIV = np.asarray(F.FLXDIV)
            F.COLP_NEW[GR.iijj] = F.COLP_OLD[GR.iijj] + GR.dt*F.dCOLPdt[GR.iijj]

        elif comp_mode == 2:
            F.dCOLPdt, F.UFLX, F.VFLX, F.FLXDIV = \
                 colp_tendency_jacobson_gpu(GR, GR.griddim, GR.blockdim, GR.stream,
                                        F.dCOLPdt, F.UFLX, F.VFLX, F.FLXDIV,
                                        F.COLP, F.UWIND, F.VWIND,
                                        GR.dy, GR.dxjsd, GR.Ad, GR.dsigmad)
            time_step_2D[GR.griddim, GR.blockdim, GR.stream]\
                                (F.COLP_NEW, F.COLP_OLD, F.dCOLPdt, GR.dt)
            GR.stream.synchronize()


        # DIAGNOSE WWIND
        if comp_mode == 1:
            F.WWIND = vertical_wind_jacobson_c(GR, F.COLP_NEW,
                                    F.dCOLPdt, F.FLXDIV, F.WWIND)
            F.WWIND = np.asarray(F.WWIND)
            F.COLP_NEW = exchange_BC(GR, F.COLP_NEW)
            F.WWIND = exchange_BC(GR, F.WWIND)

        elif comp_mode == 2:
            vertical_wind_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks, GR.stream]\
                                        (F.WWIND, F.dCOLPdt, F.FLXDIV,
                                        F.COLP_NEW, GR.sigma_vbd)
            GR.stream.synchronize()
            F.COLP_NEW = exchange_BC_gpu(F.COLP_NEW, GR.zonal, GR.merid,
                                        GR.griddim_xy, GR.blockdim_xy,
                                        GR.stream, array2D=True)
            F.WWIND = exchange_BC_gpu(F.WWIND, GR.zonalvb, GR.meridvb,
                                        GR.griddim_ks, GR.blockdim_ks, GR.stream)

    t_end = time.time()
    GR.cont_comp_time += t_end - t_start
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    # PROGNOSE WIND
    if comp_mode == 1:
        if i_run_new_style == 1:

            NF.old_to_new(F, host=True)
            Tendencies.momentum(HOST, GR_NEW,
                    **NF.get(Tendencies.fields_momentum, target=HOST))
                            
            NF.new_to_old(F, host=True)
        
        else:
            F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_c(GR, njobs,
                                        F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                        F.WWIND_UWIND, F.WWIND_VWIND, 
                                        F.COLP, F.COLP_NEW, F.PHI,
                                        F.POTT, F.PVTF, F.PVTFVB)
            F.dUFLXdt = np.asarray(F.dUFLXdt)
            F.dVFLXdt = np.asarray(F.dVFLXdt)


    elif comp_mode == 2:
        if i_run_new_style == 1:

            NF.old_to_new(F, host=False)
            Tendencies.momentum(DEVICE, GR_NEW,
                    **NF.get(Tendencies.fields_momentum, target=DEVICE))
                            
            NF.new_to_old(F, host=False)

        else:
            F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_gpu(GR,
                                        F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                        F.WWIND_UWIND, F.WWIND_VWIND, 
                                        F.COLP, F.COLP_NEW, F.PHI, F.POTT,
                                        F.PVTF, F.PVTFVB)

    t_end = time.time()
    GR.wind_comp_time += t_end - t_start
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    # PROGNOSE POTT
    if comp_mode == 1:
        if i_run_new_style == 1:

            NF.old_to_new(F, host=True)
            Tendencies.temperature(HOST, GR_NEW,
                        **NF.get(Tendencies.fields_temperature, target=HOST))
            NF.new_to_old(F, host=True)

        else:
            F.dPOTTdt = temperature_tendency_jacobson_c(GR, njobs,
                                    F.POTT, F.POTTVB, F.COLP, F.COLP_NEW,
                                    F.UFLX, F.VFLX, F.WWIND,
                                    F.dPOTTdt_RAD, F.dPOTTdt_MIC)
            F.dPOTTdt = np.asarray(F.dPOTTdt)


    elif comp_mode == 2:

        if i_run_new_style == 0:
            F.COLP          = cp.expand_dims(F.COLP, axis=2)
            F.COLP_NEW      = cp.expand_dims(F.COLP_NEW, axis=2)

        NF.old_to_new(F, host=False)
        Tendencies.temperature(DEVICE, GR_NEW,
                    **NF.get(Tendencies.fields_temperature, target=DEVICE))
        NF.new_to_old(F, host=False)

        if i_run_new_style == 0:
            F.COLP          = F.COLP.squeeze()
            F.COLP_NEW      = F.COLP_NEW.squeeze()


    t_end = time.time()
    GR.temp_comp_time += t_end - t_start
    ##############################
    ##############################


    ###############################
    ###############################
    #t_start = time.time()
    ## MOIST VARIABLES
    #if comp_mode == 0:
    #    F.dQVdt = water_vapor_tendency(GR, F.dQVdt, F.QV, F.COLP, F.COLP_NEW, \
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
    #    F.dQCdt = cloud_water_tendency(GR, F.dQCdt, F.QC, F.COLP, F.COLP_NEW, \
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)

    #elif comp_mode == 1:
    #    F.dQVdt = water_vapor_tendency_c(GR, njobs, F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
    #    F.dQVdt = np.asarray(F.dQVdt)
    #    F.dQCdt = cloud_water_tendency_c(GR, njobs, F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)
    #    F.dQCdt = np.asarray(F.dQCdt)

    #elif comp_mode == 2:
    #    water_vapor_tendency_gpu[GR.griddim, GR.blockdim, GR.stream] \
    #                                (F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
    #                                 F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC,
    #                                 GR.Ad, GR.dsigmad)
    #    GR.stream.synchronize()
    #    cloud_water_tendency_gpu[GR.griddim, GR.blockdim, GR.stream] \
    #                                (F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
    #                                 F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC,
    #                                 GR.Ad, GR.dsigmad)
    #    GR.stream.synchronize()

    #t_end = time.time()
    #GR.trac_comp_time += t_end - t_start
    ###############################
    ###############################




def diagnose_fields_jacobson(GR, GR_NEW, F, NF):

    ##############################
    ##############################
    if comp_mode == 1:
        if i_run_new_style:

            NF.old_to_new(F, host=True)
            Diagnostics.primary_diag(HOST, GR_NEW,
                    **NF.get(Diagnostics.fields_primary_diag, target=HOST))
            NF.new_to_old(F, host=True)

        else:
            F.PHI, F.PHIVB, F.PVTF, F.PVTFVB = \
                diag_geopotential_jacobson_c(GR, njobs, F.PHI, F.PHIVB, F.HSURF, 
                                                F.POTT, F.COLP, F.PVTF, F.PVTFVB)
            F.PHI = np.asarray(F.PHI)
            F.PHIVB = np.asarray(F.PHIVB)
            F.PVTF = np.asarray(F.PVTF)
            F.PVTFVB = np.asarray(F.PVTFVB)

            F.POTTVB = diagnose_POTTVB_jacobson_c(GR, njobs, F.POTTVB,
                                                    F.POTT, F.PVTF, F.PVTFVB)
            F.POTTVB = np.asarray(F.POTTVB)

    elif comp_mode == 2:
        if i_run_new_style:

            NF.old_to_new(F, host=False)
            Diagnostics.primary_diag(DEVICE, GR_NEW,
                    **NF.get(Diagnostics.fields_primary_diag, target=DEVICE))
            NF.new_to_old(F, host=False)

        else:
            F.PHI, F.PHIVB, F.PVTF, F.PVTFVB = \
                 diag_geopotential_jacobson_gpu(GR, GR.stream, F.PHI,
                                                F.PHIVB, F.HSURF, 
                                                F.POTT, F.COLP, F.PVTF, F.PVTFVB)

            diagnose_POTTVB_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks,
                                                        GR.stream] \
                                            (F.POTTVB, F.POTT, F.PVTF, F.PVTFVB)
            GR.stream.synchronize()
    ##############################
    ##############################




