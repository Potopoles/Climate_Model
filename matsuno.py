#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          matsuno.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190529
License:            MIT

Perform a matsuno time step.
###############################################################################
"""
import copy
import time
import numpy as np
import cupy as cp
from namelist import comp_mode
from org_namelist import wp_old, HOST, DEVICE
from boundaries import exchange_BC
from jacobson import tendencies_jacobson, \
                    diagnose_fields_jacobson

from bin.jacobson_cython import proceed_timestep_jacobson_c
from jacobson_cuda import proceed_timestep_jacobson_gpu

from diagnostics import interp_COLPA
from namelist import i_run_new_style

from numba import cuda, jit
if wp_old == 'float64':
    from numba import float64

from grid import tpb, bpg
from GPU import set_equal

from dyn_org_discretizations import (PrognosticsFactory) 
Prognostics = PrognosticsFactory()
###############################################################################

def step_matsuno(GR, GR_NEW, subgrids, F, NF):

    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 1:
        F.UWIND_OLD[:] = F.UWIND[:]
        F.VWIND_OLD[:] = F.VWIND[:]
        F.POTT_OLD[:]  = F.POTT[:]
        F.QV_OLD[:]    = F.QV[:]
        F.QC_OLD[:]    = F.QC[:]
        F.COLP_OLD[:]  = F.COLP[:]
    elif comp_mode == 2:
        if i_run_new_style == 1:
            set_equal[bpg, tpb](F.COLP_OLD, F.COLP)
            set_equal[bpg, tpb](F.UWIND_OLD, F.UWIND)
            set_equal[bpg, tpb](F.VWIND_OLD, F.VWIND)
            set_equal[bpg, tpb](F.POTT_OLD, F.POTT)
            set_equal[bpg, tpb](F.QV_OLD, F.QV)
            set_equal[bpg, tpb](F.QC_OLD, F.QC)
        else:
            set_equal_o  [GR.griddim_is, GR.blockdim   , GR.stream](
                                    F.UWIND_OLD, F.UWIND)
            set_equal_o  [GR.griddim_js, GR.blockdim   , GR.stream](
                                    F.VWIND_OLD, F.VWIND)
            set_equal_o  [GR.griddim   , GR.blockdim   , GR.stream](
                                    F.POTT_OLD, F.POTT)
            set_equal_o  [GR.griddim   , GR.blockdim   , GR.stream](
                                    F.QV_OLD, F.QV)
            set_equal_o  [GR.griddim   , GR.blockdim   , GR.stream](
                                    F.QC_OLD, F.QC)
            set_equal2D[GR.griddim_xy, GR.blockdim_xy, GR.stream](
                                    F.COLP_OLD, F.COLP)
    t_end = time.time()
    GR.step_comp_time += t_end - t_start
    GR.special += t_end - t_start
    ##############################
    ##############################


    ############################################################
    ############################################################
    ##########     ESTIMATE
    ############################################################
    ############################################################

    ##############################
    ##############################
    tendencies_jacobson(GR, GR_NEW, F, subgrids, NF)
    if comp_mode == 1:
        F.COLP[:] = F.COLP_NEW[:]
    elif comp_mode == 2:
        if i_run_new_style == 1:
            set_equal[bpg, tpb](F.COLP, F.COLP_NEW)
        else:
            set_equal2D[GR.griddim_xy, GR.blockdim_xy, GR.stream](
                        F.COLP, F.COLP_NEW)
    ##############################
    ##############################

    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 1:

        if i_run_new_style:

            NF.old_to_new(F, host=False)
            #NF.host['POTT'][:] = np.nan
            #NF.host['UWIND'][:] = np.nan
            #NF.host['VWIND'][:] = np.nan
            #n_iter = 10
            Prognostics.euler_forward(HOST, GR_NEW,
                    **NF.get(Prognostics.fields_prognostic, target=HOST))
            #t0 = time.time()
            #for i in range(n_iter):
            #    Prognostics.euler_forward(HOST, GR_NEW,
            #            **NF.get(Prognostics.fields_prognostic, target=HOST))
            #print((time.time() - t0)/n_iter)
            NF.new_to_old(F, host=False)

            ##TODO
            #FIELD1 = np.asarray(F.VWIND)
            #print(np.nanmean((FIELD1)))
            #print()

            #F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
            #             = proceed_timestep_jacobson_c(GR,
            #                    F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
            #                    F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
            #                    F.QV_OLD, F.QV, F.QC_OLD, F.QC,
            #                    F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)
            #F.UWIND = np.asarray(F.UWIND)
            #F.VWIND = np.asarray(F.VWIND)
            #F.COLP = np.asarray(F.COLP)
            #F.POTT = np.asarray(F.POTT)
            #F.QV = np.asarray(F.QV)
            #F.QC = np.asarray(F.QC)
            #t0 = time.time()
            #for i in range(n_iter):
            #    F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
            #             = proceed_timestep_jacobson_c(GR,
            #                    F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
            #                    F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
            #                    F.QV_OLD, F.QV, F.QC_OLD, F.QC,
            #                    F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)
            #    F.UWIND = np.asarray(F.UWIND)
            #    F.VWIND = np.asarray(F.VWIND)
            #    F.COLP = np.asarray(F.COLP)
            #    F.POTT = np.asarray(F.POTT)
            #    F.QV = np.asarray(F.QV)
            #    F.QC = np.asarray(F.QC)
            #print((time.time() - t0)/n_iter)

            ##TODO
            #FIELD2 = np.asarray(F.VWIND)
            #print(np.nanmean((FIELD2)))
            #
            #print()
            #print(np.sum(np.isnan(FIELD2[:,:,:])) -\
            #             np.sum(np.isnan(FIELD1[:,:,:])))
            #print(np.nanmean(FIELD2[:,:,:] - FIELD1[:,:,:]))
            #quit()
        else:

            F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
                         = proceed_timestep_jacobson_c(GR,
                                F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
                                F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
                                F.QV_OLD, F.QV, F.QC_OLD, F.QC,
                                F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)
            F.UWIND = np.asarray(F.UWIND)
            F.VWIND = np.asarray(F.VWIND)
            F.COLP = np.asarray(F.COLP)
            F.POTT = np.asarray(F.POTT)
            F.QV = np.asarray(F.QV)
            F.QC = np.asarray(F.QC)

    elif comp_mode == 2:
        if i_run_new_style:

            NF.old_to_new(F, host=False)
            #NF.device['POTT'][:] = np.nan
            #NF.device['UWIND'][:] = np.nan
            #NF.device['VWIND'][:] = np.nan
            #print('GPU')
            #n_iter = 10
            #t0 = time.time()
            Prognostics.euler_forward(DEVICE, GR_NEW,
                    **NF.get(Prognostics.fields_prognostic, target=DEVICE))
            #for i in range(n_iter):
            #    Prognostics.euler_forward(DEVICE, GR_NEW,
            #            **NF.get(Prognostics.fields_prognostic, target=DEVICE))
            #print((time.time() - t0)/n_iter)
            NF.new_to_old(F, host=False)

            ##TODO
            #FIELD1 = np.asarray(F.UWIND)
            #print(np.nanmean((FIELD1)))
            #print()

            #t0 = time.time()
            #F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
            #             = proceed_timestep_jacobson_gpu(GR, GR.stream,
            #                    F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
            #                    F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
            #                    F.QV_OLD, F.QV, F.QC_OLD, F.QC,
            #                    F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt,
            #                    F.dQCdt, GR.Ad)
            #for i in range(n_iter):
            #    F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
            #                 = proceed_timestep_jacobson_gpu(GR, GR.stream,
            #                        F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
            #                        F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
            #                        F.QV_OLD, F.QV, F.QC_OLD, F.QC,
            #                        F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt,
            #                        F.dQCdt, GR.Ad)
            #print((time.time() - t0)/n_iter)

            ##TODO
            #FIELD2 = np.asarray(F.UWIND)
            #print(np.nanmean((FIELD2)))
            #
            #print()
            #print(np.sum(np.isnan(FIELD2[:,:,:])) -\
            #             np.sum(np.isnan(FIELD1[:,:,:])))
            #print(np.nanmean(FIELD2[:,:,:] - FIELD1[:,:,:]))
            ##print(np.sum(np.isnan(FIELD2[:,:])) - np.sum(np.isnan(FIELD1[:,:])))
            ##print(np.nanmean(FIELD2[:,:] - FIELD1[:,:]))
            #quit()

            #import matplotlib.pyplot as plt
            ##diff = FIELD2[:,:,k] - FIELD1[:,:,k]
            #diff = FIELD2[:,:] - FIELD1[:,:,0]
            #plt.contourf(diff)
            #plt.colorbar()
            #plt.show()
            #quit()

        else:
            F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
                         = proceed_timestep_jacobson_gpu(GR, GR.stream,
                                F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
                                F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
                                F.QV_OLD, F.QV, F.QC_OLD, F.QC,
                                F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt,
                                F.dQCdt, GR.Ad)

    t_end = time.time()
    GR.step_comp_time += t_end - t_start
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    diagnose_fields_jacobson(GR, GR_NEW, F, NF)
    t_end = time.time()
    GR.diag_comp_time += t_end - t_start
    ##############################
    ##############################



    ############################################################
    ############################################################
    ##########     FINAL
    ############################################################
    ############################################################

    ##############################
    ##############################
    tendencies_jacobson(GR, GR_NEW, F, subgrids, NF)
    if comp_mode == 1:
        F.COLP[:] = F.COLP_NEW[:]
    elif comp_mode == 2:
        if i_run_new_style == 1:
            set_equal[bpg, tpb](F.COLP, F.COLP_NEW)
        else:
            set_equal2D[GR.griddim_xy, GR.blockdim_xy, GR.stream](
                        F.COLP, F.COLP_NEW)
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 1:
        if i_run_new_style:

            NF.old_to_new(F, host=False)
            Prognostics.euler_forward(HOST, GR_NEW,
                    **NF.get(Prognostics.fields_prognostic, target=HOST))
            NF.new_to_old(F, host=False)

        else:
            F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
                         = proceed_timestep_jacobson_c(GR,
                                F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
                                F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
                                F.QV_OLD, F.QV, F.QC_OLD, F.QC,
                                F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)
            F.UWIND = np.asarray(F.UWIND)
            F.VWIND = np.asarray(F.VWIND)
            F.COLP = np.asarray(F.COLP)
            F.POTT = np.asarray(F.POTT)
            F.QV = np.asarray(F.QV)
            F.QC = np.asarray(F.QC)

    elif comp_mode == 2:
        if i_run_new_style:

            NF.old_to_new(F, host=False)
            Prognostics.euler_forward(DEVICE, GR_NEW,
                    **NF.get(Prognostics.fields_prognostic, target=DEVICE))
            NF.new_to_old(F, host=False)

        else:
            F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
                 = proceed_timestep_jacobson_gpu(GR, GR.stream,
                        F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
                        F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
                        F.QV_OLD, F.QV, F.QC_OLD, F.QC,
                        F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt, GR.Ad)

    t_end = time.time()
    GR.step_comp_time += t_end - t_start
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    diagnose_fields_jacobson(GR, GR_NEW, F, NF)
    t_end = time.time()
    GR.diag_comp_time += t_end - t_start
    ##############################
    ##############################

    

@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:]'],target='gpu')
def set_equal_o(set_FIELD, get_FIELD):
    i, j, k = cuda.grid(3)
    set_FIELD[i,j,k] = get_FIELD[i,j,k] 
    cuda.syncthreads()

@jit([wp_old+'[:,:  ], '+wp_old+'[:,:  ]'],target='gpu')
def set_equal2D(set_FIELD, get_FIELD):
    i, j = cuda.grid(2)
    set_FIELD[i,j  ] = get_FIELD[i,j  ] 
    cuda.syncthreads()

