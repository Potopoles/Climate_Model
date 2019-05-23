import copy
import time
import numpy as np
from namelist import comp_mode
from org_namelist import wp_old
from boundaries import exchange_BC
from jacobson import tendencies_jacobson, proceed_timestep_jacobson, \
                    diagnose_fields_jacobson

from jacobson import proceed_timestep_jacobson
from bin.jacobson_cython import proceed_timestep_jacobson_c
from jacobson_cuda import proceed_timestep_jacobson_gpu

from diagnostics import interp_COLPA

from numba import cuda, jit
if wp_old == 'float64':
    from numba import float64

######################################################################################
######################################################################################
######################################################################################

def step_matsuno(GR, GR_NEW, subgrids, F):


    ##############################
    ##############################
    t_start = time.time()
    if comp_mode in [0,1]:
        F.UWIND_OLD[:] = F.UWIND[:]
        F.VWIND_OLD[:] = F.VWIND[:]
        F.POTT_OLD[:]  = F.POTT[:]
        F.QV_OLD[:]    = F.QV[:]
        F.QC_OLD[:]    = F.QC[:]
        F.COLP_OLD[:]  = F.COLP[:]
    elif comp_mode == 2:
        set_equal  [GR.griddim_is, GR.blockdim   , GR.stream](F.UWIND_OLD, F.UWIND)
        set_equal  [GR.griddim_js, GR.blockdim   , GR.stream](F.VWIND_OLD, F.VWIND)
        set_equal  [GR.griddim   , GR.blockdim   , GR.stream](F.POTT_OLD, F.POTT)
        set_equal  [GR.griddim   , GR.blockdim   , GR.stream](F.QV_OLD, F.QV)
        set_equal  [GR.griddim   , GR.blockdim   , GR.stream](F.QC_OLD, F.QC)
        set_equal2D[GR.griddim_xy, GR.blockdim_xy, GR.stream](F.COLP_OLD, F.COLP)
    t_end = time.time()
    GR.step_comp_time += t_end - t_start
    ##############################
    ##############################

    ############################################################
    ############################################################
    ##########     ESTIMATE
    ############################################################
    ############################################################

    ##############################
    ##############################
    tendencies_jacobson(GR, GR_NEW, F, subgrids)
    if comp_mode in [0,1]:
        F.COLP[:] = F.COLP_NEW[:]
    elif comp_mode == 2:
        set_equal2D[GR.griddim_xy, GR.blockdim_xy, GR.stream](F.COLP, F.COLP_NEW)
        GR.stream.synchronize()
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 0:
        F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
                    = proceed_timestep_jacobson(GR,
                            F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND, 
                            F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
                            F.QV_OLD, F.QV, F.QC_OLD, F.QC,
                            F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)


    elif comp_mode == 1:
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
    diagnose_fields_jacobson(GR, F)
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
    tendencies_jacobson(GR, GR_NEW, F, subgrids)
    if comp_mode in [0,1]:
        F.COLP[:] = F.COLP_NEW[:]
    elif comp_mode == 2:
        set_equal2D[GR.griddim_xy, GR.blockdim_xy, GR.stream](F.COLP, F.COLP_NEW)
        GR.stream.synchronize()
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 0:
        F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
                     = proceed_timestep_jacobson(GR,
                            F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
                            F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
                            F.QV_OLD, F.QV, F.QC_OLD, F.QC,
                            F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)
    elif comp_mode == 1:
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
    diagnose_fields_jacobson(GR, F)
    t_end = time.time()
    GR.diag_comp_time += t_end - t_start
    ##############################
    ##############################

    


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:]'],target='gpu')
def set_equal(set_FIELD, get_FIELD):
    i, j, k = cuda.grid(3)
    set_FIELD[i,j,k] = get_FIELD[i,j,k] 
    cuda.syncthreads()

@jit([wp_old+'[:,:  ], '+wp_old+'[:,:  ]'],target='gpu')
def set_equal2D(set_FIELD, get_FIELD):
    i, j = cuda.grid(2)
    set_FIELD[i,j  ] = get_FIELD[i,j  ] 
    cuda.syncthreads()

