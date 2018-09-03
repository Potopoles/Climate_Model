import copy
import time
import numpy as np
from namelist import comp_mode, wp
from boundaries import exchange_BC
from jacobson import tendencies_jacobson, proceed_timestep_jacobson, \
                    diagnose_fields_jacobson

from jacobson import proceed_timestep_jacobson
from bin.jacobson_cython import proceed_timestep_jacobson_c
from jacobson_cuda import proceed_timestep_jacobson_gpu

from diagnostics import interp_COLPA

from numba import cuda, jit
if wp == 'float64':
    from numba import float64

######################################################################################
######################################################################################
######################################################################################

def step_matsuno(GR, subgrids,
            COLP_OLD, COLP, COLP_NEW, dCOLPdt, PHI, PHIVB, \
            POTT_OLD, POTT, dPOTTdt, POTTVB,
            UWIND_OLD, UWIND, VWIND_OLD, VWIND, WWIND,
            UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
            BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
            HSURF, PVTF, PVTFVB,
            dPOTTdt_RAD, dPOTTdt_MIC,
            QV_OLD, QV, dQVdt,
            QC_OLD, QC, dQCdt,
            dQVdt_MIC, dQCdt_MIC):

    stream = GR.stream

    ##############################
    ##############################
    t_start = time.time()
    if comp_mode in [0,1]:
        UWIND_OLD[:] = UWIND[:]
        VWIND_OLD[:] = VWIND[:]
        POTT_OLD[:]  = POTT[:]
        QV_OLD[:]    = QV[:]
        QC_OLD[:]    = QC[:]
        COLP_OLD[:]  = COLP[:]
    elif comp_mode == 2:
        set_equal  [GR.griddim_is, GR.blockdim   , stream](UWIND_OLD, UWIND)
        set_equal  [GR.griddim_js, GR.blockdim   , stream](VWIND_OLD, VWIND)
        set_equal  [GR.griddim   , GR.blockdim   , stream](POTT_OLD, POTT)
        set_equal  [GR.griddim   , GR.blockdim   , stream](QV_OLD, QV)
        set_equal  [GR.griddim   , GR.blockdim   , stream](QC_OLD, QC)
        set_equal2D[GR.griddim_xy, GR.blockdim_xy, stream](COLP_OLD, COLP)
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
    if comp_mode in [0,1]:
        COLP_NEW, dUFLXdt, dVFLXdt, \
        dPOTTdt, WWIND,\
        dQVdt, dQCdt = tendencies_jacobson(GR, subgrids, stream,
                                COLP_OLD, COLP, COLP_NEW, dCOLPdt,
                                POTT, dPOTTdt, POTTVB,
                                UWIND, VWIND, WWIND,
                                UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                                BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                PHI, PVTF, PVTFVB,
                                dPOTTdt_RAD, dPOTTdt_MIC,
                                QV, dQVdt, QC, dQCdt, dQVdt_MIC, dQCdt_MIC)
        COLP[:] = COLP_NEW[:]

    elif comp_mode == 2:
        COLP_NEW, dUFLXdt, dVFLXdt, \
        dPOTTdt, WWIND,\
        dQVdt, dQCdt = tendencies_jacobson(GR, subgrids, stream,
                                COLP_OLD, COLP, COLP_NEW, dCOLPdt,
                                POTT, dPOTTdt, POTTVB,
                                UWIND, VWIND, WWIND,
                                UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                                BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                PHI, PVTF, PVTFVB,
                                dPOTTdt_RAD, dPOTTdt_MIC,
                                QV, dQVdt, QC, dQCdt, dQVdt_MIC, dQCdt_MIC)
        set_equal2D[GR.griddim_xy, GR.blockdim_xy, stream](COLP, COLP_NEW)
        stream.synchronize()
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 0:
        UWIND, VWIND, COLP, POTT, QV, QC \
                    = proceed_timestep_jacobson(GR, UWIND_OLD, UWIND, VWIND_OLD, VWIND, 
                            COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                            dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt)

    elif comp_mode == 1:
        UWIND, VWIND, COLP, POTT, QV, QC \
                     = proceed_timestep_jacobson_c(GR, UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                            COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                            dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt)
        UWIND = np.asarray(UWIND)
        VWIND = np.asarray(VWIND)
        COLP = np.asarray(COLP)
        POTT = np.asarray(POTT)
        QV = np.asarray(QV)
        QC = np.asarray(QC)

    elif comp_mode == 2:
        UWIND, VWIND, COLP, POTT, QV, QC \
                     = proceed_timestep_jacobson_gpu(GR, stream,
                            UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                            COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                            dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt, GR.Ad)

    t_end = time.time()
    GR.step_comp_time += t_end - t_start
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    PHI, PHIVB, PVTF, PVTFVB, POTTVB = \
                diagnose_fields_jacobson(GR, stream, PHI, PHIVB, COLP, POTT, \
                                        HSURF, PVTF, PVTFVB, POTTVB)
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
    if comp_mode in [0,1]:
        COLP_NEW, dUFLXdt, dVFLXdt, \
        dPOTTdt, WWIND, \
        dQVdt, dQCdt = tendencies_jacobson(GR, subgrids, stream,
                                COLP_OLD, COLP, COLP_NEW, dCOLPdt,
                                POTT, dPOTTdt, POTTVB,
                                UWIND, VWIND, WWIND,
                                UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                                BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                PHI, PVTF, PVTFVB,
                                dPOTTdt_RAD, dPOTTdt_MIC,
                                QV, dQVdt, QC, dQCdt, dQVdt_MIC, dQCdt_MIC)
        COLP[:] = COLP_NEW[:]

    elif comp_mode == 2:
        COLP_NEW, dUFLXdt, dVFLXdt, \
        dPOTTdt, WWIND,\
        dQVdt, dQCdt = tendencies_jacobson(GR, subgrids, stream,
                                COLP_OLD, COLP, COLP_NEW, dCOLPdt,
                                POTT, dPOTTdt, POTTVB,
                                UWIND, VWIND, WWIND,
                                UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                                BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                PHI, PVTF, PVTFVB,
                                dPOTTdt_RAD, dPOTTdt_MIC,
                                QV, dQVdt, QC, dQCdt, dQVdt_MIC, dQCdt_MIC)
        set_equal2D[GR.griddim_xy, GR.blockdim_xy, stream](COLP, COLP_NEW)
        stream.synchronize()
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    if comp_mode == 0:
        UWIND, VWIND, COLP, POTT, QV, QC \
                     = proceed_timestep_jacobson(GR, UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                            COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                           dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt)
    elif comp_mode == 1:
        UWIND, VWIND, COLP, POTT, QV, QC \
                     = proceed_timestep_jacobson_c(GR, UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                            COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                            dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt)
        UWIND = np.asarray(UWIND)
        VWIND = np.asarray(VWIND)
        COLP = np.asarray(COLP)
        POTT = np.asarray(POTT)
        QV = np.asarray(QV)
        QC = np.asarray(QC)

    elif comp_mode == 2:
        UWIND, VWIND, COLP, POTT, QV, QC \
                     = proceed_timestep_jacobson_gpu(GR, stream,
                            UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                            COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                            dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt, GR.Ad)

    t_end = time.time()
    GR.step_comp_time += t_end - t_start
    ##############################
    ##############################


    ##############################
    ##############################
    t_start = time.time()
    PHI, PHIVB, PVTF, PVTFVB, POTTVB = \
                diagnose_fields_jacobson(GR, stream, PHI, PHIVB, COLP, POTT, \
                                        HSURF, PVTF, PVTFVB, POTTVB)
    t_end = time.time()
    GR.diag_comp_time += t_end - t_start
    ##############################
    ##############################

    
    return(COLP, PHI, PHIVB, POTT, POTTVB,
            UWIND, VWIND, WWIND,
            UFLX, VFLX, QV, QC)




@jit([wp+'[:,:,:], '+wp+'[:,:,:]'],target='gpu')
def set_equal(set_FIELD, get_FIELD):
    i, j, k = cuda.grid(3)
    set_FIELD[i,j,k] = get_FIELD[i,j,k] 
    cuda.syncthreads()

@jit([wp+'[:,:  ], '+wp+'[:,:  ]'],target='gpu')
def set_equal2D(set_FIELD, get_FIELD):
    i, j = cuda.grid(2)
    set_FIELD[i,j  ] = get_FIELD[i,j  ] 
    cuda.syncthreads()

