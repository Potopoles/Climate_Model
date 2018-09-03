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

    if comp_mode in [0,1]:
        UWIND_OLD[:] = UWIND[:]
        VWIND_OLD[:] = VWIND[:]
        POTT_OLD[:]  = POTT[:]
        QV_OLD[:]    = QV[:]
        QC_OLD[:]    = QC[:]
        COLP_OLD[:]  = COLP[:]
    elif comp_mode == 2:
        set_equal  [GR.griddim_is, GR.blockdim, stream](UWIND_OLD, UWIND)
        set_equal  [GR.griddim_js, GR.blockdim, stream](VWIND_OLD, VWIND)
        set_equal  [GR.griddim   , GR.blockdim, stream](POTT_OLD, POTT)
        set_equal  [GR.griddim   , GR.blockdim, stream](QV_OLD, QV)
        set_equal  [GR.griddim   , GR.blockdim, stream](QC_OLD, QC)
        set_equal2D[GR.griddim   , GR.blockdim, stream](COLP_OLD, COLP)

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
        ##############################
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
        ##############################

        ###############################
        #t_start = time.time()
        #stream = cuda.stream()
        #COLP_OLDd     = cuda.to_device(COLP_OLD, stream)
        #COLPd         = cuda.to_device(COLP, stream)
        #COLP_NEWd     = cuda.to_device(COLP_NEW, stream)
        #dCOLPdtd      = cuda.to_device(dCOLPdt, stream)
        #UFLXd         = cuda.to_device(UFLX, stream)
        #dUFLXdtd      = cuda.to_device(dUFLXdt, stream)
        #VFLXd         = cuda.to_device(VFLX, stream)
        #dVFLXdtd      = cuda.to_device(dVFLXdt, stream)
        #FLXDIVd       = cuda.to_device(FLXDIV, stream)
        #BFLXd         = cuda.to_device(BFLX, stream)
        #CFLXd         = cuda.to_device(CFLX, stream)
        #DFLXd         = cuda.to_device(DFLX, stream)
        #EFLXd         = cuda.to_device(EFLX, stream)
        #RFLXd         = cuda.to_device(RFLX, stream)
        #QFLXd         = cuda.to_device(QFLX, stream)
        #SFLXd         = cuda.to_device(SFLX, stream)
        #TFLXd         = cuda.to_device(TFLX, stream)
        #UWIND_OLDd    = cuda.to_device(UWIND_OLD, stream)
        #UWINDd        = cuda.to_device(UWIND, stream)
        #VWIND_OLDd    = cuda.to_device(VWIND_OLD, stream)
        #VWINDd        = cuda.to_device(VWIND, stream)
        #WWINDd        = cuda.to_device(WWIND, stream)
        #POTT_OLDd     = cuda.to_device(POTT_OLD, stream)
        #POTTd         = cuda.to_device(POTT, stream)
        #POTTVBd       = cuda.to_device(POTTVB, stream)
        #dPOTTdtd      = cuda.to_device(dPOTTdt, stream)
        #dPOTTdt_RADd  = cuda.to_device(dPOTTdt_RAD, stream)
        #dPOTTdt_MICd  = cuda.to_device(dPOTTdt_MIC, stream)
        #PVTFd         = cuda.to_device(PVTF, stream)
        #PVTFVBd       = cuda.to_device(PVTFVB, stream)
        #PHId          = cuda.to_device(PHI, stream)
        #QV_OLDd       = cuda.to_device(QV_OLD, stream)
        #QVd           = cuda.to_device(QV, stream)
        #dQVdtd        = cuda.to_device(dQVdt, stream)
        #dQVdt_MICd    = cuda.to_device(dQVdt_MIC, stream)
        #QC_OLDd       = cuda.to_device(QC_OLD, stream)
        #QCd           = cuda.to_device(QC, stream)
        #dQCdtd        = cuda.to_device(dQCdt, stream)
        #dQCdt_MICd    = cuda.to_device(dQCdt_MIC, stream)

        #GR.Ad            = cuda.to_device(GR.A, stream)
        #GR.dsigmad       = cuda.to_device(GR.dsigma, stream)
        #GR.sigma_vbd     = cuda.to_device(GR.sigma_vb, stream)
        #GR.dxjsd         = cuda.to_device(GR.dxjs, stream)
        #GR.corfd         = cuda.to_device(GR.corf, stream)
        #GR.corf_isd      = cuda.to_device(GR.corf_is, stream)
        #GR.lat_radd      = cuda.to_device(GR.lat_rad, stream)
        #GR.latis_radd    = cuda.to_device(GR.latis_rad, stream)

        #t_end = time.time()
        #GR.copy_time += t_end - t_start
        ###############################

        ###############################
        #COLP_NEWd, dUFLXdtd, dVFLXdtd, \
        #dPOTTdtd, WWINDd,\
        #dQVdtd, dQCdtd = tendencies_jacobson(GR, subgrids, stream,
        #                        COLP_OLDd, COLPd, COLP_NEWd, dCOLPdtd,
        #                        POTTd, dPOTTdtd, POTTVBd,
        #                        UWINDd, VWINDd, WWINDd,
        #                        UFLXd, dUFLXdtd, VFLXd, dVFLXdtd, FLXDIVd,
        #                        BFLXd, CFLXd, DFLXd, EFLXd, RFLXd, QFLXd, SFLXd, TFLXd, 
        #                        PHId, PVTFd, PVTFVBd,
        #                        dPOTTdt_RADd, dPOTTdt_MICd,
        #                        QVd, dQVdtd, QCd, dQCdtd, dQVdt_MICd, dQCdt_MICd)
        #COLPd[:] = COLP_NEWd[:]
        ###############################
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

        #UWINDd, VWINDd, COLPd, POTTd, QVd, QCd \
        #             = proceed_timestep_jacobson_gpu(GR, stream,
        #                    UWIND_OLDd, UWINDd, VWIND_OLDd, VWINDd,
        #                    COLP_OLDd, COLPd, POTT_OLDd, POTTd, QV_OLDd, QVd, QC_OLDd, QCd,
        #                    dUFLXdtd, dVFLXdtd, dPOTTdtd, dQVdtd, dQCdtd, GR.Ad)

        ###############################
        #t_start = time.time()
        #COLPd     .to_host(stream)
        #COLP_NEWd .to_host(stream)
        #UWINDd    .to_host(stream)
        #VWINDd    .to_host(stream)
        #WWINDd    .to_host(stream) 
        #POTTd     .to_host(stream)
        #QVd       .to_host(stream)
        #QCd       .to_host(stream)

        #stream.synchronize()

        #t_end = time.time()
        #GR.copy_time += t_end - t_start
        ###############################


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
        ##############################
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
        ##############################

        ###############################
        #t_start = time.time()
        #stream = cuda.stream()
        #COLP_OLDd     = cuda.to_device(COLP_OLD, stream)
        #COLPd         = cuda.to_device(COLP, stream)
        #COLP_NEWd     = cuda.to_device(COLP_NEW, stream)
        #dCOLPdtd      = cuda.to_device(dCOLPdt, stream)
        #UFLXd         = cuda.to_device(UFLX, stream)
        #dUFLXdtd      = cuda.to_device(dUFLXdt, stream)
        #VFLXd         = cuda.to_device(VFLX, stream)
        #dVFLXdtd      = cuda.to_device(dVFLXdt, stream)
        #FLXDIVd       = cuda.to_device(FLXDIV, stream)
        #BFLXd         = cuda.to_device(BFLX, stream)
        #CFLXd         = cuda.to_device(CFLX, stream)
        #DFLXd         = cuda.to_device(DFLX, stream)
        #EFLXd         = cuda.to_device(EFLX, stream)
        #RFLXd         = cuda.to_device(RFLX, stream)
        #QFLXd         = cuda.to_device(QFLX, stream)
        #SFLXd         = cuda.to_device(SFLX, stream)
        #TFLXd         = cuda.to_device(TFLX, stream)
        #UWINDd        = cuda.to_device(UWIND, stream)
        #VWINDd        = cuda.to_device(VWIND, stream)
        #WWINDd        = cuda.to_device(WWIND, stream)
        #POTTd         = cuda.to_device(POTT, stream)
        #POTTVBd       = cuda.to_device(POTTVB, stream)
        #dPOTTdtd      = cuda.to_device(dPOTTdt, stream)
        #dPOTTdt_RADd  = cuda.to_device(dPOTTdt_RAD, stream)
        #dPOTTdt_MICd  = cuda.to_device(dPOTTdt_MIC, stream)
        #PVTFd         = cuda.to_device(PVTF, stream)
        #PVTFVBd       = cuda.to_device(PVTFVB, stream)
        #PHId          = cuda.to_device(PHI, stream)
        #QVd           = cuda.to_device(QV, stream)
        #dQVdtd        = cuda.to_device(dQVdt, stream)
        #dQVdt_MICd    = cuda.to_device(dQVdt_MIC, stream)
        #QCd           = cuda.to_device(QC, stream)
        #dQCdtd        = cuda.to_device(dQCdt, stream)
        #dQCdt_MICd    = cuda.to_device(dQCdt_MIC, stream)

        #t_end = time.time()
        #GR.copy_time += t_end - t_start
        ###############################

        ###############################
        #COLP_NEWd, dUFLXdtd, dVFLXdtd, \
        #dPOTTdtd, WWINDd,\
        #dQVdtd, dQCdtd = tendencies_jacobson(GR, subgrids, stream,
        #                        COLP_OLDd, COLPd, COLP_NEWd, dCOLPdtd,
        #                        POTTd, dPOTTdtd, POTTVBd,
        #                        UWINDd, VWINDd, WWINDd,
        #                        UFLXd, dUFLXdtd, VFLXd, dVFLXdtd, FLXDIVd,
        #                        BFLXd, CFLXd, DFLXd, EFLXd, RFLXd, QFLXd, SFLXd, TFLXd, 
        #                        PHId, PVTFd, PVTFVBd,
        #                        dPOTTdt_RADd, dPOTTdt_MICd,
        #                        QVd, dQVdtd, QCd, dQCdtd, dQVdt_MICd, dQCdt_MICd)
        #COLPd[:] = COLP_NEWd[:]
        ###############################
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

        #UWINDd, VWINDd, COLPd, POTTd, QVd, QCd \
        #             = proceed_timestep_jacobson_gpu(GR, stream,
        #                    UWIND_OLDd, UWINDd, VWIND_OLDd, VWINDd,
        #                    COLP_OLDd, COLPd, POTT_OLDd, POTTd, QV_OLDd, QVd, QC_OLDd, QCd,
        #                    dUFLXdtd, dVFLXdtd, dPOTTdtd, dQVdtd, dQCdtd, GR.Ad)

        ###############################
        #t_start = time.time()
        #COLPd     .to_host(stream)
        #COLP_NEWd .to_host(stream)
        #UWINDd    .to_host(stream)
        #VWINDd    .to_host(stream)
        #WWINDd    .to_host(stream) 
        #POTTd     .to_host(stream)
        #QVd       .to_host(stream)
        #QCd       .to_host(stream)

        #stream.synchronize()

        #t_end = time.time()
        #GR.copy_time += t_end - t_start
        ###############################

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

