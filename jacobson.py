import copy
import matplotlib.pyplot as plt
import numpy as np
import time
from namelist import pTop, njobs, comp_mode
from constants import con_Rd

from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from bin.continuity_cython import colp_tendency_jacobson_c, vertical_wind_jacobson_c
from continuity_cuda import colp_tendency_jacobson_gpu, vertical_wind_jacobson_gpu

from wind import wind_tendency_jacobson
from bin.wind_cython import wind_tendency_jacobson_c
from wind_cuda import wind_tendency_jacobson_gpu

from temperature import temperature_tendency_jacobson
from bin.temperature_cython import temperature_tendency_jacobson_c
from temperature_cuda import temperature_tendency_jacobson_gpu

from moisture import water_vapor_tendency, cloud_water_tendency
from bin.moisture_cython import water_vapor_tendency_c, cloud_water_tendency_c
from moisture_cuda import water_vapor_tendency_gpu, cloud_water_tendency_gpu

#from geopotential import diag_geopotential_jacobson
from bin.geopotential_cython import diag_geopotential_jacobson_c

#from diagnostics import diagnose_POTTVB_jacobson, interp_COLPA
from bin.diagnostics_cython import diagnose_POTTVB_jacobson_c, interp_COLPA_c

from boundaries import exchange_BC
from boundaries_cuda import exchange_BC_gpu

from jacobson_cuda import time_step_2D

from numba import cuda
import numba



def tendencies_jacobson(GR, subgrids, stream,\
                    COLP_OLD, COLP, COLP_NEW, dCOLPdt, POTT, dPOTTdt, POTTVB,
                    UWIND, VWIND, WWIND,
                    UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                    BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                    PHI, PVTF, PVTFVB,
                    dPOTTdt_RAD, dPOTTdt_MIC,
                    QV, dQVdt, QC, dQCdt, dQVdt_MIC, dQCdt_MIC):

    ###############################
    ###############################
    #if comp_mode == 2:
    #    t_start = time.time()

    #    stream = cuda.stream()
    #    COLP_OLDd     = cuda.to_device(COLP_OLD, stream)
    #    COLPd         = cuda.to_device(COLP, stream)
    #    COLP_NEWd     = cuda.to_device(COLP_NEW, stream)
    #    dCOLPdtd      = cuda.to_device(dCOLPdt, stream)
    #    UFLXd         = cuda.to_device(UFLX, stream)
    #    dUFLXdtd      = cuda.to_device(dUFLXdt, stream)
    #    VFLXd         = cuda.to_device(VFLX, stream)
    #    dVFLXdtd      = cuda.to_device(dVFLXdt, stream)
    #    FLXDIVd       = cuda.to_device(FLXDIV, stream)
    #    BFLXd         = cuda.to_device(BFLX, stream)
    #    CFLXd         = cuda.to_device(CFLX, stream)
    #    DFLXd         = cuda.to_device(DFLX, stream)
    #    EFLXd         = cuda.to_device(EFLX, stream)
    #    RFLXd         = cuda.to_device(RFLX, stream)
    #    QFLXd         = cuda.to_device(QFLX, stream)
    #    SFLXd         = cuda.to_device(SFLX, stream)
    #    TFLXd         = cuda.to_device(TFLX, stream)
    #    UWINDd        = cuda.to_device(UWIND, stream)
    #    VWINDd        = cuda.to_device(VWIND, stream)
    #    WWINDd        = cuda.to_device(WWIND, stream)
    #    POTTd         = cuda.to_device(POTT, stream)
    #    POTTVBd       = cuda.to_device(POTTVB, stream)
    #    dPOTTdtd      = cuda.to_device(dPOTTdt, stream)
    #    dPOTTdt_RADd  = cuda.to_device(dPOTTdt_RAD, stream)
    #    dPOTTdt_MICd  = cuda.to_device(dPOTTdt_MIC, stream)
    #    PVTFd         = cuda.to_device(PVTF, stream)
    #    PVTFVBd       = cuda.to_device(PVTFVB, stream)
    #    PHId          = cuda.to_device(PHI, stream)
    #    QVd           = cuda.to_device(QV, stream)
    #    dQVdtd        = cuda.to_device(dQVdt, stream)
    #    dQVdt_MICd    = cuda.to_device(dQVdt_MIC, stream)
    #    QCd           = cuda.to_device(QC, stream)
    #    dQCdtd        = cuda.to_device(dQCdt, stream)
    #    dQCdt_MICd    = cuda.to_device(dQCdt_MIC, stream)

    #    Ad            = cuda.to_device(GR.A, stream)
    #    dsigmad       = cuda.to_device(GR.dsigma, stream)
    #    sigma_vbd     = cuda.to_device(GR.sigma_vb, stream)
    #    dxjsd         = cuda.to_device(GR.dxjs, stream)
    #    corfd         = cuda.to_device(GR.corf, stream)
    #    corf_isd      = cuda.to_device(GR.corf_is, stream)
    #    lat_radd      = cuda.to_device(GR.lat_rad, stream)
    #    latis_radd    = cuda.to_device(GR.latis_rad, stream)

    #    stream.synchronize()

    #    t_end = time.time()
    #    GR.copy_time += t_end - t_start
    ###############################
    ###############################



    ##############################
    ##############################
    t_start = time.time()
    # PROGNOSE COLP
    if comp_mode == 0:
        dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND, VWIND, \
                                                            dCOLPdt, UFLX, VFLX, FLXDIV)
        COLP_NEW[GR.iijj] = COLP_OLD[GR.iijj] + GR.dt*dCOLPdt[GR.iijj]

    elif comp_mode == 1:
        dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson_c(GR, COLP, UWIND, VWIND, \
                                                             dCOLPdt, UFLX, VFLX, FLXDIV)
        dCOLPdt = np.asarray(dCOLPdt)
        UFLX = np.asarray(UFLX)
        VFLX = np.asarray(VFLX)
        FLXDIV = np.asarray(FLXDIV)
        COLP_NEW[GR.iijj] = COLP_OLD[GR.iijj] + GR.dt*dCOLPdt[GR.iijj]

    elif comp_mode == 2:
        #dCOLPdtd, UFLXd, VFLXd, FLXDIVd = \
        #             colp_tendency_jacobson_gpu(GR, GR.griddim, GR.blockdim, stream,\
        #                                        dCOLPdtd, UFLXd, VFLXd, FLXDIVd,\
        #                                        COLPd, UWINDd, VWINDd, \
        #                                        GR.dy, dxjsd, Ad, dsigmad)
        #time_step_2D[GR.griddim, GR.blockdim, stream]\
        #                    (COLP_NEWd, COLP_OLDd, dCOLPdtd, GR.dt)
        dCOLPdt, UFLX, VFLX, FLXDIV = \
                     colp_tendency_jacobson_gpu(GR, GR.griddim, GR.blockdim, stream,\
                                                dCOLPdt, UFLX, VFLX, FLXDIV,\
                                                COLP, UWIND, VWIND, \
                                                GR.dy, GR.dxjs, GR.A, GR.dsigma)
        time_step_2D[GR.griddim, GR.blockdim, stream]\
                            (COLP_NEW, COLP_OLD, dCOLPdt, GR.dt)


    # DIAGNOSE WWIND
    if comp_mode == 0:
        WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)
        # TODO 2 NECESSARY
        COLP_NEW = exchange_BC(GR, COLP_NEW)
        WWIND = exchange_BC(GR, WWIND)

    elif comp_mode == 1:
        WWIND = vertical_wind_jacobson_c(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)
        WWIND = np.asarray(WWIND)
        # TODO 2 NECESSARY
        COLP_NEW = exchange_BC(GR, COLP_NEW)
        WWIND = exchange_BC(GR, WWIND)

    elif comp_mode == 2:
        #vertical_wind_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks, stream]\
        #                                (WWINDd, dCOLPdtd, FLXDIVd, COLP_NEWd, sigma_vbd)
        ## TODO 2 NECESSARY
        #COLP_NEWd = exchange_BC_gpu(COLP_NEWd, GR.zonal, GR.merid,
        #                            GR.griddim_xy, GR.blockdim_xy, stream, array2D=True)
        #WWINDd = exchange_BC_gpu(WWINDd, GR.zonals, GR.merid,
        #                            GR.griddim_ks, GR.blockdim_ks, stream)
        vertical_wind_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks, stream]\
                                        (WWIND, dCOLPdt, FLXDIV, COLP_NEW, GR.sigma_vb)
        # TODO 2 NECESSARY
        COLP_NEWd = exchange_BC_gpu(COLP_NEW, GR.zonal, GR.merid,
                                    GR.griddim_xy, GR.blockdim_xy, stream, array2D=True)
        WWINDd = exchange_BC_gpu(WWIND, GR.zonals, GR.merid,
                                    GR.griddim_ks, GR.blockdim_ks, stream)

    t_end = time.time()
    GR.cont_comp_time += t_end - t_start
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    # PROGNOSE WIND
    if comp_mode == 0:
        dUFLXdt, dVFLXdt = wind_tendency_jacobson(GR, UWIND, VWIND, WWIND,
                                        UFLX, dUFLXdt, VFLX, dVFLXdt,
                                        BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                        COLP, COLP_NEW, PHI, POTT,
                                        PVTF, PVTFVB)

    elif comp_mode == 1:
        dUFLXdt, dVFLXdt = wind_tendency_jacobson_c(GR, njobs, UWIND, VWIND, WWIND,
                                        UFLX, dUFLXdt, VFLX, dVFLXdt,
                                        BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                        COLP, COLP_NEW, PHI,
                                        POTT, PVTF, PVTFVB)
        dUFLXdt = np.asarray(dUFLXdt)
        dVFLXdt = np.asarray(dVFLXdt)

    elif comp_mode == 2:
        #dUFLXdtd, dVFLXdtd = wind_tendency_jacobson_gpu(GR, UWINDd, VWINDd, WWINDd,
        #                    UFLXd, dUFLXdtd, VFLXd, dVFLXdtd,
        #                    BFLXd, CFLXd, DFLXd, EFLXd, RFLXd, QFLXd, SFLXd, TFLXd, 
        #                    COLPd, COLP_NEWd, PHId, POTTd, PVTFd, PVTFVBd,
        #                    Ad, dsigmad, sigma_vbd, corfd, corf_isd, lat_radd, latis_radd,
        #                    GR.dy, GR.dlon_rad, dxjsd,
        #                    stream)
        dUFLXdt, dVFLXdt = wind_tendency_jacobson_gpu(GR, UWIND, VWIND, WWIND,
                            UFLX, dUFLXdt, VFLX, dVFLXdt,
                            BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                            COLP, COLP_NEW, PHI, POTT, PVTF, PVTFVB,
                            GR.A, GR.dsigma, GR.sigma_vb, GR.corf, GR.corf_is,
                            GR.lat_rad, GR.latis_rad, GR.dy, GR.dlon_rad, GR.dxjs,
                            stream)

    t_end = time.time()
    GR.wind_comp_time += t_end - t_start
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    # PROGNOSE POTT
    if comp_mode == 0:
        dPOTTdt = temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW,\
                                                UFLX, VFLX, WWIND, \
                                                dPOTTdt_RAD, dPOTTdt_MIC)

    elif comp_mode == 1:
        dPOTTdt = temperature_tendency_jacobson_c(GR, njobs, POTT, POTTVB, COLP, COLP_NEW,\
                                                UFLX, VFLX, WWIND, \
                                                dPOTTdt_RAD, dPOTTdt_MIC)
        dPOTTdt = np.asarray(dPOTTdt)

    elif comp_mode == 2:
        #temperature_tendency_jacobson_gpu[GR.griddim, GR.blockdim, stream] \
        #                                    (dPOTTdtd, \
        #                                    POTTd, POTTVBd, COLPd, COLP_NEWd, \
        #                                    UFLXd, VFLXd, WWINDd, \
        #                                    dPOTTdt_RADd, dPOTTdt_MICd, \
        #                                    Ad, dsigmad)
        temperature_tendency_jacobson_gpu[GR.griddim, GR.blockdim, stream] \
                                            (dPOTTdt, 
                                            POTT, POTTVB, COLP, COLP_NEW, 
                                            UFLX, VFLX, WWIND, 
                                            dPOTTdt_RAD, dPOTTdt_MIC, 
                                            GR.A, GR.dsigma)

    t_end = time.time()
    GR.temp_comp_time += t_end - t_start
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    # MOIST VARIABLES
    if comp_mode == 0:
        dQVdt = water_vapor_tendency(GR, dQVdt, QV, COLP, COLP_NEW, \
                                        UFLX, VFLX, WWIND, dQVdt_MIC)
        dQCdt = cloud_water_tendency(GR, dQCdt, QC, COLP, COLP_NEW, \
                                        UFLX, VFLX, WWIND, dQCdt_MIC)

    elif comp_mode == 1:
        dQVdt = water_vapor_tendency_c(GR, njobs, dQVdt, QV, COLP, COLP_NEW,
                                        UFLX, VFLX, WWIND, dQVdt_MIC)
        dQVdt = np.asarray(dQVdt)
        dQCdt = cloud_water_tendency_c(GR, njobs, dQCdt, QC, COLP, COLP_NEW,
                                        UFLX, VFLX, WWIND, dQCdt_MIC)
        dQCdt = np.asarray(dQCdt)

    elif comp_mode == 2:
        #water_vapor_tendency_gpu[GR.griddim, GR.blockdim, stream] \
        #                            (dQVdtd, QVd, COLPd, COLP_NEWd,
        #                             UFLXd, VFLXd, WWINDd, dQVdt_MICd,
        #                             Ad, dsigmad)
        #cloud_water_tendency_gpu[GR.griddim, GR.blockdim, stream] \
        #                            (dQCdtd, QCd, COLPd, COLP_NEWd,
        #                             UFLXd, VFLXd, WWINDd, dQCdt_MICd,
        #                             Ad, dsigmad)
        water_vapor_tendency_gpu[GR.griddim, GR.blockdim, stream] \
                                    (dQVdt, QV, COLP, COLP_NEW,
                                     UFLX, VFLX, WWIND, dQVdt_MIC,
                                     GR.A, GR.dsigma)
        cloud_water_tendency_gpu[GR.griddim, GR.blockdim, stream] \
                                    (dQCdt, QC, COLP, COLP_NEW,
                                     UFLX, VFLX, WWIND, dQCdt_MIC,
                                     GR.A, GR.dsigma)

    t_end = time.time()
    GR.trac_comp_time += t_end - t_start
    ##############################
    ##############################


    ###############################
    ###############################
    #if comp_mode == 2:
    #    t_start = time.time()

    #    COLPd     .to_host(stream)
    #    COLP_NEWd .to_host(stream)
    #    UFLXd     .to_host(stream)
    #    dUFLXdtd  .to_host(stream)
    #    VFLXd     .to_host(stream)
    #    dVFLXdtd  .to_host(stream)
    #    WWINDd    .to_host(stream) 
    #    dPOTTdtd  .to_host(stream)
    #    dQVdtd    .to_host(stream)
    #    dQCdtd    .to_host(stream)

    #    stream.synchronize()

    #    t_end = time.time()
    #    GR.copy_time += t_end - t_start
    ###############################
    ###############################


    return(COLP_NEW, dUFLXdt, dVFLXdt, dPOTTdt, WWIND, dQVdt, dQCdt)


def proceed_timestep_jacobson(GR, UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                    COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                    dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt):

    #import copy
    #UWIND = copy.copy(UWIND_OLD)

    # TIME STEPPING
    #COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA_c(GR, njobs, COLP_OLD)

    #COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA_c(GR, njobs, COLP)

    for k in range(0,GR.nz):
        UWIND[:,:,k][GR.iisjj] = UWIND_OLD[:,:,k][GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                            + GR.dt*dUFLXdt[:,:,k][GR.iisjj]/COLPA_is_NEW
        VWIND[:,:,k][GR.iijjs] = VWIND_OLD[:,:,k][GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                            + GR.dt*dVFLXdt[:,:,k][GR.iijjs]/COLPA_js_NEW
        POTT[:,:,k][GR.iijj] = POTT_OLD[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dPOTTdt[:,:,k][GR.iijj]/COLP[GR.iijj]
        QV[:,:,k][GR.iijj] = QV_OLD[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dQVdt[:,:,k][GR.iijj]/COLP[GR.iijj]
        QC[:,:,k][GR.iijj] = QC_OLD[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dQCdt[:,:,k][GR.iijj]/COLP[GR.iijj]
    QV[QV < 0] = 0
    QC[QC < 0] = 0

    # TODO 4 NECESSARY
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)
    #POTT = exchange_BC(GR, POTT)
    QV = exchange_BC(GR, QV)
    QC = exchange_BC(GR, QC)

    stream = cuda.stream()
    POTTd        = cuda.to_device(POTT, stream)
    POTTd = exchange_BC_gpu(POTTd, GR.zonal, GR.merid, GR.griddim, GR.blockdim, stream)
    POTTd.to_host(stream)
    stream.synchronize()

    return(UWIND, VWIND, COLP, POTT, QV, QC)


def diagnose_fields_jacobson(GR, PHI, PHIVB, COLP, POTT, HSURF, PVTF, PVTFVB, POTTVB):


    #PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, PHIVB, HSURF, 
    #                                            POTT, COLP, PVTF, PVTFVB)

    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson_c(GR, njobs, PHI, PHIVB, HSURF, 
                                                    POTT, COLP, PVTF, PVTFVB)
    PHI = np.asarray(PHI)
    PHIVB = np.asarray(PHIVB)
    PVTF = np.asarray(PVTF)
    PVTFVB = np.asarray(PVTFVB)




    #POTTVB = diagnose_POTTVB_jacobson(GR, POTTVB, POTT, PVTF, PVTFVB)

    POTTVB = diagnose_POTTVB_jacobson_c(GR, njobs, POTTVB, POTT, PVTF, PVTFVB)
    POTTVB = np.asarray(POTTVB)


    #TURB.diag_rho(GR, COLP, POTT, PVTF, POTTVB, PVTFVB)
    #TURB.diag_dz(GR, PHI, PHIVB)


    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)




