import copy
import matplotlib.pyplot as plt
import numpy as np
import time
from namelist import pTop, njobs
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

#from geopotential import diag_geopotential_jacobson
from bin.geopotential_cython import diag_geopotential_jacobson_c

#from diagnostics import diagnose_POTTVB_jacobson, interp_COLPA
from bin.diagnostics_cython import diagnose_POTTVB_jacobson_c, interp_COLPA_c

#from moisture import water_vapor_tendency, cloud_water_tendency
from bin.moisture_cython import water_vapor_tendency_c, cloud_water_tendency_c

from boundaries import exchange_BC
from boundaries_cuda import exchange_BC_gpu

from jacobson_cuda import time_step_2D

from numba import cuda
import numba


def tendencies_jacobson(GR, subgrids,\
                    COLP_OLD, COLP, COLP_NEW, dCOLPdt, POTT, dPOTTdt, POTTVB, HSURF,
                    UWIND, VWIND, WWIND,
                    UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                    BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                    PHI, PVTF, PVTFVB,
                    dPOTTdt_RAD, dPOTTdt_MIC,
                    QV, QC, dQVdt_MIC, dQCdt_MIC):


    ##############################

    t_start = time.time()
    # PROGNOSE COLP
    #dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND, VWIND, \
    #                                                    dCOLPdt, UFLX, VFLX, FLXDIV)
    #dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson_c(GR, COLP, UWIND, VWIND, \
    #                                                     dCOLPdt, UFLX, VFLX, FLXDIV)
    #dCOLPdt = np.asarray(dCOLPdt)
    #UFLX = np.asarray(UFLX)
    #VFLX = np.asarray(VFLX)
    #FLXDIV = np.asarray(FLXDIV)


    #UFLX_cpu = copy.deepcopy(UFLX)
    #VFLX_cpu = copy.deepcopy(VFLX)
    #FLXDIV_cpu = copy.deepcopy(FLXDIV)
    #dCOLPdt_cpu = copy.deepcopy(dCOLPdt)
    #UFLX[GR.iisjj] = 0.
    #VFLX[GR.iijjs] = 0.

    stream = cuda.stream()

    COLPd         = cuda.to_device(COLP, stream)
    dCOLPdtd      = cuda.to_device(dCOLPdt, stream)
    UFLXd         = cuda.to_device(UFLX, stream)
    VFLXd         = cuda.to_device(VFLX, stream)
    FLXDIVd       = cuda.to_device(FLXDIV, stream)
    UWINDd        = cuda.to_device(UWIND, stream)
    VWINDd        = cuda.to_device(VWIND, stream)
    dxjsd         = cuda.to_device(GR.dxjs, stream)
    Ad            = cuda.to_device(GR.A, stream)
    dsigmad       = cuda.to_device(GR.dsigma, stream)
    dxjsd         = cuda.to_device(GR.dxjs, stream)

    dCOLPdtd, UFLXd, VFLXd, FLXDIVd = \
                 colp_tendency_jacobson_gpu(GR, GR.griddim, GR.blockdim, stream,\
                                            dCOLPdtd, UFLXd, VFLXd, FLXDIVd,\
                                            COLPd, UWINDd, VWINDd, \
                                            GR.dy, dxjsd, Ad, dsigmad)

    #quit()




    #dCOLPdtd.to_host(stream)
    #stream.synchronize()
    #COLP_NEW[GR.iijj] = COLP_OLD[GR.iijj] + GR.dt*dCOLPdt[GR.iijj]

    COLP_NEWd         = cuda.to_device(COLP_NEW, stream)
    COLP_OLDd         = cuda.to_device(COLP_OLD, stream)
    time_step_2D[GR.griddim, GR.blockdim, stream]\
                        (COLP_NEWd, COLP_OLDd, dCOLPdtd, GR.dt)
    dCOLPdtd.to_host(stream)
    stream.synchronize()
    #print(COLP_NEW)
    #quit()

    UFLXd.to_host(stream)
    VFLXd.to_host(stream)
    #COLP_OLDd.to_host(stream)
    stream.synchronize()

    # DIAGNOSE WWIND
    #WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)
    #WWIND = vertical_wind_jacobson_c(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)
    #WWIND = np.asarray(WWIND)

    WWINDd        = cuda.to_device(WWIND, stream)
    sigma_vbd     = cuda.to_device(GR.sigma_vb, stream)

    vertical_wind_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks, stream]\
                                    (WWINDd, dCOLPdtd, FLXDIVd, COLP_NEWd, sigma_vbd)

    
    #FLXDIVd.to_host(stream)
    #COLP_NEWd.to_host(stream)
    #WWINDd.to_host(stream)
    # TODO 2 NECESSARY
    #COLP_NEW = exchange_BC(GR, COLP_NEW)
    #WWIND = exchange_BC(GR, WWIND)

    # TODO 2 NECESSARY
    COLP_NEWd = exchange_BC_gpu(COLP_NEWd, GR.zonal, GR.merid,
                                GR.griddim_xy, GR.blockdim_xy, stream, array2D=True)
    WWINDd = exchange_BC_gpu(WWINDd, GR.zonals, GR.merid,
                                GR.griddim_ks, GR.blockdim_ks, stream)

    #FLXDIVd.to_host(stream)
    COLP_NEWd.to_host(stream)
    WWINDd.to_host(stream)
    stream.synchronize()

    t_end = time.time()
    GR.cont_comp_time += t_end - t_start
    ##############################



    ##############################
    #UFLX[:,0,:] = 300
    #UFLX[:,GR.ny+1,:] = 300
    #UFLX[0,:,:] = 300
    #UFLX[GR.nxs+1,:,:] = 300
    #UFLX_gpu = copy.deepcopy(UFLX)
    #UFLX_cpu = copy.deepcopy(UFLX)
    #from fields import gaussian2D
    #for k in range(1,GR.nzs-1):
    #    WWIND[:,:,k] = gaussian2D(GR, WWIND[:,:,k], 1E-4, np.pi, 0,  np.pi/3, np.pi/4)
    t_start = time.time()
    # PROGNOSE WIND
    t0 = time.time()
    dUFLXdt, dVFLXdt, \
    BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX = \
    wind_tendency_jacobson(GR, UWIND, VWIND, WWIND,
                                        UFLX, dUFLXdt, VFLX, dVFLXdt,
                                        BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                        COLP, COLP_NEW, HSURF, PHI, POTT,
                                        PVTF, PVTFVB)
    #dUFLXdt, dVFLXdt = wind_tendency_jacobson_c(GR, njobs, UWIND, VWIND, WWIND,
    #                                        UFLX, dUFLXdt, VFLX, dVFLXdt,
    #                                        BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
    #                                        COLP, COLP_NEW, PHI,
    #                                        POTT, PVTF, PVTFVB)
    #dUFLXdt = np.asarray(dUFLXdt)
    #dVFLXdt = np.asarray(dVFLXdt)
    t1 = time.time()
    print('orig ' + str(t1 - t0))


    dUFLXdt_cpu = copy.deepcopy(dUFLXdt)
    dVFLXdt_cpu = copy.deepcopy(dVFLXdt)
    TFLX_cpu = copy.deepcopy(TFLX)
    #FLXDIV_cpu = copy.deepcopy(FLXDIV)
    #dCOLPdt_cpu = copy.deepcopy(dCOLPdt)
    #UFLX[GR.iisjj] = 0.
    #VFLX[GR.iijjs] = 0.



    t0 = time.time()
    dUFLXdtd, dVFLXdtd, \
    BFLXd, CFLXd, DFLXd, EFLXd, RFLXd, QFLXd, SFLXd, TFLXd = \
    wind_tendency_jacobson_gpu(GR, UWIND, VWIND, WWIND,
                                    UFLX, dUFLXdt, VFLX, dVFLXdt,
                                    BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                    COLP, COLP_NEW, HSURF, PHI, POTT,
                                    PVTF, PVTFVB,
                                    stream)
    t1 = time.time()
    print('gpu  ' + str(t1 - t0))


    dUFLXdtd.to_host(stream)
    dVFLXdtd.to_host(stream)
    BFLXd.to_host(stream)
    CFLXd.to_host(stream)
    DFLXd.to_host(stream)
    EFLXd.to_host(stream)
    RFLXd.to_host(stream)
    QFLXd.to_host(stream)
    SFLXd.to_host(stream)
    TFLXd.to_host(stream)
    stream.synchronize()
    dUFLXdt_gpu = copy.deepcopy(dUFLXdt)
    dVFLXdt_gpu = copy.deepcopy(dVFLXdt)

    var = dVFLXdt_gpu
    var_orig = dVFLXdt_cpu
    #var = dUFLXdt_gpu
    #var_orig = dUFLXdt_cpu
    print('###################')
    nan_here = np.isnan(var)
    nan_orig = np.isnan(var_orig)
    diff = var - var_orig
    nan_diff = nan_here != nan_orig 
    print('values ' + str(np.nansum(np.abs(diff))))
    print('  nans ' + str(np.sum(nan_diff)))
    print('###################')

    quit()
    #plt.contourf(var_orig[:,:,1].T)
    #plt.contourf(var[:,:,1].T)
    plt.contourf(diff[:,:,1].T)
    plt.colorbar()
    plt.show()
    quit()




    t_end = time.time()
    GR.wind_comp_time += t_end - t_start
    ##############################



    ##############################
    t_start = time.time()
    # PROGNOSE POTT
    #dPOTTdt = temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW,\
    #                                        UFLX, VFLX, WWIND, \
    #                                        dPOTTdt_RAD, dPOTTdt_MIC)
    #dPOTTdt = temperature_tendency_jacobson_c(GR, njobs, POTT, POTTVB, COLP, COLP_NEW,\
    #                                        UFLX, VFLX, WWIND, \
    #                                        dPOTTdt_RAD, dPOTTdt_MIC)
    #dPOTTdt = np.asarray(dPOTTdt)
    t_end = time.time()
    GR.temp_comp_time += t_end - t_start
    #cpu_time = t_end - t_start


    stream = cuda.stream()
    dPOTTdtd      = cuda.to_device(dPOTTdt, stream)
    dPOTTdt_RADd  = cuda.to_device(dPOTTdt_RAD, stream)
    dPOTTdt_MICd  = cuda.to_device(dPOTTdt_MIC, stream)
    UFLXd         = cuda.to_device(UFLX, stream)
    VFLXd         = cuda.to_device(VFLX, stream)
    POTTd         = cuda.to_device(POTT, stream)
    POTTVBd       = cuda.to_device(POTTVB, stream)
    COLPd         = cuda.to_device(COLP, stream)
    COLP_NEWd     = cuda.to_device(COLP_NEW, stream)
    WWINDd        = cuda.to_device(WWIND, stream)
    Ad            = cuda.to_device(GR.A, stream)
    dsigmad       = cuda.to_device(GR.dsigma, stream)

    t_start = time.time()
    temperature_tendency_jacobson_gpu[GR.griddim, GR.blockdim, stream] \
                                        (dPOTTdtd, \
                                        POTTd, POTTVBd, COLPd, COLP_NEWd, \
                                        UFLXd, VFLXd, WWINDd, \
                                        dPOTTdt_RADd, dPOTTdt_MICd, \
                                        Ad, dsigmad)
    t_end = time.time()
    time_gpu = t_end - t_start

    dPOTTdtd.to_host(stream)
    stream.synchronize()
    #dPOTTdt_gpu = copy.deepcopy(dPOTTdt)

    #for k in range(0,GR.nz):
    #    POTT_gpu[:,:,k][GR.iijj] = POTT[:,:,k][GR.iijj] + GR.dt*dPOTTdt[:,:,k][GR.iijj]/COLP[GR.iijj]

    #POTT_gpud       = cuda.to_device(POTT_gpu, stream)

    #t_start = time.time()
    #POTT_gpud = exchange_BC_gpu(GR, POTT_gpud, zonald, meridd, stream)
    #t_end = time.time()
    #cpu_time += t_end - t_start
    #print('gpu ' + str(time_gpu))

    #POTT_gpud.to_host(stream)


    ##############################



    ##############################
    t_start = time.time()
    # MOIST VARIABLES
    #dQVdt = water_vapor_tendency(GR, QV, COLP, COLP_NEW, UFLX, VFLX, WWIND)
    #dQCdt = cloud_water_tendency(GR, QC, COLP, COLP_NEW, UFLX, VFLX, WWIND)
    dQVdt = water_vapor_tendency_c(GR, njobs, QV, COLP, COLP_NEW,
                                    UFLX, VFLX, WWIND, dQVdt_MIC)
    dQVdt = np.asarray(dQVdt)
    dQCdt = cloud_water_tendency_c(GR, njobs, QC, COLP, COLP_NEW,
                                    UFLX, VFLX, WWIND, dQCdt_MIC)
    dQCdt = np.asarray(dQCdt)
    t_end = time.time()
    GR.trac_comp_time += t_end - t_start
    ##############################

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
                            + GR.dt*dQVdt[:,:,k]/COLP[GR.iijj]
        QC[:,:,k][GR.iijj] = QC_OLD[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dQCdt[:,:,k]/COLP[GR.iijj]
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




