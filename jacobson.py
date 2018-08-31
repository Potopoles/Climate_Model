import copy
import matplotlib.pyplot as plt
import numpy as np
import time
from namelist import pTop, njobs
from constants import con_Rd

#from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from bin.continuity_cython import colp_tendency_jacobson_c, vertical_wind_jacobson_c
#from wind import wind_tendency_jacobson
from bin.wind_cython import wind_tendency_jacobson_c
from temperature import temperature_tendency_jacobson
#from bin.temperature_cython import temperature_tendency_jacobson_c
from temperature_cuda import temperature_tendency_jacobson_gpu
#from geopotential import diag_geopotential_jacobson
from bin.geopotential_cython import diag_geopotential_jacobson_c
#from diagnostics import diagnose_POTTVB_jacobson, interp_COLPA
from bin.diagnostics_cython import diagnose_POTTVB_jacobson_c, interp_COLPA_c
#from moisture import water_vapor_tendency, cloud_water_tendency
from bin.moisture_cython import water_vapor_tendency_c, cloud_water_tendency_c
from boundaries import exchange_BC
from boundaries_gpu import exchange_BC_gpu

from numba import cuda
import numba


def tendencies_jacobson(GR, subgrids,\
                    COLP_OLD, COLP, POTT, POTTVB, HSURF,
                    UWIND, VWIND, WWIND,
                    UFLX, VFLX, PHI, PVTF, PVTFVB,
                    dPOTTdt_RAD, dPOTTdt_MIC,
                    QV, QC, dQVdt_MIC, dQCdt_MIC):


    ##############################
    t_start = time.time()
    # PROGNOSE COLP
    #dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND,\
    #                                                    VWIND, UFLX, VFLX)
    dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson_c(GR, COLP, UWIND,\
                                                        VWIND, UFLX, VFLX)
    dCOLPdt = np.asarray(dCOLPdt)
    UFLX = np.asarray(UFLX)
    VFLX = np.asarray(VFLX)
    FLXDIV = np.asarray(FLXDIV)


    COLP_NEW = copy.deepcopy(COLP)
    COLP_NEW[GR.iijj] = COLP_OLD[GR.iijj] + GR.dt*dCOLPdt

    # DIAGNOSE WWIND
    #WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)
    WWIND = vertical_wind_jacobson_c(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)
    WWIND = np.asarray(WWIND)


    # TODO 2 NECESSARY
    COLP_NEW = exchange_BC(GR, COLP_NEW)
    WWIND = exchange_BC(GR, WWIND)

    t_end = time.time()
    GR.cont_comp_time += t_end - t_start
    ##############################



    ##############################
    t_start = time.time()
    # PROGNOSE WIND
    #dUFLXdt, dVFLXdt = wind_tendency_jacobson(GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
    #                                                COLP, COLP_NEW, HSURF, PHI, POTT,
    #                                                PVTF, PVTFVB)
    dUFLXdt, dVFLXdt = wind_tendency_jacobson_c(GR, njobs, UWIND, VWIND, WWIND, UFLX, VFLX,
                                                    COLP, COLP_NEW, PHI,
                                                    POTT, PVTF, PVTFVB)
    dUFLXdt = np.asarray(dUFLXdt)
    dVFLXdt = np.asarray(dVFLXdt)
    t_end = time.time()
    GR.wind_comp_time += t_end - t_start
    ##############################



    ##############################
    POTT[:,0,:] = 300
    POTT[:,GR.ny+1,:] = 300
    POTT[0,:,:] = 300
    POTT[GR.nx+1,:,:] = 300
    POTT_gpu = copy.deepcopy(POTT)
    POTT_cpu = copy.deepcopy(POTT)

    t_start = time.time()
    # PROGNOSE POTT
    dPOTTdt = temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW,\
                                            UFLX, VFLX, WWIND, \
                                            dPOTTdt_RAD, dPOTTdt_MIC)
    #dPOTTdt = temperature_tendency_jacobson_c(GR, njobs, POTT, POTTVB, COLP, COLP_NEW,\
    #                                        UFLX, VFLX, WWIND, \
    #                                        dPOTTdt_RAD, dPOTTdt_MIC)
    #dPOTTdt = np.asarray(dPOTTdt)
    t_end = time.time()
    cpu_time = t_end - t_start
    GR.temp_comp_time += t_end - t_start

    dPOTTdt_cpu = copy.deepcopy(dPOTTdt)
    for k in range(0,GR.nz):
        POTT_cpu[:,:,k][GR.iijj] = POTT[:,:,k][GR.iijj] + GR.dt*dPOTTdt[:,:,k]/COLP[GR.iijj]
    t_start = time.time()
    POTT_cpu = exchange_BC(GR, POTT_cpu)
    t_end = time.time()
    cpu_time += t_end - t_start
    print('cpu ' + str(cpu_time))
    #print(POTT)
    #quit()
    
    dPOTTdt = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
    zonal   = np.zeros((2,GR.ny +2*GR.nb,GR.nz),      np.float64)
    merid   = np.zeros((GR.nx +2*GR.nb,2,GR.nz),      np.float64)

    stream = cuda.stream()
    dPOTTdtd    = cuda.to_device(dPOTTdt, stream)
    zonald      = cuda.to_device(zonal, stream)
    meridd      = cuda.to_device(merid, stream)
    UFLXd       = cuda.to_device(UFLX, stream)
    VFLXd       = cuda.to_device(VFLX, stream)
    POTTd       = cuda.to_device(POTT, stream)
    Ad          = cuda.to_device(GR.A, stream)

    t_start = time.time()
    temperature_tendency_jacobson_gpu[GR.griddim, GR.blockdim, stream] \
                                        (dPOTTdtd, POTTd, UFLXd, VFLXd, 
                                        Ad)
    stream.synchronize()
    t_end = time.time()
    time_gpu = t_end - t_start

    dPOTTdtd.to_host(stream)
    dPOTTdt_gpu = copy.deepcopy(dPOTTdt)

    for k in range(0,GR.nz):
        POTT_gpu[:,:,k][GR.iijj] = POTT[:,:,k][GR.iijj] + GR.dt*dPOTTdt[:,:,k][GR.iijj]/COLP[GR.iijj]

    POTT_gpud       = cuda.to_device(POTT_gpu, stream)

    t_start = time.time()
    POTT_gpud = exchange_BC_gpu(GR, POTT_gpud, zonald, meridd, stream)
    t_end = time.time()
    cpu_time += t_end - t_start
    print('gpu ' + str(time_gpu))

    POTT_gpud.to_host(stream)

    var = POTT_gpu
    var_orig = POTT_cpu
    #var = dPOTTdt_gpu[GR.iijj]
    #var_orig = dPOTTdt_cpu
    print('###################')
    nan_here = np.isnan(var)
    nan_orig = np.isnan(var_orig)
    diff = var - var_orig
    print('values ' + str(np.nansum(np.abs(diff))))
    print('  nans ' + str(np.sum(nan_here != nan_orig)))
    print('###################')

    quit()
    #plt.contourf(var_orig[:,:,1].T)
    #plt.contourf(var[:,:,1].T)
    plt.contourf(diff[:,:,1].T)
    plt.colorbar()
    plt.show()
    quit()

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


def proceed_timestep_jacobson(GR, UWIND, VWIND,
                    COLP_OLD, COLP, POTT, QV, QC,
                    dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt):

    # TIME STEPPING
    #COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA_c(GR, njobs, COLP_OLD)

    #COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA_c(GR, njobs, COLP)

    for k in range(0,GR.nz):
        UWIND[:,:,k][GR.iisjj] = UWIND[:,:,k][GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                            + GR.dt*dUFLXdt[:,:,k]/COLPA_is_NEW
        VWIND[:,:,k][GR.iijjs] = VWIND[:,:,k][GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                            + GR.dt*dVFLXdt[:,:,k]/COLPA_js_NEW
        POTT[:,:,k][GR.iijj] = POTT[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dPOTTdt[:,:,k]/COLP[GR.iijj]
        QV[:,:,k][GR.iijj] = QV[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dQVdt[:,:,k]/COLP[GR.iijj]
        QC[:,:,k][GR.iijj] = QC[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dQCdt[:,:,k]/COLP[GR.iijj]
    QV[QV < 0] = 0
    QC[QC < 0] = 0

    # TODO 4 NECESSARY
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)
    POTT = exchange_BC(GR, POTT)
    QV = exchange_BC(GR, QV)
    QC = exchange_BC(GR, QC)

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




