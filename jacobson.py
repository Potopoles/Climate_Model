import copy
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
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
from temperature_cuda import calc_dPOTTdt_orig
###### NEW
from tendencies import TendencyFactory
Tendencies = TendencyFactory(target='GPU')
###### NEW


from moisture import water_vapor_tendency, cloud_water_tendency
from bin.moisture_cython import water_vapor_tendency_c, cloud_water_tendency_c
from moisture_cuda import water_vapor_tendency_gpu, cloud_water_tendency_gpu

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


def tendencies_jacobson(GR, F, subgrids):


    ##############################
    ##############################
    t_start = time.time()
    # PROGNOSE COLP
    if comp_mode == 0:
        F.dCOLPdt, F.UFLX, F.VFLX, F.FLXDIV = colp_tendency_jacobson(GR,
                                                    F.COLP, F.UWIND, F.VWIND,
                                                    F.dCOLPdt, F.UFLX, F.VFLX, F.FLXDIV)
        F.COLP_NEW[GR.iijj] = F.COLP_OLD[GR.iijj] + GR.dt*F.dCOLPdt[GR.iijj]

    elif comp_mode == 1:
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
    if comp_mode == 0:
        F.WWIND = vertical_wind_jacobson(GR, F.COLP_NEW, F.dCOLPdt, F.FLXDIV, F.WWIND)
        # TODO 2 NECESSARY
        F.COLP_NEW = exchange_BC(GR, F.COLP_NEW)
        F.WWIND = exchange_BC(GR, F.WWIND)

    elif comp_mode == 1:
        F.WWIND = vertical_wind_jacobson_c(GR, F.COLP_NEW, F.dCOLPdt, F.FLXDIV, F.WWIND)
        F.WWIND = np.asarray(F.WWIND)
        # TODO 2 NECESSARY
        F.COLP_NEW = exchange_BC(GR, F.COLP_NEW)
        F.WWIND = exchange_BC(GR, F.WWIND)

    elif comp_mode == 2:
        vertical_wind_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks, GR.stream]\
                                    (F.WWIND, F.dCOLPdt, F.FLXDIV, F.COLP_NEW, GR.sigma_vbd)
        GR.stream.synchronize()
        # TODO 2 NECESSARY
        F.COLP_NEW = exchange_BC_gpu(F.COLP_NEW, GR.zonal, GR.merid,
                                    GR.griddim_xy, GR.blockdim_xy, GR.stream, array2D=True)
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
    if comp_mode == 0:
        F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson(GR, F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                        F.COLP, F.COLP_NEW, F.PHI, F.POTT,
                                        F.PVTF, F.PVTFVB)

    elif comp_mode == 1:
        F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_c(GR, njobs, F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                        F.COLP, F.COLP_NEW, F.PHI,
                                        F.POTT, F.PVTF, F.PVTFVB)
        F.dUFLXdt = np.asarray(F.dUFLXdt)
        F.dVFLXdt = np.asarray(F.dVFLXdt)

    elif comp_mode == 2:
        F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_gpu(GR, F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
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
    if comp_mode == 0:
        F.dPOTTdt = temperature_tendency_jacobson(GR,
                                            F.POTT, F.POTTVB, F.COLP, F.COLP_NEW,
                                            F.UFLX, F.VFLX, F.WWIND,
                                            F.dPOTTdt_RAD, F.dPOTTdt_MIC)

    elif comp_mode == 1:
        F.dPOTTdt = temperature_tendency_jacobson_c(GR, njobs,
                                            F.POTT, F.POTTVB, F.COLP, F.COLP_NEW,
                                            F.UFLX, F.VFLX, F.WWIND,
                                            F.dPOTTdt_RAD, F.dPOTTdt_MIC)
        F.dPOTTdt = np.asarray(F.dPOTTdt)

    elif comp_mode == 2:
        #F.COLP = cp.expand_dims(F.COLP, axis=2)
        #GR.Ad = cp.expand_dims(GR.Ad, axis=2)
        #F.COLP_NEW = cp.expand_dims(F.COLP_NEW, axis=2)
        #GR.dsigma = cp.expand_dims(cp.expand_dims(GR.dsigma, 0),0)
        #F.dPOTTdt = Tendencies.POTT_tendency(
        #                F.dPOTTdt, F.POTT, F.UFLX, F.VFLX, F.COLP, GR.Ad,
        #                F.POTTVB, F.WWIND, F.COLP_NEW, GR.dsigma)
        #F.COLP = F.COLP.squeeze()
        #GR.Ad = GR.Ad.squeeze()
        #F.COLP_NEW = F.COLP_NEW.squeeze()
        #GR.dsigma = GR.dsigma.squeeze()

        calc_dPOTTdt_orig[GR.griddim, GR.blockdim, GR.stream] \
                                    (F.dPOTTdt, 
                                    F.POTT, F.POTTVB, F.COLP, F.COLP_NEW, 
                                    F.UFLX, F.VFLX, F.WWIND, 
                                            #F.dPOTTdt_RAD, F.dPOTTdt_MIC, 
                                            GR.Ad, GR.dsigmad)
        GR.stream.synchronize()



    t_end = time.time()
    GR.temp_comp_time += t_end - t_start
    ##############################
    ##############################



    ##############################
    ##############################
    t_start = time.time()
    # MOIST VARIABLES
    if comp_mode == 0:
        F.dQVdt = water_vapor_tendency(GR, F.dQVdt, F.QV, F.COLP, F.COLP_NEW, \
                                        F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
        F.dQCdt = cloud_water_tendency(GR, F.dQCdt, F.QC, F.COLP, F.COLP_NEW, \
                                        F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)

    elif comp_mode == 1:
        F.dQVdt = water_vapor_tendency_c(GR, njobs, F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
                                        F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
        F.dQVdt = np.asarray(F.dQVdt)
        F.dQCdt = cloud_water_tendency_c(GR, njobs, F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
                                        F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)
        F.dQCdt = np.asarray(F.dQCdt)

    elif comp_mode == 2:
        water_vapor_tendency_gpu[GR.griddim, GR.blockdim, GR.stream] \
                                    (F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
                                     F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC,
                                     GR.Ad, GR.dsigmad)
        GR.stream.synchronize()
        cloud_water_tendency_gpu[GR.griddim, GR.blockdim, GR.stream] \
                                    (F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
                                     F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC,
                                     GR.Ad, GR.dsigmad)
        GR.stream.synchronize()

    t_end = time.time()
    GR.trac_comp_time += t_end - t_start
    ##############################
    ##############################





def proceed_timestep_jacobson(GR, UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                    COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                    dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt):

    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)

    # TIME STEPPING
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
    POTT = exchange_BC(GR, POTT)
    QV = exchange_BC(GR, QV)
    QC = exchange_BC(GR, QC)

    return(UWIND, VWIND, COLP, POTT, QV, QC)




def diagnose_fields_jacobson(GR, F, on_host=False):
    # on_host forces the execution to be performed on CPU.
    # The function is called in initialize_fields() in field.py like this
    # for first time setup, when GPU_Fields are not yet initialized.

    ##############################
    ##############################
    if comp_mode == 0 or on_host:
        F.PHI, F.PHIVB, F.PVTF, F.PVTFVB = \
                     diag_geopotential_jacobson(GR, F.PHI, F.PHIVB, F.HSURF, 
                                                    F.POTT, F.COLP, F.PVTF, F.PVTFVB)

    elif comp_mode == 1:
        F.PHI, F.PHIVB, F.PVTF, F.PVTFVB = \
                    diag_geopotential_jacobson_c(GR, njobs, F.PHI, F.PHIVB, F.HSURF, 
                                                    F.POTT, F.COLP, F.PVTF, F.PVTFVB)
        F.PHI = np.asarray(F.PHI)
        F.PHIVB = np.asarray(F.PHIVB)
        F.PVTF = np.asarray(F.PVTF)
        F.PVTFVB = np.asarray(F.PVTFVB)

    elif comp_mode == 2:
        F.PHI, F.PHIVB, F.PVTF, F.PVTFVB = \
                 diag_geopotential_jacobson_gpu(GR, GR.stream, F.PHI, F.PHIVB, F.HSURF, 
                                                    F.POTT, F.COLP, F.PVTF, F.PVTFVB)
    ##############################
    ##############################



    ##############################
    ##############################
    if comp_mode == 0 or on_host:
        F.POTTVB = diagnose_POTTVB_jacobson(GR, F.POTTVB, F.POTT, F.PVTF, F.PVTFVB)

    elif comp_mode == 1:
        F.POTTVB = diagnose_POTTVB_jacobson_c(GR, njobs, F.POTTVB, F.POTT, F.PVTF, F.PVTFVB)
        F.POTTVB = np.asarray(F.POTTVB)

    elif comp_mode == 2:
        diagnose_POTTVB_jacobson_gpu[GR.griddim_ks, GR.blockdim_ks, GR.stream] \
                                        (F.POTTVB, F.POTT, F.PVTF, F.PVTFVB)
        GR.stream.synchronize()
    ##############################
    ##############################


    #TURB.diag_rho(GR, COLP, POTT, PVTF, POTTVB, PVTFVB)
    #TURB.diag_dz(GR, PHI, PHIVB)






##var = PHIVB
##var_orig = PHIVB_orig
#var = PVTF
#var_orig = PVTF_orig
#print('###################')
#nan_here = np.isnan(var)
#nan_orig = np.isnan(var_orig)
#diff = var - var_orig
#nan_diff = nan_here != nan_orig 
#print('values ' + str(np.nansum(np.abs(diff))))
#print('  nans ' + str(np.sum(nan_diff)))
#print('###################')

##quit()
##plt.contourf(var_orig[:,:,0].T)
##plt.contourf(var[:,:,0].T)
#plt.contourf(diff[:,:,0].T)
#plt.colorbar()
#plt.show()
#quit()
