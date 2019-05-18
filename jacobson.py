import copy
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
from namelist import pTop, njobs, comp_mode
from namelist import i_run_new_style
from constants import con_Rd

from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from bin.continuity_cython import colp_tendency_jacobson_c, vertical_wind_jacobson_c
from continuity_cuda import colp_tendency_jacobson_gpu, vertical_wind_jacobson_gpu

from wind import wind_tendency_jacobson
from bin.wind_cython import wind_tendency_jacobson_c
from wind_cuda import wind_tendency_jacobson_gpu

from bin.temperature_cython import temperature_tendency_jacobson_c
###### NEW
from org_tendencies import TendencyFactory
Tendencies_GPU = TendencyFactory(target='GPU')
Tendencies_CPU = TendencyFactory(target='CPU')
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
                                    (F.WWIND, F.dCOLPdt, F.FLXDIV,
                                    F.COLP_NEW, GR.sigma_vbd)
        GR.stream.synchronize()
        # TODO 2 NECESSARY
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
            # TODO
            F.COLP          = np.expand_dims(F.COLP, axis=2)
            F.COLP_NEW      = np.expand_dims(F.COLP_NEW, axis=2)

            GR.corf         = np.expand_dims(GR.corf   , axis=2)
            GR.corf_is      = np.expand_dims(GR.corf_is, axis=2)
            GR.lat_rad      = np.expand_dims(GR.lat_rad, axis=2)
            GR.lat_is_rad   = np.expand_dims(GR.lat_is_rad, axis=2)
            GR.dlon_rad_2D  = np.expand_dims(GR.dlon_rad_2D, axis=2)
            GR.dlat_rad_2D  = np.expand_dims(GR.dlat_rad_2D, axis=2)
            GR.A            = np.expand_dims(GR.A, axis=2)
            GR.dyis         = np.expand_dims(GR.dyis, axis=2)
            GR.dxjs         = np.expand_dims(GR.dxjs, axis=2)

            GR.dsigma       = np.expand_dims(cp.expand_dims(GR.dsigma, 0),0)
            GR.sigma_vb     = np.expand_dims(cp.expand_dims(GR.sigma_vb, 0),0)

            F.dUFLXdt, F.dVFLXdt = Tendencies_CPU.UVFLX_tendency(
                            F.dUFLXdt, F.dVFLXdt,
                            F.UWIND, F.VWIND, F.WWIND,
                            F.UFLX, F.VFLX,
                            F.PHI, F.COLP, F.COLP_NEW, F.POTT,
                            F.PVTF, F.PVTFVB,
                            F.WWIND_UWIND, F.WWIND_VWIND,
                            GR.A, GR.corf_is, GR.corf,
                            GR.lat_rad, GR.lat_is_rad,
                            GR.dlon_rad_2D, GR.dlat_rad_2D,
                            GR.dyis, GR.dxjs,
                            GR.dsigma, GR.sigma_vb)
            print('CPU')
            t0 = time.time()
            for i in range(10):
                F.dUFLXdt, F.dVFLXdt = Tendencies_CPU.UVFLX_tendency(
                                F.dUFLXdt, F.dVFLXdt,
                                F.UWIND, F.VWIND, F.WWIND,
                                F.UFLX, F.VFLX,
                                F.PHI, F.COLP, F.COLP_NEW, F.POTT,
                                F.PVTF, F.PVTFVB,
                                F.WWIND_UWIND, F.WWIND_VWIND,
                                GR.A, GR.corf_is, GR.corf,
                                GR.lat_rad, GR.lat_is_rad,
                                GR.dlon_rad_2D, GR.dlat_rad_2D,
                                GR.dyis, GR.dxjs,
                                GR.dsigma, GR.sigma_vb)
            print(time.time() - t0)

            F.COLP          = F.COLP.squeeze()
            F.COLP_NEW      = F.COLP_NEW.squeeze()

            GR.corf         = GR.corf       .squeeze()
            GR.corf_is      = GR.corf_is    .squeeze()
            GR.lat_rad      = GR.lat_rad    .squeeze()
            GR.lat_is_rad   = GR.lat_is_rad .squeeze()
            GR.dlon_rad_2D  = GR.dlon_rad_2D.squeeze()
            GR.dlat_rad_2D  = GR.dlat_rad_2D.squeeze()
            GR.A            = GR.A          .squeeze()
            GR.dyis         = GR.dyis       .squeeze()
            GR.dxjs         = GR.dxjs       .squeeze()

            GR.dsigma       = GR.dsigma     .squeeze()
            GR.sigma_vb     = GR.sigma_vb   .squeeze()



            # TODO
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
            t0 = time.time()
            for i in range(10):
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
            print(time.time() - t0)
            quit()
        
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
            # TODO
            F.COLP          = cp.expand_dims(F.COLP, axis=2)
            F.COLP_NEW      = cp.expand_dims(F.COLP_NEW, axis=2)

            GR.corfd        = cp.expand_dims(GR.corfd  , axis=2)
            GR.corf_isd     = cp.expand_dims(GR.corf_isd, axis=2)
            GR.lat_radd     = cp.expand_dims(GR.lat_radd, axis=2)
            GR.lat_is_radd  = cp.expand_dims(GR.lat_is_radd, axis=2)
            GR.dlon_rad_2Dd = cp.expand_dims(GR.dlon_rad_2Dd,axis=2)
            GR.dlat_rad_2Dd = cp.expand_dims(GR.dlat_rad_2Dd,axis=2)
            GR.Ad           = cp.expand_dims(GR.Ad, axis=2)
            GR.dyisd        = cp.expand_dims(GR.dyisd, axis=2)
            GR.dxjsd        = cp.expand_dims(GR.dxjsd, axis=2)

            GR.dsigmad      = cp.expand_dims(cp.expand_dims(GR.dsigmad, 0),0)
            GR.sigma_vbd    = cp.expand_dims(cp.expand_dims(GR.sigma_vbd, 0),0)

            F.dUFLXdt, F.dVFLXdt = Tendencies_GPU.UVFLX_tendency(
                            F.dUFLXdt, F.dVFLXdt,
                            F.UWIND, F.VWIND, F.WWIND,
                            F.UFLX, F.VFLX,
                            F.CFLX, F.QFLX, F.DFLX, F.EFLX,
                            F.SFLX, F.TFLX, F.BFLX, F.RFLX,
                            F.PHI, F.COLP, F.COLP_NEW, F.POTT,
                            F.PVTF, F.PVTFVB,
                            F.WWIND_UWIND, F.WWIND_VWIND,
                            GR.Ad, GR.corf_isd, GR.corfd,
                            GR.lat_radd, GR.lat_is_radd,
                            GR.dlon_rad_2Dd, GR.dlat_rad_2Dd,
                            GR.dyisd, GR.dxjsd,
                            GR.dsigmad, GR.sigma_vbd)
            cuda.synchronize()
            print('GPU')
            t0 = time.time()
            for i in range(50):
                F.dUFLXdt, F.dVFLXdt = Tendencies_GPU.UVFLX_tendency(
                                F.dUFLXdt, F.dVFLXdt,
                                F.UWIND, F.VWIND, F.WWIND,
                                F.UFLX, F.VFLX,
                                F.CFLX, F.QFLX, F.DFLX, F.EFLX,
                                F.SFLX, F.TFLX, F.BFLX, F.RFLX,
                                F.PHI, F.COLP, F.COLP_NEW, F.POTT,
                                F.PVTF, F.PVTFVB,
                                F.WWIND_UWIND, F.WWIND_VWIND,
                                GR.Ad, GR.corf_isd, GR.corfd,
                                GR.lat_radd, GR.lat_is_radd,
                                GR.dlon_rad_2Dd, GR.dlat_rad_2Dd,
                                GR.dyisd, GR.dxjsd,
                                GR.dsigmad, GR.sigma_vbd)
                cuda.synchronize()
            print(time.time() - t0)
            
            
            #TODO
            CFLX_new = np.asarray(F.CFLX)
            QFLX_new = np.asarray(F.QFLX)
            DFLX_new = np.asarray(F.DFLX)
            EFLX_new = np.asarray(F.EFLX)
            SFLX_new = np.asarray(F.SFLX)
            TFLX_new = np.asarray(F.TFLX)
            BFLX_new = np.asarray(F.BFLX)
            RFLX_new = np.asarray(F.RFLX)

            F.COLP          = F.COLP.squeeze()
            F.COLP_NEW      = F.COLP_NEW.squeeze()

            GR.corfd        = GR.corfd          .squeeze()
            GR.corf_isd     = GR.corf_isd       .squeeze()
            GR.lat_radd     = GR.lat_radd       .squeeze()
            GR.lat_is_radd   = GR.lat_is_radd   .squeeze()
            GR.dlon_rad_2Dd = GR.dlon_rad_2Dd   .squeeze()
            GR.dlat_rad_2Dd = GR.dlat_rad_2Dd   .squeeze()
            GR.Ad           = GR.Ad             .squeeze()
            GR.dyisd        = GR.dyisd          .squeeze()
            GR.dxjsd        = GR.dxjsd          .squeeze()

            GR.dsigmad      = GR.dsigmad        .squeeze()
            GR.sigma_vbd    = GR.sigma_vbd      .squeeze()


            # TODO
            F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_gpu(GR,
                                        F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                        F.WWIND_UWIND, F.WWIND_VWIND, 
                                        F.COLP, F.COLP_NEW, F.PHI, F.POTT,
                                        F.PVTF, F.PVTFVB)
            cuda.synchronize()
            t0 = time.time()
            for i in range(50):
                F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_gpu(GR,
                                            F.UWIND, F.VWIND, F.WWIND,
                                            F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                            F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                            F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                            F.WWIND_UWIND, F.WWIND_VWIND, 
                                            F.COLP, F.COLP_NEW, F.PHI, F.POTT,
                                            F.PVTF, F.PVTFVB)
                cuda.synchronize()
            print(time.time() - t0)

            #TODO
            CFLX_old = np.asarray(F.CFLX)
            QFLX_old = np.asarray(F.QFLX)
            DFLX_old = np.asarray(F.DFLX)
            EFLX_old = np.asarray(F.EFLX)
            SFLX_old = np.asarray(F.SFLX)
            TFLX_old = np.asarray(F.TFLX)
            BFLX_old = np.asarray(F.BFLX)
            RFLX_old = np.asarray(F.RFLX)
            
            print()

            print(np.nanmean(CFLX_new - CFLX_old))
            print(np.sum(np.isnan(CFLX_new)) - np.sum(np.isnan(CFLX_old)))

            print(np.nanmean(QFLX_new - QFLX_old))
            print(np.sum(np.isnan(QFLX_new)) - np.sum(np.isnan(QFLX_old)))

            print(np.nanmean(DFLX_new - DFLX_old))
            print(np.sum(np.isnan(DFLX_new)) - np.sum(np.isnan(DFLX_old)))

            print(np.nanmean(EFLX_new - EFLX_old))
            print(np.sum(np.isnan(EFLX_new)) - np.sum(np.isnan(EFLX_old)))

            print(np.nanmean(SFLX_new - SFLX_old))
            print(np.sum(np.isnan(SFLX_new)) - np.sum(np.isnan(SFLX_old)))

            print(np.nanmean(TFLX_new - TFLX_old))
            print(np.sum(np.isnan(TFLX_new)) - np.sum(np.isnan(TFLX_old)))

            print(np.nanmean(BFLX_new - BFLX_old))
            print(np.sum(np.isnan(BFLX_new)) - np.sum(np.isnan(BFLX_old)))

            print(np.nanmean(RFLX_new - RFLX_old))
            print(np.sum(np.isnan(RFLX_new)) - np.sum(np.isnan(RFLX_old)))
            
            quit()
            diff = CFLX_new - CFLX_old
            import matplotlib.pyplot as plt
            k = 0
            plt.contourf(diff[:,:,k].squeeze())
            plt.colorbar()
            plt.show()
            #print(diff[:,:,k].squeeze())
            quit()



        else:
            t0 = time.time()
            F.dUFLXdt, F.dVFLXdt = wind_tendency_jacobson_gpu(GR,
                                        F.UWIND, F.VWIND, F.WWIND,
                                        F.UFLX, F.dUFLXdt, F.VFLX, F.dVFLXdt,
                                        F.BFLX, F.CFLX, F.DFLX, F.EFLX,
                                        F.RFLX, F.QFLX, F.SFLX, F.TFLX, 
                                        F.WWIND_UWIND, F.WWIND_VWIND, 
                                        F.COLP, F.COLP_NEW, F.PHI, F.POTT,
                                        F.PVTF, F.PVTFVB)
            print(time.time() - t0)
            #quit()



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
            # TODO
            F.COLP = np.expand_dims(F.COLP, axis=2)
            GR.A = np.expand_dims(GR.A, axis=2)
            F.COLP_NEW = np.expand_dims(F.COLP_NEW, axis=2)
            GR.dsigma = np.expand_dims(np.expand_dims(GR.dsigma, 0),0)
            F.dPOTTdt = Tendencies_CPU.POTT_tendency(
                            F.dPOTTdt, F.POTT, F.UFLX, F.VFLX, F.COLP, GR.A,
                            F.POTTVB, F.WWIND, F.COLP_NEW, GR.dsigma)
            F.COLP = F.COLP.squeeze()
            GR.A = GR.A.squeeze()
            F.COLP_NEW = F.COLP_NEW.squeeze()
            GR.dsigma = GR.dsigma.squeeze()
        else:
            F.dPOTTdt = temperature_tendency_jacobson_c(GR, njobs,
                                                F.POTT, F.POTTVB, F.COLP, F.COLP_NEW,
                                                F.UFLX, F.VFLX, F.WWIND,
                                                F.dPOTTdt_RAD, F.dPOTTdt_MIC)
            F.dPOTTdt = np.asarray(F.dPOTTdt)


    elif comp_mode == 2:
        # TODO
        F.COLP = cp.expand_dims(F.COLP, axis=2)
        GR.Ad = cp.expand_dims(GR.Ad, axis=2)
        F.COLP_NEW = cp.expand_dims(F.COLP_NEW, axis=2)
        GR.dsigmad = cp.expand_dims(cp.expand_dims(GR.dsigmad, 0),0)

        #print('GPU POTT')
        #t0 = time.time()
        F.dPOTTdt = Tendencies_GPU.POTT_tendency(
                        F.dPOTTdt, F.POTT, F.UFLX, F.VFLX, F.COLP, GR.Ad,
                        F.POTTVB, F.WWIND, F.COLP_NEW, GR.dsigmad)
        #print(time.time() - t0)
        #quit()

        F.COLP = F.COLP.squeeze()
        GR.Ad = GR.Ad.squeeze()
        F.COLP_NEW = F.COLP_NEW.squeeze()
        GR.dsigmad = GR.dsigmad.squeeze()





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


