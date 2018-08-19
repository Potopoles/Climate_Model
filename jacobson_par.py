import copy
import numpy as np
import time
from continuity import colp_tendency_jacobson, vertical_wind_jacobson

#from wind import wind_tendency_jacobson
from wind_cython import wind_tendency_jacobson_par

#from temperature import temperature_tendency_jacobson
from temperature_cython import temperature_tendency_jacobson_par
#from geopotential import diag_geopotential_jacobson
from geopotential_cython import diag_geopotential_jacobson_par
#from diagnostics import diagnose_POTTVB_jacobson, interp_COLPA
from diagnostics_cython import diagnose_POTTVB_jacobson_par, interp_COLPA_par
from boundaries import exchange_BC
#from moisture import water_vapor_tendency, cloud_water_tendency
from moisture_cython import water_vapor_tendency_par, cloud_water_tendency_par
from namelist import pTop, njobs
from constants import con_Rd

# parallel
import multiprocessing as mp

def tendencies_jacobson(GR,\
                    COLP, POTT, POTTVB, HSURF,
                    UWIND, VWIND, WWIND,
                    UFLX, VFLX, PHI, PVTF, PVTFVB,
                    dPOTTdt_RAD, dPOTTdt_MIC,
                    QV, QC, dQVdt_MIC, dQCdt_MIC):

    # PROGNOSE COLP
    dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND,\
                                                        VWIND, UFLX, VFLX)
    COLP_NEW = copy.deepcopy(COLP)
    COLP_NEW[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    COLP_NEW = exchange_BC(GR, COLP_NEW)

    # DIAGNOSE WWIND
    WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)

    # PROGNOSE WIND
    t_start = time.time()
    #dUFLXdt, dVFLXdt = wind_tendency_jacobson(GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
    #                                                COLP, COLP_NEW, HSURF, PHI, POTT,
    #                                                PVTF, PVTFVB)
    dUFLXdt, dVFLXdt = wind_tendency_jacobson_par(GR, njobs, UWIND, VWIND, WWIND, UFLX, VFLX,
                                                    COLP, COLP_NEW, PHI,
                                                    POTT, PVTF, PVTFVB)
    dUFLXdt = np.asarray(dUFLXdt)
    dVFLXdt = np.asarray(dVFLXdt)

    t_end = time.time()
    GR.wind_comp_time += t_end - t_start

    # PROGNOSE POTT
    t_start = time.time()

    #dPOTTdt = temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW,\
    #                                        UFLX, VFLX, WWIND, \
    #                                        dPOTTdt_RAD, dPOTTdt_MIC)
    dPOTTdt = temperature_tendency_jacobson_par(GR, njobs, POTT, POTTVB, COLP, COLP_NEW,\
                                            UFLX, VFLX, WWIND, \
                                            dPOTTdt_RAD, dPOTTdt_MIC)
    dPOTTdt = np.asarray(dPOTTdt)

    t_end = time.time()
    GR.temp_comp_time += t_end - t_start

    # MOIST VARIABLES
    t_start = time.time()

    #dQVdt = water_vapor_tendency(GR, QV, COLP, COLP_NEW, UFLX, VFLX, WWIND)
    #dQCdt = cloud_water_tendency(GR, QC, COLP, COLP_NEW, UFLX, VFLX, WWIND)

    dQVdt = water_vapor_tendency_par(GR, njobs, QV, COLP, COLP_NEW,
                                    UFLX, VFLX, WWIND, dQVdt_MIC)
    dQVdt = np.asarray(dQVdt)
    dQCdt = cloud_water_tendency_par(GR, njobs, QC, COLP, COLP_NEW,
                                    UFLX, VFLX, WWIND, dQCdt_MIC)
    dQCdt = np.asarray(dQCdt)


    t_end = time.time()
    GR.trac_comp_time += t_end - t_start

    return(dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt, WWIND, dQVdt, dQCdt)


def proceed_timestep_jacobson(GR, UWIND, VWIND,
                    COLP, POTT, QV, QC,
                    dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt):

    # TIME STEPPING
    COLP_OLD = copy.deepcopy(COLP)
    #COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA_par(GR, njobs, COLP_OLD)

    COLP[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    COLP = exchange_BC(GR, COLP)
    #COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA_par(GR, njobs, COLP)

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

    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)
    POTT = exchange_BC(GR, POTT)
    QV = exchange_BC(GR, QV)
    QC = exchange_BC(GR, QC)

    return(UWIND, VWIND, COLP, POTT, QV, QC)


def diagnose_fields_jacobson(GR, PHI, PHIVB, COLP, POTT, HSURF, PVTF, PVTFVB, POTTVB):

    t_start = time.time()

    #PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, PHIVB, HSURF, 
    #                                            POTT, COLP, PVTF, PVTFVB)

    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson_par(GR, njobs, PHI, PHIVB, HSURF, 
                                                    POTT, COLP, PVTF, PVTFVB)
    PHI = np.asarray(PHI)
    PHIVB = np.asarray(PHIVB)
    PVTF = np.asarray(PVTF)
    PVTFVB = np.asarray(PVTFVB)

    #import pickle
    #with open('testarray.pkl', 'rb') as f:
    #    out = pickle.load(f)
    #PHI_orig = out['PHI']
    #PHIVB_orig = out['PHIVB']
    #PVTF_orig = out['PVTF']
    #PVTFVB_orig = out['PVTFVB']
    #print('###################')
    #nan_here = np.isnan(PHIVB)
    #nan_orig = np.isnan(PHIVB_orig)
    #print('u values ' + str(np.nansum(np.abs(PVTF - PVTF_orig))))
    #print('u nan ' + str(np.sum(nan_here != nan_orig)))
    #print('###################')
    #quit()



    #POTTVB = diagnose_POTTVB_jacobson(GR, POTTVB, POTT, PVTF, PVTFVB)

    POTTVB = diagnose_POTTVB_jacobson_par(GR, njobs, POTTVB, POTT, PVTF, PVTFVB)
    POTTVB = np.asarray(POTTVB)


    #TURB.diag_rho(GR, COLP, POTT, PVTF, POTTVB, PVTFVB)
    #TURB.diag_dz(GR, PHI, PHIVB)

    t_end = time.time()
    GR.diag_comp_time += t_end - t_start

    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)




