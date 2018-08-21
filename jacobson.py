import copy
import numpy as np
import time
from continuity import colp_tendency_jacobson, vertical_wind_jacobson

from wind import wind_tendency_jacobson
#from wind_cython_par import wind_tendency_jacobson_par
from wind_cython import wind_tendency_jacobson_c

#from temperature import temperature_tendency_jacobson
from temperature_cython import temperature_tendency_jacobson_c
#from geopotential import diag_geopotential_jacobson
from geopotential_cython import diag_geopotential_jacobson_c
#from diagnostics import diagnose_POTTVB_jacobson, interp_COLPA
from diagnostics_cython import diagnose_POTTVB_jacobson_c, interp_COLPA_c
from boundaries import exchange_BC
#from moisture import water_vapor_tendency, cloud_water_tendency
from moisture_cython import water_vapor_tendency_c, cloud_water_tendency_c
from namelist import pTop, njobs
from constants import con_Rd

# parallel
import multiprocessing as mp

def tendencies_jacobson(GR, subgrids,\
                    COLP_OLD, COLP, POTT, POTTVB, HSURF,
                    UWIND, VWIND, WWIND,
                    UFLX, VFLX, PHI, PVTF, PVTFVB,
                    dPOTTdt_RAD, dPOTTdt_MIC,
                    QV, QC, dQVdt_MIC, dQCdt_MIC):


    # PROGNOSE COLP
    dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND,\
                                                        VWIND, UFLX, VFLX)


    COLP_NEW = copy.deepcopy(COLP)
    COLP_NEW[GR.iijj] = COLP_OLD[GR.iijj] + GR.dt*dCOLPdt

    # DIAGNOSE WWIND
    WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)

    #import pickle
    #print('did it')
    #out = {}
    #out['UFLX'] = UFLX
    #out['VFLX'] = VFLX
    #out['COLP'] = COLP_NEW
    #out['WWIND'] = WWIND
    ##out['UWIND'] = UWIND
    ##out['VWIND'] = VWIND
    #with open('testarray.pkl', 'wb') as f:
    #    pickle.dump(out, f)
    #quit()

    # TODO 2 NECESSARY
    COLP_NEW = exchange_BC(GR, COLP_NEW)
    WWIND = exchange_BC(GR, WWIND)

    # PROGNOSE WIND
    t_start = time.time()
    #dUFLXdt, dVFLXdt = wind_tendency_jacobson(GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
    #                                                COLP, COLP_NEW, HSURF, PHI, POTT,
    #                                                PVTF, PVTFVB)
    dUFLXdt, dVFLXdt = wind_tendency_jacobson_c(GR, njobs, UWIND, VWIND, WWIND, UFLX, VFLX,
                                                    COLP, COLP_NEW, PHI,
                                                    POTT, PVTF, PVTFVB)
    dUFLXdt = np.asarray(dUFLXdt)
    dVFLXdt = np.asarray(dVFLXdt)


    #output = mp.Queue()
    #processes = []
    #for job_ind in range(0,njobs):

    #    SGR = subgrids[job_ind]
    #    processes.append(
    #        mp.Process(\
    #            target=wind_tendency_jacobson_c,
    #            args = (job_ind, output, subgrids[job_ind], 1,
    #                    UWIND[SGR.map_iisjj], VWIND[SGR.map_iijjs],
    #                    WWIND[SGR.map_iijj], 
    #                    UFLX[SGR.map_iisjj], VFLX[SGR.map_iijjs],
    #                    COLP[SGR.map_iijj], COLP_NEW[SGR.map_iijj],
    #                    PHI[SGR.map_iijj],
    #                    POTT[SGR.map_iijj], PVTF[SGR.map_iijj],
    #                    PVTFVB[SGR.map_iijj])))
    #for proc in processes:
    #    proc.start()

    #results = [output.get() for p in processes]
    #results.sort()
    #dUFLXdt = np.zeros( (GR.nxs, GR.ny, GR.nz) )
    #dVFLXdt = np.zeros( (GR.nx, GR.nys, GR.nz) )
    #for job_ind in range(0,njobs):
    #    SGR = subgrids[job_ind]
    #    #res = results[job_ind][1]

    #    dUFLXdt[SGR.mapin_iisjj] = np.asarray(results[job_ind][1]['dUFLXdt'])
    #    dVFLXdt[SGR.mapin_iijjs] = np.asarray(results[job_ind][1]['dVFLXdt'])

    #for proc in processes:
    #    proc.join()



    t_end = time.time()
    GR.wind_comp_time += t_end - t_start
    #print(t_end - t_start)
    #quit()

    # PROGNOSE POTT
    t_start = time.time()

    #dPOTTdt = temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW,\
    #                                        UFLX, VFLX, WWIND, \
    #                                        dPOTTdt_RAD, dPOTTdt_MIC)
    dPOTTdt = temperature_tendency_jacobson_c(GR, njobs, POTT, POTTVB, COLP, COLP_NEW,\
                                            UFLX, VFLX, WWIND, \
                                            dPOTTdt_RAD, dPOTTdt_MIC)
    dPOTTdt = np.asarray(dPOTTdt)

    t_end = time.time()
    GR.temp_comp_time += t_end - t_start

    # MOIST VARIABLES
    t_start = time.time()

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

    t_start = time.time()

    #PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, PHIVB, HSURF, 
    #                                            POTT, COLP, PVTF, PVTFVB)

    #import pickle
    #out = {}
    #out['PHI'] = PHI
    #out['PHIVB'] = PHIVB
    #out['PVTF'] = PVTF
    #out['PVTFVB'] = PVTFVB
    #with open('testarray.pkl', 'wb') as f:
    #    pickle.dump(out, f)

    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson_c(GR, njobs, PHI, PHIVB, HSURF, 
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

    POTTVB = diagnose_POTTVB_jacobson_c(GR, njobs, POTTVB, POTT, PVTF, PVTFVB)
    POTTVB = np.asarray(POTTVB)


    #TURB.diag_rho(GR, COLP, POTT, PVTF, POTTVB, PVTFVB)
    #TURB.diag_dz(GR, PHI, PHIVB)

    t_end = time.time()
    GR.diag_comp_time += t_end - t_start

    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)




