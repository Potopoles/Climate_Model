#import pyximport; pyximport.install()
from wind_cython_par import wind_tendency_jacobson_c
import numpy as np
import time
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt


from namelist import *
from grid import Grid
from fields import initialize_fields
from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from temperature_cython import temperature_tendency_jacobson_c

from boundaries import exchange_BC
from multiproc import create_subgrids 
from namelist import njobs
from time_integration_par import matsuno as time_stepper


if __name__ == '__main__':


    GR = Grid()
    GR, subgrids = create_subgrids(GR, njobs)

    COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
    UFLX, VFLX, \
    HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
    RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids)

    MIC.dPOTTdt_MIC[:] = 0
    RAD.dPOTTdt_RAD[:] = 0
    MIC.QV[:] = 0
    MIC.QV[:,:,1] = 3
    MIC.QC[:] = 0
    MIC.QC[:,:,1] = 3

    time0 = time.time()

    status = mp.Value('i', 0)
    uvflx_helix = mp.Array('f', GR.helix_size['uvflx'])
    contin_helix = mp.Array('f', GR.helix_size['contin'])
    brflx_helix = mp.Array('f', GR.helix_size['brflx'])
    prog_helix = mp.Array('f', GR.helix_size['prog'])

    lock = mp.Lock()
    barrier = mp.Barrier(parties=njobs)

    # MULTIPROCESSING
    output = mp.Queue()
    processes = []
    for job_ind in range(0,njobs):


        SGR = subgrids[job_ind]
        
        processes.append(
            mp.Process(\
                target=time_stepper,
                args = (job_ind, output, status,
                        uvflx_helix, contin_helix, brflx_helix, prog_helix,
                        lock, barrier, subgrids[job_ind],
                        COLP[SGR.GRmap_out_iijj],
                        PHI[SGR.GRmap_out_iijj], PHIVB[SGR.GRmap_out_iijj], 
                        POTT[SGR.GRmap_out_iijj], POTTVB[SGR.GRmap_out_iijj],
                        UWIND[SGR.GRmap_out_iisjj], VWIND[SGR.GRmap_out_iijjs],
                        WWIND[SGR.GRmap_out_iijj], 
                        UFLX[SGR.GRmap_out_iisjj], VFLX[SGR.GRmap_out_iijjs],
                        HSURF[SGR.GRmap_out_iijj],
                        PVTF[SGR.GRmap_out_iijj], PVTFVB[SGR.GRmap_out_iijj],
                        RAD.dPOTTdt_RAD[SGR.GRimap_out_iijj],
                        MIC.dPOTTdt_MIC[SGR.GRimap_out_iijj],
                        MIC.QV[SGR.GRmap_out_iijj], MIC.QC[SGR.GRmap_out_iijj],
                        MIC.dQVdt_MIC[SGR.GRimap_out_iijj],
                        MIC.dQCdt_MIC[SGR.GRimap_out_iijj])))

    for proc in processes:
        proc.start()


    results = [output.get() for p in processes]
    results.sort()
    dUFLXdt = np.zeros( (GR.nxs, GR.ny, GR.nz) )
    dVFLXdt = np.zeros( (GR.nx, GR.nys, GR.nz) )
    #for job_ind in range(0,njobs):
    for job_ind in range(0,njobs):
        SGR = subgrids[job_ind]
        #res = results[job_ind][1]

        #print(SGR.SGRmap_out_iijj)
        #print(SGR.SGRmap_out_iijjs)
        #quit()

        COLP[SGR.GRmap_in_iijj]     = np.asarray(results[job_ind][1]['COLP'] \
                                                    [SGR.SGRmap_out_iijj])
        PHI[SGR.GRmap_in_iijj]      = np.asarray(results[job_ind][1]['PHI'] \
                                                    [SGR.SGRmap_out_iijj])
        #PHIVB[SGR.map_iijj]    = np.asarray(results[job_ind][1]['PHIVB'])
        POTT[SGR.GRmap_in_iijj]     = np.asarray(results[job_ind][1]['POTT'] \
                                                    [SGR.SGRmap_out_iijj])
        POTTVB[SGR.GRmap_in_iijj]   = np.asarray(results[job_ind][1]['POTTVB'] \
                                                    [SGR.SGRmap_out_iijj])
        #PHIVB[SGR.map_iijj]    = np.asarray(results[job_ind][1]['PHIVB'])
        UWIND[SGR.GRmap_in_iisjj]  = np.asarray(results[job_ind][1]['UWIND'] \
                                                    [SGR.SGRmap_out_iisjj])
        VWIND[SGR.GRmap_in_iijjs]  = np.asarray(results[job_ind][1]['VWIND'] \
                                                    [SGR.SGRmap_out_iijjs])
        WWIND[SGR.GRmap_in_iijj]    = np.asarray(results[job_ind][1]['WWIND'] \
                                                    [SGR.SGRmap_out_iijj])
        UFLX[SGR.GRmap_in_iisjj]  = np.asarray(results[job_ind][1]['UFLX'] \
                                                    [SGR.SGRmap_out_iisjj])
        VFLX[SGR.GRmap_in_iijjs]  = np.asarray(results[job_ind][1]['VFLX'] \
                                                    [SGR.SGRmap_out_iijjs])
        MIC.QV[SGR.GRmap_in_iijj]   = np.asarray(results[job_ind][1]['QV'] \
                                                    [SGR.SGRmap_out_iijj])
        MIC.QC[SGR.GRmap_in_iijj]   = np.asarray(results[job_ind][1]['QC'] \
                                                    [SGR.SGRmap_out_iijj])

    for job_ind in range(0,njobs):
        SGR = subgrids[job_ind]

    for proc in processes:
        proc.join()

    #quit()

    time1 = time.time()
    print(time1 - time0)

    UFLX = exchange_BC(GR, UFLX)
    VFLX = exchange_BC(GR, VFLX)
    COLP = exchange_BC(GR, COLP)
    WWIND = exchange_BC(GR, WWIND)
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)
    POTT = exchange_BC(GR, POTT)
    MIC.QV = exchange_BC(GR, MIC.QV)
    MIC.QC = exchange_BC(GR, MIC.QC)
    PHI = exchange_BC(GR, PHI)

    #print(UFLX[:,:,1].T)
    #print(VFLX[:,:,1].T)
    #print()

    #print(UFLX.shape)
    #print(VFLX.shape)

    import pickle
    with open('testarray.pkl', 'rb') as f:
        out = pickle.load(f)
    #var_orig = out['UFLX']
    #var_orig = out['VFLX']
    var_orig = out['COLP']
    var_orig = out['WWIND']
    var_orig = out['UWIND']
    #var_orig = out['VWIND']
    var_orig = out['POTT']
    #var_orig = out['QV']
    var_orig = out['PHI']
    #var = UFLX
    #var = VFLX
    var = COLP
    var = WWIND
    var = UWIND
    #var = VWIND
    var = POTT
    var = PHI
    print('###################')
    nan_here = np.isnan(var)
    nan_orig = np.isnan(var_orig)
    diff = var - var_orig
    print('values ' + str(np.nansum(np.abs(diff))))
    print('  nans ' + str(np.sum(nan_here != nan_orig)))
    print('###################')

    #print(diff[:,:,1].T)
    #print(diff[:,:].T)
    #quit()

    plt.contourf(diff[:,:,1].T)
    #plt.contourf(var[:,:,1].T)
    #plt.contourf(var_orig[:,:,1].T)

    #plt.contourf(diff[:,:].T)
    #plt.contourf(var[:,:].T)
    #plt.contourf(var_orig[:,:].T)

    plt.colorbar()
    plt.show()
    quit()

    quit()

    #dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND,\
    #                                                    VWIND, UFLX, VFLX)
    #COLP_NEW = copy.deepcopy(COLP)
    #COLP_NEW[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    #COLP_NEW = exchange_BC(GR, COLP_NEW)

    ## DIAGNOSE WWIND
    #WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)

    #time0 = time.time()
    #
    #dUFLXdt, dVFLXdt = wind_tendency_jacobson_c(GR, njobs, UWIND, VWIND, WWIND, UFLX, VFLX,
    #                                                COLP, COLP_NEW, PHI,
    #                                                POTT, PVTF, PVTFVB)

    #print('###################')
    #time1 = time.time()
    #print('sec: ' + str(time1 - time0))

    #import pickle
    #with open('testarray.pkl', 'rb') as f:
    #    out = pickle.load(f)

    #dUFLXdt_orig = out['dUFLXdt']
    #dVFLXdt_orig = out['dVFLXdt']

    #print('###################')
    #nan_here = np.isnan(dUFLXdt)
    #nan_orig = np.isnan(dUFLXdt_orig)
    #print('u values ' + str(np.nansum(np.abs(dUFLXdt - dUFLXdt_orig))))
    #print('u nan ' + str(np.sum(nan_here != nan_orig)))
    #nan_here = np.isnan(dVFLXdt)
    #nan_orig = np.isnan(dVFLXdt_orig)
    #print('v values ' + str(np.nansum(np.abs(dVFLXdt - dVFLXdt_orig))))
    #print('v nan ' + str(np.sum(nan_here != nan_orig)))
    #print('###################')

    #import matplotlib.pyplot as plt
    #k = 5
    #fig,axes = plt.subplots(1,2, figsize=(14,5))
    ##CB1 = axes[0].contourf(dUFLXdt[:,:,k].T)
    ##plt.colorbar(CB1, ax=axes[0])
    ##CB2 = axes[1].contourf(dUFLXdt_orig[:,:,k].T)
    ##plt.colorbar(CB2, ax=axes[1])
    #CB1 = axes[0].contourf(dVFLXdt[:,:,k].T)
    #plt.colorbar(CB1, ax=axes[0])
    #CB2 = axes[1].contourf(dVFLXdt_orig[:,:,k].T)
    #plt.colorbar(CB2, ax=axes[1])
    #plt.show()




    #dPOTTdt = temperature_tendency_jacobson_c(GR, njobs, POTT, POTTVB, COLP, COLP_NEW,\
    #                                        UFLX, VFLX, WWIND, \
    #                                        RAD.dPOTTdt_RAD, MIC.dPOTTdt_MIC, \
    #                                        MIC.i_microphysics, RAD.i_radiation)
    #dPOTTdt = np.asarray(dPOTTdt)

    #import pickle
    #with open('testarray.pkl', 'rb') as f:
    #    out = pickle.load(f)
    #dPOTTdt_orig = out['dPOTTdt']
    #print('###################')
    #nan_here = np.isnan(dPOTTdt)
    #nan_orig = np.isnan(dPOTTdt_orig)
    #print('pott values ' + str(np.nansum(np.abs(dPOTTdt - dPOTTdt_orig))))
    #print('pott nan ' + str(np.sum(nan_here != nan_orig)))
    #print('###################')
    #quit()


