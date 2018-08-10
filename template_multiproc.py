#import pyximport; pyximport.install()
from wind_cython import wind_tendency_jacobson_par
import numpy as np
import time
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt


from namelist import *
from grid import Grid
from fields import initialize_fields
from continuity import colp_tendency_jacobson, vertical_wind_jacobson

from boundaries import exchange_BC
from multiproc import create_subgrids 
from namelist import njobs


if __name__ == '__main__':


    GR = Grid()
    subgrids = create_subgrids(GR, njobs)

    COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
    UFLX, VFLX, UFLXMP, VFLXMP, \
    HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
    RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids)

    dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND,\
                                                        VWIND, UFLX, VFLX)
    COLP_NEW = copy.deepcopy(COLP)
    COLP_NEW[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    COLP_NEW = exchange_BC(GR, COLP_NEW)

    # DIAGNOSE WWIND
    WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)


    #print(UWIND.shape)
    #print(UWIND[GR.iijj].shape)
    #print(UWIND[GR.nb:(GR.nx+GR.nb),1:10,:].shape)
    #print(UWIND[GR.nb+0:GR.nx,1:10,:].shape)
    #quit()

    time0 = time.time()
    
    dUFLXdt, dVFLXdt = wind_tendency_jacobson_par(GR, njobs, UWIND, VWIND, WWIND, UFLX, VFLX,
                                                    COLP, COLP_NEW, PHI,
                                                    POTT, PVTF, PVTFVB)

    print('###################')
    time1 = time.time()
    print('sec: ' + str(time1 - time0))

    import pickle
    with open('testarray.pkl', 'rb') as f:
        out = pickle.load(f)

    dUFLXdt_orig = out['dUFLXdt']
    dVFLXdt_orig = out['dVFLXdt']

    print('###################')
    nan_here = np.isnan(dUFLXdt)
    nan_orig = np.isnan(dUFLXdt_orig)
    print('u values ' + str(np.nansum(np.abs(dUFLXdt - dUFLXdt_orig))))
    print('u nan ' + str(np.sum(nan_here != nan_orig)))
    nan_here = np.isnan(dVFLXdt)
    nan_orig = np.isnan(dVFLXdt_orig)
    print('v values ' + str(np.nansum(np.abs(dVFLXdt - dVFLXdt_orig))))
    print('v nan ' + str(np.sum(nan_here != nan_orig)))
    print('###################')

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




