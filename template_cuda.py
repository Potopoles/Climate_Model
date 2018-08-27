#import pyximport; pyximport.install()
#from wind_cython_par import wind_tendency_jacobson_c
#import numpy as np
import time
import copy
#import multiprocessing as mp
import matplotlib.pyplot as plt


from namelist import *
from grid import Grid
from fields import initialize_fields
from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from temperature import temperature_tendency_jacobson
from temperature_cuda import temperature_tendency_jacobson_cuda

from boundaries import exchange_BC
from multiproc import create_subgrids 
from namelist import njobs
from time_integration import matsuno as time_stepper

from numba import vectorize, cuda, jit, float32

nsteps = 1

GR = Grid()
GR, subgrids = create_subgrids(GR, njobs)
nx = GR.nx
ny = GR.ny
nz = GR.nz
nb = GR.nb

#@cuda.jit('(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:])')
@jit(['float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:]'], target='gpu')
def temp_cuda(dPOTTdt, UFLX, VFLX, POTT):

    i, j, k = cuda.grid(3)
    for t in range(1):
        if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
            dPOTTdt[i,j,k] = ( UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/2. -
                               UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/2. -
                               VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/2. -
                               VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/2. )
        #dPOTTdt[i  ,j  ,k] = POTT[i,j,k] 

    cuda.syncthreads()



def temp_norm(GR, UFLX, VFLX, POTT):
    dPOTTdt = np.zeros( (GR.nx ,GR.ny ,GR.nz) )
    for k in range(0,GR.nz):
        dPOTTdt[:,:,k] = (+ UFLX[:,:,k][GR.iijj    ] *\
                             (POTT[:,:,k][GR.iijj_im1] + POTT[:,:,k][GR.iijj    ])/2 \
                          - UFLX[:,:,k][GR.iijj_ip1] *\
                             (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_ip1])/2 \
                          + VFLX[:,:,k][GR.iijj    ] *\
                         (POTT[:,:,k][GR.iijj_jm1] + POTT[:,:,k][GR.iijj    ])/2 \
                  - VFLX[:,:,k][GR.iijj_jp1] *\
                     (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_jp1])/2 \
                             )
    return(dPOTTdt)





if __name__ == '__main__':



    COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
    UFLX, VFLX, \
    HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
    RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids)
    #COLP_NEW = copy.deepcopy(COLP)
    #dPOTTdt_RAD = np.zeros((GR.nx,GR.ny,GR.nz))
    #dPOTTdt_MIC = np.zeros((GR.nx,GR.ny,GR.nz))

    #MIC.dPOTTdt_MIC[:] = 0
    #RAD.dPOTTdt_RAD[:] = 0
    #MIC.QV[:] = 0
    #MIC.QV[:,:,1] = 3
    #MIC.QC[:] = 0
    #MIC.QC[:,:,1] = 3

    UFLX    = np.zeros((GR.nxs+2*nb,GR.ny +2*nb,GR.nz),np.float64)
    VFLX    = np.zeros((GR.nx +2*nb,GR.nys+2*nb,GR.nz),np.float64)
    #POTT    = np.ones((GR.nx +2*nb,GR.ny +2*nb,GR.nz),np.float64)
    dPOTTdt = np.zeros((GR.nx +2*nb,GR.ny +2*nb,GR.nz),np.float64)
    UFLX[:] = 10.

    tpbh = 1
    tpbv = GR.nz
    blockdim = (tpbh, tpbh, tpbv)
    griddim = ((GR.nx+2*nb)//blockdim[0], (GR.ny+2*nb)//blockdim[1], GR.nz//blockdim[2])

    time0   = time.time()

    stream = cuda.stream()
    UFLXd   = cuda.to_device(UFLX, stream)
    VFLXd   = cuda.to_device(VFLX, stream)
    POTTd   = cuda.to_device(POTT, stream)
    dPOTTdtd = cuda.to_device(dPOTTdt, stream)

    time1 = time.time()
    print('copy ',str(time1 - time0))

    time0 = time.time()
    for i in range(0,nsteps):
        temp_cuda[griddim, blockdim, stream](dPOTTdtd, UFLXd, VFLXd, POTTd)

    
    stream.synchronize()
    time1 = time.time()
    print(time1 - time0)

    dPOTTdtd.to_host(stream)
    


    time0 = time.time()

    for i in range(0,nsteps):
        ref_dPOTTdt = temp_norm(GR, UFLX, VFLX, POTT)

    
    time1 = time.time()
    print(time1 - time0)

    quit()


    k=2
    #print(dPOTTdt[:,:,k])
    plt.contourf(dPOTTdt[:,:,k].T)
    plt.colorbar()
    plt.show()
    quit()

