import numpy as np
from constants import con_g, con_Rd, con_kappa, con_cp
from namelist import pTop, wp
from boundaries_cuda import exchange_BC_gpu

from numba import cuda, jit
if wp == 'float64':
    from numba import float64
from math import pow



@jit([wp+'[:,:  ], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:]'], target='gpu')
def diag_pvt_factor(COLP, PVTF, PVTFVB, sigma_vb):


    nx = PVTF.shape[0] - 2
    ny = PVTF.shape[1] - 2
    nz = PVTF.shape[2]
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        
        pairvb_km12 = pTop + sigma_vb[k  ] * COLP[i,j]
        pairvb_kp12 = pTop + sigma_vb[k+1] * COLP[i,j]
        
        PVTF[i,j,k] = 1./(1.+con_kappa) * \
                    ( pow( pairvb_kp12/100000. , con_kappa ) * pairvb_kp12 - \
                      pow( pairvb_km12/100000. , con_kappa ) * pairvb_km12 ) / \
                    ( pairvb_kp12 - pairvb_km12 )

        PVTFVB[i,j,k] = pow( pairvb_km12/100000. , con_kappa )
        if k == nz-1:
            PVTFVB[i,j,k+1] = pow( pairvb_kp12/100000. , con_kappa )

    cuda.syncthreads()



@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
      wp+'[:,:,:], '+wp+'[:,:]'], target='gpu')
def get_geopotential(PHI, PHIVB, PVTF, PVTFVB, \
                     POTT, HSURF):
    nx  = PHIVB.shape[0] - 2
    ny  = PHIVB.shape[1] - 2
    nzs = PHIVB.shape[2]
    i, j, ks = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        kiter = nzs-1
        if ks == kiter:
            PHIVB[i,j,ks] = HSURF[i,j]*con_g
        kiter = kiter - 1
        cuda.syncthreads()

        while kiter >= 0:
            if ks == kiter:
                PHI  [i,j,ks] = PHIVB[i,j,ks+1] - con_cp*  \
                                        ( POTT[i,j,ks] * (   PVTF  [i,j,ks  ] \
                                                           - PVTFVB[i,j,ks+1] ) )
                PHIVB[i,j,ks] = PHI  [i,j,ks  ] - con_cp * \
                                        ( POTT[i,j,ks] * (   PVTFVB[i,j,ks  ] \
                                                           - PVTF  [i,j,ks  ] ) )

            kiter = kiter - 1
            cuda.syncthreads()





def diag_geopotential_jacobson_gpu(GR, stream, PHI, PHIVB, HSURF, POTT, COLP,
                                   PVTF, PVTFVB):

    diag_pvt_factor[GR.griddim, GR.blockdim, stream] \
                        (COLP, PVTF, PVTFVB, GR.sigma_vbd)

    get_geopotential[GR.griddim_ks, GR.blockdim_ks, stream] \
                       (PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 

    # TODO 5 NECESSARY
    PVTF  = exchange_BC_gpu(PVTF, GR.zonal, GR.merid,   \
                            GR.griddim, GR.blockdim, stream)
    PVTFVB  = exchange_BC_gpu(PVTFVB, GR.zonalvb, GR.meridvb,   \
                            GR.griddim_ks, GR.blockdim_ks, stream)
    PHI  = exchange_BC_gpu(PHI, GR.zonal, GR.merid,   \
                            GR.griddim, GR.blockdim, stream)

    return(PHI, PHIVB, PVTF, PVTFVB)



