import numpy as np
from constants import con_kappa, con_g, con_Rd
from namelist import pTop, wp

from numba import cuda, jit


@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
def diagnose_secondary_fields_vb_gpu(POTTVB, TAIRVB, PVTFVB):
    nx = POTTVB.shape[0] - 2
    ny = POTTVB.shape[1] - 2
    i, j, ks = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        TAIRVB[i,j,ks] = POTTVB[i,j,ks] * PVTFVB[i,j,ks]

@jit([wp+'[:,:  ], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ 
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]  '], target='gpu')
def diagnose_secondary_fields_gpu(COLP, PAIR, PHI, POTT, 
                                    TAIR, RHO, PVTF,
                                    UWIND, VWIND, WIND):
    nx = COLP.shape[0] - 2
    ny = COLP.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        TAIR[i,j,k] = POTT[i,j,k] * PVTF[i,j,k]
        PAIR[i,j,k] = 100000.*(PVTF[i,j,k])**(1./con_kappa)
        RHO [i,j,k] = PAIR[i,j,k] / (con_Rd * TAIR[i,j,k])
        WIND[i,j,k] = ( ((UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k])/2.)**2. + \
                        ((VWIND[i  ,j  ,k] + VWIND[i  ,j+1,k])/2.)**2. ) ** (1./2.)



@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
def diagnose_POTTVB_jacobson_gpu(POTTVB, POTT, PVTF, PVTFVB):
    nx  = POTTVB.shape[0] - 2
    ny  = POTTVB.shape[1] - 2
    nzs = POTTVB.shape[2]
    i, j, ks = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        if ks > 0 and ks < nzs-1:
            POTTVB[i,j,ks] =   ( \
                        +   (PVTFVB[i,j,ks] - PVTF  [i,j,ks-1]) * POTT[i,j,ks-1]
                        +   (PVTF  [i,j,ks] - PVTFVB[i,j,ks  ]) * POTT[i,j,ks  ]
                                ) / (PVTF[i,j,ks] - PVTF[i,j,ks-1])
            if ks == 1:
                # extrapolate model top POTTVB
                POTTVB[i,j,ks-1] = POTT[i,j,ks-1] - ( POTTVB[i,j,ks] - POTT[i,j,ks-1] )
            elif ks == nzs-2:
                # extrapolate model bottom POTTVB
                POTTVB[i,j,ks+1] = POTT[i,j,ks  ] - ( POTTVB[i,j,ks] - POTT[i,j,ks  ] )

        


