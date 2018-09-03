import numpy as np
from constants import con_kappa, con_g, con_Rd
from namelist import pTop, wp

from numba import cuda, jit
if wp == 'float64':
    from numba import float64


def diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB, TAIR, TAIRVB, RHO,\
                                PVTF, PVTFVB, UWIND, VWIND, WIND):

    TAIR[GR.iijj] = POTT[GR.iijj] * PVTF[GR.iijj]
    TAIRVB[GR.iijj] = POTTVB[GR.iijj] * PVTFVB[GR.iijj]
    PAIR[GR.iijj] = 100000*np.power(PVTF[GR.iijj], 1/con_kappa)
    RHO[GR.iijj] = PAIR[GR.iijj] / (con_Rd * TAIR[GR.iijj])

    for k in range(0,GR.nz):
        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + \
                                        UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )

    return(PAIR, TAIR, TAIRVB, RHO, WIND)


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

        


