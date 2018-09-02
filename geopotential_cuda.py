import numpy as np
from constants import con_g, con_Rd, con_kappa, con_cp
from namelist import pTop, wp, tpbv
from boundaries_cuda import exchange_BC_gpu
from geopotential import diag_pvt_factor as tst

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



@jit([wp+'[:,:  ], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:]'], target='gpu')
def get_geopotential(COLP, PVTF, PVTFVB, sigma_vb):
    nx  = FLXDIV.shape[0] - 2
    ny  = FLXDIV.shape[1] - 2
    nzs = FLXDIV.shape[2]

    geopot = cuda.shared.array(tpbv, dtype=float64)

    i, j, k = cuda.grid(3)
    vert_sum[k] = FLXDIV[i,j,k]

    if i > 0 and i < nx+1 and j > 0 and j < ny+1 and ks > 0 and ks < nzs:
        # cumulative-sum-reduce vert_sum vertically
        k = 0
        while k < nzs-1:
            if k == ks-1:
                vert_sum[ks] = get_sum(vert_sum[ks], vert_sum[ks-1])
                fluxdivsum = vert_sum[k]
            k = k + 1
            cuda.syncthreads()

        WWIND[i,j,ks] = - fluxdivsum / COLP_NEW[i,j] \
                        - sigma_vb[ks] * dCOLPdt[i,j] / COLP_NEW[i,j]

    cuda.syncthreads()


def diag_geopotential_jacobson_gpu(GR, stream, PHIh, PHIVBh, HSURFh, POTTh, COLPh,
                               PVTFh, PVTFVBh):

    #PVTFh, PVTFVBh = tst(GR, COLPh, PVTFh, PVTFVBh)

    PHI     = cuda.to_device(PHIh, stream)
    PHIVB         = cuda.to_device(PHIVBh, stream)
    HSURF     = cuda.to_device(HSURFh, stream)
    POTT      = cuda.to_device(POTTh, stream)
    COLP         = cuda.to_device(COLPh, stream)
    PVTF         = cuda.to_device(PVTFh, stream)
    PVTFVB         = cuda.to_device(PVTFVBh, stream)
    GR.sigma_vbd = cuda.to_device(GR.sigma_vb, stream)

    diag_pvt_factor[GR.griddim, GR.blockdim, stream] \
                        (COLP, PVTF, PVTFVB, GR.sigma_vbd)


    ##phi_vb = HSURF[GR.iijj]*con_g
    #PHIVB[:,:,GR.nzs-1][GR.iijj] = HSURF[GR.iijj]*con_g
    #PHI[:,:,GR.nz-1][GR.iijj] = PHIVB[:,:,GR.nzs-1][GR.iijj] - con_cp*  \
    #                            ( POTT[:,:,GR.nz-1][GR.iijj] * \
    #                                (   PVTF  [:,:,GR.nz-1 ][GR.iijj]  \
    #                                  - PVTFVB[:,:,GR.nzs-1][GR.iijj]  ) )
    #for k in range(GR.nz-2,-1,-1):
    #    kp1 = k + 1

    #    dphi = con_cp * POTT[:,:,kp1][GR.iijj] * \
    #                    (PVTFVB[:,:,kp1][GR.iijj] - PVTF[:,:,kp1][GR.iijj])
    #    #phi_vb = PHI[:,:,kp1][GR.iijj] - dphi
    #    PHIVB[:,:,kp1][GR.iijj] = PHI[:,:,kp1][GR.iijj] - dphi

    #    # phi_k
    #    dphi = con_cp * POTT[:,:,k][GR.iijj] * \
    #                        (PVTF[:,:,k][GR.iijj] - PVTFVB[:,:,kp1][GR.iijj])
    #    #PHI[:,:,k][GR.iijj] = phi_vb - dphi
    #    PHI[:,:,k][GR.iijj] = PHIVB[:,:,kp1][GR.iijj] - dphi

    #dphi = con_cp * POTT[:,:,0][GR.iijj] * \
    #                (PVTFVB[:,:,0][GR.iijj] - PVTF[:,:,0][GR.iijj])
    #PHIVB[:,:,0][GR.iijj] = PHI[:,:,0][GR.iijj] - dphi

    # TODO 5 NECESSARY
    PVTF  = exchange_BC_gpu(PVTF, GR.zonal, GR.merid,   \
                            GR.griddim, GR.blockdim, stream)
    PVTFVB  = exchange_BC_gpu(PVTFVB, GR.zonalvb, GR.meridvb,   \
                            GR.griddim_ks, GR.blockdim_ks, stream)
    PHI  = exchange_BC_gpu(PHI, GR.zonal, GR.merid,   \
                            GR.griddim, GR.blockdim, stream)

    PVTF     .to_host(stream)
    PVTFVB .to_host(stream)
    PHI    .to_host(stream)
    PHIVB    .to_host(stream)

    return(PHIh, PHIVBh, PVTFh, PVTFVBh)



