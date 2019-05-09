import numpy as np
from namelist import  i_colp_tendency, COLP_hor_dif_tau
from org_namelist import wp_old
from grid import tpbv, tpbvs
from boundaries_cuda import exchange_BC_gpu
import time
from numba import cuda, jit
if wp_old == 'float64':
    from numba import float64 as wp_nb
if wp_old == 'float32':
    from numba import float32 as wp_nb

@jit([wp_old+'[:,:,:], '+wp_old+'[:,:  ], '+wp_old+'[:,:,:], '+wp_old], target='gpu')
def calc_UFLX(UFLX, COLP, UWIND, dy):
    nxs = UFLX.shape[0] - 2
    ny  = UFLX.shape[1] - 2

    i_s, j, k = cuda.grid(3)
    if i_s > 0 and i_s < nxs+1 and j > 0 and j < ny+1:
        UFLX[i_s ,j  ,k] = \
                (COLP[i_s-1,j  ] + COLP[i_s  ,j  ])/2. *\
                UWIND[i_s  ,j  ,k] * dy

    cuda.syncthreads()


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:  ], '+wp_old+'[:,:,:], '+wp_old+'[:,:  ]'], target='gpu')
def calc_VFLX(VFLX, COLP, VWIND, dxjs):
    nx  = VFLX.shape[0] - 2
    nys = VFLX.shape[1] - 2

    i, js, k = cuda.grid(3)
    if i > 0 and i < nx+1 and js > 0 and js < nys+1:
        VFLX[i  ,js  ,k] = \
                (COLP[i  ,js-1] + COLP[i  ,js  ])/2. *\
                VWIND[i  ,js  ,k] * dxjs[i  ,js  ]

    cuda.syncthreads()


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
      wp_old+'[:    ], '+wp_old+'[:,:  ]'], target='gpu')
def calc_FLXDIV(FLXDIV, UFLX, VFLX, dsigma, A):
    nx = FLXDIV.shape[0] - 2
    ny = FLXDIV.shape[1] - 2

    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        FLXDIV[i  ,j  ,k] = \
                ( + UFLX[i+1,j  ,k] - UFLX[i  ,j  ,k] \
                  + VFLX[i  ,j+1,k] - VFLX[i  ,j  ,k] ) \
                  * dsigma[k] / A[i  ,j  ]

    cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def get_sum(a, b):
    return a + b


@jit([wp_old+'[:,:  ], '+wp_old+'[:,:,:]'], target='gpu')
def calc_dCOLPdt(dCOLPdt, FLXDIV):

    if i_colp_tendency:
        nx = dCOLPdt.shape[0] - 2
        ny = dCOLPdt.shape[1] - 2

        tz = cuda.threadIdx.z

        vert_sum = cuda.shared.array(tpbv, dtype=wp_nb)
        i, j, k = cuda.grid(3)
        vert_sum[tz] = FLXDIV[i,j,k]

        cuda.syncthreads()

        # sum-reduce vert_sum vertically
        t = tpbv // 2
        while t > 0:
            if tz < t:
                vert_sum[tz] = get_sum(vert_sum[tz], vert_sum[tz+t])
            t //= 2
            cuda.syncthreads()

        if tz == 0:
            dCOLPdt[i,j] = - vert_sum[0]

        if COLP_hor_dif_tau > 0:
            raise NotImplementedError('no pressure difusion in gpu implemented')
    else:
        dCOLPdt[i,j] = 0.

    cuda.syncthreads()


def colp_tendency_jacobson_gpu(GR, griddim, blockdim, stream,
                                dCOLPdt, UFLX, VFLX, FLXDIV, \
                                COLP, UWIND, VWIND, \
                                dy, dxjs, A, dsigma):
    
    calc_UFLX[GR.griddim_is, blockdim, stream](UFLX, COLP, UWIND, dy)
    stream.synchronize()
    calc_VFLX[GR.griddim_js, blockdim, stream](VFLX, COLP, VWIND, dxjs)
    stream.synchronize()
    
    # TODO 1 NECESSARY
    UFLX = exchange_BC_gpu(UFLX, GR.zonal , GR.merids, GR.griddim_is, blockdim, stream, \
                            stagx=True)
    VFLX = exchange_BC_gpu(VFLX, GR.zonals, GR.merid , GR.griddim_js, blockdim, stream, \
                            stagy=True)

    calc_FLXDIV[GR.griddim, blockdim, stream](FLXDIV, UFLX, VFLX, dsigma, A)
    stream.synchronize()
    calc_dCOLPdt[GR.griddim, blockdim, stream](dCOLPdt, FLXDIV)
    stream.synchronize()

    return(dCOLPdt, UFLX, VFLX, FLXDIV)


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:  ], '+wp_old+'[:,:,:], '+wp_old+'[:,:  ], '+wp_old+'[:]'], target='gpu')
def vertical_wind_jacobson_gpu(WWIND, dCOLPdt, FLXDIV, COLP_NEW, sigma_vb):

    nx  = FLXDIV.shape[0] - 2
    ny  = FLXDIV.shape[1] - 2
    nzs = FLXDIV.shape[2]

    vert_sum = cuda.shared.array(tpbvs, dtype=wp_nb)

    i, j, ks = cuda.grid(3)
    vert_sum[ks] = FLXDIV[i,j,ks]

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


