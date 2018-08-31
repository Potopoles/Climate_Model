import numpy as np
from namelist import  i_colp_tendency, COLP_hor_dif_tau, wp
from boundaries_cuda import exchange_BC_gpu
import time

# TODO remove
from boundaries import exchange_BC

from numba import cuda, jit
if wp == 'float64':
    from numba import float64

@jit([wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:,:], '+wp], target='gpu')
def calc_UFLX(UFLX, COLP, UWIND, dy):
    nxs = UFLX.shape[0] - 2
    ny  = UFLX.shape[1] - 2

    i_s, j, k = cuda.grid(3)
    if i_s > 0 and i_s < nxs+1 and j > 0 and j < ny+1:
        UFLX[i_s ,j  ,k] = \
                (COLP[i_s-1,j  ] + COLP[i_s  ,j  ])/2. *\
                UWIND[i_s  ,j  ,k] * dy


@jit([wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:,:], '+wp+'[:,:  ]'], target='gpu')
def calc_VFLX(VFLX, COLP, VWIND, dxjs):
    nx  = VFLX.shape[0] - 2
    nys = VFLX.shape[1] - 2

    i, js, k = cuda.grid(3)
    if i > 0 and i < nx+1 and js > 0 and js < nys+1:
        VFLX[i  ,js  ,k] = \
                (COLP[i  ,js-1] + COLP[i  ,js  ])/2. *\
                VWIND[i  ,js  ,k] * dxjs[i  ,js  ]


@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
      wp+'[:    ], '+wp+'[:,:  ]'], target='gpu')
def calc_FLXDIV(FLXDIV, UFLX, VFLX, dsigma, A):
    nx = FLXDIV.shape[0] - 2
    ny = FLXDIV.shape[1] - 2

    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        FLXDIV[i  ,j  ,k] = \
                ( + UFLX[i+1,j  ,k] - UFLX[i  ,j  ,k] \
                  + VFLX[i  ,j+1,k] - VFLX[i  ,j  ,k] ) \
                  * dsigma[k] / A[i  ,j  ]


@jit([wp+'[:,:  ], '+wp+'[:,:,:]'], target='gpu')
def calc_dCOLPdt(dCOLPdt, FLXDIV):
    nx = dCOLPdt.shape[0] - 2
    ny = dCOLPdt.shape[1] - 2

    if i_colp_tendency:
        i, j, k = cuda.grid(3)
        if i > 0 and i < nx+1 and j > 0 and j < ny+1:
            TODO
            #dCOLPdt = - np.sum(FLXDIV, axis=2)

            if COLP_hor_dif_tau > 0:
                raise NotImplementedError('no pressure difusion in gpu implemented')
    else:
        dCOLPdt[i,j,k] = 0.

    #if i_colp_tendency:
    #    dCOLPdt = - np.sum(FLXDIV, axis=2)

    #    if COLP_hor_dif_tau > 0:
    #        raise NotImplementedError('no pressure difusion in gpu implemented')
    #else:
    #    dCOLPdt =  np.zeros( (GR.nx+2*GR.nb,GR.ny+2*GR.nb) )


#@jit() #TODO test whether performance changes with jit
def colp_tendency_jacobson_gpu(GR, griddim, blockdim, stream,
                                dCOLPdt, UFLX, VFLX, FLXDIV, \
                                COLP, UWIND, VWIND, \
                                dy, dxjs):
    
    # TODO REMOVE GR

    #nxs = UFLX.shape[0] - 2
    #ny  = UFLX.shape[1] - 2
    #nz  = UFLX.shape[2]


    COLPd         = cuda.to_device(COLP, stream)
    dCOLPdtd      = cuda.to_device(dCOLPdt, stream)
    UFLXd         = cuda.to_device(UFLX, stream)
    VFLXd         = cuda.to_device(VFLX, stream)
    FLXDIVd       = cuda.to_device(FLXDIV, stream)
    UWINDd        = cuda.to_device(UWIND, stream)
    VWINDd        = cuda.to_device(VWIND, stream)
    dxjsd         = cuda.to_device(GR.dxjs, stream)
    Ad            = cuda.to_device(GR.A, stream)
    dsigmad       = cuda.to_device(GR.dsigma, stream)

    calc_UFLX[GR.griddim_is, blockdim, stream](UFLXd, COLPd, UWINDd, dy)
    calc_VFLX[GR.griddim_js, blockdim, stream](VFLXd, COLPd, VWINDd, dxjsd)
    stream.synchronize()
    
    # TODO 1 NECESSARY
    UFLXd = exchange_BC_gpu(UFLXd, GR.zonal , GR.merids, GR.griddim_is, blockdim, stream, \
                            stagx=True)
    VFLXd = exchange_BC_gpu(VFLXd, GR.zonals, GR.merid , GR.griddim_js, blockdim, stream, \
                            stagy=True)
    stream.synchronize()

    calc_FLXDIV[GR.griddim, blockdim, stream](FLXDIVd, UFLXd, VFLXd, dsigmad, Ad)
    calc_dCOLPdt[GR.griddim, blockdim, stream](dCOLPdtd, FLXDIVd)

    UFLXd.to_host(stream)
    VFLXd.to_host(stream)
    FLXDIVd.to_host(stream)
    dCOLPdtd.to_host(stream)



    return(dCOLPdt, UFLX, VFLX, FLXDIV)


def vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND):


    for ks in range(1,GR.nzs-1):

        WWIND[:,:,ks][GR.iijj] = - np.sum(FLXDIV[:,:,:ks][GR.iijj], axis=2) / \
                                    COLP_NEW[GR.iijj] \
                                 - GR.sigma_vb[ks] * dCOLPdt[GR.iijj] / \
                                    COLP_NEW[GR.iijj]



    return(WWIND)


