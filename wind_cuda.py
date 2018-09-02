import time
import numpy as np
from boundaries_cuda import exchange_BC_gpu
from constants import con_cp, con_rE, con_Rd
from namelist import WIND_hor_dif_tau, i_wind_tendency, wp

from numba import cuda, jit
if wp == 'float64':
    from numba import float64
from math import cos, sin

i_hor_adv  = 1
i_vert_adv = 1
i_coriolis = 1
i_pre_grad = 1
i_num_dif  = 1


def wind_tendency_jacobson_gpu(GR, UWIND, VWIND, WWIND, UFLX, dUFLXdt, VFLX, dVFLXdt,
                                BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                                COLP, COLP_NEW, PHI, POTT, PVTF, PVTFVB,
                                A, dsigma, sigma_vb, corf, corf_is, lat_rad, latis_rad,
                                dy, dlon_rad, dxjs,
                                stream):
    
    set_to[GR.griddim_is, GR.blockdim, stream](dUFLXdt, 0.)
    set_to[GR.griddim_js, GR.blockdim, stream](dVFLXdt, 0.)

    WWIND_UWIND_ks = cuda.device_array( (GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nzs), \
                                        dtype=UWIND.dtype )
    WWIND_VWIND_ks = cuda.device_array( (GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nzs), \
                                        dtype=VWIND.dtype )


    if i_wind_tendency:

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            # CALCULATE MOMENTUM FLUXES
            calc_fluxes_ij[GR.griddim, GR.blockdim, stream] \
                            (BFLX, RFLX, UFLX, VFLX)
            BFLX = exchange_BC_gpu(BFLX, GR.zonal, GR.merid, GR.griddim,
                                    GR.blockdim, stream)
            RFLX = exchange_BC_gpu(RFLX, GR.zonal, GR.merid, GR.griddim,
                                    GR.blockdim, stream)

            calc_fluxes_isj[GR.griddim_is, GR.blockdim, stream] \
                            (SFLX, TFLX, UFLX, VFLX)
            SFLX = exchange_BC_gpu(SFLX, GR.zonal, GR.merids, GR.griddim_is,
                                    GR.blockdim, stream, stagx=True)
            TFLX = exchange_BC_gpu(TFLX, GR.zonal, GR.merids, GR.griddim_is,
                                    GR.blockdim, stream, stagx=True)

            calc_fluxes_ijs[GR.griddim_js, GR.blockdim, stream] \
                            (DFLX, EFLX, UFLX, VFLX)
            DFLX = exchange_BC_gpu(DFLX, GR.zonals, GR.merid, GR.griddim_js,
                                    GR.blockdim, stream, stagy=True)
            EFLX = exchange_BC_gpu(EFLX, GR.zonals, GR.merid, GR.griddim_js,
                                    GR.blockdim, stream, stagy=True)

            calc_fluxes_isjs[GR.griddim_is_js, GR.blockdim, stream] \
                            (CFLX, QFLX, UFLX, VFLX)
            CFLX = exchange_BC_gpu(CFLX, GR.zonals, GR.merids, GR.griddim_is_js,
                                    GR.blockdim, stream, stagx=True, stagy=True)
            QFLX = exchange_BC_gpu(QFLX, GR.zonals, GR.merids, GR.griddim_is_js,
                                    GR.blockdim, stream, stagx=True, stagy=True)

        # VERTICAL ADVECTION
        if i_vert_adv:
            calc_WWIND_VWIND_ks[GR.griddim_js_ks, GR.blockdim_ks, stream] \
                                (WWIND_VWIND_ks, VWIND, COLP_NEW, WWIND, A, 
                                dsigma)
            calc_WWIND_UWIND_ks[GR.griddim_is_ks, GR.blockdim_ks, stream] \
                                (WWIND_UWIND_ks, UWIND, COLP_NEW, WWIND, A, 
                                dsigma)

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

        run_UWIND[GR.griddim_is, GR.blockdim, stream] \
                        (dUFLXdt, UWIND, VWIND, COLP,
                        UFLX, BFLX, CFLX, DFLX, EFLX,
                        PHI, POTT, PVTF, PVTFVB, WWIND_UWIND_ks,
                        corf_is, latis_rad, dlon_rad,
                        dsigma, sigma_vb, dy)

        run_VWIND[GR.griddim_js, GR.blockdim, stream] \
                        (dVFLXdt, UWIND, VWIND, COLP,
                        VFLX, RFLX, QFLX, SFLX, TFLX,
                        PHI, POTT, PVTF, PVTFVB, WWIND_VWIND_ks,
                        corf, lat_rad, dlon_rad,
                        dsigma, sigma_vb, dxjs)

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

    return(dUFLXdt, dVFLXdt)


@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+ \
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
      wp+'[:,:  ], '+wp+'[:,:  ], '+wp+', '+ \
      wp+'[:    ], '+wp+'[:    ], '+wp], target='gpu')
def run_UWIND(dUFLXdt, UWIND, VWIND, COLP,
                UFLX, BFLX, CFLX, DFLX, EFLX,
                PHI, POTT, PVTF, PVTFVB, WWIND_UWIND_ks,
                corf_is, latis_rad, dlon_rad,
                dsigma, sigma_vb, dy):
    nx = dUFLXdt.shape[0] - 2
    ny = dUFLXdt.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
                    + BFLX [i-1,j  ,k] * \
                    ( UWIND[i-1,j  ,k] + UWIND[i  ,j  ,k] )/2. \
                    - BFLX [i  ,j  ,k] * \
                    ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. \
                    \
                    + CFLX [i  ,j  ,k] * \
                    ( UWIND[i  ,j-1,k] + UWIND[i  ,j  ,k] )/2. \
                    - CFLX [i  ,j+1,k] * \
                    ( UWIND[i  ,j  ,k] + UWIND[i  ,j+1,k] )/2. \
                    \
                    + DFLX [i-1,j  ,k] * \
                    ( UWIND[i-1,j-1,k] + UWIND[i  ,j  ,k] )/2. \
                    - DFLX [i  ,j+1,k] * \
                    ( UWIND[i  ,j  ,k] + UWIND[i+1,j+1,k] )/2. \
                    \
                    + EFLX [i  ,j  ,k] * \
                    ( UWIND[i+1,j-1,k] + UWIND[i  ,j  ,k] )/2. \
                    - EFLX [i-1,j+1,k] * \
                    ( UWIND[i  ,j  ,k] + UWIND[i-1,j+1,k] )/2. 


        # VERTICAL ADVECTION
        if i_vert_adv:
            dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
                                (WWIND_UWIND_ks[i  ,j  ,k  ] - \
                                 WWIND_UWIND_ks[i  ,j  ,k+1]  ) / dsigma[k]


        # CORIOLIS AND SPHERICAL GRID CONVERSION
        if i_coriolis:
            dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
                con_rE*dlon_rad*dlon_rad/2.*(\
                  COLP [i-1,j    ] * \
                ( VWIND[i-1,j  ,k] + VWIND[i-1,j+1,k] )/2. * \
                ( corf_is[i  ,j  ] * con_rE *\
                  cos(latis_rad[i  ,j  ]) + \
                  ( UWIND[i-1,j  ,k] + UWIND[i  ,j  ,k] )/2. * \
                  sin(latis_rad[i  ,j  ]) )\
                + COLP [i  ,j    ] * \
                ( VWIND[i  ,j  ,k] + VWIND[i  ,j+1,k] )/2. * \
                ( corf_is[i  ,j  ] * con_rE * \
                  cos(latis_rad[i  ,j  ]) + \
                  ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
                  sin(latis_rad[i  ,j  ]) )\
                )


        # PRESSURE GRADIENT
        if i_pre_grad:
            dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
                 - dy * ( \
                ( PHI [i  ,j  ,k]  - PHI [i-1,j  ,k] ) * \
                ( COLP[i  ,j    ]  + COLP[i-1,j    ] )/2. + \
                ( COLP[i  ,j    ]  - COLP[i-1,j    ] ) * con_cp/2. * \
                (\
                  + POTT[i-1,j  ,k] / dsigma[k] * \
                    ( \
                        sigma_vb[k+1] * \
                        ( PVTFVB[i-1,j  ,k+1] - PVTF  [i-1,j  ,k] ) + \
                        sigma_vb[k  ] * \
                        ( PVTF  [i-1,j  ,k  ] - PVTFVB[i-1,j  ,k] )   \
                    ) \
                  + POTT[i  ,j  ,k] / dsigma[k] * \
                    ( \
                        sigma_vb[k+1] * \
                        ( PVTFVB[i  ,j  ,k+1] - PVTF  [i  ,j  ,k] ) + \
                        sigma_vb[k  ] * \
                        ( PVTF  [i  ,j  ,k  ] - PVTFVB[i  ,j  ,k] )   \
                    ) \
                ) )


        # HORIZONTAL DIFFUSION
        if i_num_dif and (WIND_hor_dif_tau > 0):
            dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
                                WIND_hor_dif_tau * \
                             (  UFLX[i-1,j  ,k] + UFLX[i+1,j  ,k] \
                              + UFLX[i  ,j-1,k] + UFLX[i  ,j+1,k] \
                           - 4.*UFLX[i  ,j  ,k] )

    cuda.syncthreads()


@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+ \
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
      wp+'[:,:  ], '+wp+'[:,:  ], '+wp+', '+ \
      wp+'[:    ], '+wp+'[:    ], '+wp+'[:,:  ]'], target='gpu')
def run_VWIND(dVFLXdt, UWIND, VWIND, COLP,
                VFLX, RFLX, QFLX, SFLX, TFLX,
                PHI, POTT, PVTF, PVTFVB, WWIND_VWIND_ks,
                corf, lat_rad, dlon_rad,
                dsigma, sigma_vb, dxjs):
    nx = dVFLXdt.shape[0] - 2
    ny = dVFLXdt.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
                      RFLX [i  ,j-1,k] * \
                    ( VWIND[i  ,j-1,k] + VWIND[i  ,j  ,k] )/2. \
                    - RFLX [i  ,j  ,k] * \
                    ( VWIND[i  ,j  ,k] + VWIND[i  ,j+1,k] )/2. \
                    \
                    + QFLX [i  ,j  ,k] * \
                    ( VWIND[i-1,j  ,k] + VWIND[i  ,j  ,k] )/2. \
                    - QFLX [i+1,j  ,k] * \
                    ( VWIND[i  ,j  ,k] + VWIND[i+1,j  ,k] )/2. \
                    \
                    + SFLX [i  ,j-1,k] * \
                    ( VWIND[i-1,j-1,k] + VWIND[i  ,j  ,k] )/2. \
                    - SFLX [i+1,j  ,k] * \
                    ( VWIND[i  ,j  ,k] + VWIND[i+1,j+1,k] )/2. \
                    \
                    + TFLX [i+1,j-1,k] * \
                    ( VWIND[i+1,j-1,k] + VWIND[i  ,j  ,k] )/2. \
                    - TFLX [i  ,j  ,k] * \
                    ( VWIND[i  ,j  ,k] + VWIND[i-1,j+1,k] )/2. 


        # VERTICAL ADVECTION
        if i_vert_adv:
            dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
                                (WWIND_VWIND_ks[i  ,j  ,k  ] - \
                                 WWIND_VWIND_ks[i  ,j  ,k+1]  ) / dsigma[k]


        # CORIOLIS AND SPHERICAL GRID CONVERSION
        if i_coriolis:
            dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
                 - con_rE*dlon_rad*dlon_rad/2.*(\
                  COLP[i  ,j-1  ] * \
                ( UWIND[i  ,j-1,k] + UWIND[i+1,j-1,k] )/2. * \
                ( corf[i  ,j-1  ] * con_rE *\
                  cos(lat_rad[i  ,j-1  ]) +\
                  ( UWIND[i  ,j-1,k] + UWIND[i+1,j-1,k] )/2. * \
                  sin(lat_rad[i  ,j-1  ]) )\

                + COLP [i  ,j    ] * \
                ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
                ( corf [i  ,j    ] * con_rE *\
                  cos(lat_rad[i  ,j    ]) +\
                  ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
                  sin(lat_rad[i  ,j    ]) )\
                )


        # PRESSURE GRADIENT
        if i_pre_grad:
            dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
                - dxjs[i  ,j    ] * ( \
                ( PHI [i  ,j  ,k] - PHI [i  ,j-1,k] ) *\
                ( COLP[i  ,j    ] + COLP[i  ,j-1  ] )/2. + \
                ( COLP[i  ,j    ] - COLP[i  ,j-1  ] ) * con_cp/2. * \
                (\
                    POTT[i  ,j-1,k] / dsigma[k] * \
                    ( \
                        + sigma_vb[k+1] * \
                        ( PVTFVB[i  ,j-1,k+1] - PVTF  [i  ,j-1,k] ) \
                        + sigma_vb[k  ] * \
                        ( PVTF  [i  ,j-1,k  ] - PVTFVB[i  ,j-1,k] ) \
                    ) +\
                    POTT[i  ,j  ,k] / dsigma[k] * \
                    ( \
                        + sigma_vb[k+1] * \
                        ( PVTFVB[i  ,j  ,k+1] - PVTF  [i  ,j  ,k] ) \
                        + sigma_vb[k  ] * \
                        ( PVTF  [i  ,j  ,k  ] - PVTFVB[i  ,j  ,k] ) \
                    ) \
                ) )


        # HORIZONTAL DIFFUSION
        if i_num_dif and (WIND_hor_dif_tau > 0):
            dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
                                WIND_hor_dif_tau * \
                             (  VFLX[i-1,j  ,k] + VFLX[i+1,j  ,k] \
                              + VFLX[i  ,j-1,k] + VFLX[i  ,j+1,k] \
                           - 4.*VFLX[i  ,j  ,k] )

    cuda.syncthreads()








def vertical_interp_UWIND(GR, COLP_NEW, UWIND, WWIND, WWIND_UWIND, ks):

    COLPAWWIND_is_ks = 1/8*( COLP_NEW[GR.iisjj_im1_jp1] * GR.A[GR.iisjj_im1_jp1] * \
                                                 WWIND[:,:,ks][GR.iisjj_im1_jp1] + \
                             COLP_NEW[GR.iisjj_jp1    ] * GR.A[GR.iisjj_jp1    ] * \
                                                 WWIND[:,:,ks][GR.iisjj_jp1    ] + \
                         2 * COLP_NEW[GR.iisjj_im1    ] * GR.A[GR.iisjj_im1    ] * \
                                                 WWIND[:,:,ks][GR.iisjj_im1    ] + \
                         2 * COLP_NEW[GR.iisjj        ] * GR.A[GR.iisjj        ] * \
                                                 WWIND[:,:,ks][GR.iisjj        ] + \
                             COLP_NEW[GR.iisjj_im1_jm1] * GR.A[GR.iisjj_im1_jm1] * \
                                                 WWIND[:,:,ks][GR.iisjj_im1_jm1] + \
                             COLP_NEW[GR.iisjj_jm1    ] * GR.A[GR.iisjj_jm1    ] * \
                                                 WWIND[:,:,ks][GR.iisjj_jm1    ]   )

    # ATTEMPT TO INTERPOLATE ONLY WITH TWO NEIGHBORING POINTS (JACOBSON)
    COLPAWWIND_is_ks[:,-1] = 1/4*( \
                            COLP_NEW[GR.iis-1,GR.jj[-1]] * GR.A[GR.iis-1,GR.jj[-1]] * \
                                                  WWIND[:,:,ks][GR.iis-1,GR.jj[-1]] + \
                            COLP_NEW[GR.iis  ,GR.jj[-1]] * GR.A[GR.iis  ,GR.jj[-1]] * \
                                                  WWIND[:,:,ks][GR.iis  ,GR.jj[-1]] + \
                            COLP_NEW[GR.iis-1,GR.jj[-2]] * GR.A[GR.iis-1,GR.jj[-2]] * \
                                                  WWIND[:,:,ks][GR.iis-1,GR.jj[-2]] + \
                            COLP_NEW[GR.iis  ,GR.jj[-2]] * GR.A[GR.iis  ,GR.jj[-2]] * \
                                                  WWIND[:,:,ks][GR.iis  ,GR.jj[-2]]  )

    COLPAWWIND_is_ks[:, 0] = 1/4*( \
                            COLP_NEW[GR.iis-1,GR.jj[ 0]] * GR.A[GR.iis-1,GR.jj[ 0]] * \
                                                  WWIND[:,:,ks][GR.iis-1,GR.jj[ 0]] + \
                            COLP_NEW[GR.iis  ,GR.jj[ 0]] * GR.A[GR.iis  ,GR.jj[ 0]] * \
                                                  WWIND[:,:,ks][GR.iis  ,GR.jj[ 0]] + \
                            COLP_NEW[GR.iis-1,GR.jj[ 1]] * GR.A[GR.iis-1,GR.jj[ 1]] * \
                                                  WWIND[:,:,ks][GR.iis-1,GR.jj[ 1]] + \
                            COLP_NEW[GR.iis  ,GR.jj[ 1]] * GR.A[GR.iis  ,GR.jj[ 1]] * \
                                                  WWIND[:,:,ks][GR.iis  ,GR.jj[ 1]]   )


    UWIND_ks = ( GR.dsigma[ks  ] * UWIND[:,:,ks-1][GR.iisjj] +   \
                 GR.dsigma[ks-1] * UWIND[:,:,ks  ][GR.iisjj] ) / \
               ( GR.dsigma[ks  ] + GR.dsigma[ks-1] )

    WWIND_UWIND_ks = COLPAWWIND_is_ks * UWIND_ks

    return(WWIND_UWIND_ks)

def vertical_interp_VWIND(GR, COLP_NEW, VWIND, WWIND, WWINDVWIND, ks):

    COLPAWWIND_js_ks = 1/8*( COLP_NEW[GR.iijjs_ip1_jm1] * GR.A[GR.iijjs_ip1_jm1] * \
                                                 WWIND[:,:,ks][GR.iijjs_ip1_jm1] + \
                             COLP_NEW[GR.iijjs_ip1    ] * GR.A[GR.iijjs_ip1    ] * \
                                                 WWIND[:,:,ks][GR.iijjs_ip1    ] + \
                         2 * COLP_NEW[GR.iijjs_jm1    ] * GR.A[GR.iijjs_jm1    ] * \
                                                 WWIND[:,:,ks][GR.iijjs_jm1    ] + \
                         2 * COLP_NEW[GR.iijjs        ] * GR.A[GR.iijjs        ] * \
                                                 WWIND[:,:,ks][GR.iijjs        ] + \
                             COLP_NEW[GR.iijjs_im1_jm1] * GR.A[GR.iijjs_im1_jm1] * \
                                                 WWIND[:,:,ks][GR.iijjs_im1_jm1] + \
                             COLP_NEW[GR.iijjs_im1    ] * GR.A[GR.iijjs_im1    ] * \
                                                 WWIND[:,:,ks][GR.iijjs_im1    ]   )

    VWIND_ks = ( GR.dsigma[ks  ] * VWIND[:,:,ks-1][GR.iijjs] +   \
                 GR.dsigma[ks-1] * VWIND[:,:,ks  ][GR.iijjs] ) / \
               ( GR.dsigma[ks  ] + GR.dsigma[ks-1] )

    WWIND_VWIND_ks = COLPAWWIND_js_ks * VWIND_ks

    return(WWIND_VWIND_ks)

def vertical_advection(GR, WWIND_UWIND, WWIND_VWIND, k):

    vertAdv_UWIND = (WWIND_UWIND[:,:,k  ] - WWIND_UWIND[:,:,k+1]) / GR.dsigma[k]
    vertAdv_VWIND = (WWIND_VWIND[:,:,k  ] - WWIND_VWIND[:,:,k+1]) / GR.dsigma[k]

    return(vertAdv_UWIND, vertAdv_VWIND)


def horizontal_diffusion(GR, UFLX, VFLX, k):

    diff_UWIND = WIND_hor_dif_tau * \
                 (  UFLX[:,:,k][GR.iisjj_im1] + UFLX[:,:,k][GR.iisjj_ip1] \
                  + UFLX[:,:,k][GR.iisjj_jm1] + UFLX[:,:,k][GR.iisjj_jp1] - 4*UFLX[:,:,k][GR.iisjj])

    diff_VWIND = WIND_hor_dif_tau * \
                 (  VFLX[:,:,k][GR.iijjs_im1] + VFLX[:,:,k][GR.iijjs_ip1] \
                  + VFLX[:,:,k][GR.iijjs_jm1] + VFLX[:,:,k][GR.iijjs_jp1] - 4*VFLX[:,:,k][GR.iijjs])

    return(diff_UWIND, diff_VWIND)






@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
def calc_fluxes_ij(BFLX, RFLX, UFLX, VFLX):
    nx = BFLX.shape[0] - 2
    ny = BFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        BFLX[i  ,j  ,k] = 1./12. * (      UFLX[i  ,j-1,k] + \
                                          UFLX[i+1,j-1,k]   + \
                                     2.*( UFLX[i  ,j  ,k] + \
                                          UFLX[i+1,j  ,k] ) + \
                                          UFLX[i  ,j+1,k] + \
                                          UFLX[i+1,j+1,k]     )

        RFLX[i  ,j  ,k] = 1./12. * (      VFLX[i-1,j  ,k] + \
                                          VFLX[i-1,j+1,k]   +\
                                     2.*( VFLX[i  ,j  ,k] + \
                                          VFLX[i  ,j+1,k] ) +\
                                          VFLX[i+1,j  ,k] + \
                                          VFLX[i+1,j+1,k]    )
    cuda.syncthreads()

@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
def calc_fluxes_isj(SFLX, TFLX, UFLX, VFLX):
    nx = SFLX.shape[0] - 2
    ny = SFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        SFLX[i  ,j  ,k]  = 1./24. * (  VFLX[i-1,j  ,k]  + \
                                       VFLX[i-1,j+1,k] +\
                                       VFLX[i  ,j  ,k]  +   \
                                       VFLX[i  ,j+1,k] +\
                                       UFLX[i-1,j  ,k]  + \
                                    2.*UFLX[i  ,j  ,k] +\
                                       UFLX[i+1,j  ,k]   )

        TFLX[i  ,j  ,k]  = 1./24. * (  VFLX[i-1,j  ,k] + \
                                       VFLX[i-1,j+1,k] +\
                                       VFLX[i  ,j  ,k] + \
                                       VFLX[i  ,j+1,k] +\
                                     - UFLX[i-1,j  ,k] - \
                                    2.*UFLX[i  ,j  ,k] +\
                                     - UFLX[i+1,j  ,k]   )
    cuda.syncthreads()


@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
def calc_fluxes_ijs(DFLX, EFLX, UFLX, VFLX):
    nx = DFLX.shape[0] - 2
    ny = DFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        DFLX[i  ,j  ,k]  = 1./24. * (  VFLX[i  ,j-1,k]   + \
                                    2.*VFLX[i  ,j  ,k] +\
                                       VFLX[i  ,j+1,k]   + \
                                       UFLX[i  ,j-1,k]   +\
                                       UFLX[i  ,j  ,k]   + \
                                       UFLX[i+1,j-1,k]   +\
                                       UFLX[i+1,j  ,k]   )

        EFLX[i  ,j  ,k]  = 1./24. * (  VFLX[i  ,j-1,k]    + \
                                    2.*VFLX[i  ,j  ,k]  +\
                                       VFLX[i  ,j+1,k]    - \
                                       UFLX[i  ,j-1,k]    +\
                                     - UFLX[i  ,j  ,k]    - \
                                       UFLX[i+1,j-1,k]    +\
                                     - UFLX[i+1,j  ,k]    )
    cuda.syncthreads()

@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
def calc_fluxes_isjs(CFLX, QFLX, UFLX, VFLX):
    nx = CFLX.shape[0] - 2
    ny = CFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        CFLX[i  ,j  ,k] = 1./12. * (  VFLX[i-1,j-1,k]   + \
                                      VFLX[i  ,j-1,k]   +\
                                 2.*( VFLX[i-1,j  ,k]   + \
                                      VFLX[i  ,j  ,k] ) +\
                                      VFLX[i-1,j+1,k]   + \
                                      VFLX[i  ,j+1,k]   )

        QFLX[i  ,j  ,k] = 1./12. * (  UFLX[i-1,j-1,k]   + \
                                      UFLX[i-1,j  ,k]   +\
                                 2.*( UFLX[i  ,j-1,k]   + \
                                      UFLX[i  ,j  ,k] ) +\
                                      UFLX[i+1,j-1,k]   + \
                                      UFLX[i+1,j  ,k]    )
    cuda.syncthreads()




@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:,:], '+wp+'[:,:  ], '+ \
      wp+'[:    ]'], target='gpu')
def calc_WWIND_UWIND_ks(WWIND_UWIND_ks, UWIND, COLP_NEW, WWIND, A,
                        dsigma):
    nx = WWIND_UWIND_ks.shape[0] - 2
    ny = WWIND_UWIND_ks.shape[1] - 2
    nz = WWIND_UWIND_ks.shape[2]
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1 and k > 0 and k < nz-1:
        if j == 1:
            # INTERPOLATE DIFFERENTLY AT MERID. BOUNDARIES (JACOBSON)
            COLPAWWIND_is_ks = 1./4.*( \
                COLP_NEW[i-1,j  ] * A[i-1,j  ] * \
                                WWIND[i-1,j  ,k ] + \
                COLP_NEW[i  ,j  ] * A[i  ,j  ] * \
                                WWIND[i  ,j  ,k ] + \
                COLP_NEW[i-1,j+1] * A[i-1,j+1] * \
                                WWIND[i-1,j+1,k ] + \
                COLP_NEW[i ,j+1] * A[i   ,j+1] * \
                                WWIND[i  ,j+1,k ]   )
        elif j == ny:
            # INTERPOLATE DIFFERENTLY AT MERID. BOUNDARIES (JACOBSON)
            COLPAWWIND_is_ks = 1./4.*( \
                COLP_NEW[i-1,j  ] * A[i-1,j  ] * \
                                WWIND[i-1,j  ,k ] + \
                COLP_NEW[i  ,j  ] * A[i  ,j  ] * \
                                WWIND[i  ,j  ,k ] + \
                COLP_NEW[i-1,j-1] * A[i-1,j-1] * \
                                WWIND[i-1,j-1,k ] + \
                COLP_NEW[i  ,j-1] * A[i  ,j-1] * \
                                WWIND[i  ,j-1,k ]  )
        else:
            COLPAWWIND_is_ks = 1./8.*( \
                COLP_NEW[i-1,j+1] * A[-1,j+1] * \
                                WWIND[i-1,j+1,k] + \
                COLP_NEW[i  ,j+1] * A[i  ,j+1] * \
                                WWIND[i  ,j+1,k] + \
           2. * COLP_NEW[i-1,j  ] * A[i-1,j  ] * \
                                WWIND[i-1,j  ,k] + \
           2. * COLP_NEW[i  ,j  ] * A[i  ,j  ] * \
                                WWIND[i  ,j  ,k] + \
                COLP_NEW[i-1,j-1] * A[i-1,j-1] * \
                                WWIND[i-1,j-1,k] + \
                COLP_NEW[i  ,j-1] * A[i  ,j-1] * \
                                WWIND[i  ,j-1,k]   )

        UWIND_ks = ( dsigma[k  ] * UWIND[i  ,j  ,k-1] +   \
                     dsigma[k-1] * UWIND[i  ,j  ,k  ] ) / \
                   ( dsigma[k  ] + dsigma[k-1] )
        WWIND_UWIND_ks[i  ,j  ,k ] = COLPAWWIND_is_ks * UWIND_ks

    if k == 0 or k == nz-1:
        WWIND_UWIND_ks[i  ,j  ,k ] = 0.

    cuda.syncthreads()





@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:,:], '+wp+'[:,:  ], '+ \
      wp+'[:    ]'], target='gpu')
def calc_WWIND_VWIND_ks(WWIND_VWIND_ks, VWIND, COLP_NEW, WWIND, A,
                        dsigma):
    nx = WWIND_VWIND_ks.shape[0] - 2
    ny = WWIND_VWIND_ks.shape[1] - 2
    nz = WWIND_VWIND_ks.shape[2]
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1 and k > 0 and k < nz-1:
        COLPAWWIND_js_ks = 1./8.*( \
                 COLP_NEW[i+1,j-1] * A[i+1,j-1] * \
                                 WWIND[i+1,j-1,k] + \
                 COLP_NEW[i+1,j  ] * A[i+1,j  ] * \
                                 WWIND[i+1,j  ,k] + \
            2. * COLP_NEW[i  ,j-1] * A[i  ,j-1] * \
                                 WWIND[i  ,j-1,k] + \
            2. * COLP_NEW[i  ,j  ] * A[i  ,j  ] * \
                                 WWIND[i  ,j  ,k] + \
                 COLP_NEW[i-1,j-1] * A[i-1,j-1] * \
                                 WWIND[i-1,j-1,k] + \
                 COLP_NEW[i-1,j  ] * A[i-1,j  ] * \
                                 WWIND[i-1,j  ,k]   )

        VWIND_ks = ( dsigma[k  ] * VWIND[i  ,j  ,k-1] +   \
                     dsigma[k-1] * VWIND[i  ,j  ,k  ] ) / \
                   ( dsigma[k  ] + dsigma[k-1] )
        WWIND_VWIND_ks[i  ,j  ,k ] = COLPAWWIND_js_ks * VWIND_ks

    if k == 0 or k == nz-1:
        WWIND_VWIND_ks[i  ,j  ,k ] = 0.

    cuda.syncthreads()


@jit([wp+'[:,:,:], '+wp],target='gpu')
def set_to(FIELD, number):
    i, j, k = cuda.grid(3)
    FIELD[i,j,k] = number 
    cuda.syncthreads()




# SOME SEPARATE FUNCTIONS. NOT USED


#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+\
#      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
#def horizontal_advection_UWIND(dUFLXdt, UWIND, BFLX, CFLX, DFLX, EFLX):
#    nx = dUFLXdt.shape[0] - 2
#    ny = dUFLXdt.shape[1] - 2
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#        dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
#                + BFLX [i-1,j  ,k] * \
#                ( UWIND[i-1,j  ,k] + UWIND[i  ,j  ,k] )/2. \
#                - BFLX [i  ,j  ,k] * \
#                ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. \
#                \
#                + CFLX [i  ,j  ,k] * \
#                ( UWIND[i  ,j-1,k] + UWIND[i  ,j  ,k] )/2. \
#                - CFLX [i  ,j+1,k] * \
#                ( UWIND[i  ,j  ,k] + UWIND[i  ,j+1,k] )/2. \
#                \
#                + DFLX [i-1,j  ,k] * \
#                ( UWIND[i-1,j-1,k] + UWIND[i  ,j  ,k] )/2. \
#                - DFLX [i  ,j+1,k] * \
#                ( UWIND[i  ,j  ,k] + UWIND[i+1,j+1,k] )/2. \
#                \
#                + EFLX [i  ,j  ,k] * \
#                ( UWIND[i+1,j-1,k] + UWIND[i  ,j  ,k] )/2. \
#                - EFLX [i-1,j+1,k] * \
#                ( UWIND[i  ,j  ,k] + UWIND[i-1,j+1,k] )/2. 
#    cuda.syncthreads()
#
#
#
#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+\
#      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
#def horizontal_advection_VWIND(dVFLXdt, VWIND, RFLX, QFLX, SFLX, TFLX):
#    nx = dVFLXdt.shape[0] - 2
#    ny = dVFLXdt.shape[1] - 2
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#        dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
#                  RFLX [i  ,j-1,k] * \
#                ( VWIND[i  ,j-1,k] + VWIND[i  ,j  ,k] )/2. \
#                - RFLX [i  ,j  ,k] * \
#                ( VWIND[i  ,j  ,k] + VWIND[i  ,j+1,k] )/2. \
#                \
#                + QFLX [i  ,j  ,k] * \
#                ( VWIND[i-1,j  ,k] + VWIND[i  ,j  ,k] )/2. \
#                - QFLX [i+1,j  ,k] * \
#                ( VWIND[i  ,j  ,k] + VWIND[i+1,j  ,k] )/2. \
#                \
#                + SFLX [i  ,j-1,k] * \
#                ( VWIND[i-1,j-1,k] + VWIND[i  ,j  ,k] )/2. \
#                - SFLX [i+1,j  ,k] * \
#                ( VWIND[i  ,j  ,k] + VWIND[i+1,j+1,k] )/2. \
#                \
#                + TFLX [i+1,j-1,k] * \
#                ( VWIND[i+1,j-1,k] + VWIND[i  ,j  ,k] )/2. \
#                - TFLX [i  ,j  ,k] * \
#                ( VWIND[i  ,j  ,k] + VWIND[i-1,j+1,k] )/2. 
#    cuda.syncthreads()
#
#
#
#@jit([wp+'[:,:,:], '+wp+'[:,:  ], '+\
#      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
#      wp+'[:    ], '+wp+'[:    ], '+wp], target='gpu')
#def pressure_gradient_UWIND(dUFLXdt, COLP, PHI, POTT, PVTF, PVTFVB,
#                            dsigma, sigma_vb, dy):
#    nx = dUFLXdt.shape[0] - 2
#    ny = dUFLXdt.shape[1] - 2
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#        dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
#             - dy * ( \
#            ( PHI [i  ,j  ,k]  - PHI [i-1,j  ,k] ) * \
#            ( COLP[i  ,j    ]  + COLP[i-1,j    ] )/2. + \
#            ( COLP[i  ,j    ]  - COLP[i-1,j    ] ) * con_cp/2. * \
#            (\
#              + POTT[i-1,j  ,k] / dsigma[k] * \
#                ( \
#                    sigma_vb[k+1] * \
#                    ( PVTFVB[i-1,j  ,k+1] - PVTF  [i-1,j  ,k] ) + \
#                    sigma_vb[k  ] * \
#                    ( PVTF  [i-1,j  ,k  ] - PVTFVB[i-1,j  ,k] )   \
#                ) \
#              + POTT[i  ,j  ,k] / dsigma[k] * \
#                ( \
#                    sigma_vb[k+1] * \
#                    ( PVTFVB[i  ,j  ,k+1] - PVTF  [i  ,j  ,k] ) + \
#                    sigma_vb[k  ] * \
#                    ( PVTF  [i  ,j  ,k  ] - PVTFVB[i  ,j  ,k] )   \
#                ) \
#            ) )
#    cuda.syncthreads()
#
#@jit([wp+'[:,:,:], '+wp+'[:,:  ], '+\
#      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
#      wp+'[:    ], '+wp+'[:    ], '+wp+'[:,:  ]'], target='gpu')
#def pressure_gradient_VWIND(dVFLXdt, COLP, PHI, POTT, PVTF, PVTFVB,
#                            dsigma, sigma_vb, dxjs):
#    nx = dVFLXdt.shape[0] - 2
#    ny = dVFLXdt.shape[1] - 2
#    i, j, k = cuda.grid(3)
#    kp1 = k + 1
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#        dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
#            - dxjs[i  ,j    ] * ( \
#            ( PHI [i  ,j  ,k] - PHI [i  ,j-1,k] ) *\
#            ( COLP[i  ,j    ] + COLP[i  ,j-1  ] )/2. + \
#            ( COLP[i  ,j    ] - COLP[i  ,j-1  ] ) * con_cp/2. * \
#            (\
#                POTT[i  ,j-1,k] / dsigma[k] * \
#                ( \
#                    + sigma_vb[k+1] * \
#                    ( PVTFVB[i  ,j-1,k+1] - PVTF  [i  ,j-1,k] ) \
#                    + sigma_vb[k  ] * \
#                    ( PVTF  [i  ,j-1,k  ] - PVTFVB[i  ,j-1,k] ) \
#                ) +\
#                POTT[i  ,j  ,k] / dsigma[k] * \
#                ( \
#                    + sigma_vb[k+1] * \
#                    ( PVTFVB[i  ,j  ,k+1] - PVTF  [i  ,j  ,k] ) \
#                    + sigma_vb[k  ] * \
#                    ( PVTF  [i  ,j  ,k  ] - PVTFVB[i  ,j  ,k] ) \
#                ) \
#            ) )
#    cuda.syncthreads()
#
#
#
#
#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+ \
#      wp+'[:,:  ], '+wp+'[:,:  ], '+wp], target='gpu')
#def coriolis_UWIND(dUFLXdt, UWIND, VWIND, COLP,
#                    corf_is, latis_rad, dlon_rad):
#    nx = dUFLXdt.shape[0] - 2
#    ny = dUFLXdt.shape[1] - 2
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#        dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
#            con_rE*dlon_rad*dlon_rad/2.*(\
#              COLP [i-1,j    ] * \
#            ( VWIND[i-1,j  ,k] + VWIND[i-1,j+1,k] )/2. * \
#            ( corf_is[i  ,j  ] * con_rE *\
#              cos(latis_rad[i  ,j  ]) + \
#              ( UWIND[i-1,j  ,k] + UWIND[i  ,j  ,k] )/2. * \
#              sin(latis_rad[i  ,j  ]) )\
#            + COLP [i  ,j    ] * \
#            ( VWIND[i  ,j  ,k] + VWIND[i  ,j+1,k] )/2. * \
#            ( corf_is[i  ,j  ] * con_rE * \
#              cos(latis_rad[i  ,j  ]) + \
#              ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
#              sin(latis_rad[i  ,j  ]) )\
#            )
#    cuda.syncthreads()
#
#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+ \
#      wp+'[:,:  ], '+wp+'[:,:  ], '+wp], target='gpu')
#def coriolis_VWIND(dVFLXdt, UWIND, VWIND, COLP,
#                    corf, lat_rad, dlon_rad):
#
#    nx = dVFLXdt.shape[0] - 2
#    ny = dVFLXdt.shape[1] - 2
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#        dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
#             - con_rE*dlon_rad*dlon_rad/2.*(\
#              COLP[i  ,j-1  ] * \
#            ( UWIND[i  ,j-1,k] + UWIND[i+1,j-1,k] )/2. * \
#            ( corf[i  ,j-1  ] * con_rE *\
#              cos(lat_rad[i  ,j-1  ]) +\
#              ( UWIND[i  ,j-1,k] + UWIND[i+1,j-1,k] )/2. * \
#              sin(lat_rad[i  ,j-1  ]) )\
#
#            + COLP [i  ,j    ] * \
#            ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
#            ( corf [i  ,j    ] * con_rE *\
#              cos(lat_rad[i  ,j    ]) +\
#              ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
#              sin(lat_rad[i  ,j    ]) )\
#            )
#    cuda.syncthreads()

