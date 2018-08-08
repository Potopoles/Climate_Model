import copy
import numpy as np
#import time
from boundaries import exchange_BC
from constants import con_cp, con_rE, con_Rd
from namelist import WIND_hor_dif_tau, i_wind_tendency

i_hor_adv  = 1
i_vert_adv = 1
i_coriolis = 1
i_pre_grad = 1
i_num_dif  = 1

def wind_tendency_jacobson(GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
                                COLP, COLP_NEW, HSURF, PHI, POTT, PVTF, PVTFVB):
#def wind_tendency_jacobson(job_ind, output, GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
#                                COLP, COLP_NEW, HSURF, PHI, POTT, PVTF, PVTFVB):

    #t_start = time.time()

    dUFLXdt = np.zeros( (GR.nxs,GR.ny ,GR.nz) )
    dVFLXdt = np.zeros( (GR.nx ,GR.nys,GR.nz) )

    if i_wind_tendency:

        if i_vert_adv:
            WWIND_UWIND = np.zeros( (GR.nxs,GR.ny ,GR.nzs) )
            WWIND_VWIND = np.zeros( (GR.nx ,GR.nys,GR.nzs) )
            for ks in range(1,GR.nzs-1):
                WWIND_UWIND[:,:,ks] = vertical_interp_UWIND(GR, COLP_NEW, UWIND,
                                                        WWIND, WWIND_UWIND, ks)
                WWIND_VWIND[:,:,ks] = vertical_interp_VWIND(GR, COLP_NEW, VWIND,
                                                    WWIND, WWIND_VWIND, ks)

        for k in range(0,GR.nz):

            # HORIZONTAL ADVECTION
            if i_hor_adv:
                horAdv_UWIND = horizontal_advection_UWIND(GR, UWIND, UFLX, VFLX, k)
                horAdv_VWIND = horizontal_advection_VWIND(GR, VWIND, UFLX, VFLX, k)
                dUFLXdt[:,:,k] += horAdv_UWIND
                dVFLXdt[:,:,k] += horAdv_VWIND

            if i_vert_adv:
                # VERTICAL ADVECTION
                vertAdv_UWIND, vertAdv_VWIND = vertical_advection(GR, WWIND_UWIND, 
                                                                    WWIND_VWIND, k)
                dUFLXdt[:,:,k] += vertAdv_UWIND
                dVFLXdt[:,:,k] += vertAdv_VWIND

            if i_coriolis:
                # CORIOLIS AND SPHERICAL GRID CONVERSION
                coriolis_UWIND, coriolis_VWIND = coriolis_term(GR, UWIND, VWIND, COLP, k)
                dUFLXdt[:,:,k] += coriolis_UWIND
                dVFLXdt[:,:,k] += coriolis_VWIND

            if i_pre_grad:
                # PRESSURE GRADIENT
                preGrad_UWIND, preGrad_VWIND = pressure_gradient_term(GR, COLP, PHI, POTT,
                                                                    PVTF, PVTFVB, k)
                dUFLXdt[:,:,k] += preGrad_UWIND
                dVFLXdt[:,:,k] += preGrad_VWIND

            if i_num_dif and (WIND_hor_dif_tau > 0):
                # HORIZONTAL DIFFUSION
                diff_UWIND, diff_VWIND = horizontal_diffusion(GR, UFLX, VFLX, k)
                dUFLXdt[:,:,k] += diff_UWIND
                dVFLXdt[:,:,k] += diff_VWIND

    #t_end = time.time()
    #GR.wind_comp_time += t_end - t_start

    #print('yolo')
    #print(dUFLXdt)

    #out = {}
    #out['dUFLXdt'] = dUFLXdt
    #out['dVFLXdt'] = dVFLXdt
    #output.put( (job_ind, out) )

    return(dUFLXdt, dVFLXdt)


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

def coriolis_term(GR, UWIND, VWIND, COLP, k):

    coriolis_UWIND = con_rE*GR.dlon_rad*GR.dlon_rad/2*(\

              COLP[GR.iisjj_im1] * \
            ( VWIND[:,:,k][GR.iisjj_im1    ] + VWIND[:,:,k][GR.iisjj_im1_jp1] )/2 * \
            ( GR.corf_is[GR.iisjj] * con_rE *\
              np.cos(GR.latis_rad[GR.iisjj]) + \
              ( UWIND[:,:,k][GR.iisjj_im1    ] + UWIND[:,:,k][GR.iisjj        ] )/2 * \
              np.sin(GR.latis_rad[GR.iisjj]) )\

            + COLP[GR.iisjj    ] * \
            ( VWIND[:,:,k][GR.iisjj        ] + VWIND[:,:,k][GR.iisjj_jp1    ] )/2 * \
            ( GR.corf_is[GR.iisjj] * con_rE * \
              np.cos(GR.latis_rad[GR.iisjj]) + \
              ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_ip1    ] )/2 * \
              np.sin(GR.latis_rad[GR.iisjj]) )\
            )

    coriolis_VWIND = - con_rE*GR.dlon_rad*GR.dlon_rad/2*(\

              COLP[GR.iijjs_jm1] * \
            ( UWIND[:,:,k][GR.iijjs_jm1    ] + UWIND[:,:,k][GR.iijjs_ip1_jm1] )/2 * \
            ( GR.corf[GR.iijjs_jm1] * con_rE *\
              np.cos(GR.lat_rad[GR.iijjs_jm1]) +\
              ( UWIND[:,:,k][GR.iijjs_jm1    ] + UWIND[:,:,k][GR.iijjs_ip1_jm1] )/2 * \
              np.sin(GR.lat_rad[GR.iijjs_jm1]) )\

            + COLP[GR.iijjs    ] * \
            ( UWIND[:,:,k][GR.iijjs        ] + UWIND[:,:,k][GR.iijjs_ip1    ] )/2 * \
            ( GR.corf[GR.iijjs    ] * con_rE *\
              np.cos(GR.lat_rad[GR.iijjs    ]) +\
              ( UWIND[:,:,k][GR.iijjs        ] + UWIND[:,:,k][GR.iijjs_ip1    ] )/2 * \
              np.sin(GR.lat_rad[GR.iijjs    ]) )\
            )

    return(coriolis_UWIND, coriolis_VWIND)


def pressure_gradient_term(GR, COLP, PHI, POTT, PVTF, PVTFVB, k):

    kp1 = k + 1

    preGrad_UWIND = - GR.dy * ( \
            ( PHI[:,:,k][GR.iisjj]  - PHI[:,:,k][GR.iisjj_im1] ) * \
            ( COLP      [GR.iisjj]  + COLP      [GR.iisjj_im1] )/2 + \
            ( COLP      [GR.iisjj]  - COLP      [GR.iisjj_im1] ) * con_cp/2 * \
            (\
              + POTT[:,:,k][GR.iisjj_im1] / GR.dsigma[k] * \
                ( \
                    GR.sigma_vb[kp1] * \
                    ( PVTFVB[:,:,kp1][GR.iisjj_im1] - PVTF  [:,:,k  ][GR.iisjj_im1] ) + \
                    GR.sigma_vb[k  ] * \
                    ( PVTF  [:,:,k  ][GR.iisjj_im1] - PVTFVB[:,:,k  ][GR.iisjj_im1] )   \
                ) \
              + POTT[:,:,k][GR.iisjj    ] / GR.dsigma[k] * \
                ( \
                    GR.sigma_vb[kp1] * \
                    ( PVTFVB[:,:,kp1][GR.iisjj    ] - PVTF  [:,:,k  ][GR.iisjj    ] ) + \
                    GR.sigma_vb[k  ] * \
                    ( PVTF  [:,:,k  ][GR.iisjj    ] - PVTFVB[:,:,k  ][GR.iisjj    ] )   \
                ) \
            ) )

    preGrad_VWIND = - GR.dxjs[GR.iijjs] * ( \
            ( PHI[:,:,k][GR.iijjs] - PHI[:,:,k][GR.iijjs_jm1] ) *\
            ( COLP      [GR.iijjs] + COLP      [GR.iijjs_jm1] )/2 + \
            ( COLP      [GR.iijjs] - COLP      [GR.iijjs_jm1] ) * con_cp/2 * \
            (\
                POTT[:,:,k][GR.iijjs_jm1] / GR.dsigma[k] * \
                ( \
                    + GR.sigma_vb[kp1] * \
                    ( PVTFVB[:,:,kp1][GR.iijjs_jm1] - PVTF  [:,:,k  ][GR.iijjs_jm1] ) \
                    + GR.sigma_vb[k  ] * \
                    ( PVTF  [:,:,k  ][GR.iijjs_jm1] - PVTFVB[:,:,k  ][GR.iijjs_jm1] ) \
                ) +\
                POTT[:,:,k][GR.iijjs    ] / GR.dsigma[k] * \
                ( \
                    + GR.sigma_vb[kp1] * \
                    ( PVTFVB[:,:,kp1][GR.iijjs    ] - PVTF  [:,:,k  ][GR.iijjs    ] ) \
                    + GR.sigma_vb[k  ] * \
                    ( PVTF  [:,:,k  ][GR.iijjs    ] - PVTFVB[:,:,k  ][GR.iijjs    ] ) \
                ) \
            ) )


    return(preGrad_UWIND, preGrad_VWIND)


def horizontal_advection_UWIND(GR, UWIND, UFLX, VFLX, k):
    BFLX = np.full( (GR.nx +2*GR.nb,GR.ny +2*GR.nb), np.nan )
    CFLX = np.full( (GR.nxs+2*GR.nb,GR.nys+2*GR.nb), np.nan )
    DFLX = np.full( (GR.nx +2*GR.nb,GR.nys+2*GR.nb), np.nan )
    EFLX = np.full( (GR.nx +2*GR.nb,GR.nys+2*GR.nb), np.nan )

    BFLX[GR.iijj]   = 1/12 * (  UFLX[:,:,k][GR.iijj_jm1      ]   + \
                                UFLX[:,:,k][GR.iijj_ip1_jm1  ]   + \
                            2*( UFLX[:,:,k][GR.iijj          ]   + \
                                UFLX[:,:,k][GR.iijj_ip1      ] ) + \
                                UFLX[:,:,k][GR.iijj_jp1      ]   + \
                                UFLX[:,:,k][GR.iijj_ip1_jp1  ]   )
    BFLX = exchange_BC(GR, BFLX)

    CFLX[GR.iisjjs] = 1/12 * (  VFLX[:,:,k][GR.iisjjs_im1_jm1]   + \
                                VFLX[:,:,k][GR.iisjjs_jm1    ]   +\
                            2*( VFLX[:,:,k][GR.iisjjs_im1    ]   + \
                                VFLX[:,:,k][GR.iisjjs        ] ) +\
                                VFLX[:,:,k][GR.iisjjs_im1_jp1]   + \
                                VFLX[:,:,k][GR.iisjjs_jp1    ]   )
    CFLX = exchange_BC(GR, CFLX)

    DFLX[GR.iijjs]  = 1/24 * (  VFLX[:,:,k][GR.iijjs_jm1     ]    + \
                                2*VFLX[:,:,k][GR.iijjs       ]    +\
                                VFLX[:,:,k][GR.iijjs_jp1     ]    + \
                                UFLX[:,:,k][GR.iijjs_jm1     ]    +\
                                UFLX[:,:,k][GR.iijjs         ]    + \
                                UFLX[:,:,k][GR.iijjs_ip1_jm1 ]    +\
                                UFLX[:,:,k][GR.iijjs_ip1     ]    )
    DFLX = exchange_BC(GR, DFLX)

    EFLX[GR.iijjs]  = 1/24 * (  VFLX[:,:,k][GR.iijjs_jm1     ]     + \
                                2*VFLX[:,:,k][GR.iijjs       ]     +\
                                VFLX[:,:,k][GR.iijjs_jp1     ]     - \
                                UFLX[:,:,k][GR.iijjs_jm1     ]     +\
                              - UFLX[:,:,k][GR.iijjs         ]     - \
                                UFLX[:,:,k][GR.iijjs_ip1_jm1 ]     +\
                              - UFLX[:,:,k][GR.iijjs_ip1     ]     )
    EFLX = exchange_BC(GR, EFLX)

    horAdv_UWIND =  + BFLX [GR.iisjj_im1    ] * \
                    ( UWIND[:,:,k][GR.iisjj_im1    ] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - BFLX [GR.iisjj        ] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_ip1    ] )/2 \
                    \
                    + CFLX [GR.iisjj        ] * \
                    ( UWIND[:,:,k][GR.iisjj_jm1    ] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - CFLX [GR.iisjj_jp1    ] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_jp1    ] )/2 \
                    \
                    + DFLX [GR.iisjj_im1    ] * \
                    ( UWIND[:,:,k][GR.iisjj_im1_jm1] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - DFLX [GR.iisjj_jp1    ] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_ip1_jp1] )/2 \
                    \
                    + EFLX [GR.iisjj        ] * \
                    ( UWIND[:,:,k][GR.iisjj_ip1_jm1] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - EFLX [GR.iisjj_im1_jp1] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_im1_jp1] )/2 

    return( horAdv_UWIND )


def horizontal_advection_VWIND(GR, VWIND, UFLX, VFLX, k):
    RFLX = np.zeros( (GR.nx +2*GR.nb,GR.ny +2*GR.nb) )
    QFLX = np.zeros( (GR.nxs+2*GR.nb,GR.nys+2*GR.nb) )
    SFLX = np.zeros( (GR.nxs+2*GR.nb,GR.ny +2*GR.nb) )
    TFLX = np.zeros( (GR.nxs+2*GR.nb,GR.ny +2*GR.nb) )

    RFLX[GR.iijj] = 1/12 * (    VFLX[:,:,k][GR.iijj_im1    ]   +   VFLX[:,:,k][GR.iijj_im1_jp1]   +\
                            2*( VFLX[:,:,k][GR.iijj        ]   +   VFLX[:,:,k][GR.iijj_jp1    ] ) +\
                                VFLX[:,:,k][GR.iijj_ip1    ]   +   VFLX[:,:,k][GR.iijj_ip1_jp1]    )
    RFLX = exchange_BC(GR, RFLX)

    QFLX[GR.iisjjs] = 1/12 * (  UFLX[:,:,k][GR.iisjjs_im1_jm1]   +   UFLX[:,:,k][GR.iisjjs_im1    ]   +\
                            2*( UFLX[:,:,k][GR.iisjjs_jm1    ]   +   UFLX[:,:,k][GR.iisjjs        ] ) +\
                                UFLX[:,:,k][GR.iisjjs_ip1_jm1]   +   UFLX[:,:,k][GR.iisjjs_ip1    ]    )
    QFLX = exchange_BC(GR, QFLX)

    SFLX[GR.iisjj]  = 1/24 * (  VFLX[:,:,k][GR.iisjj_im1     ]   +   VFLX[:,:,k][GR.iisjj_im1_jp1]   +\
                                VFLX[:,:,k][GR.iisjj         ]   +   VFLX[:,:,k][GR.iisjj_jp1    ]   +\
                                UFLX[:,:,k][GR.iisjj_im1     ]   + 2*UFLX[:,:,k][GR.iisjj        ]   +\
                                UFLX[:,:,k][GR.iisjj_ip1     ]                                  )
    SFLX = exchange_BC(GR, SFLX)

    TFLX[GR.iisjj]  = 1/24 * (  VFLX[:,:,k][GR.iisjj_im1    ]   +   VFLX[:,:,k][GR.iisjj_im1_jp1]   +\
                                VFLX[:,:,k][GR.iisjj        ]   +   VFLX[:,:,k][GR.iisjj_jp1    ]   +\
                              - UFLX[:,:,k][GR.iisjj_im1    ]   - 2*UFLX[:,:,k][GR.iisjj        ]   +\
                              - UFLX[:,:,k][GR.iisjj_ip1    ]                                  )
    TFLX = exchange_BC(GR, TFLX)

    horAdv_VWIND =  + RFLX [GR.iijjs_jm1    ] * \
                    ( VWIND[:,:,k][GR.iijjs_jm1    ] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - RFLX [GR.iijjs        ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_jp1    ] )/2 \
                    \
                    + QFLX [GR.iijjs        ] * \
                    ( VWIND[:,:,k][GR.iijjs_im1    ] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - QFLX        [GR.iijjs_ip1    ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_ip1    ] )/2 \
                    \
                    + SFLX [GR.iijjs_jm1    ] * \
                    ( VWIND[:,:,k][GR.iijjs_im1_jm1] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - SFLX        [GR.iijjs_ip1    ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_ip1_jp1] )/2 \
                    \
                    + TFLX [GR.iijjs_ip1_jm1] * \
                    ( VWIND[:,:,k][GR.iijjs_ip1_jm1] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - TFLX        [GR.iijjs        ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_im1_jp1] )/2 

    return( horAdv_VWIND )





#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################



def masspoint_flux_tendency_upwind(GR, UFLXMP, VFLXMP, COLP,
                            UWIND, VWIND,
                            UUFLX, VUFLX, UVFLX, VVFLX,
                            HSURF):

    UFLXMP[GR.iijj] = COLP[GR.iijj]*(UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2
    VFLXMP[GR.iijj] = COLP[GR.iijj]*(VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2
    UFLXMP = exchange_BC(GR, UFLXMP)
    VFLXMP = exchange_BC(GR, VFLXMP)

    UUFLX[GR.iisjj] = GR.dy * ( \
            np.maximum(UWIND[GR.iisjj],0) * UFLXMP[GR.iisjj_im1] + \
            np.minimum(UWIND[GR.iisjj],0) * UFLXMP[GR.iisjj] )
    VUFLX[GR.iijjs] = GR.dxjs[GR.iijjs] * ( \
            np.maximum(VWIND[GR.iijjs],0) * UFLXMP[GR.iijjs_jm1] + \
            np.minimum(VWIND[GR.iijjs],0) * UFLXMP[GR.iijjs] )

    UVFLX[GR.iisjj] = GR.dy * ( \
            np.maximum(UWIND[GR.iisjj],0) * VFLXMP[GR.iisjj_im1] + \
            np.minimum(UWIND[GR.iisjj],0) * VFLXMP[GR.iisjj] )
    VVFLX[GR.iijjs] = GR.dxjs[GR.iijjs] * ( \
            np.maximum(VWIND[GR.iijjs],0) * VFLXMP[GR.iijjs_jm1] + \
            np.minimum(VWIND[GR.iijjs],0) * VFLXMP[GR.iijjs] )  

    corx = GR.corf[GR.iijj] * COLP[GR.iijj] * (VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2
    cory = GR.corf[GR.iijj] * COLP[GR.iijj] * (UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2

    dUFLXMPdt = - ( UUFLX[GR.iijj_ip1] - UUFLX[GR.iijj] + \
                    VUFLX[GR.iijj_jp1] - VUFLX[GR.iijj]) / GR.A[GR.iijj] + \
            - con_g*COLP[GR.iijj]*( HSURF[GR.iijj_ip1] - HSURF[GR.iijj_im1] + \
                                    COLP[GR.iijj_ip1] - COLP[GR.iijj_im1] ) / \
                                    (2*GR.dx[GR.iijj]) + corx
    dVFLXMPdt = - ( UVFLX[GR.iijj_ip1] - UVFLX[GR.iijj] + \
                    VVFLX[GR.iijj_jp1] - VVFLX[GR.iijj]) / GR.A[GR.iijj] + \
            - con_g*COLP[GR.iijj]*( HSURF[GR.iijj_jp1] - HSURF[GR.iijj_jm1] + \
                                    COLP[GR.iijj_jp1] - COLP[GR.iijj_jm1] ) / \
                                    (2*GR.dy) - cory


    return(dUFLXMPdt, dVFLXMPdt)
