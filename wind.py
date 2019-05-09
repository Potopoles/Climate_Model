import time
import numpy as np
from boundaries import exchange_BC
from constants import con_cp, con_rE, con_Rd
from namelist import WIND_hor_dif_tau, i_wind_tendency
from namelist import wp

i_hor_adv  = 1
i_vert_adv = 1
i_coriolis = 1
i_pre_grad = 1
i_num_dif  = 1



def wind_tendency_jacobson(GR, UWIND, VWIND, WWIND, UFLX, dUFLXdt, VFLX, dVFLXdt,
                            BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                            COLP, COLP_NEW, PHI, POTT, PVTF, PVTFVB):

    dUFLXdt[:] = 0.
    dVFLXdt[:] = 0.

    if i_wind_tendency:

        if i_vert_adv:
            WWIND_UWIND = np.zeros( (GR.nxs+2*GR.nb,GR.ny +2*GR.nb,GR.nzs) , dtype=wp)
            WWIND_VWIND = np.zeros( (GR.nx +2*GR.nb,GR.nys+2*GR.nb,GR.nzs) , dtype=wp)
            for ks in range(1,GR.nzs-1):
                WWIND_UWIND[:,:,ks][GR.iisjj] = vertical_interp_UWIND(GR, COLP_NEW, UWIND,
                                                        WWIND, WWIND_UWIND, ks)
                WWIND_VWIND[:,:,ks][GR.iijjs] = vertical_interp_VWIND(GR, COLP_NEW, VWIND,
                                                    WWIND, WWIND_VWIND, ks)

        for k in range(0,GR.nz):

            # HORIZONTAL ADVECTION
            if i_hor_adv:
                horAdv_UWIND = horizontal_advection_UWIND(GR, UWIND, UFLX, VFLX,
                                                        BFLX, CFLX, DFLX, EFLX, k)
                horAdv_VWIND = horizontal_advection_VWIND(GR, VWIND, UFLX, VFLX,
                                                        RFLX, QFLX, SFLX, TFLX, k)
                dUFLXdt[:,:,k][GR.iisjj] += horAdv_UWIND
                dVFLXdt[:,:,k][GR.iijjs] += horAdv_VWIND

            if i_vert_adv:
                # VERTICAL ADVECTION
                vertAdv_UWIND, vertAdv_VWIND = vertical_advection(GR, WWIND_UWIND, 
                                                                    WWIND_VWIND, k)
                dUFLXdt[:,:,k][GR.iisjj] += vertAdv_UWIND
                dVFLXdt[:,:,k][GR.iijjs] += vertAdv_VWIND

            if i_coriolis:
                # CORIOLIS AND SPHERICAL GRID CONVERSION
                coriolis_UWIND, coriolis_VWIND = coriolis_term(GR, UWIND, VWIND, COLP, k)
                dUFLXdt[:,:,k][GR.iisjj] += coriolis_UWIND
                dVFLXdt[:,:,k][GR.iijjs] += coriolis_VWIND

            if i_pre_grad:
                # PRESSURE GRADIENT
                preGrad_UWIND, preGrad_VWIND = pressure_gradient_term(GR, COLP, PHI, POTT,
                                                                    PVTF, PVTFVB, k)
                dUFLXdt[:,:,k][GR.iisjj] += preGrad_UWIND
                dVFLXdt[:,:,k][GR.iijjs] += preGrad_VWIND

            if i_num_dif and (WIND_hor_dif_tau > 0):
                # HORIZONTAL DIFFUSION
                diff_UWIND, diff_VWIND = horizontal_diffusion(GR, UFLX, VFLX, k)
                dUFLXdt[:,:,k][GR.iisjj] += diff_UWIND
                dVFLXdt[:,:,k][GR.iijjs] += diff_VWIND


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

    vertAdv_UWIND = (WWIND_UWIND[:,:,k  ][GR.iisjj] - WWIND_UWIND[:,:,k+1][GR.iisjj]) / \
                    GR.dsigma[k]
    vertAdv_VWIND = (WWIND_VWIND[:,:,k  ][GR.iijjs] - WWIND_VWIND[:,:,k+1][GR.iijjs]) / \
                    GR.dsigma[k]

    return(vertAdv_UWIND, vertAdv_VWIND)


def horizontal_diffusion(GR, UFLX, VFLX, k):

    diff_UWIND = WIND_hor_dif_tau * \
                 (  UFLX[:,:,k][GR.iisjj_im1] + UFLX[:,:,k][GR.iisjj_ip1] \
                  + UFLX[:,:,k][GR.iisjj_jm1] + UFLX[:,:,k][GR.iisjj_jp1] \
                - 4*UFLX[:,:,k][GR.iisjj])

    diff_VWIND = WIND_hor_dif_tau * \
                 (  VFLX[:,:,k][GR.iijjs_im1] + VFLX[:,:,k][GR.iijjs_ip1] \
                  + VFLX[:,:,k][GR.iijjs_jm1] + VFLX[:,:,k][GR.iijjs_jp1] \
                - 4*VFLX[:,:,k][GR.iijjs])

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


def horizontal_advection_UWIND(GR, UWIND, UFLX, VFLX,
                                    BFLX, CFLX, DFLX, EFLX, k):

    BFLX[:,:,k][GR.iijj]   = 1/12 * (  UFLX[:,:,k][GR.iijj_jm1      ]   + \
                                UFLX[:,:,k][GR.iijj_ip1_jm1  ]   + \
                            2*( UFLX[:,:,k][GR.iijj          ]   + \
                                UFLX[:,:,k][GR.iijj_ip1      ] ) + \
                                UFLX[:,:,k][GR.iijj_jp1      ]   + \
                                UFLX[:,:,k][GR.iijj_ip1_jp1  ]   )
    BFLX = exchange_BC(GR, BFLX)

    CFLX[:,:,k][GR.iisjjs] = 1/12 * (  VFLX[:,:,k][GR.iisjjs_im1_jm1]   + \
                                VFLX[:,:,k][GR.iisjjs_jm1    ]   +\
                            2*( VFLX[:,:,k][GR.iisjjs_im1    ]   + \
                                VFLX[:,:,k][GR.iisjjs        ] ) +\
                                VFLX[:,:,k][GR.iisjjs_im1_jp1]   + \
                                VFLX[:,:,k][GR.iisjjs_jp1    ]   )
    CFLX = exchange_BC(GR, CFLX)

    DFLX[:,:,k][GR.iijjs]  = 1/24 * (  VFLX[:,:,k][GR.iijjs_jm1     ]    + \
                                2*VFLX[:,:,k][GR.iijjs       ]    +\
                                VFLX[:,:,k][GR.iijjs_jp1     ]    + \
                                UFLX[:,:,k][GR.iijjs_jm1     ]    +\
                                UFLX[:,:,k][GR.iijjs         ]    + \
                                UFLX[:,:,k][GR.iijjs_ip1_jm1 ]    +\
                                UFLX[:,:,k][GR.iijjs_ip1     ]    )
    DFLX = exchange_BC(GR, DFLX)

    EFLX[:,:,k][GR.iijjs]  = 1/24 * (  VFLX[:,:,k][GR.iijjs_jm1     ]     + \
                                2*VFLX[:,:,k][GR.iijjs       ]     +\
                                VFLX[:,:,k][GR.iijjs_jp1     ]     - \
                                UFLX[:,:,k][GR.iijjs_jm1     ]     +\
                              - UFLX[:,:,k][GR.iijjs         ]     - \
                                UFLX[:,:,k][GR.iijjs_ip1_jm1 ]     +\
                              - UFLX[:,:,k][GR.iijjs_ip1     ]     )
    EFLX = exchange_BC(GR, EFLX)

    horAdv_UWIND =  + BFLX [:,:,k][GR.iisjj_im1    ] * \
                    ( UWIND[:,:,k][GR.iisjj_im1    ] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - BFLX [:,:,k][GR.iisjj        ] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_ip1    ] )/2 \
                    \
                    + CFLX [:,:,k][GR.iisjj        ] * \
                    ( UWIND[:,:,k][GR.iisjj_jm1    ] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - CFLX [:,:,k][GR.iisjj_jp1    ] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_jp1    ] )/2 \
                    \
                    + DFLX [:,:,k][GR.iisjj_im1    ] * \
                    ( UWIND[:,:,k][GR.iisjj_im1_jm1] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - DFLX [:,:,k][GR.iisjj_jp1    ] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_ip1_jp1] )/2 \
                    \
                    + EFLX [:,:,k][GR.iisjj        ] * \
                    ( UWIND[:,:,k][GR.iisjj_ip1_jm1] + UWIND[:,:,k][GR.iisjj        ] )/2 \
                    - EFLX [:,:,k][GR.iisjj_im1_jp1] * \
                    ( UWIND[:,:,k][GR.iisjj        ] + UWIND[:,:,k][GR.iisjj_im1_jp1] )/2 

    return( horAdv_UWIND )


def horizontal_advection_VWIND(GR, VWIND, UFLX, VFLX,
                                RFLX, QFLX, SFLX, TFLX, k):

    RFLX[:,:,k][GR.iijj] = 1/12 * (  VFLX[:,:,k][GR.iijj_im1    ]   + \
                                 VFLX[:,:,k][GR.iijj_im1_jp1]   +\
                             2*( VFLX[:,:,k][GR.iijj        ]   + \
                                 VFLX[:,:,k][GR.iijj_jp1    ] ) +\
                                 VFLX[:,:,k][GR.iijj_ip1    ]   + \
                                 VFLX[:,:,k][GR.iijj_ip1_jp1]    )
    RFLX = exchange_BC(GR, RFLX)

    QFLX[:,:,k][GR.iisjjs] = 1/12 * (  UFLX[:,:,k][GR.iisjjs_im1_jm1]   + \
                                 UFLX[:,:,k][GR.iisjjs_im1    ]   +\
                             2*( UFLX[:,:,k][GR.iisjjs_jm1    ]   + \
                                 UFLX[:,:,k][GR.iisjjs        ] ) +\
                                 UFLX[:,:,k][GR.iisjjs_ip1_jm1]   + \
                                 UFLX[:,:,k][GR.iisjjs_ip1    ]    )
    QFLX = exchange_BC(GR, QFLX)

    SFLX[:,:,k][GR.iisjj]  = 1/24 * (  VFLX[:,:,k][GR.iisjj_im1     ]   + \
                                 VFLX[:,:,k][GR.iisjj_im1_jp1]   +\
                                 VFLX[:,:,k][GR.iisjj         ]   +   \
                                 VFLX[:,:,k][GR.iisjj_jp1    ]   +\
                                 UFLX[:,:,k][GR.iisjj_im1     ]   + \
                               2*UFLX[:,:,k][GR.iisjj        ]   +\
                                 UFLX[:,:,k][GR.iisjj_ip1     ]    )
    SFLX = exchange_BC(GR, SFLX)

    TFLX[:,:,k][GR.iisjj]  = 1/24 * (  VFLX[:,:,k][GR.iisjj_im1    ]   + \
                                 VFLX[:,:,k][GR.iisjj_im1_jp1]   +\
                                 VFLX[:,:,k][GR.iisjj        ]   + \
                                 VFLX[:,:,k][GR.iisjj_jp1    ]   +\
                               - UFLX[:,:,k][GR.iisjj_im1    ]   - \
                               2*UFLX[:,:,k][GR.iisjj        ]   +\
                               - UFLX[:,:,k][GR.iisjj_ip1    ]     )
    TFLX = exchange_BC(GR, TFLX)

    horAdv_VWIND =  + RFLX [:,:,k][GR.iijjs_jm1    ] * \
                    ( VWIND[:,:,k][GR.iijjs_jm1    ] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - RFLX [:,:,k][GR.iijjs        ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_jp1    ] )/2 \
                    \
                    + QFLX [:,:,k][GR.iijjs        ] * \
                    ( VWIND[:,:,k][GR.iijjs_im1    ] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - QFLX [:,:,k][GR.iijjs_ip1    ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_ip1    ] )/2 \
                    \
                    + SFLX [:,:,k][GR.iijjs_jm1    ] * \
                    ( VWIND[:,:,k][GR.iijjs_im1_jm1] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - SFLX [:,:,k][GR.iijjs_ip1    ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_ip1_jp1] )/2 \
                    \
                    + TFLX [:,:,k][GR.iijjs_ip1_jm1] * \
                    ( VWIND[:,:,k][GR.iijjs_ip1_jm1] + VWIND[:,:,k][GR.iijjs        ] )/2 \
                    - TFLX [:,:,k][GR.iijjs        ] * \
                    ( VWIND[:,:,k][GR.iijjs        ] + VWIND[:,:,k][GR.iijjs_im1_jp1] )/2 

    return( horAdv_VWIND )


