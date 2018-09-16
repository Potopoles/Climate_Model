import numpy as np
import time
from namelist import wp, QV_hor_dif_tau
if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np


i_hor_adv      = 1
i_vert_adv     = 1
i_num_dif      = 1
i_microphysics = 1
i_turb         = 0


def water_vapor_tendency(GR, dQVdt, QV, COLP, COLP_NEW, UFLX, VFLX, WWIND, dQVdt_MIC):

    QVVB = np.zeros( (GR.nx ,GR.ny ,GR.nzs), dtype=wp_np)
    QVVB[:,:,1:(GR.nzs-1)] = (QV[:,:,:-1][GR.iijj] + QV[:,:,1:][GR.iijj])/2

    if (i_turb and TURB.i_turbulence):
        turb_flux_div = TURB.turbulent_flux_divergence(GR, QV)

    for k in range(0,GR.nz):

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            dQVdt[:,:,k][GR.iijj] = (+ UFLX[:,:,k][GR.iijj    ] *\
                                 (QV[:,:,k][GR.iijj_im1] + QV[:,:,k][GR.iijj    ])/2 \
                              - UFLX[:,:,k][GR.iijj_ip1] *\
                                 (QV[:,:,k][GR.iijj    ] + QV[:,:,k][GR.iijj_ip1])/2 \
                              + VFLX[:,:,k][GR.iijj    ] *\
                                 (QV[:,:,k][GR.iijj_jm1] + QV[:,:,k][GR.iijj    ])/2 \
                              - VFLX[:,:,k][GR.iijj_jp1] *\
                                 (QV[:,:,k][GR.iijj    ] + QV[:,:,k][GR.iijj_jp1])/2 \
                             ) / GR.A[GR.iijj]

        # VERTICAL ADVECTION
        if i_vert_adv:
            if k == 0:
                vertAdv_QV = COLP_NEW[GR.iijj] * (\
                        - WWIND[:,:,k+1][GR.iijj] * QVVB[:,:,k+1] \
                                               ) / GR.dsigma[k]
            elif k == GR.nz:
                vertAdv_QV = COLP_NEW[GR.iijj] * (\
                        + WWIND[:,:,k  ][GR.iijj] * QVVB[:,:,k  ] \
                                               ) / GR.dsigma[k]
            else:
                vertAdv_QV = COLP_NEW[GR.iijj] * (\
                        + WWIND[:,:,k  ][GR.iijj] * QVVB[:,:,k  ] \
                        - WWIND[:,:,k+1][GR.iijj] * QVVB[:,:,k+1] \
                                               ) / GR.dsigma[k]

            dQVdt[:,:,k][GR.iijj] = dQVdt[:,:,k][GR.iijj] + vertAdv_QV

        # TURBULENCE
        if (i_turb and TURB.i_turbulence):
            dQVdt[:,:,k][GR.iijj] = dQVdt[:,:,k][GR.iijj] + turb_flux_div[:,:,k] * COLP[GR.iijj]


        # NUMERICAL DIFUSION 
        if i_num_dif and (QV_hor_dif_tau > 0):
            num_diff = QV_hor_dif_tau * \
                         (+ COLP[GR.iijj_im1] * QV[:,:,k][GR.iijj_im1] \
                          + COLP[GR.iijj_ip1] * QV[:,:,k][GR.iijj_ip1] \
                          + COLP[GR.iijj_jm1] * QV[:,:,k][GR.iijj_jm1] \
                          + COLP[GR.iijj_jp1] * QV[:,:,k][GR.iijj_jp1] \
                          - 4*COLP[GR.iijj] * QV[:,:,k][GR.iijj] ) 
            dQVdt[:,:,k][GR.iijj] = dQVdt[:,:,k][GR.iijj] + num_diff

        # MICROPHYSICS
        if i_microphysics:
            dQVdt[:,:,k][GR.iijj] = dQVdt[:,:,k][GR.iijj] + \
                                     dQVdt_MIC[:,:,k] * COLP[GR.iijj]

    return(dQVdt)






def cloud_water_tendency(GR, dQCdt, QC, COLP, COLP_NEW, UFLX, VFLX, WWIND, dQCdt_MIC):

    QCVB = np.zeros( (GR.nx ,GR.ny ,GR.nzs), dtype=wp_np)
    QCVB[:,:,1:(GR.nzs-1)] = (QC[:,:,:-1][GR.iijj] + QC[:,:,1:][GR.iijj])/2


    for k in range(0,GR.nz):

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            dQCdt[:,:,k][GR.iijj] = (+ UFLX[:,:,k][GR.iijj    ] *\
                                 (QC[:,:,k][GR.iijj_im1] + QC[:,:,k][GR.iijj    ])/2 \
                              - UFLX[:,:,k][GR.iijj_ip1] *\
                                 (QC[:,:,k][GR.iijj    ] + QC[:,:,k][GR.iijj_ip1])/2 \
                              + VFLX[:,:,k][GR.iijj    ] *\
                                 (QC[:,:,k][GR.iijj_jm1] + QC[:,:,k][GR.iijj    ])/2 \
                              - VFLX[:,:,k][GR.iijj_jp1] *\
                                 (QC[:,:,k][GR.iijj    ] + QC[:,:,k][GR.iijj_jp1])/2 \
                             ) / GR.A[GR.iijj]

        # VERTICAL ADVECTION
        if i_vert_adv:
            if k == 0:
                vertAdv_QC = COLP_NEW[GR.iijj] * (\
                        - WWIND[:,:,k+1][GR.iijj] * QCVB[:,:,k+1] \
                                               ) / GR.dsigma[k]
            elif k == GR.nz:
                vertAdv_QC = COLP_NEW[GR.iijj] * (\
                        + WWIND[:,:,k  ][GR.iijj] * QCVB[:,:,k  ] \
                                               ) / GR.dsigma[k]
            else:
                vertAdv_QC = COLP_NEW[GR.iijj] * (\
                        + WWIND[:,:,k  ][GR.iijj] * QCVB[:,:,k  ] \
                        - WWIND[:,:,k+1][GR.iijj] * QCVB[:,:,k+1] \
                                               ) / GR.dsigma[k]

            dQCdt[:,:,k][GR.iijj] = dQCdt[:,:,k][GR.iijj] + vertAdv_QC


        # NUMERICAL DIFUSION 
        if i_num_dif and (QV_hor_dif_tau > 0):
            num_diff = QV_hor_dif_tau * \
                         (+ COLP[GR.iijj_im1] * QC[:,:,k][GR.iijj_im1] \
                          + COLP[GR.iijj_ip1] * QC[:,:,k][GR.iijj_ip1] \
                          + COLP[GR.iijj_jm1] * QC[:,:,k][GR.iijj_jm1] \
                          + COLP[GR.iijj_jp1] * QC[:,:,k][GR.iijj_jp1] \
                          - 4*COLP[GR.iijj] * QC[:,:,k][GR.iijj] ) 
            dQCdt[:,:,k][GR.iijj] = dQCdt[:,:,k][GR.iijj] + num_diff

        # MICROPHYSICS
        if i_microphysics:
            dQCdt[:,:,k][GR.iijj] = dQCdt[:,:,k][GR.iijj] +  \
                                    dQCdt_MIC[:,:,k] * COLP[GR.iijj]

    return(dQCdt)



