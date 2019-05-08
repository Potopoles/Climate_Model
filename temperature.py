import numpy as np
import time
from namelist import POTT_dif_coef, i_temperature_tendency, \
                    i_radiation, i_microphysics
from namelist import wp
if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np


i_vert_adv  = 1
i_hor_adv   = 1
i_num_dif   = 1


def temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW, \
                                    UFLX, VFLX, WWIND, dPOTTdt_RAD, dPOTTdt_MIC):

    dPOTTdt = np.zeros( (GR.nx+2*GR.nb ,GR.ny+2*GR.nb ,GR.nz) , dtype=wp_np)

    if i_temperature_tendency:

        for k in range(0,GR.nz):

            # HORIZONTAL ADVECTION
            if i_hor_adv:
                horAdv = (+ UFLX[:,:,k][GR.iijj    ] *\
                             (POTT[:,:,k][GR.iijj_im1] + POTT[:,:,k][GR.iijj    ])/2 \
                          - UFLX[:,:,k][GR.iijj_ip1] *\
                             (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_ip1])/2 \
                          + VFLX[:,:,k][GR.iijj    ] *\
                             (POTT[:,:,k][GR.iijj_jm1] + POTT[:,:,k][GR.iijj    ])/2 \
                          - VFLX[:,:,k][GR.iijj_jp1] *\
                             (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_jp1])/2 \
                         ) / GR.A[GR.iijj]
                dPOTTdt[:,:,k][GR.iijj] = dPOTTdt[:,:,k][GR.iijj] + horAdv

            # VERTICAL ADVECTION
            if i_vert_adv:
                if k == 0:
                    vertAdv = COLP_NEW[GR.iijj] * (\
                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
                                                   ) / GR.dsigma[k]
                elif k == GR.nz:
                    vertAdv = COLP_NEW[GR.iijj] * (\
                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
                                                   ) / GR.dsigma[k]
                else:
                    vertAdv = COLP_NEW[GR.iijj] * (\
                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
                                                   ) / GR.dsigma[k]

                dPOTTdt[:,:,k][GR.iijj] = dPOTTdt[:,:,k][GR.iijj] + vertAdv


            # NUMERICAL DIFUSION 
            if i_num_dif and (POTT_dif_coef > 0):
                num_dif = POTT_dif_coef * np.exp(-(GR.nz-k-1)) *\
                            ( + COLP[GR.iijj_im1] * POTT[:,:,k][GR.iijj_im1] \
                              + COLP[GR.iijj_ip1] * POTT[:,:,k][GR.iijj_ip1] \
                              + COLP[GR.iijj_jm1] * POTT[:,:,k][GR.iijj_jm1] \
                              + COLP[GR.iijj_jp1] * POTT[:,:,k][GR.iijj_jp1] \
                            - 4*COLP[GR.iijj    ] * POTT[:,:,k][GR.iijj    ] )
                dPOTTdt[:,:,k][GR.iijj] = dPOTTdt[:,:,k][GR.iijj] + num_dif

            # RADIATION 
            if i_radiation:
                dPOTTdt[:,:,k][GR.iijj] = dPOTTdt[:,:,k][GR.iijj] + \
                                    dPOTTdt_RAD[:,:,k]*COLP[GR.iijj]
            # MICROPHYSICS
            if i_microphysics:
                dPOTTdt[:,:,k][GR.iijj] = dPOTTdt[:,:,k][GR.iijj] + \
                                    dPOTTdt_MIC[:,:,k]*COLP[GR.iijj]

    return(dPOTTdt)



