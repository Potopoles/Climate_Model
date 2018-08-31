import numpy as np
import time
from namelist import POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics


i_vert_adv  = 0
i_hor_adv   = 1
i_num_dif   = 0


def temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW, \
                                    UFLX, VFLX, WWIND, dPOTTdt_RAD, dPOTTdt_MIC):

    #t_start = time.time()

    dPOTTdt = np.zeros( (GR.nx ,GR.ny ,GR.nz) )

    if i_temperature_tendency:

        for k in range(0,GR.nz):

            # HORIZONTAL ADVECTION
            if i_hor_adv:
                dPOTTdt[:,:,k] = (+ UFLX[:,:,k][GR.iijj    ] *\
                                     (POTT[:,:,k][GR.iijj_im1] + POTT[:,:,k][GR.iijj    ])/2 \
                                  - UFLX[:,:,k][GR.iijj_ip1] *\
                                     (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_ip1])/2 \
                                  + VFLX[:,:,k][GR.iijj    ] *\
                                     (POTT[:,:,k][GR.iijj_jm1] + POTT[:,:,k][GR.iijj    ])/2 \
                                  - VFLX[:,:,k][GR.iijj_jp1] *\
                                     (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_jp1])/2 \
                                 ) / GR.A[GR.iijj]

            # VERTICAL ADVECTION
            if i_vert_adv:
                if k == 0:
                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
                                                   ) / GR.dsigma[k]
                elif k == GR.nz:
                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
                                                   ) / GR.dsigma[k]
                else:
                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
                                                   ) / GR.dsigma[k]

                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + vertAdv_POTT


            # NUMERICAL DIFUSION 
            if i_num_dif and (POTT_hor_dif_tau > 0):
                num_diff = POTT_hor_dif_tau * \
                             (+ COLP[GR.iijj_im1] * POTT[:,:,k][GR.iijj_im1] \
                              + COLP[GR.iijj_ip1] * POTT[:,:,k][GR.iijj_ip1] \
                              + COLP[GR.iijj_jm1] * POTT[:,:,k][GR.iijj_jm1] \
                              + COLP[GR.iijj_jp1] * POTT[:,:,k][GR.iijj_jp1] \
                              - 4*COLP[GR.iijj] * POTT[:,:,k][GR.iijj] )
                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + num_diff

            # RADIATION 
            if i_radiation:
                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + \
                                    dPOTTdt_RAD[:,:,k]*COLP[GR.iijj]
            if i_microphysics:
                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + \
                                    dPOTTdt_MIC[:,:,k]*COLP[GR.iijj]

    #t_end = time.time()
    #GR.temp_comp_time += t_end - t_start


    return(dPOTTdt)



#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################



def temperature_tendency_upwind(GR, POTT, COLP, UWIND, VWIND, UFLX, VFLX):
    UFLX[GR.iisjj] = \
            GR.dy * (np.maximum(UWIND[GR.iisjj],0) * POTT[GR.iisjj_im1] * \
                    COLP[GR.iisjj_im1] + \
                    np.minimum(UWIND[GR.iisjj],0) * POTT[GR.iisjj] * \
                    COLP[GR.iisjj])

    VFLX[GR.iijjs] = \
            GR.dxjs[GR.iijjs] * ( np.maximum(VWIND[GR.iijjs],0) * POTT[GR.iijjs_jm1] * \
                                    COLP[GR.iijjs_jm1] + \
                                    np.minimum(VWIND[GR.iijjs],0) * POTT[GR.iijjs] * \
                                    COLP[GR.iijjs] )

    dPOTTdt = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - \
                    (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) / \
            (GR.A[GR.iijj] * COLP[GR.iijj]) + \
            np.maximum(0., 0.00001*WIND[GR.iijj]*(5.0 - POTT[GR.iijj]))

    return(dPOTTdt)
