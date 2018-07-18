import numpy as np
from namelist import i_pseudo_radiation, outRate, inpRate
from namelist import POTT_hor_dif_tau, i_temperature_tendency

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





def temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW, UWIND, VWIND, \
                                    UFLX, VFLX, WWIND):

    dPOTTdt = np.full( (GR.nx ,GR.ny ,GR.nz), np.nan)

    i_vertAdv = 1

    if i_temperature_tendency:

        for k in range(0,GR.nz):
            dPOTTdt[:,:,k] = (+ UFLX[:,:,k][GR.iijj    ] *\
                                 (POTT[:,:,k][GR.iijj_im1] + POTT[:,:,k][GR.iijj    ])/2 \
                              - UFLX[:,:,k][GR.iijj_ip1] *\
                                 (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_ip1])/2 \
                              + VFLX[:,:,k][GR.iijj    ] *\
                                 (POTT[:,:,k][GR.iijj_jm1] + POTT[:,:,k][GR.iijj    ])/2 \
                              - VFLX[:,:,k][GR.iijj_jp1] *\
                                 (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_jp1])/2 \
                             ) / GR.A[GR.iijj]

            #print(np.max(dPOTTdt[:,:,k]))

            # VERTICAL ADVECTION
            if i_vertAdv:
                #print(k)
                if k == 0:
                    #print('yolo')
                    #print(POTTVB[:,:,k+1][GR.iijj])
                    #quit()
                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
                                                   ) / GR.dsigma[k]
                elif k == GR.nz:
                    #print(WWIND[:,:,k][GR.iijj])
                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
                                                   ) / GR.dsigma[k]
                else:
                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
                                                   ) / GR.dsigma[k]

                #print(np.mean(WWIND[:,:,k][GR.iijj]))
                #print('yolo')
                #print(np.nanmax(vertAdv_POTT))
                #quit()

                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + vertAdv_POTT

            if POTT_hor_dif_tau > 0:
                #num_diff = COLP[GR.iijj] * POTT_hor_dif_tau * \
                #             (  POTT[:,:,k][GR.iijj_im1] - 2*POTT[:,:,k][GR.iijj] + \
                #                POTT[:,:,k][GR.iijj_ip1] \
                #              + POTT[:,:,k][GR.iijj_jm1] - 2*POTT[:,:,k][GR.iijj] + \
                #                POTT[:,:,k][GR.iijj_jp1] )
                num_diff = POTT_hor_dif_tau * \
                             (+ COLP[GR.iijj_im1] * POTT[:,:,k][GR.iijj_im1] \
                              + COLP[GR.iijj_ip1] * POTT[:,:,k][GR.iijj_ip1] \
                              + COLP[GR.iijj_jm1] * POTT[:,:,k][GR.iijj_jm1] \
                              + COLP[GR.iijj_jp1] * POTT[:,:,k][GR.iijj_jp1] \
                              - 4*COLP[GR.iijj] * POTT[:,:,k][GR.iijj] )
                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + num_diff

            if i_pseudo_radiation:
                radiation = - COLP[GR.iijj]*outRate*POTT[:,:,k][GR.iijj]**1 + \
                                COLP[GR.iijj]*inpRate*np.cos(GR.lat_rad[GR.iijj])
                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + radiation


    else:
        dPOTTdt[:] = 0

    return(dPOTTdt)

