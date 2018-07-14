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





def temperature_tendency_jacobson(GR, POTT, COLP, UWIND, VWIND, UFLX, VFLX):


    dPOTTdt = ( + UFLX[GR.iijj    ] *\
                     (POTT[GR.iijj_im1] + POTT[GR.iijj    ])/2 \
                  - UFLX[GR.iijj_ip1] *\
                     (POTT[GR.iijj    ] + POTT[GR.iijj_ip1])/2 \
                  + VFLX[GR.iijj    ] *\
                     (POTT[GR.iijj_jm1] + POTT[GR.iijj    ])/2 \
                  - VFLX[GR.iijj_jp1] *\
                     (POTT[GR.iijj    ] + POTT[GR.iijj_jp1])/2 \
                ) / (GR.A[GR.iijj] * COLP[GR.iijj])

    if POTT_hor_dif_tau > 0:
        num_diff = POTT_hor_dif_tau * \
                     (  POTT[GR.iijj_im1] - 2*POTT[GR.iijj] + POTT[GR.iijj_ip1] \
                      + POTT[GR.iijj_jm1] - 2*POTT[GR.iijj] + POTT[GR.iijj_jp1] )
        dPOTTdt = dPOTTdt + num_diff

    if i_pseudo_radiation:
        radiation = - outRate*POTT[GR.iijj]**4 + inpRate*np.cos(GR.lat_rad[GR.iijj])
        dPOTTdt = dPOTTdt + radiation


    if i_temperature_tendency == 0:
        dPOTTdt[:] = 0

    return(dPOTTdt)

