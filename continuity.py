import numpy as np
from namelist import inpRate, outRate, i_pseudo_radiation, i_colp_tendency
from boundaries import exchange_BC


def colp_tendency_upwind(GR, COLP, UWIND, VWIND, UFLX, VFLX):
    UFLX[GR.iisjj] = \
            GR.dy * (np.maximum(UWIND[GR.iisjj],0) * COLP[GR.iisjj_im1] + \
                        np.minimum(UWIND[GR.iisjj],0) * COLP[GR.iisjj])
    UFLX = exchange_BC(GR, UFLX)

    VFLX[GR.iijjs] = \
            GR.dxjs[GR.iijjs] * ( np.maximum(VWIND[GR.iijjs],0) * COLP[GR.iijjs_jm1] + \
                                    np.minimum(VWIND[GR.iijjs],0) * COLP[GR.iijjs] )
    VFLX = exchange_BC(GR, VFLX)


    fluxdiv = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - \
                    (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) \
                / GR.A[GR.iijj]

    dCOLPdt = fluxdiv

    return(dCOLPdt)




def colp_tendency_jacobson(GR, COLP, UWIND, VWIND, UFLX, VFLX):

    UFLX[GR.iisjj] = \
            (COLP[GR.iisjj_im1] + COLP[GR.iisjj])/2 * UWIND[GR.iisjj] * GR.dy
    UFLX = exchange_BC(GR, UFLX)

    VFLX[GR.iijjs] = \
            (COLP[GR.iijjs_jm1] + COLP[GR.iijjs])/2 * VWIND[GR.iijjs] * GR.dxjs[GR.iijjs]
    VFLX = exchange_BC(GR, VFLX)

    fluxdiv = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - \
                    (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) \
                / GR.A[GR.iijj]

    dCOLPdt = fluxdiv

    if i_colp_tendency == 0:
        dCOLPdt[:] = 0

    return(dCOLPdt, UFLX, VFLX)

