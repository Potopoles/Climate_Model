import numpy as np

def colp_tendency_upstream(GR, COLP, UWIND, VWIND, UFLX, VFLX):
    UFLX[GR.iisjj] = \
            GR.dy * ( np.maximum(UWIND[GR.iisjj],0) * COLP[GR.iisjj_im1] + \
                        np.minimum(UWIND[GR.iisjj],0) * COLP[GR.iisjj] )

    VFLX[GR.iijjs] = \
            GR.dxjs[GR.iijjs] * ( np.maximum(VWIND[GR.iijjs],0) * COLP[GR.iijjs_jm1] + \
                                    np.minimum(VWIND[GR.iijjs],0) * COLP[GR.iijjs] )

    dCOLPdt = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) / GR.A[GR.iijj]

    return(dCOLPdt)
