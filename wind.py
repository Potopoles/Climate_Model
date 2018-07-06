import numpy as np
from boundaries import exchange_BC
from constants import con_Rd


def masspoint_flux_tendency_upstream(GR, UFLXMP, VFLXMP, COLP, PHI,
                            UWIND, VWIND,
                            UUFLX, VUFLX, UVFLX, VVFLX,
                            TAIR):

    UFLXMP[GR.iijj] = COLP[GR.iijj]*(UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2
    VFLXMP[GR.iijj] = COLP[GR.iijj]*(VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2
    UFLXMP = exchange_BC(GR, UFLXMP)
    VFLXMP = exchange_BC(GR, VFLXMP)

    UUFLX[GR.iisjj] = GR.dy * ( np.maximum(UWIND[GR.iisjj],0) * UFLXMP[GR.iisjj_im1] + \
                                np.minimum(UWIND[GR.iisjj],0) * UFLXMP[GR.iisjj] )
    VUFLX[GR.iijjs] = GR.dxjs[GR.iijjs] * ( np.maximum(VWIND[GR.iijjs],0) * UFLXMP[GR.iijjs_jm1] + \
                                            np.minimum(VWIND[GR.iijjs],0) * UFLXMP[GR.iijjs] )

    UVFLX[GR.iisjj] = GR.dy * ( np.maximum(UWIND[GR.iisjj],0) * VFLXMP[GR.iisjj_im1] + \
                                np.minimum(UWIND[GR.iisjj],0) * VFLXMP[GR.iisjj] )
    VVFLX[GR.iijjs] = GR.dxjs[GR.iijjs] * ( np.maximum(VWIND[GR.iijjs],0) * VFLXMP[GR.iijjs_jm1] + \
                                            np.minimum(VWIND[GR.iijjs],0) * VFLXMP[GR.iijjs] )  

    corx = GR.corf[GR.iijj] * COLP[GR.iijj] * (VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2
    cory = GR.corf[GR.iijj] * COLP[GR.iijj] * (UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2

    dUFLXMPdt = - ( UUFLX[GR.iijj_ip1] - UUFLX[GR.iijj] + VUFLX[GR.iijj_jp1] - VUFLX[GR.iijj]) / GR.A[GR.iijj] + \
            - con_Rd*TAIR[GR.iijj]*( COLP[GR.iijj_ip1] - COLP[GR.iijj_im1] ) / (2*GR.dx[GR.iijj]) + corx \
            - COLP[GR.iijj]*( PHI[GR.iijj_ip1] - PHI[GR.iijj_im1] ) / (2*GR.dx[GR.iijj]) 
    dVFLXMPdt = - ( UVFLX[GR.iijj_ip1] - UVFLX[GR.iijj] + VVFLX[GR.iijj_jp1] - VVFLX[GR.iijj]) / GR.A[GR.iijj] + \
            - con_Rd*TAIR[GR.iijj]*( COLP[GR.iijj_jp1] - COLP[GR.iijj_jm1] ) / (2*GR.dy) - cory \
            - COLP[GR.iijj]*( PHI[GR.iijj_jp1] - PHI[GR.iijj_jm1] ) / (2*GR.dy)

    return(dUFLXMPdt, dVFLXMPdt)
