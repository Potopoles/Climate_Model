import numpy as np
from constants import con_Rd, con_cp
from namelist import inpRate, outRate, i_pseudo_radiation

def temperature_tendency_upstream(GR, TAIR, COLP, UWIND, VWIND, UFLX, VFLX, dCOLPdt):

    # ADVECTION
    UFLX[GR.iisjj] = \
            np.maximum(UWIND[GR.iisjj],0) * TAIR[GR.iisjj_im1] * COLP[GR.iisjj_im1] * GR.dy + \
            np.minimum(UWIND[GR.iisjj],0) * TAIR[GR.iisjj] * COLP[GR.iisjj] * GR.dy

    VFLX[GR.iijjs] = \
            np.maximum(VWIND[GR.iijjs],0) * TAIR[GR.iijjs_jm1] * COLP[GR.iijjs_jm1] * GR.dx[GR.iijjs_jm1] + \
            np.minimum(VWIND[GR.iijjs],0) * TAIR[GR.iijjs] * COLP[GR.iijjs] * GR.dx[GR.iijjs]

    dTAIRdt = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) / \
            (GR.A[GR.iijj] * COLP[GR.iijj])

    # COMPRESSION
    ## colp flux
    #UFLX[GR.iisjj] = \
    #        np.maximum(UWIND[GR.iisjj],0) * COLP[GR.iisjj_im1] * GR.dy + \
    #        np.minimum(UWIND[GR.iisjj],0) * COLP[GR.iisjj] * GR.dy
    #VFLX[GR.iijjs] = \
    #        np.maximum(VWIND[GR.iijjs],0) * COLP[GR.iijjs_jm1] * GR.dx[GR.iijjs_jm1] + \
    #        np.minimum(VWIND[GR.iijjs],0) * COLP[GR.iijjs] * GR.dx[GR.iijjs]

    #colp_adv = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) / GR.A[GR.iijj]
    
    u = (UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2
    v = (VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2

    sigma_k = 0.5
    omega = sigma_k*( dCOLPdt + \
        u*(COLP[GR.iijj_ip1] - COLP[GR.iijj_im1])/(2*GR.dx[GR.iijj]) + \
        v*(COLP[GR.iijj_jp1] - COLP[GR.iijj_jm1])/(2*GR.dy)  )
    #omega = sigma_k*( dCOLPdt + colp_adv )
    compression = con_Rd*TAIR[GR.iijj]/(con_cp*sigma_k*COLP[GR.iijj])*omega

    if i_pseudo_radiation:
        radiation = - outRate*TAIR[GR.iijj] + inpRate*np.cos(GR.lat_rad[GR.iijj])

    if i_pseudo_radiation:
        dTAIRdt = dTAIRdt + compression + radiation
    else:
        dTAIRdt = dTAIRdt + compression

    return(dTAIRdt)

