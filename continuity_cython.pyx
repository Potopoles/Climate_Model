import numpy as np
from namelist import  i_colp_tendency, COLP_hor_dif_tau
from boundaries import exchange_BC


cpdef colp_tendency_jacobson(GR, COLP, UWIND, VWIND, UFLX, VFLX):


    for k in range(0,GR.nz):
        UFLX[:,:,k][GR.iisjj] = \
                (COLP[GR.iisjj_im1] + COLP[GR.iisjj])/2 *\
                UWIND[:,:,k][GR.iisjj] * GR.dy
        VFLX[:,:,k][GR.iijjs] = \
                (COLP[GR.iijjs_jm1] + COLP[GR.iijjs])/2 *\
                VWIND[:,:,k][GR.iijjs] * GR.dxjs[GR.iijjs]

    # TODO 1 NECESSARY
    UFLX = exchange_BC(GR, UFLX)
    VFLX = exchange_BC(GR, VFLX)

    FLXDIV =  np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb,GR.nz), np.nan)
    for k in range(0,GR.nz):
        if k == 0:
            FLXDIV[:,:,k][GR.iijj] = \
                    ( + UFLX[:,:,k][GR.iijj_ip1] - UFLX[:,:,k][GR.iijj] \
                      + VFLX[:,:,k][GR.iijj_jp1] - VFLX[:,:,k][GR.iijj] ) \
                      * GR.dsigma[k]
        else:
            FLXDIV[:,:,k][GR.iijj] = FLXDIV[:,:,k-1][GR.iijj] + \
                    ( + UFLX[:,:,k][GR.iijj_ip1] - UFLX[:,:,k][GR.iijj] \
                      + VFLX[:,:,k][GR.iijj_jp1] - VFLX[:,:,k][GR.iijj] ) \
                      * GR.dsigma[k]

        FLXDIV[:,:,k][GR.iijj] = FLXDIV[:,:,k][GR.iijj] / GR.A[GR.iijj]

    if i_colp_tendency:
        dCOLPdt = - np.sum(FLXDIV[GR.iijj], axis=2)

        if COLP_hor_dif_tau > 0:
            num_diff = COLP_hor_dif_tau * \
                         ( +   COLP[GR.iijj_im1] \
                           +   COLP[GR.iijj_ip1] \
                           +   COLP[GR.iijj_jm1] \
                           +   COLP[GR.iijj_jp1] \
                           - 2*COLP[GR.iijj    ] )
            dCOLPdt = dCOLPdt + num_diff
    else:
        dCOLPdt =  np.zeros( (GR.nx,GR.ny) )


    return(dCOLPdt, UFLX, VFLX, FLXDIV)



def vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND):


    for ks in range(1,GR.nzs-1):

        WWIND[:,:,ks][GR.iijj] = - np.sum(FLXDIV[:,:,:ks][GR.iijj], axis=2) / \
                                    COLP_NEW[GR.iijj] \
                                 - GR.sigma_vb[ks] * dCOLPdt / \
                                    COLP_NEW[GR.iijj]

    return(WWIND)

