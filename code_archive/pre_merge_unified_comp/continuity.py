import numpy as np
from namelist import  i_colp_tendency, COLP_dif_coef
from boundaries import exchange_BC



def colp_tendency_jacobson(GR, COLP, UWIND, VWIND, \
                            dCOLPdt, UFLX, VFLX, FLXDIV):

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

    for k in range(0,GR.nz):
        FLXDIV[:,:,k][GR.iijj] = \
                ( + UFLX[:,:,k][GR.iijj_ip1] - UFLX[:,:,k][GR.iijj] \
                  + VFLX[:,:,k][GR.iijj_jp1] - VFLX[:,:,k][GR.iijj] ) \
                  * GR.dsigma[k] / GR.A[GR.iijj]


    if i_colp_tendency:
        dCOLPdt[GR.iijj] = - np.sum(FLXDIV[GR.iijj], axis=2)

        if COLP_dif_coef > 0:
            num_diff = COLP_dif_coef * \
                         ( +   COLP[GR.iijj_im1] \
                           +   COLP[GR.iijj_ip1] \
                           +   COLP[GR.iijj_jm1] \
                           +   COLP[GR.iijj_jp1] \
                           - 2*COLP[GR.iijj    ] )
            dCOLPdt[GR.iijj] = dCOLPdt[GR.iijj] + num_diff


    return(dCOLPdt, UFLX, VFLX, FLXDIV)


def vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND):


    for ks in range(1,GR.nzs-1):

        WWIND[:,:,ks][GR.iijj] = - np.sum(FLXDIV[:,:,:ks][GR.iijj], axis=2) / \
                                    COLP_NEW[GR.iijj] \
                                 - GR.sigma_vb[ks] * dCOLPdt[GR.iijj] / \
                                    COLP_NEW[GR.iijj]

    return(WWIND)


