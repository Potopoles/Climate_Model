import numpy as np
from constants import con_g, con_Rd, con_kappa, con_cp
from namelist import pTop
from boundaries import exchange_BC


def diag_geopotential_upwind(GR, PHI, HSURF, POTT, COLP):

    dsig = 1
    sigma = 0.5
    PHI[GR.iijj] = HSURF[GR.iijj]*con_g + con_Rd*POTT[GR.iijj]*dsig*COLP[GR.iijj] /\
                    (COLP[GR.iijj]*sigma + pTop)
    # TODO DEBUG
    PHI[GR.iijj] = 1

    PHI = exchange_BC(GR, PHI)

    return(PHI)






def diag_geopotential_jacobson(GR, PHI, HSURF, POTT, COLP,
                                PVTF, PVTFVB):

    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, 2), np.nan )

    for ks in range(0,2):
        PAIRVB[:,:,ks][GR.iijj] = pTop + GR.sigma_vb[ks] * COLP[GR.iijj]

    PVTFVB = np.power(PAIRVB/100000, con_kappa)
    for k in range(0,1):
        kp1 = k + 1
        PVTF[:,:,k][GR.iijj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.iijj] * PAIRVB[:,:,kp1][GR.iijj] - \
                      PVTFVB[:,:,k  ][GR.iijj] * PAIRVB[:,:,k  ][GR.iijj] ) / \
                    ( PAIRVB[:,:,kp1][GR.iijj] - PAIRVB[:,:,k  ][GR.iijj] )

    phi_vb = HSURF[GR.iijj]*con_g
    nz = 1
    for k in range(nz-1,-1,-1):
        kp1 = k + 1
        
        # phi_k
        dphi = - con_cp * POTT[GR.iijj] * (PVTF[:,:,k][GR.iijj] - PVTFVB[:,:,kp1][GR.iijj])
        PHI[:,:,k][GR.iijj] = phi_vb + dphi

        if k > 0:
            # phi_km1/2
            dphi = - con_cp * POTT[GR.iijj] * (PVTFVB[:,:,k][GR.iijj] - PVTF[:,:,k][GR.iijj])
            phi_vb = PHI[:,:,k][GR.iijj] + dphi
    
    PHI = exchange_BC(GR, PHI)
    PVTF = exchange_BC(GR, PVTF)
    PVTFVB = exchange_BC(GR, PVTFVB)

    ## DEBUG TODO
    #PHI[:] = 50000

    return(PHI, PVTF, PVTFVB)
