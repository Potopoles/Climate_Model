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



def diag_pvt_factor(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan )

    for ks in range(0,GR.nzs):
        PAIRVB[:,:,ks][GR.iijj] = pTop + GR.sigma_vb[ks] * COLP[GR.iijj]

    PVTFVB = np.power(PAIRVB/100000, con_kappa)
    for k in range(0,GR.nz):
        kp1 = k + 1
        PVTF[:,:,k][GR.iijj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.iijj] * PAIRVB[:,:,kp1][GR.iijj] - \
                      PVTFVB[:,:,k  ][GR.iijj] * PAIRVB[:,:,k  ][GR.iijj] ) / \
                    ( PAIRVB[:,:,kp1][GR.iijj] - PAIRVB[:,:,k  ][GR.iijj] )

    PVTF = exchange_BC(GR, PVTF)
    PVTFVB = exchange_BC(GR, PVTFVB)
    #print(np.mean(np.mean(PVTFVB[GR.iijj], axis=0),axis=0))
    #print(np.mean(np.mean(PVTF[GR.iijj], axis=0),axis=0))
    #quit()

    return(PVTF, PVTFVB)



def diag_geopotential_jacobson(GR, PHI, HSURF, POTT, COLP,
                                PVTF, PVTFVB):

    PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)

    phi_vb = HSURF[GR.iijj]*con_g
    PHI[:,:,GR.nz-1][GR.iijj] = phi_vb - con_cp*  \
                                ( POTT[:,:,GR.nz-1][GR.iijj] * \
                                    (   PVTF  [:,:,GR.nz-1 ][GR.iijj]  \
                                      - PVTFVB[:,:,GR.nzs-1][GR.iijj]  ) )
    for k in range(GR.nz-2,-1,-1):
        kp1 = k + 1

        dphi = con_cp * POTT[:,:,kp1][GR.iijj] * \
                        (PVTFVB[:,:,kp1][GR.iijj] - PVTF[:,:,kp1][GR.iijj])
        phi_vb = PHI[:,:,kp1][GR.iijj] - dphi

        # phi_k
        dphi = con_cp * POTT[:,:,k][GR.iijj] * \
                            (PVTF[:,:,k][GR.iijj] - PVTFVB[:,:,kp1][GR.iijj])
        PHI[:,:,k][GR.iijj] = phi_vb - dphi

    #print(np.mean(np.mean(PHI[:,:,1][GR.iijj], axis=0),axis=0))

    #phi_vb = HSURF[GR.iijj]*con_g
    #for k in range(GR.nz-1,-1,-1):
    #    kp1 = k + 1

    #    # phi_k
    #    dphi = - con_cp * POTT[:,:,k][GR.iijj] * \
    #                        (PVTF[:,:,k][GR.iijj] - PVTFVB[:,:,kp1][GR.iijj])
    #    PHI[:,:,k][GR.iijj] = phi_vb + dphi

    #    if k > 0:
    #        # phi_km1/2
    #        dphi = - con_cp * POTT[:,:,k][GR.iijj] * \
    #                        (PVTFVB[:,:,k][GR.iijj] - PVTF[:,:,k][GR.iijj])
    #        phi_vb = PHI[:,:,k][GR.iijj] + dphi
    
    PHI = exchange_BC(GR, PHI)

    return(PHI, PVTF, PVTFVB)
