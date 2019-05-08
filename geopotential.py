import numpy as np
from constants import con_g, con_Rd, con_kappa, con_cp
from namelist import wp, pTop
from boundaries import exchange_BC



def diag_pvt_factor(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan, dtype=wp)

    # TODO: WHY IS PAIRVB NOT FILLED AT UPPERMOST AND LOWER MOST HALFLEVEL??? 
    for ks in range(0,GR.nzs):
        PAIRVB[:,:,ks][GR.iijj] = pTop + GR.sigma_vb[ks] * COLP[GR.iijj]
    
    PVTFVB[:] = np.power(PAIRVB/100000, con_kappa)

    for k in range(0,GR.nz):
        kp1 = k + 1
        PVTF[:,:,k][GR.iijj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.iijj] * PAIRVB[:,:,kp1][GR.iijj] - \
                      PVTFVB[:,:,k  ][GR.iijj] * PAIRVB[:,:,k  ][GR.iijj] ) / \
                    ( PAIRVB[:,:,kp1][GR.iijj] - PAIRVB[:,:,k  ][GR.iijj] )

    return(PVTF, PVTFVB)



def diag_geopotential_jacobson(GR, PHI, PHIVB, HSURF, POTT, COLP,
                               PVTF, PVTFVB):

    PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)

    #phi_vb = HSURF[GR.iijj]*con_g
    PHIVB[:,:,GR.nzs-1][GR.iijj] = HSURF[GR.iijj]*con_g
    PHI[:,:,GR.nz-1][GR.iijj] = PHIVB[:,:,GR.nzs-1][GR.iijj] - con_cp*  \
                                ( POTT[:,:,GR.nz-1][GR.iijj] * \
                                    (   PVTF  [:,:,GR.nz-1 ][GR.iijj]  \
                                      - PVTFVB[:,:,GR.nzs-1][GR.iijj]  ) )
    for k in range(GR.nz-2,-1,-1):
        kp1 = k + 1

        dphi = con_cp * POTT[:,:,kp1][GR.iijj] * \
                        (PVTFVB[:,:,kp1][GR.iijj] - PVTF[:,:,kp1][GR.iijj])
        #phi_vb = PHI[:,:,kp1][GR.iijj] - dphi
        PHIVB[:,:,kp1][GR.iijj] = PHI[:,:,kp1][GR.iijj] - dphi

        # phi_k
        dphi = con_cp * POTT[:,:,k][GR.iijj] * \
                            (PVTF[:,:,k][GR.iijj] - PVTFVB[:,:,kp1][GR.iijj])
        #PHI[:,:,k][GR.iijj] = phi_vb - dphi
        PHI[:,:,k][GR.iijj] = PHIVB[:,:,kp1][GR.iijj] - dphi

    dphi = con_cp * POTT[:,:,0][GR.iijj] * \
                    (PVTFVB[:,:,0][GR.iijj] - PVTF[:,:,0][GR.iijj])
    PHIVB[:,:,0][GR.iijj] = PHI[:,:,0][GR.iijj] - dphi

    # TODO 5 NECESSARY
    PVTF = exchange_BC(GR, PVTF)
    PVTFVB = exchange_BC(GR, PVTFVB)
    PHI = exchange_BC(GR, PHI)

    return(PHI, PHIVB, PVTF, PVTFVB)



