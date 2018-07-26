import copy
import numpy as np
from continuity import colp_tendency_jacobson, vertical_wind_jacobson
from wind import wind_tendency_jacobson
from temperature import temperature_tendency_jacobson
from geopotential import diag_geopotential_jacobson
from boundaries import exchange_BC

def tendencies_jacobson(GR, COLP, POTT, POTTVB, HSURF,
                    UWIND, VWIND, WWIND,
                    UFLX, VFLX, PHI, PVTF, PVTFVB,
                    RAD):

    # PROGNOSE COLP
    dCOLPdt, UFLX, VFLX, FLXDIV = colp_tendency_jacobson(GR, COLP, UWIND,\
                                                        VWIND, UFLX, VFLX)
    COLP_NEW = copy.deepcopy(COLP)
    COLP_NEW[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    COLP_NEW = exchange_BC(GR, COLP_NEW)

    # DIAGNOSE WWIND
    WWIND = vertical_wind_jacobson(GR, COLP_NEW, dCOLPdt, FLXDIV, WWIND)

    # PROGNOSE WIND
    dUFLXdt, dVFLXdt = wind_tendency_jacobson(GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
                                                    COLP, COLP_NEW, HSURF, PHI, POTT,
                                                    PVTF, PVTFVB)

    # PROGNOSE POTT
    dPOTTdt = temperature_tendency_jacobson(GR, POTT, POTTVB, COLP, COLP_NEW, UWIND, VWIND,\
                                            UFLX, VFLX, WWIND, RAD)


    return(dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt, WWIND)


def proceed_timestep_jacobson(GR, UWIND, VWIND, COLP,
                    POTT, dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt):

    # TIME STEPPING
    COLP_OLD = copy.deepcopy(COLP)
    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)

    COLP[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    COLP = exchange_BC(GR, COLP)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)

    for k in range(0,GR.nz):
        UWIND[:,:,k][GR.iisjj] = UWIND[:,:,k][GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                            + GR.dt*dUFLXdt[:,:,k]/COLPA_is_NEW
        VWIND[:,:,k][GR.iijjs] = VWIND[:,:,k][GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                            + GR.dt*dVFLXdt[:,:,k]/COLPA_js_NEW
        #VWIND[:,:,k][GR.ii,GR.nb] = 0
        #VWIND[:,:,k][GR.ii,-1-GR.nb] = 0
        POTT[:,:,k][GR.iijj] = POTT[:,:,k][GR.iijj] * COLP_OLD[GR.iijj]/COLP[GR.iijj] \
                            + GR.dt*dPOTTdt[:,:,k]/COLP[GR.iijj]
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)
    POTT = exchange_BC(GR, POTT)

    return(UWIND, VWIND, COLP, POTT)


def diagnose_fields_jacobson(GR, PHI, PHIVB, COLP, POTT, HSURF, PVTF, PVTVB, POTTVB):

    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, PHIVB, HSURF, 
                                                POTT, COLP, PVTF, PVTVB)

    for ks in range(1,GR.nzs-1):
        POTTVB[:,:,ks][GR.iijj] =   ( \
                    +   (PVTFVB[:,:,ks][GR.iijj] - PVTF[:,:,ks-1][GR.iijj]) * \
                        POTT[:,:,ks-1][GR.iijj]
                    +   (PVTF[:,:,ks][GR.iijj] - PVTFVB[:,:,ks][GR.iijj]) * \
                        POTT[:,:,ks][GR.iijj]
                                    ) / (PVTF[:,:,ks][GR.iijj] - PVTF[:,:,ks-1][GR.iijj])


    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)




def interp_COLPA(GR, COLP):

    COLPA_is = 1/8*(    COLP[GR.iisjj_im1_jp1] * GR.A[GR.iisjj_im1_jp1] + \
                        COLP[GR.iisjj_jp1    ] * GR.A[GR.iisjj_jp1    ] + \
                    2 * COLP[GR.iisjj_im1    ] * GR.A[GR.iisjj_im1    ] + \
                    2 * COLP[GR.iisjj        ] * GR.A[GR.iisjj        ] + \
                        COLP[GR.iisjj_im1_jm1] * GR.A[GR.iisjj_im1_jm1] + \
                        COLP[GR.iisjj_jm1    ] * GR.A[GR.iisjj_jm1    ]   )

    # ATTEMPT TO INTERPOLATE ONLY WITH TWO NEIGHBORING POINTS
    #COLPA_is[:, 0] = 1/2*( COLP[GR.iis-1,GR.jj[ 0]] * GR.A[GR.iis-1,GR.jj[ 0]] + \
    #                       COLP[GR.iis  ,GR.jj[ 0]] * GR.A[GR.iis  ,GR.jj[ 0]]   )
    #COLPA_is[:,-1] = 1/2*( COLP[GR.iis-1,GR.jj[-1]] * GR.A[GR.iis-1,GR.jj[-1]] + \
    #                       COLP[GR.iis  ,GR.jj[-1]] * GR.A[GR.iis  ,GR.jj[-1]]   )

    # ATTEMPT TO SET BOUNDARY AREA TO ZERO (see grid)
    # consequently change 1/8 to 1/6
    #COLPA_is[:, 0] = COLPA_is[:, 0]*4/3
    #COLPA_is[:, -1] = COLPA_is[:, -1]*4/3

    # ATTEMPT TO INTERPOLATE ONLY WITH TWO NEIGHBORING POINTS (JACOBSON)
    COLPA_is[:,-1] = 1/4*(    COLP[GR.iis-1,GR.jj[-1]] * GR.A[GR.iis-1,GR.jj[-1]] + \
                              COLP[GR.iis  ,GR.jj[-1]] * GR.A[GR.iis  ,GR.jj[-1]] + \
                              COLP[GR.iis-1,GR.jj[-2]] * GR.A[GR.iis-1,GR.jj[-2]] + \
                              COLP[GR.iis  ,GR.jj[-2]] * GR.A[GR.iis  ,GR.jj[-2]]   )

    COLPA_is[:, 0] = 1/4*(    COLP[GR.iis-1,GR.jj[0]] * GR.A[GR.iis-1,GR.jj[0]] + \
                              COLP[GR.iis  ,GR.jj[0]] * GR.A[GR.iis  ,GR.jj[0]] + \
                              COLP[GR.iis-1,GR.jj[1]] * GR.A[GR.iis-1,GR.jj[1]] + \
                              COLP[GR.iis  ,GR.jj[1]] * GR.A[GR.iis  ,GR.jj[1]]   )




    COLPA_js = 1/8*(    COLP[GR.iijjs_ip1_jm1] * GR.A[GR.iijjs_ip1_jm1] + \
                        COLP[GR.iijjs_ip1    ] * GR.A[GR.iijjs_ip1    ] + \
                    2 * COLP[GR.iijjs_jm1    ] * GR.A[GR.iijjs_jm1    ] + \
                    2 * COLP[GR.iijjs        ] * GR.A[GR.iijjs        ] + \
                        COLP[GR.iijjs_im1_jm1] * GR.A[GR.iijjs_im1_jm1] + \
                        COLP[GR.iijjs_im1    ] * GR.A[GR.iijjs_im1    ]   )


    return(COLPA_is, COLPA_js)




