import numpy as np
from constants import con_kappa, con_g, con_Rd
from namelist import pTop


def diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB, TAIR, TAIRVB, RHO,\
                                PVTF, PVTFVB, UWIND, VWIND, WIND):

    TAIR[GR.iijj] = POTT[GR.iijj] * PVTF[GR.iijj]
    TAIRVB[GR.iijj] = POTTVB[GR.iijj] * PVTFVB[GR.iijj]
    PAIR[GR.iijj] = 100000*np.power(PVTF[GR.iijj], 1/con_kappa)
    RHO[GR.iijj] = PAIR[GR.iijj] / (con_Rd * TAIR[GR.iijj])

    for k in range(0,GR.nz):
        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + \
                                        UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )


    return(PAIR, TAIR, TAIRVB, RHO, WIND)


def diagnose_POTTVB_jacobson(GR, POTTVB, POTT, PVTF, PVTFVB):

    for ks in range(1,GR.nzs-1):
        POTTVB[:,:,ks][GR.iijj] =   ( \
                    +   (PVTFVB[:,:,ks][GR.iijj] - PVTF[:,:,ks-1][GR.iijj]) * \
                        POTT[:,:,ks-1][GR.iijj]
                    +   (PVTF[:,:,ks][GR.iijj] - PVTFVB[:,:,ks][GR.iijj]) * \
                        POTT[:,:,ks][GR.iijj]
                                    ) / (PVTF[:,:,ks][GR.iijj] - PVTF[:,:,ks-1][GR.iijj])

    # extrapolate model bottom and model top POTTVB
    POTTVB[:,:,0][GR.iijj] = POTT[:,:,0][GR.iijj] - \
            ( POTTVB[:,:,1][GR.iijj] - POTT[:,:,0][GR.iijj] )
    POTTVB[:,:,-1][GR.iijj] = POTT[:,:,-1][GR.iijj] - \
            ( POTTVB[:,:,-2][GR.iijj] - POTT[:,:,-1][GR.iijj] )

    return(POTTVB)




def interp_COLPA(GR, COLP):

    COLPA_is = 1/8*(    COLP[GR.iisjj_im1_jp1] * GR.A[GR.iisjj_im1_jp1] + \
                        COLP[GR.iisjj_jp1    ] * GR.A[GR.iisjj_jp1    ] + \
                    2 * COLP[GR.iisjj_im1    ] * GR.A[GR.iisjj_im1    ] + \
                    2 * COLP[GR.iisjj        ] * GR.A[GR.iisjj        ] + \
                        COLP[GR.iisjj_im1_jm1] * GR.A[GR.iisjj_im1_jm1] + \
                        COLP[GR.iisjj_jm1    ] * GR.A[GR.iisjj_jm1    ]   )

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




