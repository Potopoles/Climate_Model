import copy
import numpy as np
from continuity import colp_tendency_jacobson
from wind import wind_tendency_jacobson
from temperature import temperature_tendency_jacobson
from geopotential import diag_geopotential_jacobson
from boundaries import exchange_BC

def tendencies_jacobson(GR, COLP, POTT, HSURF,
                    UWIND, VWIND, WIND,
                    UFLX, VFLX, PHI, PVTF, PVTFVB):

    # PROGNOSE COLP
    dCOLPdt, UFLX, VFLX = colp_tendency_jacobson(GR, COLP, UWIND, VWIND, UFLX, VFLX)

    # PROGNOSE WIND
    dUFLXdt, dVFLXdt = wind_tendency_jacobson(GR, UWIND, VWIND, UFLX, VFLX, 
                                                    COLP, HSURF, PHI, POTT, PVTF, PVTFVB)

    # PROGNOSE POTT
    dPOTTdt = temperature_tendency_jacobson(GR, POTT, COLP, UWIND, VWIND, UFLX, VFLX)


    return(dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt)


def proceed_timestep_jacobson(GR, UWIND, VWIND, COLP,
                    POTT,
                    dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt):

    # TIME STEPPING
    COLP_OLD = copy.deepcopy(COLP)
    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)

    COLP[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    COLP = exchange_BC(GR, COLP)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)

    UWIND[GR.iisjj] = UWIND[GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                        + GR.dt*dUFLXdt/COLPA_is_NEW
    VWIND[GR.iijjs] = VWIND[GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                        + GR.dt*dVFLXdt/COLPA_js_NEW
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)

    POTT[GR.iijj] = POTT[GR.iijj] + GR.dt*dPOTTdt
    POTT = exchange_BC(GR, POTT)

    return(UWIND, VWIND, COLP, POTT)


def diagnose_fields_jacobson(GR, PHI, COLP, POTT, HSURF, PVTF, PVTVB):

    PHI, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, HSURF, POTT, COLP,
                                    PVTF, PVTVB)

    return(PHI, PVTF, PVTFVB)




def interp_COLPA(GR, COLP):

    COLPA_is = 1/8*(    COLP[GR.iisjj_im1_jp1] * GR.A[GR.iisjj_im1_jp1] + \
                        COLP[GR.iisjj_jp1]     * GR.A[GR.iisjj_jp1] + \
                    2 * COLP[GR.iisjj_im1]     * GR.A[GR.iisjj_im1] + \
                    2 * COLP[GR.iisjj]         * GR.A[GR.iisjj] + \
                        COLP[GR.iisjj_im1_jm1] * GR.A[GR.iisjj_im1_jm1] + \
                        COLP[GR.iisjj_jm1]     * GR.A[GR.iisjj_jm1]     )


    COLPA_js = 1/8*(    COLP[GR.iijjs_ip1_jm1] * GR.A[GR.iijjs_ip1_jm1] + \
                        COLP[GR.iijjs_ip1]     * GR.A[GR.iijjs_ip1] + \
                    2 * COLP[GR.iijjs_im1_jm1]     * GR.A[GR.iijjs_im1_jm1] + \
                    2 * COLP[GR.iijjs]         * GR.A[GR.iijjs] + \
                        COLP[GR.iijjs_im1_jm1] * GR.A[GR.iijjs_im1_jm1] + \
                        COLP[GR.iijjs_im1]     * GR.A[GR.iijjs_im1]     )


    #import matplotlib.pyplot as plt
    ##plt.plot(COLPA_js[3,:])
    ##plt.grid()
    #plt.contourf(COLPA_js.T)
    #plt.show()
    #print(COLPA_js[3,:])
    #quit()

    return(COLPA_is, COLPA_js)




