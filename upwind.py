import numpy as np
from continuity import colp_tendency_upwind
from wind import masspoint_flux_tendency_upwind
from temperature import temperature_tendency_upwind
from boundaries import exchange_BC


def tendencies_upwind(GR, COLP, POTT, HSURF,
                    UWIND, VWIND, WIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX):

    # PROGNOSE COLP
    dCOLPdt = height_tendency_upstream(GR, COLP, UWIND, VWIND, UFLX, VFLX)

    # PROGNOSE WIND
    dUFLXMPdt, dVFLXMPdt = masspoint_flux_tendency_upstream(GR, UFLXMP, VFLXMP, COLP,
                                                    UWIND, VWIND,
                                                    UUFLX, VUFLX, UVFLX, VVFLX,
                                                    HSURF)

    # PROGNOSE POTT
    dPOTTdt = temperature_tendency_upstream(GR, POTT, COLP, UWIND, VWIND, UFLX, VFLX)


    return(dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt)


def proceed_timestep_upwind(GR, UFLXMP, VFLXMP, COLP, POTT,
                    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt):

    # TIME STEPPING
    UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + GR.dt*dUFLXMPdt
    VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + GR.dt*dVFLXMPdt
    COLP[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    POTT[GR.iijj] = POTT[GR.iijj] + GR.dt*dPOTTdt

    UFLXMP = exchange_BC(GR, UFLXMP)
    VFLXMP = exchange_BC(GR, VFLXMP)
    COLP = exchange_BC(GR, COLP)
    POTT = exchange_BC(GR, POTT)

    return(UFLXMP, VFLXMP, COLP, POTT)


def diagnose_fields_upwind(GR, COLP, POTT,
                    UWIND, VWIND, UFLXMP, VFLXMP, HSURF):

    # DIAGNOSTICS 
    UWIND[GR.iisjj] = ( UFLXMP[GR.iisjj_im1] + UFLXMP[GR.iisjj] ) \
                    / ( COLP[GR.iisjj_im1] + COLP[GR.iisjj] )
    VWIND[GR.iijjs] = ( VFLXMP[GR.iijjs_jm1] + VFLXMP[GR.iijjs] ) \
                    / ( COLP[GR.iijjs_jm1] + COLP[GR.iijjs] )
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)

    return(UWIND, VWIND)
