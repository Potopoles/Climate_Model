import copy
import numpy as np
from continuity import colp_tendency_upstream
from wind import masspoint_flux_tendency_upstream
from temperature import temperature_tendency_upstream
from geopotential import diag_geopotential
from boundaries import exchange_BC

def tendencies(GR, COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX):
    # PROGNOSE COLP
    dCOLPdt = colp_tendency_upstream(GR, COLP, UWIND, VWIND, UFLX, VFLX)

    # PROGNOSE WIND
    dUFLXMPdt, dVFLXMPdt = masspoint_flux_tendency_upstream(GR, UFLXMP, VFLXMP, COLP, PHI,
                                                    UWIND, VWIND,
                                                    UUFLX, VUFLX, UVFLX, VVFLX,
                                                    TAIR)
    # PROGNOSE TAIR
    dTAIRdt = temperature_tendency_upstream(GR, TAIR, COLP, UWIND, VWIND, UFLX, VFLX, dCOLPdt)


    return(dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt)


def proceed_timestep(GR, UFLXMP, VFLXMP, COLP, TAIR,
                    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt):

    # TIME STEPPING
    UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + GR.dt*dUFLXMPdt
    VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + GR.dt*dVFLXMPdt
    COLP[GR.iijj] = COLP[GR.iijj] + GR.dt*dCOLPdt
    TAIR[GR.iijj] = TAIR[GR.iijj] + GR.dt*dTAIRdt

    UFLXMP = exchange_BC(GR, UFLXMP)
    VFLXMP = exchange_BC(GR, VFLXMP)
    COLP = exchange_BC(GR, COLP)
    TAIR = exchange_BC(GR, TAIR)

    return(UFLXMP, VFLXMP, COLP, TAIR)


def diagnose_fields(GR, COLP, PHI, TAIR,
                    UWIND, VWIND, UFLXMP, VFLXMP, HSURF):

    # DIAGNOSTICS 
    UWIND[GR.iisjj] = ( UFLXMP[GR.iisjj_im1] + UFLXMP[GR.iisjj] ) \
                    / ( COLP[GR.iisjj_im1] + COLP[GR.iisjj] )
    VWIND[GR.iijjs] = ( VFLXMP[GR.iijjs_jm1] + VFLXMP[GR.iijjs] ) \
                    / ( COLP[GR.iijjs_jm1] + COLP[GR.iijjs] )
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)

    PHI = diag_geopotential(GR, PHI, HSURF, TAIR, COLP)

    return(UWIND, VWIND, PHI)






def euler_forward(GR, COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF):

    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    UFLXMP, VFLXMP, COLP, TAIR = proceed_timestep(GR, UFLXMP, VFLXMP, COLP, TAIR,
                                                dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt)

    UWIND, VWIND, PHI = diagnose_fields(GR, COLP, PHI, TAIR,
                    UWIND, VWIND, UFLXMP, VFLXMP, HSURF)

    return(COLP, PHI, TAIR,
            UWIND, VWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, UVFLX, VUFLX, VVFLX,
            HSURF)





def matsuno(GR, COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF):

    ########## ESTIMATE
    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    # has to happen after masspoint_flux_tendency function
    UFLXMP_OLD = copy.deepcopy(UFLXMP)
    VFLXMP_OLD = copy.deepcopy(VFLXMP)
    COLP_OLD = copy.deepcopy(COLP)
    TAIR_OLD = copy.deepcopy(TAIR)

    UFLXMP, VFLXMP, COLP, TAIR = proceed_timestep(GR, UFLXMP, VFLXMP, COLP, TAIR,
                                                dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt)

    UWIND, VWIND, PHI = diagnose_fields(GR, COLP, PHI, TAIR,
                    UWIND, VWIND, UFLXMP, VFLXMP, HSURF)

    ########## FINAL
    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    UFLXMP, VFLXMP, COLP, TAIR = proceed_timestep(GR, UFLXMP_OLD, VFLXMP_OLD, COLP_OLD, TAIR_OLD,
                                                dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt)

    UWIND, VWIND, PHI = diagnose_fields(GR, COLP, PHI, TAIR,
                    UWIND, VWIND, UFLXMP, VFLXMP, HSURF)

    return(COLP, PHI, TAIR,
            UWIND, VWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, UVFLX, VUFLX, VVFLX,
            HSURF)







def RK4(GR, COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF):

    ########## level 1
    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP, PHI, TAIR,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    # has to happen after masspoint_flux_tendency function
    UFLXMP_OLD = copy.deepcopy(UFLXMP)
    VFLXMP_OLD = copy.deepcopy(VFLXMP)
    COLP_OLD = copy.deepcopy(COLP)
    TAIR_OLD = copy.deepcopy(TAIR)

    UFLXMP_INT = copy.deepcopy(UFLXMP)
    VFLXMP_INT = copy.deepcopy(VFLXMP)
    COLP_INT = copy.deepcopy(COLP)
    TAIR_INT = copy.deepcopy(TAIR)

    dUFLXMP = GR.dt*dUFLXMPdt
    dVFLXMP = GR.dt*dVFLXMPdt
    dCOLP = GR.dt*dCOLPdt
    dTAIR = GR.dt*dTAIRdt

    UFLXMP_INT[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP/2
    VFLXMP_INT[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP/2
    COLP_INT[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP/2
    TAIR_INT[GR.iijj] = TAIR_OLD[GR.iijj] + dTAIR/2
    UFLXMP_INT = exchange_BC(GR, UFLXMP_INT)
    VFLXMP_INT = exchange_BC(GR, VFLXMP_INT)
    COLP_INT = exchange_BC(GR, COLP_INT)
    TAIR_INT = exchange_BC(GR, TAIR_INT)

    UFLXMP[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP/6
    VFLXMP[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP/6
    COLP[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP/6
    TAIR[GR.iijj] = TAIR_OLD[GR.iijj] + dTAIR/6
    
    UWIND, VWIND, PHI = diagnose_fields(GR, COLP_INT, PHI, TAIR_INT,
                    UWIND, VWIND, UFLXMP_INT, VFLXMP_INT, HSURF)

    ########## level 2
    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP_INT, PHI, TAIR_INT,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP_INT, VFLXMP_INT,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    dUFLXMP = GR.dt*dUFLXMPdt
    dVFLXMP = GR.dt*dVFLXMPdt
    dCOLP = GR.dt*dCOLPdt
    dTAIR = GR.dt*dTAIRdt

    UFLXMP_INT[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP/2
    VFLXMP_INT[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP/2
    COLP_INT[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP/2
    TAIR_INT[GR.iijj] = TAIR_OLD[GR.iijj] + dTAIR/2
    UFLXMP_INT = exchange_BC(GR, UFLXMP_INT)
    VFLXMP_INT = exchange_BC(GR, VFLXMP_INT)
    COLP_INT = exchange_BC(GR, COLP_INT)
    TAIR_INT = exchange_BC(GR, TAIR_INT)

    UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + dUFLXMP/3
    VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + dVFLXMP/3
    COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/3
    TAIR[GR.iijj] = TAIR[GR.iijj] + dTAIR/3
    
    UWIND, VWIND, PHI = diagnose_fields(GR, COLP_INT, PHI, TAIR_INT,
                    UWIND, VWIND, UFLXMP_INT, VFLXMP_INT, HSURF)

    ########## level 3
    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP_INT, PHI, TAIR_INT,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP_INT, VFLXMP_INT,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    dUFLXMP = GR.dt*dUFLXMPdt
    dVFLXMP = GR.dt*dVFLXMPdt
    dCOLP = GR.dt*dCOLPdt
    dTAIR = GR.dt*dTAIRdt

    UFLXMP_INT[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP
    VFLXMP_INT[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP
    COLP_INT[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP
    TAIR_INT[GR.iijj] = TAIR_OLD[GR.iijj] + dTAIR
    UFLXMP_INT = exchange_BC(GR, UFLXMP_INT)
    VFLXMP_INT = exchange_BC(GR, VFLXMP_INT)
    COLP_INT = exchange_BC(GR, COLP_INT)
    TAIR_INT = exchange_BC(GR, TAIR_INT)

    UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + dUFLXMP/3
    VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + dVFLXMP/3
    COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/3
    TAIR[GR.iijj] = TAIR[GR.iijj] + dTAIR/3
    
    UWIND, VWIND, PHI = diagnose_fields(GR, COLP_INT, PHI, TAIR_INT,
                    UWIND, VWIND, UFLXMP_INT, VFLXMP_INT, HSURF)

    ########## level 4
    dCOLPdt, dUFLXMPdt, dVFLXMPdt, dTAIRdt = tendencies(GR, 
                    COLP_INT, PHI, TAIR_INT,
                    UWIND, VWIND,
                    UFLX, VFLX, UFLXMP_INT, VFLXMP_INT,
                    UUFLX, UVFLX, VUFLX, VVFLX)

    dUFLXMP = GR.dt*dUFLXMPdt
    dVFLXMP = GR.dt*dVFLXMPdt
    dCOLP = GR.dt*dCOLPdt
    dTAIR = GR.dt*dTAIRdt

    UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + dUFLXMP/6
    VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + dVFLXMP/6
    COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/6
    TAIR[GR.iijj] = TAIR[GR.iijj] + dTAIR/6
    UFLXMP = exchange_BC(GR, UFLXMP)
    VFLXMP = exchange_BC(GR, VFLXMP)
    COLP = exchange_BC(GR, COLP)
    TAIR = exchange_BC(GR, TAIR)
    
    UWIND, VWIND, PHI = diagnose_fields(GR, COLP, PHI, TAIR,
                    UWIND, VWIND, UFLXMP, VFLXMP, HSURF)

    return(COLP, PHI, TAIR,
            UWIND, VWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, UVFLX, VUFLX, VVFLX,
            HSURF)
