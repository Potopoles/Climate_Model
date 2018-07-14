import copy
import numpy as np
from boundaries import exchange_BC
from upwind import tendencies_upwind, proceed_timestep_upwind, diagnose_fields_upwind
from jacobson import tendencies_jacobson, proceed_timestep_jacobson, \
                    diagnose_fields_jacobson, interp_COLPA

######################################################################################
######################################################################################
######################################################################################

def euler_forward(GR, COLP, PHI, POTT,
                    UWIND, VWIND, WIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF, PVTF, PVTFVB, i_spatial_discretization):

    if i_spatial_discretization == 'UPWIND':
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP, POTT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP, VFLXMP,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        UFLXMP, VFLXMP, COLP, POTT = proceed_timestep_upwind(GR, UFLXMP, VFLXMP, 
                                            COLP, POTT, dCOLPdt, dUFLXMPdt, 
                                            dVFLXMPdt, dPOTTdt)

        UWIND, VWIND = diagnose_fields_upwind(GR, COLP, POTT,
                        UWIND, VWIND, UFLXMP, VFLXMP, HSURF)

    elif i_spatial_discretization == 'JACOBSON':
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP, POTT, HSURF,
                                                                    UWIND, VWIND, WIND,
                                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        UWIND, VWIND, COLP, POTT = proceed_timestep_jacobson(GR, UWIND, VWIND, COLP,
                                                    POTT,
                                                    dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt)

        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP, POTT, HSURF, PVTF, PVTFVB)


    return(COLP, PHI, POTT,
            UWIND, VWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, UVFLX, VUFLX, VVFLX,
            HSURF)




######################################################################################
######################################################################################
######################################################################################

def matsuno(GR, COLP, PHI, POTT,
                    UWIND, VWIND, WIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF, PVTF, PVTFVB, i_spatial_discretization):

    if i_spatial_discretization == 'UPWIND':
        ########## ESTIMATE
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP, POTT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP, VFLXMP,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        # has to happen after masspoint_flux_tendency function
        UFLXMP_OLD = copy.deepcopy(UFLXMP)
        VFLXMP_OLD = copy.deepcopy(VFLXMP)
        COLP_OLD = copy.deepcopy(COLP)
        POTT_OLD = copy.deepcopy(POTT)


        UFLXMP, VFLXMP, COLP, POTT = proceed_timestep_upwind(GR, UFLXMP, VFLXMP, COLP,
                                            POTT, dCOLPdt, dUFLXMPdt, dVFLXMPdt,
                                            dPOTTdt)

        UWIND, VWIND = diagnose_fields_upwind(GR, COLP, POTT,
                        UWIND, VWIND, UFLXMP, VFLXMP, HSURF)

        ########## FINAL
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP, POTT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP, VFLXMP,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        UFLXMP, VFLXMP, COLP, POTT = proceed_timestep_upwind(GR, UFLXMP_OLD,
                                            VFLXMP_OLD, COLP_OLD, POTT_OLD,
                                            dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt)

        UWIND, VWIND = diagnose_fields_upwind(GR, COLP, POTT,
                        UWIND, VWIND, UFLXMP, VFLXMP, HSURF)


    elif i_spatial_discretization == 'JACOBSON':
        ########## ESTIMATE
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP, POTT, HSURF,
                                                                    UWIND, VWIND, WIND,
                                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        # has to happen after masspoint_flux_tendency function
        UWIND_OLD = copy.deepcopy(UWIND)
        VWIND_OLD = copy.deepcopy(VWIND)
        COLP_OLD = copy.deepcopy(COLP)
        POTT_OLD = copy.deepcopy(POTT)

        UWIND, VWIND, COLP, POTT = proceed_timestep_jacobson(GR, UWIND, VWIND, COLP,
                                                    POTT,
                                                    dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt)

        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP, POTT, HSURF, PVTF, PVTFVB)

        ########## FINAL
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP, POTT, HSURF,
                                                                    UWIND, VWIND, WIND,
                                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        UWIND, VWIND, COLP, POTT = proceed_timestep_jacobson(GR, UWIND_OLD, VWIND_OLD,
                                                    COLP_OLD, POTT_OLD,
                                                    dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt)

        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP, POTT, HSURF, PVTF, PVTFVB)

    return(COLP, PHI, POTT,
            UWIND, VWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, UVFLX, VUFLX, VVFLX,
            HSURF)






######################################################################################
######################################################################################
######################################################################################

def RK4(GR, COLP, PHI, POTT,
                    UWIND, VWIND, WIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF, PVTF, PVTFVB, i_spatial_discretization):

    if i_spatial_discretization == 'UPWIND':
        ########## level 1
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP, POTT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP, VFLXMP,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        # has to happen after masspoint_flux_tendency function
        UFLXMP_OLD = copy.deepcopy(UFLXMP)
        VFLXMP_OLD = copy.deepcopy(VFLXMP)
        COLP_OLD = copy.deepcopy(COLP)
        POTT_OLD = copy.deepcopy(POTT)

        UFLXMP_INT = copy.deepcopy(UFLXMP)
        VFLXMP_INT = copy.deepcopy(VFLXMP)
        COLP_INT = copy.deepcopy(COLP)
        POTT_INT = copy.deepcopy(POTT)

        dUFLXMP = GR.dt*dUFLXMPdt
        dVFLXMP = GR.dt*dVFLXMPdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        UFLXMP_INT[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP/2
        VFLXMP_INT[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP/2
        COLP_INT[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP/2
        POTT_INT[GR.iijj] = POTT_OLD[GR.iijj] + dPOTT/2
        UFLXMP_INT = exchange_BC(GR, UFLXMP_INT)
        VFLXMP_INT = exchange_BC(GR, VFLXMP_INT)
        COLP_INT = exchange_BC(GR, COLP_INT)
        POTT_INT = exchange_BC(GR, POTT_INT)

        UFLXMP[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP/6
        VFLXMP[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP/6
        COLP[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP/6
        POTT[GR.iijj] = POTT_OLD[GR.iijj] + dPOTT/6
        
        UWIND, VWIND = diagnose_fields_upwind(GR, COLP_INT, POTT_INT,
                        UWIND, VWIND, UFLXMP_INT, VFLXMP_INT, HSURF)

        ########## level 2
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP_INT, POTT_INT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP_INT, VFLXMP_INT,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        dUFLXMP = GR.dt*dUFLXMPdt
        dVFLXMP = GR.dt*dVFLXMPdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        UFLXMP_INT[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP/2
        VFLXMP_INT[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP/2
        COLP_INT[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP/2
        POTT_INT[GR.iijj] = POTT_OLD[GR.iijj] + dPOTT/2
        UFLXMP_INT = exchange_BC(GR, UFLXMP_INT)
        VFLXMP_INT = exchange_BC(GR, VFLXMP_INT)
        COLP_INT = exchange_BC(GR, COLP_INT)
        POTT_INT = exchange_BC(GR, POTT_INT)

        UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + dUFLXMP/3
        VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + dVFLXMP/3
        COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/3
        POTT[GR.iijj] = POTT[GR.iijj] + dPOTT/3
        
        UWIND, VWIND = diagnose_fields_upwind(GR, COLP_INT, POTT_INT,
                        UWIND, VWIND, UFLXMP_INT, VFLXMP_INT, HSURF)

        ########## level 3
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP_INT, POTT_INT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP_INT, VFLXMP_INT,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        dUFLXMP = GR.dt*dUFLXMPdt
        dVFLXMP = GR.dt*dVFLXMPdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        UFLXMP_INT[GR.iijj] = UFLXMP_OLD[GR.iijj] + dUFLXMP
        VFLXMP_INT[GR.iijj] = VFLXMP_OLD[GR.iijj] + dVFLXMP
        COLP_INT[GR.iijj] = COLP_OLD[GR.iijj] + dCOLP
        POTT_INT[GR.iijj] = POTT_OLD[GR.iijj] + dPOTT
        UFLXMP_INT = exchange_BC(GR, UFLXMP_INT)
        VFLXMP_INT = exchange_BC(GR, VFLXMP_INT)
        COLP_INT = exchange_BC(GR, COLP_INT)
        POTT_INT = exchange_BC(GR, POTT_INT)

        UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + dUFLXMP/3
        VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + dVFLXMP/3
        COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/3
        POTT[GR.iijj] = POTT[GR.iijj] + dPOTT/3
        
        UWIND, VWIND = diagnose_fields_upwind(GR, COLP_INT, POTT_INT,
                        UWIND, VWIND, UFLXMP_INT, VFLXMP_INT, HSURF)

        ########## level 4
        dCOLPdt, dUFLXMPdt, dVFLXMPdt, dPOTTdt = tendencies_upwind(GR, 
                        COLP_INT, POTT_INT, HSURF,
                        UWIND, VWIND, WIND,
                        UFLX, VFLX, UFLXMP_INT, VFLXMP_INT,
                        UUFLX, UVFLX, VUFLX, VVFLX)

        dUFLXMP = GR.dt*dUFLXMPdt
        dVFLXMP = GR.dt*dVFLXMPdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + dUFLXMP/6
        VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + dVFLXMP/6
        COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/6
        POTT[GR.iijj] = POTT[GR.iijj] + dPOTT/6
        UFLXMP = exchange_BC(GR, UFLXMP)
        VFLXMP = exchange_BC(GR, VFLXMP)
        COLP = exchange_BC(GR, COLP)
        POTT = exchange_BC(GR, POTT)
        
        UWIND, VWIND = diagnose_fields_upwind(GR, COLP, POTT,
                        UWIND, VWIND, UFLXMP, VFLXMP, HSURF)


    elif i_spatial_discretization == 'JACOBSON':
        ########## level 1
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP, POTT, HSURF,
                                                                    UWIND, VWIND, WIND,
                                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        # has to happen after masspoint_flux_tendency function
        UWIND_START = copy.deepcopy(UWIND)
        VWIND_START = copy.deepcopy(VWIND)
        COLP_START = copy.deepcopy(COLP)
        POTT_START = copy.deepcopy(POTT)

        UWIND_INT = copy.deepcopy(UWIND)
        VWIND_INT = copy.deepcopy(VWIND)
        COLP_INT = copy.deepcopy(COLP)
        POTT_INT = copy.deepcopy(POTT)

        dUFLX = GR.dt*dUFLXdt
        dVFLX = GR.dt*dVFLXdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        # TIME STEPPING
        COLPA_is_START, COLPA_js_START = interp_COLPA(GR, COLP_START)
        COLP_INT[GR.iijj] = COLP_START[GR.iijj] + dCOLP/2
        COLP_INT = exchange_BC(GR, COLP_INT)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP_INT)
        UWIND_INT[GR.iisjj] = UWIND_START[GR.iisjj] * COLPA_is_START/COLPA_is_NEW \
                            + dUFLX/2/COLPA_is_NEW
        VWIND_INT[GR.iijjs] = VWIND_START[GR.iijjs] * COLPA_js_START/COLPA_js_NEW \
                            + dVFLX/2/COLPA_js_NEW
        UWIND_INT = exchange_BC(GR, UWIND_INT)
        VWIND_INT = exchange_BC(GR, VWIND_INT)
        POTT_INT[GR.iijj] = POTT_START[GR.iijj] + dPOTT/2
        POTT_INT = exchange_BC(GR, POTT_INT)

        COLPA_is_START, COLPA_js_START = interp_COLPA(GR, COLP_START)
        COLP[GR.iijj] = COLP_START[GR.iijj] + dCOLP/6
        COLP = exchange_BC(GR, COLP)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
        UWIND[GR.iisjj] = UWIND_START[GR.iisjj] * COLPA_is_START/COLPA_is_NEW \
                        + dUFLX/6/COLPA_is_NEW
        VWIND[GR.iijjs] = VWIND_START[GR.iijjs] * COLPA_js_START/COLPA_js_NEW \
                            + dVFLX/6/COLPA_js_NEW
        UWIND = exchange_BC(GR, UWIND)
        VWIND = exchange_BC(GR, VWIND)
        POTT[GR.iijj] = POTT_START[GR.iijj] + dPOTT/6
        POTT = exchange_BC(GR, POTT)

        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP_INT, POTT_INT, HSURF,
                                                    PVTF, PVTFVB)

        ########## level 2
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP_INT, POTT_INT,
                                                    HSURF, UWIND_INT, VWIND_INT, WIND,
                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        dUFLX = GR.dt*dUFLXdt
        dVFLX = GR.dt*dVFLXdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        COLP_INT[GR.iijj] = COLP_START[GR.iijj] + dCOLP/2
        COLP_INT = exchange_BC(GR, COLP_INT)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP_INT)
        UWIND_INT[GR.iisjj] = UWIND_START[GR.iisjj] * COLPA_is_START/COLPA_is_NEW \
                            + dUFLX/2/COLPA_is_NEW
        VWIND_INT[GR.iijjs] = VWIND_START[GR.iijjs] * COLPA_js_START/COLPA_js_NEW \
                            + dVFLX/2/COLPA_js_NEW
        UWIND_INT = exchange_BC(GR, UWIND_INT)
        VWIND_INT = exchange_BC(GR, VWIND_INT)
        POTT_INT[GR.iijj] = POTT_START[GR.iijj] + dPOTT/2
        POTT_INT = exchange_BC(GR, POTT_INT)

        COLP_OLD = copy.deepcopy(COLP)
        COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
        COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/3
        COLP = exchange_BC(GR, COLP)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
        UWIND[GR.iisjj] = UWIND[GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                        + dUFLX/3/COLPA_is_NEW
        VWIND[GR.iijjs] = VWIND[GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                            + dVFLX/3/COLPA_js_NEW
        UWIND = exchange_BC(GR, UWIND)
        VWIND = exchange_BC(GR, VWIND)
        POTT[GR.iijj] = POTT[GR.iijj] + dPOTT/3
        POTT = exchange_BC(GR, POTT)
        
        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP_INT, POTT_INT, HSURF,
                                                        PVTF, PVTFVB)

        ########## level 3
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP_INT, POTT_INT,
                                                    HSURF, UWIND_INT, VWIND_INT, WIND,
                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        dUFLX = GR.dt*dUFLXdt
        dVFLX = GR.dt*dVFLXdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        COLP_INT[GR.iijj] = COLP_START[GR.iijj] + dCOLP
        COLP_INT = exchange_BC(GR, COLP_INT)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP_INT)
        UWIND_INT[GR.iisjj] = UWIND_START[GR.iisjj] * COLPA_is_START/COLPA_is_NEW \
                            + dUFLX/COLPA_is_NEW
        VWIND_INT[GR.iijjs] = VWIND_START[GR.iijjs] * COLPA_js_START/COLPA_js_NEW \
                            + dVFLX/COLPA_js_NEW
        UWIND_INT = exchange_BC(GR, UWIND_INT)
        VWIND_INT = exchange_BC(GR, VWIND_INT)
        POTT_INT[GR.iijj] = POTT_START[GR.iijj] + dPOTT
        POTT_INT = exchange_BC(GR, POTT_INT)

        COLP_OLD = copy.deepcopy(COLP)
        COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
        COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/3
        COLP = exchange_BC(GR, COLP)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
        UWIND[GR.iisjj] = UWIND[GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                        + dUFLX/3/COLPA_is_NEW
        VWIND[GR.iijjs] = VWIND[GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                            + dVFLX/3/COLPA_js_NEW
        UWIND = exchange_BC(GR, UWIND)
        VWIND = exchange_BC(GR, VWIND)
        POTT[GR.iijj] = POTT[GR.iijj] + dPOTT/3
        POTT = exchange_BC(GR, POTT)
        
        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP_INT, POTT_INT, HSURF,
                                                        PVTF, PVTFVB)

        ########## level 4
        dCOLPdt, dUFLXdt, dVFLXdt, dPOTTdt = tendencies_jacobson(GR, COLP_INT, POTT_INT,
                                                    HSURF, UWIND_INT, VWIND_INT, WIND,
                                                    UFLX, VFLX, PHI, PVTF, PVTFVB)

        dUFLX = GR.dt*dUFLXdt
        dVFLX = GR.dt*dVFLXdt
        dCOLP = GR.dt*dCOLPdt
        dPOTT = GR.dt*dPOTTdt

        COLP_OLD = copy.deepcopy(COLP)
        COLPA_is_OLD, COLPA_js_OLD = interp_COLPA(GR, COLP_OLD)
        COLP[GR.iijj] = COLP[GR.iijj] + dCOLP/6
        COLP = exchange_BC(GR, COLP)
        COLPA_is_NEW, COLPA_js_NEW = interp_COLPA(GR, COLP)
        UWIND[GR.iisjj] = UWIND[GR.iisjj] * COLPA_is_OLD/COLPA_is_NEW \
                        + dUFLX/6/COLPA_is_NEW
        VWIND[GR.iijjs] = VWIND[GR.iijjs] * COLPA_js_OLD/COLPA_js_NEW \
                            + dVFLX/6/COLPA_js_NEW
        UWIND = exchange_BC(GR, UWIND)
        VWIND = exchange_BC(GR, VWIND)
        POTT[GR.iijj] = POTT[GR.iijj] + dPOTT/6
        POTT = exchange_BC(GR, POTT)

        PHI, PVTF, PVTFVB = diagnose_fields_jacobson(GR, PHI, COLP, POTT, HSURF, PVTF, PVTFVB)

    return(COLP, PHI, POTT,
            UWIND, VWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, UVFLX, VUFLX, VVFLX,
            HSURF)
