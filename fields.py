import numpy as np
from namelist import *
from boundaries import exchange_BC
from IO import load_topo, load_restart_fields, load_profile
from diagnostics import diagnose_secondary_fields, diagnose_POTTVB_jacobson
#from geopotential import diag_geopotential_jacobson
from radiation.org_radiation import radiation
from soil_model import soil
from org_microphysics import microphysics
from org_turbulence import turbulence
from constants import con_g, con_Rd, con_kappa, con_cp

def initialize_fields(GR, subgrids):
    if i_load_from_restart:
        COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND, \
        UFLX, VFLX, \
        HSURF, POTT, TAIR, TAIRVB, RHO, \
        POTTVB, PVTF, PVTFVB, \
        RAD, SOIL, MIC, TURB = load_restart_fields(GR)
    else:
        # CREATE ARRAYS
        # scalars
        COLP_OLD =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        COLP =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        COLP_NEW =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        dCOLPdt = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb), np.nan)
        PSURF =  np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        PAIR =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        PHI =    np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        PHIVB =  np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        POTT_OLD =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        POTT =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        dPOTTdt = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        TAIR =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        TAIRVB = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        RHO  =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz   ), np.nan)
        POTTVB = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        POTTVB[:] = 0
        # wind velocities
        UWIND_OLD =  np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        UWIND     =  np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        VWIND_OLD =  np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        VWIND     =  np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        WIND =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        WWIND =  np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        WWIND[:] = 0
        # mass fluxes at v elocity points
        UFLX =   np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        dUFLXdt = np.zeros( (GR.nxs+2*GR.nb,GR.ny +2*GR.nb,GR.nz) )
        VFLX =   np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        dVFLXdt = np.zeros( (GR.nx +2*GR.nb,GR.nys+2*GR.nb,GR.nz) )
        FLXDIV  = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz), np.nan)

        BFLX = np.full( (GR.nx +2*GR.nb,GR.ny +2*GR.nb, GR.nz  ), np.nan )
        CFLX = np.full( (GR.nxs+2*GR.nb,GR.nys+2*GR.nb, GR.nz  ), np.nan )
        DFLX = np.full( (GR.nx +2*GR.nb,GR.nys+2*GR.nb, GR.nz  ), np.nan )
        EFLX = np.full( (GR.nx +2*GR.nb,GR.nys+2*GR.nb, GR.nz  ), np.nan )
        RFLX = np.full( (GR.nx +2*GR.nb,GR.ny +2*GR.nb, GR.nz  ), np.nan )
        QFLX = np.full( (GR.nxs+2*GR.nb,GR.nys+2*GR.nb, GR.nz  ), np.nan )
        SFLX = np.full( (GR.nxs+2*GR.nb,GR.ny +2*GR.nb, GR.nz  ), np.nan )
        TFLX = np.full( (GR.nxs+2*GR.nb,GR.ny +2*GR.nb, GR.nz  ), np.nan )

        # vertical profile
        PVTF   = np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz ), np.nan)
        PVTFVB = np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs), np.nan)

        # surface height
        HSURF = load_topo(GR) 
        if not i_use_topo:
            HSURF[GR.iijj] = 0.
            HSURF = exchange_BC(GR, HSURF)
        
        # INITIAL CONDITIONS
        GR, COLP, PSURF, POTT, TAIR \
                = load_profile(GR, subgrids, COLP, HSURF, PSURF, PVTF, \
                                PVTFVB, POTT, TAIR)
        COLP = gaussian2D(GR, COLP, COLP_gaussian_pert, np.pi*3/4, 0, \
                            gaussian_dlon, gaussian_dlat)
        COLP = random2D(GR, COLP, COLP_random_pert)

        for k in range(0,GR.nz):
            UWIND[:,:,k][GR.iisjj] = u0   
            UWIND[:,:,k] = gaussian2D(GR, UWIND[:,:,k], UWIND_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            UWIND[:,:,k] = random2D(GR, UWIND[:,:,k], UWIND_random_pert)
            VWIND[:,:,k][GR.iijjs] = v0
            VWIND[:,:,k] = gaussian2D(GR, VWIND[:,:,k], VWIND_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            VWIND[:,:,k] = random2D(GR, VWIND[:,:,k], VWIND_random_pert)

            POTT[:,:,k] = gaussian2D(GR, POTT[:,:,k], POTT_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            POTT[:,:,k] = random2D(GR, POTT[:,:,k], POTT_random_pert)

        # BOUNDARY CONDITIONS
        COLP  = exchange_BC(GR, COLP)
        UWIND  = exchange_BC(GR, UWIND)
        VWIND  = exchange_BC(GR, VWIND)
        POTT  = exchange_BC(GR, POTT)


        # TURBULENCE 
        TURB = turbulence(GR, i_turbulence) 

        PHI, PHIVB, PVTF, PVTFVB, POTTVB \
                    = diagnose_fields_first_time(GR, PHI, PHIVB, COLP, POTT, \
                                            HSURF, PVTF, PVTFVB, POTTVB)

        PAIR, TAIR, TAIRVB, RHO, WIND = \
                diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB,
                                        TAIR, TAIRVB, RHO,\
                                        PVTF, PVTFVB, UWIND, VWIND, WIND)

        # SOIL MODEL
        SOIL = soil(GR, HSURF)

        # MOISTURE & MICROPHYSICS
        MIC = microphysics(GR, i_microphysics, TAIR, PAIR) 

        # RADIATION
        RAD = radiation(GR, i_radiation)
        RAD.calc_radiation(GR, TAIR, TAIRVB, RHO, PHIVB, SOIL, MIC)



    return(COLP_OLD, COLP, COLP_NEW, dCOLPdt, PAIR, PHI, PHIVB, \
            UWIND_OLD, UWIND, VWIND_OLD, VWIND, WIND, WWIND,
            UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
            BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
            HSURF, POTT_OLD, POTT, dPOTTdt, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB,
            RAD, SOIL, MIC, TURB)



def random2D(GR, FIELD, pert):
    FIELD = FIELD + pert*np.random.rand(FIELD.shape[0], FIELD.shape[1])
    return(FIELD)

def gaussian2D(GR, FIELD, pert, lon0_rad, lat0_rad, lonSig_rad, latSig_rad):

    dimx,dimy = FIELD.shape

    if (dimy == GR.nys+2*GR.nb): # staggered in y 
        selinds = GR.iijjs
        perturb = pert*np.exp( \
                - np.power(GR.lon_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
                - np.power(GR.latjs_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )
    elif (dimx == GR.nxs+2*GR.nb): # staggered in x 
        selinds = GR.iisjj
        perturb = pert*np.exp( \
                - np.power(GR.lonis_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
                - np.power(GR.lat_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )
    else: # unstaggered in y and x 
        selinds = GR.iijj
        perturb = pert*np.exp( \
                - np.power(GR.lon_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
                - np.power(GR.lat_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )

    FIELD[selinds] = FIELD[selinds] + perturb

    return(FIELD)



def diagnose_fields_first_time(GR, PHI, PHIVB, COLP, POTT, HSURF, PVTF, PVTFVB, POTTVB):

    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, PHIVB, HSURF, 
                                                    POTT, COLP, PVTF, PVTFVB)
    PHI = np.asarray(PHI)
    PHIVB = np.asarray(PHIVB)
    PVTF = np.asarray(PVTF)
    PVTFVB = np.asarray(PVTFVB)

    POTTVB = diagnose_POTTVB_jacobson(GR, POTTVB, POTT, PVTF, PVTFVB)
    POTTVB = np.asarray(POTTVB)

    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)



def diag_pvt_factor(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan )

    # TODO: WHY IS PAIRVB NOT FILLED AT UPPERMOST AND LOWER MOST HALFLEVEL??? 
    for ks in range(0,GR.nzs):
        PAIRVB[:,:,ks][GR.iijj] = pTop + GR.sigma_vb[ks] * COLP[GR.iijj]
    
    PVTFVB = np.power(PAIRVB/100000, con_kappa)

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
