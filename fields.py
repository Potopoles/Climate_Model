import numpy as np
from namelist import *
from boundaries import exchange_BC_all, exchange_BC
from IO import load_topo, load_restart_fields
from geopotential import diag_geopotential_upwind, diag_geopotential_jacobson

def initialize_fields(GR):
    if i_load_from_restart:
        COLP, PHI, UWIND, VWIND, WIND, \
        UFLX, VFLX, UFLXMP, VFLXMP, \
        UUFLX, VUFLX, UVFLX, VVFLX, \
        HSURF, POTT, PVTF, PVTFVB = load_restart_fields(GR)
    else:
        # CREATE ARRAYS
        # scalars
        COLP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        PHI = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, 1), np.nan)
        POTT = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        # wind velocities
        UWIND = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        VWIND = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)
        WIND = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        # mass fluxes at velocity points
        UFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        VFLX = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)

        # FOR MASSPOINT_FLUX_TENDENCY_UPSTREAM:
        # mass fluxes at mass points
        UFLXMP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        VFLXMP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        # momentum fluxes at velocity points
        UUFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        VUFLX = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)
        UVFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        VVFLX = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)

        # vertical profile
        PVTF   = np.full( (GR.nxs+2*GR.nb, GR.ny+2*GR.nb, 1), np.nan)
        PVTFVB = np.full( (GR.nxs+2*GR.nb, GR.ny+2*GR.nb, 2), np.nan)

        # surface height
        HSURF = load_topo(GR) 
        HSURF[GR.iijj] = 0.
        HSURF = exchange_BC(GR, HSURF)
        
        # INITIAL CONDITIONS
        COLP[GR.iijj] = pSurf - pTop
        #COLP[GR.iijj] = 10000 # TODO DEBUG
        COLP = gaussian2D(GR, COLP, COLP_gaussian_pert, np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        COLP = random2D(GR, COLP, COLP_random_pert)

        UWIND[GR.iisjj] = u0   
        UWIND = gaussian2D(GR, UWIND, UWIND_gaussian_pert, np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        UWIND = random2D(GR, UWIND, UWIND_random_pert)
        VWIND[GR.iijjs] = v0
        VWIND = gaussian2D(GR, VWIND, VWIND_gaussian_pert, np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        VWIND = random2D(GR, VWIND, VWIND_random_pert)

        POTT[GR.iijj] = 260.
        POTT = gaussian2D(GR, POTT, POTT_gaussian_pert, np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        POTT = random2D(GR, POTT, POTT_random_pert)

        # BOUNDARY CONDITIONS
        COLP, UWIND, VWIND, POTT = exchange_BC_all(GR, COLP, UWIND, VWIND, POTT)

        if i_spatial_discretization == 'UPWIND':
            PHI, PVTF, PVTFVB = diag_geopotential_upwind(GR, PHI, HSURF, TAIR, COLP)
        if i_spatial_discretization == 'JACOBSON':
            PHI, PVTF, PVTFVB = diag_geopotential_jacobson(GR, PHI, HSURF, POTT, COLP,
                                    PVTF, PVTFVB)

    return(COLP, PHI, UWIND, VWIND, WIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, VUFLX, UVFLX, VVFLX,
            HSURF, POTT, PVTF, PVTFVB)



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
