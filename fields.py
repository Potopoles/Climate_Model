import numpy as np
from namelist import *
from boundaries import exchange_BC_all, exchange_BC
from IO import load_topo, load_restart_fields, load_profile
from jacobson import diagnose_fields_jacobson
from diagnostics import diagnose_secondary_fields
from radiation.org_radiation import radiation
from soil_model import soil

def initialize_fields(GR):
    if i_load_from_restart:
        COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND, \
        UFLX, VFLX, UFLXMP, VFLXMP, \
        HSURF, POTT, TAIR, TAIRVB, RHO, \
        POTTVB, PVTF, PVTFVB, \
        RAD, SOIL = load_restart_fields(GR)
    else:
        # CREATE ARRAYS
        # scalars
        COLP =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        PSURF =  np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        PAIR =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        PHI =    np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        PHIVB =  np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        POTT =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        TAIR =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        TAIRVB = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        RHO  =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz   ), np.nan)
        POTTVB = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        POTTVB[:] = 0
        # wind velocities
        UWIND =  np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        VWIND =  np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        WIND =   np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        WWIND =  np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        WWIND[:] = 0
        # mass fluxes at v elocity points
        UFLX =   np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        VFLX =   np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        # vertical profile
        PVTF   = np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz ), np.nan)
        PVTFVB = np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs), np.nan)

        # FOR MASSPOINT_FLUX_TENDENCY_UPSTREAM:
        # mass fluxes at mass points
        UFLXMP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
        VFLXMP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)

        # surface height
        HSURF = load_topo(GR) 
        if not i_use_topo:
            HSURF[GR.iijj] = 0.
            HSURF = exchange_BC(GR, HSURF)
        
        # INITIAL CONDITIONS
        GR, COLP, PSURF, POTT, TAIR \
                = load_profile(GR, COLP, HSURF, PSURF, PVTF, \
                                PVTFVB, POTT, TAIR)
        #quit()
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
        COLP, UWIND, VWIND, POTT = exchange_BC_all(GR, COLP, UWIND, VWIND, POTT)

        if i_spatial_discretization == 'UPWIND':
            raise NotImplementedError()
        if i_spatial_discretization == 'JACOBSON':
            PHI, PHIVB, PVTF, PVTFVB, POTTVB \
                        = diagnose_fields_jacobson(GR, PHI, PHIVB, COLP, POTT, \
                                                HSURF, PVTF, PVTFVB, POTTVB)

        PAIR, TAIR, TAIRVB, RHO, WIND = \
                diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB, TAIR, TAIRVB, RHO,\
                                        PVTF, PVTFVB, UWIND, VWIND, WIND)

        # SOIL MODEL
        SOIL = soil(GR, HSURF)

        # RADIATION
        RAD = radiation(GR, i_radiation)
        RAD.calc_radiation(GR, TAIR, TAIRVB, RHO, PHIVB, SOIL)



    return(COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB,
            RAD, SOIL)



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
