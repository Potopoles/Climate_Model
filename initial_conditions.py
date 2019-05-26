#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          initial_conditions.py  
Author:             Christoph Heim (CH)
Date created:       20190525
Last modified:      20190526
License:            MIT

Functions to initialize the model fields and set up an average
atmospheric profile.
###############################################################################
"""
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d, interp2d

from namelist import (i_load_from_restart, i_use_topo,
                    n_topo_smooth, pair_top,
                    uwind_0, vwind_0,
                    gaussian_dlon, gaussian_dlat,
                    COLP_gaussian_pert, COLP_random_pert,
                    UWIND_gaussian_pert, UWIND_random_pert,
                    VWIND_gaussian_pert, VWIND_random_pert,
                    POTT_gaussian_pert, POTT_random_pert)
from org_namelist import wp
from constants import con_kappa, con_g, con_cp, con_Rd
from boundaries import exchange_BC

def initialize_fields(GR, POTTVB, WWIND, HSURF,
                        COLP, PSURF, PVTF, PVTFVB,
                        POTT, TAIR, TAIRVB, PAIR,
                        UWIND, VWIND, WIND, RHO,
                        PHI, PHIVB,
                    ):
    if i_load_from_restart:
        raise NotImplementedError
        #CF, RAD, SURF, MIC, TURB = load_restart_fields(GR)
    else:

        #######################################################################
        # SET INITIAL FIELD VALUES
        #######################################################################

        # need to have non-nan-values because values from
        # half-level 0 and GR.nzs are used in calculation.
        POTTVB[:] = 0
        WWIND[:] = 0

        #  LOAD TOPOGRAPHY (HSURF)
        if i_use_topo:
            HSURF = load_topo(GR, HSURF) 
        else:
            HSURF[:] = 0.

        ## INITIALIZE PROFILE
        COLP, PSURF, POTT, TAIR, PAIR = set_up_profile(
                                GR, COLP, HSURF, PSURF, PVTF,
                                PVTFVB, POTT, TAIR, PAIR)


        # INITIAL CONDITIONS
        COLP[:,:,0] = gaussian2D(GR, COLP[:,:,0], COLP_gaussian_pert,
                                np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        COLP[:,:,0] = random2D(GR, COLP[:,:,0], COLP_random_pert)

        for k in range(0,GR.nz):
            UWIND[GR.iis,GR.jj,k] = uwind_0
            UWIND[:,:,k] = gaussian2D(GR, UWIND[:,:,k], UWIND_gaussian_pert, 
                         np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            UWIND[:,:,k] = random2D(GR, UWIND[:,:,k], UWIND_random_pert)
            VWIND[:,:,k][GR.iijjs] = vwind_0
            VWIND[:,:,k] = gaussian2D(GR, VWIND[:,:,k], VWIND_gaussian_pert,
                         np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            VWIND[:,:,k] = random2D(GR, VWIND[:,:,k], VWIND_random_pert)

            POTT[:,:,k] = gaussian2D(GR, POTT[:,:,k], POTT_gaussian_pert,
                         np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            POTT[:,:,k] = random2D(GR, POTT[:,:,k], POTT_random_pert)

        # BOUNDARY EXCHANGE OF INITIAL CONDITIONS
        COLP    = exchange_BC(GR, COLP)
        UWIND   = exchange_BC(GR, UWIND)
        VWIND   = exchange_BC(GR, VWIND)
        POTT    = exchange_BC(GR, POTT)

        ## PRIMARY DIAGNOSTIC FIELDS
        diagnose_fields_initializaiton(GR, PHI, PHIVB, PVTF, PVTFVB,
                                            HSURF, POTT, COLP, POTTVB)

        ## SECONDARY DIAGNOSTIC FIELDS
        PAIR, TAIR, TAIRVB, RHO, WIND = diagnose_secondary_fields(
                                    GR, COLP, PAIR, PHI, POTT, POTTVB,
                                    TAIR, TAIRVB, RHO,
                                    PVTF, PVTFVB, UWIND, VWIND, WIND)

        ########################################################################
        ## INITIALIZE NON-ATMOSPHERIC COMPONENTS
        ########################################################################

        ## SURF MODEL
        #if i_surface:
        #    SURF = surface(GR, CF)
        #else:
        #    SURF = None

        ########################################################################
        ## INITIALIZE PROCESSES
        ########################################################################

        ## MOISTURE & MICROPHYSICS
        #if i_microphysics:
        #    MIC = microphysics(GR, CF, i_microphysics, CF.TAIR, CF.PAIR) 
        #else:
        #    MIC = None

        ## RADIATION
        #if i_radiation:
        #    if SURF is None:
        #        raise ValueError('Soil model must be used for i_radiation > 0')
        #    RAD = radiation(GR, i_radiation)
        #    rad_njobs_orig = RAD.njobs_rad
        #    RAD.njobs_rad = 4
        #    #t_start = time.time()
        #    RAD.calc_radiation(GR, CF)
        #    #t_end = time.time()
        #    #GR.rad_comp_time += t_end - t_start
        #    RAD.njobs_rad = rad_njobs_orig
        #else:
        #    RAD = None

        ## TURBULENCE 
        #if i_turbulence:
        #    raise NotImplementedError('Baustelle')
        #    TURB = turbulence(GR, i_turbulence) 
        #else:
        #    TURB = None


    out = {}
    out['COLP']        = COLP
    out['PSURF']        = PSURF
    out['PAIR']        = PAIR

    out['UWIND']        = UWIND
    out['VWIND']        = VWIND
    out['WWIND']        = WWIND
    out['WIND']        = WIND

    out['PVTF']        = PVTF
    out['PVTFVB']       = PVTFVB
    out['POTT']        = POTT
    out['POTTVB']       = POTTVB
    out['TAIR']        = TAIR
    out['TAIRVB']        = TAIRVB

    out['PHI']        = PHI
    out['PHIVB']        = PHIVB
    out['RHO']        = RHO

    return(out)



def diag_pvt_factor(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan, dtype=wp)

    # TODO: WHY IS PAIRVB NOT FILLED AT UPPERMOST AND LOWER MOST HALFLEVEL??? 
    for ks in range(0,GR.nzs):
        PAIRVB[GR.ii,GR.jj,ks] = pair_top + (GR.sigma_vb[0,0,ks] * 
                                            COLP[GR.ii,GR.jj,0])
    
    PVTFVB[:] = np.power(PAIRVB/100000., con_kappa)

    for k in range(0,GR.nz):
        kp1 = k + 1
        PVTF[:,:,k][GR.iijj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.iijj] * PAIRVB[:,:,kp1][GR.iijj] - \
                      PVTFVB[:,:,k  ][GR.iijj] * PAIRVB[:,:,k  ][GR.iijj] ) / \
                    ( PAIRVB[:,:,kp1][GR.iijj] - PAIRVB[:,:,k  ][GR.iijj] )

    return(PVTF, PVTFVB)



def set_up_profile(GR, COLP, HSURF, PSURF, PVTF,
                    PVTFVB, POTT, TAIR, PAIR):
    filename = 'verticalProfileTable.dat'
    profile = np.loadtxt(filename)

    for i in GR.ii:
        for j in GR.jj:
            PSURF[i,j,0] = np.interp(HSURF[i,j,0], profile[:,0], profile[:,2])

    COLP[GR.ii,GR.jj,0] = PSURF[GR.ii,GR.jj,0] - pair_top
    PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)

    for k in range(0,GR.nz):
        PAIR[:,:,k][GR.iijj] = 100000.*np.power(PVTF[:,:,k][GR.iijj], 1/con_kappa)

    interp = interp1d(profile[:,2], profile[:,3])
    for i in GR.ii:
        for j in GR.jj:
            TAIR[i,j,:] = interp(PAIR[i,j,:])

    for k in range(0,GR.nz):
        POTT[:,:,k][GR.iijj] = TAIR[:,:,k][GR.iijj] * \
                np.power(100000./PAIR[:,:,k][GR.iijj], con_kappa)

    return(COLP, PSURF, POTT, TAIR, PAIR)




def load_topo(GR, HSURF):
    filename = '../elevation/elev.1-deg.nc'
    ncf = Dataset(filename, 'r', format='NETCDF4')
    lon_inp = ncf['lon'][:]
    lat_inp = ncf['lat'][:]
    hsurf_inp = ncf['data'][0,:,:]
    interp = interp2d(lon_inp, lat_inp, hsurf_inp)
    HSURF[GR.ii,GR.jj,0] = interp(GR.lon_deg[GR.i,GR.nb+1],
                                  GR.lat_deg[GR.nb+1,GR.j]).T
    HSURF[HSURF < 0] = 0
    HSURF = exchange_BC(GR, HSURF)

    # smooth topography
    tau_topo_smooth = 0.1
    for i in range(0,n_topo_smooth):
        HSURF[GR.ii,GR.jj] = HSURF[GR.ii,GR.jj] + (tau_topo_smooth*
                          (  HSURF[GR.ii-1,GR.jj] + HSURF[GR.ii+1,GR.jj] +
                             HSURF[GR.ii,GR.jj-1] + HSURF[GR.ii,GR.jj+1] -
                           4*HSURF[GR.ii,GR.jj  ]))
        HSURF = exchange_BC(GR, HSURF)

    return(HSURF)




def random2D(GR, FIELD, pert):
    FIELD[:] = FIELD[:] + pert*np.random.rand(FIELD.shape[0],
                                      FIELD.shape[1])
    return(FIELD)

def gaussian2D(GR, FIELD, pert, lon0_rad, lat0_rad, lonSig_rad, latSig_rad):
    dimx,dimy = FIELD.shape

    if (dimy == GR.nys+2*GR.nb): # staggered in y 
        selinds = (GR.ii, GR.jjs)
        selinds_lat = (GR.ii, GR.jjs, 0)
    elif (dimx == GR.nxs+2*GR.nb): # staggered in x 
        selinds = (GR.iis, GR.jj)
        selinds_lat = (GR.iis, GR.jj, 0)
    else: # unstaggered in y and x 
        selinds = (GR.ii, GR.jj)
        selinds_lat = (GR.ii, GR.jj, 0)

    perturb = pert*np.exp(
            - np.power(GR.lon_rad[selinds_lat] - lon0_rad, 2)/
                        (2*lonSig_rad**2)
            - np.power(GR.lat_rad[selinds_lat] - lat0_rad, 2)/
                        (2*latSig_rad**2) )

    FIELD[selinds] = FIELD[selinds] + perturb

    return(FIELD)




def diagnose_fields_initializaiton(GR, PHI, PHIVB, PVTF, PVTFVB,
                                        HSURF, POTT, COLP, POTTVB):
    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_jacobson(
                                GR, PHI, PHIVB, HSURF, 
                                POTT, COLP, PVTF, PVTFVB)

    POTTVB = diagnose_POTTVB_jacobson(GR, POTTVB, POTT, PVTF, PVTFVB)
    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)




def diag_pvt_factor(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan, dtype=wp)

    # TODO: WHY IS PAIRVB NOT FILLED AT UPPERMOST AND LOWER MOST HALFLEVEL??? 
    for ks in range(0,GR.nzs):
        PAIRVB[GR.ii,GR.jj,ks] = (pair_top + GR.sigma_vb[0,0,ks] * 
                                            COLP[GR.ii,GR.jj,0])
    
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
    PHIVB[GR.ii,GR.jj,GR.nzs-1] = HSURF[GR.ii,GR.jj,0]*con_g
    PHI[GR.ii,GR.jj,GR.nz-1] = (PHIVB[GR.ii,GR.jj,GR.nzs-1] - 
                        con_cp*( POTT[GR.ii,GR.jj,GR.nz-1] *
                                (   PVTF  [GR.ii,GR.jj,GR.nz-1 ]
                                  - PVTFVB[GR.ii,GR.jj,GR.nzs-1]  ) ))
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





def diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB,
                                    TAIR, TAIRVB, RHO,
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
