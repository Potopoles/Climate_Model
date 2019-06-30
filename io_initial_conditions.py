#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190525
Last modified:      20190629
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
                    POTT_gaussian_pert, POTT_random_pert,
                    QV_gaussian_pert, QV_random_pert)
from io_read_namelist import wp, pair_top
from io_constants import con_kappa, con_g, con_cp, con_Rd
###############################################################################

def initialize_fields(GR, POTTVB, WWIND, HSURF,
                        COLP, PSURF, PVTF, PVTFVB,
                        POTT, TAIR, TAIRVB, PAIR,
                        UWIND, VWIND, WIND, RHO,
                        PHI, PHIVB, QV, QC):

    np.random.seed(seed=3)

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

    QV[GR.ii,GR.jj,:] = 0.
    QC[GR.ii,GR.jj,:] = 0.

    # INITIAL CONDITIONS
    COLP[:,:,0] = gaussian2D(GR, COLP[:,:,0], COLP_gaussian_pert,
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
    COLP[:,:,0] = random2D(GR, COLP[:,:,0], COLP_random_pert)

    for k in range(0,GR.nz):
        UWIND[GR.iis,GR.jj,k] = uwind_0
        UWIND[:,:,k] = gaussian2D(GR, UWIND[:,:,k], UWIND_gaussian_pert, 
                     np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)*(
                        1-(k+1)/GR.nz)**(1/2)
        UWIND[:,:,k] = random2D(GR, UWIND[:,:,k], UWIND_random_pert)

        VWIND[:,:,k][GR.ii,GR.jjs] = vwind_0
        VWIND[:,:,k] = gaussian2D(GR, VWIND[:,:,k], VWIND_gaussian_pert,
                     np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)*(
                        1-(k+1)/GR.nz)**(1/2)
        VWIND[:,:,k] = random2D(GR, VWIND[:,:,k], VWIND_random_pert)

        POTT[:,:,k] = gaussian2D(GR, POTT[:,:,k], POTT_gaussian_pert,
                     np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        POTT[:,:,k] = random2D(GR, POTT[:,:,k], POTT_random_pert)

        QV  [:,:,k] = gaussian2D(GR, QV[:,:,k], QV_gaussian_pert,
                     np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
        QV  [:,:,k] = random2D(GR, QV[:,:,k], QV_random_pert)

    # BOUNDARY EXCHANGE OF INITIAL CONDITIONS
    COLP    = GR.exchange_BC(COLP)
    UWIND   = GR.exchange_BC(UWIND)
    VWIND   = GR.exchange_BC(VWIND)
    POTT    = GR.exchange_BC(POTT)
    QV      = GR.exchange_BC(QV)
    QC      = GR.exchange_BC(QC)

    ## PRIMARY DIAGNOSTIC FIELDS
    diagnose_fields_init(GR, PHI, PHIVB, PVTF, PVTFVB,
                                        HSURF, POTT, COLP, POTTVB)

    ## SECONDARY DIAGNOSTIC FIELDS
    PAIR, TAIR, TAIRVB, RHO, WIND = diagnose_secondary_fields(
                                GR, COLP, PAIR, PHI, POTT, POTTVB,
                                TAIR, TAIRVB, RHO,
                                PVTF, PVTFVB, UWIND, VWIND, WIND)


    out = {}
    out['COLP']        = COLP
    out['PSURF']        = PSURF
    out['PAIR']        = PAIR

    out['UWIND']        = UWIND
    out['VWIND']        = VWIND
    out['WWIND']        = WWIND
    out['WIND']         = WIND

    out['PVTF']         = PVTF
    out['PVTFVB']       = PVTFVB
    out['POTT']         = POTT
    out['POTTVB']       = POTTVB
    out['TAIR']         = TAIR
    out['TAIRVB']       = TAIRVB

    out['PHI']          = PHI
    out['PHIVB']        = PHIVB
    out['RHO']          = RHO

    out['QV']           = QV
    out['QC']           = QC

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
        PVTF[:,:,k][GR.ii,GR.jj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.ii,GR.jj] * PAIRVB[:,:,kp1][GR.ii,GR.jj] - \
                      PVTFVB[:,:,k  ][GR.ii,GR.jj] * PAIRVB[:,:,k  ][GR.ii,GR.jj] ) / \
                    ( PAIRVB[:,:,kp1][GR.ii,GR.jj] - PAIRVB[:,:,k  ][GR.ii,GR.jj] )

    return(PVTF, PVTFVB)



def set_up_profile(GR, COLP, HSURF, PSURF, PVTF,
                    PVTFVB, POTT, TAIR, PAIR):
    filename = 'data/mean_vert_prof.dat'
    profile = np.loadtxt(filename)

    for i in GR.ii:
        for j in GR.jj:
            PSURF[i,j,0] = np.interp(HSURF[i,j,0], profile[:,0], profile[:,2])

    COLP[GR.ii,GR.jj,0] = PSURF[GR.ii,GR.jj,0] - pair_top
    PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)

    for k in range(0,GR.nz):
        PAIR[:,:,k][GR.ii,GR.jj] = 100000.*np.power(PVTF[:,:,k][GR.ii,GR.jj], 1/con_kappa)

    interp = interp1d(profile[:,2], profile[:,3])
    for i in GR.ii:
        for j in GR.jj:
            TAIR[i,j,:] = interp(PAIR[i,j,:])

    for k in range(0,GR.nz):
        POTT[:,:,k][GR.ii,GR.jj] = TAIR[:,:,k][GR.ii,GR.jj] * \
                np.power(100000./PAIR[:,:,k][GR.ii,GR.jj], con_kappa)

    return(COLP, PSURF, POTT, TAIR, PAIR)




def load_topo(GR, HSURF):
    filename = 'data/elev.1-deg.nc'
    ncf = Dataset(filename, 'r', format='NETCDF4')
    lon_inp = ncf['lon'][:]
    lat_inp = ncf['lat'][:]
    hsurf_inp = ncf['data'][0,:,:]
    interp = interp2d(lon_inp, lat_inp, hsurf_inp)
    HSURF[GR.ii,GR.jj,0] = interp(GR.lon_deg[GR.ii,GR.nb+1,0].squeeze(),
                                  GR.lat_deg[GR.nb+1,GR.jj,0].squeeze()).T
    HSURF = GR.exchange_BC(HSURF)
    HSURF[HSURF < 0] = 0
    HSURF = GR.exchange_BC(HSURF)

    # smooth topography
    tau_smooth_min = 0.05
    tau_smooth_max = 0.15
    tau = np.full((GR.nx+2*GR.nb,GR.ny+2*GR.nb,1),np.nan)
    for j in range(GR.nb,GR.nb+GR.ny):
        tau[:,j,0] = tau_smooth_min + np.sin(GR.lat_rad[5,j])**2 * (
                                    tau_smooth_max - tau_smooth_min)
    
    for i in range(0,n_topo_smooth):
        HSURF[GR.ii,GR.jj] = HSURF[GR.ii,GR.jj] + (tau[GR.ii,GR.jj]*
                          (  HSURF[GR.ii-1,GR.jj] + HSURF[GR.ii+1,GR.jj] +
                             HSURF[GR.ii,GR.jj-1] + HSURF[GR.ii,GR.jj+1] -
                           4*HSURF[GR.ii,GR.jj  ]))
        HSURF = GR.exchange_BC(HSURF)

    return(HSURF)




def random2D(GR, FIELD, pert):
    FIELD[:] = FIELD[:] + pert*np.random.rand(FIELD.shape[0],
                                      FIELD.shape[1])
    return(FIELD)


def gaussian2D(GR, FIELD, pert, lon0_rad, lat0_rad, lonSig_rad, latSig_rad):
    dimx,dimy = FIELD.shape

    if (dimy == GR.nys+2*GR.nb): # staggered in y 
        selinds = (GR.ii, GR.jjs)
        lat = GR.lat_js_rad[GR.ii, GR.jjs, 0]
        lon = GR.lon_js_rad[GR.ii, GR.jjs, 0]
    elif (dimx == GR.nxs+2*GR.nb): # staggered in x 
        selinds = (GR.iis, GR.jj)
        lat = GR.lat_is_rad[GR.iis, GR.jj, 0]
        lon = GR.lon_is_rad[GR.iis, GR.jj, 0]
    else: # unstaggered in y and x 
        selinds = (GR.ii, GR.jj)
        lat = GR.lat_rad[GR.ii, GR.jj, 0]
        lon = GR.lon_rad[GR.ii, GR.jj, 0]

    perturb = pert * np.exp(
            - np.power(lon - lon0_rad, 2)/
                        (2*lonSig_rad**2)
            - np.power(lat - lat0_rad, 2)/
                        (2*latSig_rad**2) ) 

    FIELD[selinds] = FIELD[selinds] + perturb.squeeze()


    return(FIELD)



def set_up_sigma_levels(GR):
    HSURF       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb    ,1   ), 
                                np.nan, dtype=wp)
    HSURF = load_topo(GR, HSURF)
    filename = 'data/mean_vert_prof.dat'
    profile = np.loadtxt(filename)
    zsurf_test = np.mean(HSURF[GR.ii,GR.jj])
    top_ind = np.argwhere(profile[:,2] >= pair_top).squeeze()[-1]
    ztop_test = profile[top_ind,0] + (profile[top_ind,2] - pair_top)/ \
                            (profile[top_ind,4]*profile[top_ind,1])

    ks = np.arange(0,GR.nzs)
    z_vb_test   = np.zeros(GR.nzs, dtype=wp)
    p_vb_test   = np.zeros(GR.nzs, dtype=wp)
    rho_vb_test = np.zeros(GR.nzs, dtype=wp)
    g_vb_test   = np.zeros(GR.nzs, dtype=wp)

    z_vb_test[0] = ztop_test
    z_vb_test[ks] = zsurf_test + (ztop_test - zsurf_test)*(1 - ks/GR.nz)**(2)
    #z_vb_test[ks] = zsurf_test + (ztop_test - zsurf_test)*(1 - ks/GR.nz)
    #print(z_vb_test)
    #print(np.diff(z_vb_test))
    #quit()

    rho_vb_test = np.interp(z_vb_test, profile[:,0], profile[:,4]) 
    g_vb_test = np.interp(z_vb_test, profile[:,0], profile[:,1]) 
    p_vb_test[0] = pair_top
    ks = 1
    for ks in range(1,GR.nzs):
        p_vb_test[ks] = p_vb_test[ks-1] + \
                        rho_vb_test[ks]*g_vb_test[ks] * \
                        (z_vb_test[ks-1] - z_vb_test[ks])
    
    GR.sigma_vb[:] = (p_vb_test - pair_top)/(p_vb_test[-1] - pair_top)
    GR.dsigma[:] = np.diff(GR.sigma_vb)




###############################################################################
### OLD AND UGLY FUNCTIONS TO INITIALIZE ATMOSPHERIC PROFILE
###############################################################################

def diagnose_fields_init(GR, PHI, PHIVB, PVTF, PVTFVB,
                                        HSURF, POTT, COLP, POTTVB):
    PHI, PHIVB, PVTF, PVTFVB = diag_geopotential_init(
                                GR, PHI, PHIVB, HSURF, 
                                POTT, COLP, PVTF, PVTFVB)

    POTTVB = diagnose_POTTVB_init(GR, POTTVB, POTT, PVTF, PVTFVB)
    return(PHI, PHIVB, PVTF, PVTFVB, POTTVB)




def diag_pvt_factor_init(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan, dtype=wp)

    # TODO: WHY IS PAIRVB NOT FILLED AT UPPERMOST AND LOWER MOST HALFLEVEL??? 
    for ks in range(0,GR.nzs):
        PAIRVB[GR.ii,GR.jj,ks] = (pair_top + GR.sigma_vb[0,0,ks] * 
                                            COLP[GR.ii,GR.jj,0])
    
    PVTFVB[:] = np.power(PAIRVB/100000, con_kappa)

    for k in range(0,GR.nz):
        kp1 = k + 1
        PVTF[:,:,k][GR.ii,GR.jj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.ii,GR.jj] * PAIRVB[:,:,kp1][GR.ii,GR.jj] - \
                      PVTFVB[:,:,k  ][GR.ii,GR.jj] * PAIRVB[:,:,k  ][GR.ii,GR.jj] ) / \
                    ( PAIRVB[:,:,kp1][GR.ii,GR.jj] - PAIRVB[:,:,k  ][GR.ii,GR.jj] )

    return(PVTF, PVTFVB)



def diag_geopotential_init(GR, PHI, PHIVB, HSURF, POTT, COLP,
                               PVTF, PVTFVB):

    PVTF, PVTFVB = diag_pvt_factor_init(GR, COLP, PVTF, PVTFVB)

    #phi_vb = HSURF[GR.iijj]*con_g
    PHIVB[GR.ii,GR.jj,GR.nzs-1] = HSURF[GR.ii,GR.jj,0]*con_g
    PHI[GR.ii,GR.jj,GR.nz-1] = (PHIVB[GR.ii,GR.jj,GR.nzs-1] - 
                        con_cp*( POTT[GR.ii,GR.jj,GR.nz-1] *
                                (   PVTF  [GR.ii,GR.jj,GR.nz-1 ]
                                  - PVTFVB[GR.ii,GR.jj,GR.nzs-1]  ) ))
    for k in range(GR.nz-2,-1,-1):
        kp1 = k + 1

        dphi = con_cp * POTT[:,:,kp1][GR.ii,GR.jj] * \
                        (PVTFVB[:,:,kp1][GR.ii,GR.jj] - PVTF[:,:,kp1][GR.ii,GR.jj])
        #phi_vb = PHI[:,:,kp1][GR.iijj] - dphi
        PHIVB[:,:,kp1][GR.ii,GR.jj] = PHI[:,:,kp1][GR.ii,GR.jj] - dphi

        # phi_k
        dphi = con_cp * POTT[:,:,k][GR.ii,GR.jj] * \
                            (PVTF[:,:,k][GR.ii,GR.jj] - PVTFVB[:,:,kp1][GR.ii,GR.jj])
        #PHI[:,:,k][GR.iijj] = phi_vb - dphi
        PHI[:,:,k][GR.ii,GR.jj] = PHIVB[:,:,kp1][GR.ii,GR.jj] - dphi

    dphi = con_cp * POTT[:,:,0][GR.ii,GR.jj] * \
                    (PVTFVB[:,:,0][GR.ii,GR.jj] - PVTF[:,:,0][GR.ii,GR.jj])
    PHIVB[:,:,0][GR.ii,GR.jj] = PHI[:,:,0][GR.ii,GR.jj] - dphi

    # TODO 5 NECESSARY
    PVTF    = GR.exchange_BC(PVTF)
    PVTFVB  = GR.exchange_BC(PVTFVB)
    PHI     = GR.exchange_BC(PHI)

    return(PHI, PHIVB, PVTF, PVTFVB)





def diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB,
                                    TAIR, TAIRVB, RHO,
                                PVTF, PVTFVB, UWIND, VWIND, WIND):

    TAIR[GR.ii,GR.jj] = POTT[GR.ii,GR.jj] * PVTF[GR.ii,GR.jj]
    TAIRVB[GR.ii,GR.jj] = POTTVB[GR.ii,GR.jj] * PVTFVB[GR.ii,GR.jj]
    PAIR[GR.ii,GR.jj] = 100000*np.power(PVTF[GR.ii,GR.jj], 1/con_kappa)
    RHO[GR.ii,GR.jj] = PAIR[GR.ii,GR.jj] / (con_Rd * TAIR[GR.ii,GR.jj])

    for k in range(0,GR.nz):
        WIND[:,:,k][GR.ii,GR.jj] = np.sqrt( ((UWIND[:,:,k][GR.ii,GR.jj] + \
                                        UWIND[:,:,k][GR.ii+1,GR.jj])/2)**2 + \
                        ((VWIND[:,:,k][GR.ii,GR.jj] + VWIND[:,:,k][GR.ii,GR.jj+1])/2)**2 )


    return(PAIR, TAIR, TAIRVB, RHO, WIND)


def diagnose_POTTVB_init(GR, POTTVB, POTT, PVTF, PVTFVB):

    for ks in range(1,GR.nzs-1):
        POTTVB[:,:,ks][GR.ii,GR.jj] =   ( \
                    +   (PVTFVB[:,:,ks][GR.ii,GR.jj] - PVTF[:,:,ks-1][GR.ii,GR.jj]) * \
                        POTT[:,:,ks-1][GR.ii,GR.jj]
                    +   (PVTF[:,:,ks][GR.ii,GR.jj] - PVTFVB[:,:,ks][GR.ii,GR.jj]) * \
                        POTT[:,:,ks][GR.ii,GR.jj]
                                    ) / (PVTF[:,:,ks][GR.ii,GR.jj] - PVTF[:,:,ks-1][GR.ii,GR.jj])

    # extrapolate model bottom and model top POTTVB
    POTTVB[:,:,0][GR.ii,GR.jj] = POTT[:,:,0][GR.ii,GR.jj] - \
            ( POTTVB[:,:,1][GR.ii,GR.jj] - POTT[:,:,0][GR.ii,GR.jj] )
    POTTVB[:,:,-1][GR.ii,GR.jj] = POTT[:,:,-1][GR.ii,GR.jj] - \
            ( POTTVB[:,:,-2][GR.ii,GR.jj] - POTT[:,:,-1][GR.ii,GR.jj] )

    return(POTTVB)
