#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_functions.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190512
License:            MIT

Collection of generally applicable finite difference tendency
kernels in pure python form. Can later be specified to GPU
or CPU using numba/cuda jit.
Functions are based on sigma pressure vertical coordinate.
"""
from org_namelist import wp, wp_int
from grid import nx,nxs,ny,nys,nz,nzs,nb
from constants import con_cp
####################################################################


def hor_adv_py(VAR, VAR_im1, VAR_ip1, VAR_jm1, VAR_jp1,
            UFLX, UFLX_ip1, VFLX, VFLX_jp1, A):
    """
    Horizontal advection using momentum fluxes UFLX and VFLX.
    """
    return(
        ( + UFLX * (VAR_im1 + VAR)/wp(2.)
          - UFLX_ip1 * (VAR + VAR_ip1)/wp(2.)
          + VFLX * (VAR_jm1 + VAR)/wp(2.)
          - VFLX_jp1 * (VAR + VAR_jp1)/wp(2.) )/A )



def vert_adv_py(VARVB, VARVB_kp1, WWIND, WWIND_kp1, COLP_NEW,
            dsigma, k):
    """
    Vertical advection. 
    """

    if k == wp_int(0):
        return(
            COLP_NEW * (
                - WWIND_kp1 * VARVB_kp1) / dsigma)
    elif k == nz:
        return(
            COLP_NEW * (
                + WWIND     * VARVB    ) / dsigma)
    else:
        return(
            COLP_NEW * (
                + WWIND     * VARVB 
                - WWIND_kp1 * VARVB_kp1) / dsigma)




def num_dif_pw_py(VAR, VAR_im1, VAR_ip1, VAR_jm1, VAR_jp1,
            COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1,
            VAR_dif_coef):
    """
    Numerical diffusion with pressure weighting.
    """
    return(
            VAR_dif_coef * (
                + COLP_im1 * VAR_im1
                + COLP_ip1 * VAR_ip1
                + COLP_jm1 * VAR_jm1
                + COLP_jp1 * VAR_jp1
                - wp(4.) * COLP * VAR )
            )


def num_dif_py(VAR, VAR_im1, VAR_ip1, VAR_jm1, VAR_jp1,
            VAR_dif_coef):
    """
    Numerical diffusion non-pressure weighting.
    """
    return(
            VAR_dif_coef * (
                + VAR_im1
                + VAR_ip1
                + VAR_jm1
                + VAR_jp1
                - wp(4.) * VAR )
            )



 


def pre_grad_py(PHI, PHI_dm1, COLP, COLP_dm1,
                POTT, POTT_dm1,
                PVTF, PVTF_dm1,
                PVTFVB, PVTFVB_dm1,
                PVTFVB_dm1_kp1, PVTFVB_kp1,
                dsigma, sigma_vb, sigma_vb_kp1,
                dgrid):
    """
    Pressure gradient term for horizontal velocities.
    dm1 & dp1 mean in the horizontal direction of interest
    -1 & +1 grid point.
    """
    return(
            - dgrid * (
                ( PHI - PHI_dm1 ) *
                ( COLP + COLP_dm1 ) / wp(2.) +
                ( COLP - COLP_dm1 ) * con_cp/wp(2.) *
                (
                    + POTT_dm1 / dsigma *
                    ( 
                    sigma_vb_kp1 * ( PVTFVB_dm1_kp1 - PVTF_dm1   ) + 
                    sigma_vb *     ( PVTF_dm1       - PVTFVB_dm1 )  
                    )
                    + POTT     / dsigma *
                    ( 
                    sigma_vb_kp1 * ( PVTFVB_kp1     - PVTF       ) + 
                    sigma_vb *     ( PVTF           - PVTFVB     )  
                    )
                )
            )
        )



def interp_WWIND_UVWIND_py(
            DWIND, DWIND_km1,
            WWIND, WWIND_dm1,
            WWIND_pm1, WWIND_pp1,
            WWIND_pm1_dm1, WWIND_pp1_dm1,
            COLP_NEW, COLP_NEW_dm1,
            COLP_NEW_pm1, COLP_NEW_pp1,
            COLP_NEW_pm1_dm1, COLP_NEW_pp1_dm1, 
            A, A_dm1,
            A_pm1, A_pp1,
            A_pm1_dm1, A_pp1_dm1, 
            dsigma, dsigma_km1,
            rigid_wall, p_ind, np, k):
    """
    Interpolate WWIND * UVWIND (U or V, depending on direction) 
    onto position of repective horizontal wind.
    d inds (e.g. dm1 = d minus 1) are in direction of hor. wind
    vector.
    p inds are perpendicular to direction of hor. wind vector.
    if rigid_wall: Special BC for rigid wall parallel to flow
    direction according to hint of Mark Jacobson during mail
    conversation. np is number of grid cells and p_ind current
    index in direction perpeindular to flow.
    """
    if k == wp_int(0) or k == nzs-wp_int(1):
        WWIND_DWIND = wp(0.)
    else:
        # left rigid wall 
        if rigid_wall and (p_ind == nb):
            COLPAWWIND_ds_ks = wp(0.25)*( 
                    COLP_NEW_pp1_dm1 * A_pp1_dm1 * WWIND_pp1_dm1 +
                    COLP_NEW_pp1     * A_pp1     * WWIND_pp1     +
                    COLP_NEW_dm1     * A_dm1     * WWIND_dm1     +
                    COLP_NEW         * A         * WWIND         )
        # right rigid wall 
        elif rigid_wall and (p_ind == np):
            COLPAWWIND_ds_ks = wp(0.25)*( 
                    COLP_NEW_dm1     * A_dm1     * WWIND_dm1     +
                    COLP_NEW         * A         * WWIND         +
                    COLP_NEW_pm1_dm1 * A_pm1_dm1 * WWIND_pm1_dm1 +
                    COLP_NEW_pm1     * A_pm1     * WWIND_pm1     )
        # inside domain (not at boundary of perpendicular dimension)
        else:
            COLPAWWIND_ds_ks = wp(0.125)*( 
                    COLP_NEW_pp1_dm1 * A_pp1_dm1 * WWIND_pp1_dm1 +
                    COLP_NEW_pp1     * A_pp1     * WWIND_pp1     +
           wp(2.) * COLP_NEW_dm1     * A_dm1     * WWIND_dm1     +
           wp(2.) * COLP_NEW         * A         * WWIND         +
                    COLP_NEW_pm1_dm1 * A_pm1_dm1 * WWIND_pm1_dm1 +
                    COLP_NEW_pm1     * A_pm1     * WWIND_pm1     )

        # interpolate hor. wind on vertical interface
        DWIND_ks = (( dsigma     * DWIND_km1  +
                      dsigma_km1 * DWIND      ) /
                    ( dsigma     + dsigma_km1 ) )

        # combine
        WWIND_DWIND = COLPAWWIND_ds_ks * DWIND_ks

    return(WWIND_DWIND)









#    nx = WWIND_UWIND.shape[0] - 2
#    ny = WWIND_UWIND.shape[1] - 2
#    nz = WWIND_UWIND.shape[2]
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1 and k > 0 and k < nz-1:
#####################################################################
#        if j == nb:
#            # meridional rigid wall boundaries (personal comm.
#            # with Mark Z. Jacobson.)
#            COLPAWWIND_is_ks = wp(0.25)*( 
#                    COLP_NEW_im1_jp1 * A_im1_jp1 * WWIND_im1_jp1 +
#                    COLP_NEW_jp1     * A_jp1     * WWIND_jp1     +
#                    COLP_NEW_im1     * A_im1     * WWIND_im1     +
#                    COLP_NEW         * A         * WWIND         )
#        elif j == ny:
#            # meridional rigid wall boundaries (personal comm.
#            # with Mark Z. Jacobson.)
#            COLPAWWIND_is_ks = wp(0.25)*( 
#                    COLP_NEW_im1     * A_im1     * WWIND_im1     +
#                    COLP_NEW         * A         * WWIND         +
#                    COLP_NEW_im1_jm1 * A_im1_jm1 * WWIND_im1_jm1 +
#                    COLP_NEW_jm1     * A_jm1     * WWIND_jm1     )
#        else:
#            COLPAWWIND_is_ks = wp(0.125)*( 
#                    COLP_NEW_im1_jp1 * A_im1_jp1 * WWIND_im1_jp1 +
#                    COLP_NEW_jp1     * A_jp1     * WWIND_jp1     +
#           wp(2.) * COLP_NEW_im1     * A_im1     * WWIND_im1     +
#           wp(2.) * COLP_NEW         * A         * WWIND         +
#                    COLP_NEW_im1_jm1 * A_im1_jm1 * WWIND_im1_jm1 +
#                    COLP_NEW_jm1     * A_jm1     * WWIND_jm1     )
#
#
#
#
#        UWIND_ks = ( dsigma[k  ] * UWIND[i  ,j  ,k-1] +   \
#                     dsigma[k-1] * UWIND[i  ,j  ,k  ] ) / \
#                   ( dsigma[k  ] + dsigma[k-1] )
#        WWIND_UWIND[i  ,j  ,k ] = COLPAWWIND_is_ks * UWIND_ks
#
#    if k == 0 or k == nz-1:
#        WWIND_UWIND[i  ,j  ,k ] = 0.
#
#    cuda.syncthreads()
#
#
#
#
#
#    nx = WWIND_VWIND.shape[0] - 2
#    ny = WWIND_VWIND.shape[1] - 2
#    nz = WWIND_VWIND.shape[2]
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1 and k > 0 and k < nz-1:
#        COLPAWWIND_js_ks = wp(0.125)*( 
#                COLP_NEW_ip1_jm1 * A_ip1_jm1 * WWIND_ip1_jm1 +
#                COLP_NEW_ip1     * A_ip1     * WWIND_ip1     +
#       wp(2.) * COLP_NEW_jm1     * A_jm1     * WWIND_jm1     +
#       wp(2.) * COLP_NEW         * A         * WWIND         +
#                COLP_NEW_im1_jm1 * A_im1_jm1 * WWIND_im1_jm1 +
#                COLP_NEW_im1     * A_im1     * WWIND_im1     )
#
#        VWIND_ks = ( dsigma[k  ] * VWIND[i  ,j  ,k-1] +   \
#                     dsigma[k-1] * VWIND[i  ,j  ,k  ] ) / \
#                   ( dsigma[k  ] + dsigma[k-1] )
#        WWIND_VWIND[i  ,j  ,k ] = COLPAWWIND_js_ks * VWIND_ks
#
#    if k == 0 or k == nz-1:
#        WWIND_VWIND[i  ,j  ,k ] = 0.
#
#    cuda.syncthreads()
#
#
#
#
#
#
#
#    dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
#                        (WWIND_UWIND[i  ,j  ,k  ] - \
#                         WWIND_UWIND[i  ,j  ,k+1]  ) / dsigma[k]
#
#
#    dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
#                        (WWIND_VWIND[i  ,j  ,k  ] - \
#                         WWIND_VWIND[i  ,j  ,k+1]  ) / dsigma[k]












#dUFLXdt[i  ,j  ,k] = dUFLXdt[i  ,j  ,k] + \
#    con_rE*dlon_rad*dlon_rad/2.*(\
#      COLP [i-1,j    ] * \
#    ( VWIND[i-1,j  ,k] + VWIND[i-1,j+1,k] )/2. * \
#    ( corf_is[i  ,j  ] * con_rE *\
#      cos(latis_rad[i  ,j  ]) + \
#      ( UWIND[i-1,j  ,k] + UWIND[i  ,j  ,k] )/2. * \
#      sin(latis_rad[i  ,j  ]) )\
#    + COLP [i  ,j    ] * \
#    ( VWIND[i  ,j  ,k] + VWIND[i  ,j+1,k] )/2. * \
#    ( corf_is[i  ,j  ] * con_rE * \
#      cos(latis_rad[i  ,j  ]) + \
#      ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
#      sin(latis_rad[i  ,j  ]) )\
#    )
#
#dVFLXdt[i  ,j  ,k] = dVFLXdt[i  ,j  ,k] + \
#     - con_rE*dlon_rad*dlon_rad/2.*(\
#      COLP[i  ,j-1  ] * \
#    ( UWIND[i  ,j-1,k] + UWIND[i+1,j-1,k] )/2. * \
#    ( corf[i  ,j-1  ] * con_rE *\
#      cos(lat_rad[i  ,j-1  ]) +\
#      ( UWIND[i  ,j-1,k] + UWIND[i+1,j-1,k] )/2. * \
#      sin(lat_rad[i  ,j-1  ]) )\
#    + COLP [i  ,j    ] * \
#    ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
#    ( corf [i  ,j    ] * con_rE *\
#      cos(lat_rad[i  ,j    ]) +\
#      ( UWIND[i  ,j  ,k] + UWIND[i+1,j  ,k] )/2. * \
#      sin(lat_rad[i  ,j    ]) )\
#    )
