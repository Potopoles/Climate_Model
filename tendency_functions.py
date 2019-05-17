#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_functions.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190513
License:            MIT

Collection of generally applicable finite difference tendency
kernels in pure python form. Can later be specified to GPU
or CPU using numba/cuda jit.
Functions are based on sigma pressure vertical coordinate.
Taken from Jacobson 2005:
Fundamentals of Atmospheric Modeling, Second Edition Chapter 7
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




# isjs
if i > 0 and i < nx+1 and j > 0 and j < ny+1:
if i > 0 and i < nx+1 and j > 0 and j < ny+1:


def calc_momentum_fluxes(
            ):


if i >= nb and i < nxs+nb and j >= nb and j < nys+nb:

    # FLUXES is js
    CFLX = wp(1.)/wp(12.) * (
                VFLX_im1_jm1 + VFLX_jm1         +
       wp(2.)*( VFLX_im1     + VFLX         )   +
                VFLX_im1_jp1 + VFLX_jp1         )

    QFLX = wp(1.)/wp(12.) * (
                UFLX_im1_jm1 + UFLX_im1         +
       wp(2.)*( UFLX_jm1     + UFLX         )   +
                UFLX_ip1_jm1 + UFLX_ip1         )


    DFLX  = wp(1.)/wp(24.) * (
        VFLX_jm1 + wp(2.) * VFLX + VFLX_jp1     +
                UFLX_jm1     + UFLX             +
                UFLX_ip1_jm1 + UFLX_ip1         )

    EFLX  = wp(1.)/wp(24.) * (  VFLX_jm1    + \
                                wp(2.)*VFLX  +\
                                   VFLX_jp1    - \
                                   UFLX_jm1    +\
                                 - UFLX    - \
                                   UFLX_ip1_jm1    +\
                                 - UFLX_ip1    )


    # FLUXES i j
    if i < nx+nb and j < ny+nb:
        BFLX = wp(1.)/wp(12.) * (
                    UFLX_jm1     + UFLX_ip1_jm1     +
           wp(2.)*( UFLX         + UFLX_ip1     )   +
                    UFLX_jp1 + UFLX_ip1_jp1         )

        RFLX = wp(1.)/wp(12.) * (
                    VFLX_im1     + VFLX_im1_jp1     +
           wp(2.)*( VFLX         + VFLX_jp1     )   +
                    VFLX_ip1 + VFLX_ip1_jp1         )
    # FLUXES i js
    elif i < nx+nb:

#if i >= nb and i < nxs+nb and j >= nb and j < ny +nb:
#if i >= nb and i < nx +nb and j >= nb and j < ny +nb:


def calc_fluxes_ij(BFLX, RFLX, UFLX, VFLX):
    nx = BFLX.shape[0] - 2
    ny = BFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:

def calc_fluxes_isj(SFLX, TFLX, UFLX, VFLX):
    nx = SFLX.shape[0] - 2
    ny = SFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        SFLX  = 1./24. * (  VFLX_im1  + \
                                       VFLX_im1_jp1 +\
                                       VFLX  +   \
                                       VFLX_jp1 +\
                                       UFLX_im1  + \
                                    2.*UFLX +\
                                       UFLX_ip1   )

        TFLX  = 1./24. * (  VFLX_im1 + \
                                       VFLX_im1_jp1 +\
                                       VFLX + \
                                       VFLX_jp1 +\
                                     - UFLX_im1 - \
                                    2.*UFLX +\
                                     - UFLX_ip1   )


def calc_fluxes_ijs(DFLX, EFLX, UFLX, VFLX):
    nx = DFLX.shape[0] - 2
    ny = DFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        DFLX  = 1./24. * (  VFLX_jm1   + \
                                    2.*VFLX +\
                                       VFLX_jp1   + \
                                       UFLX_jm1   +\
                                       UFLX   + \
                                       UFLX_ip1_jm1   +\
                                       UFLX_ip1   )

        EFLX  = 1./24. * (  VFLX_jm1    + \
                                    2.*VFLX  +\
                                       VFLX_jp1    - \
                                       UFLX_jm1    +\
                                     - UFLX    - \
                                       UFLX_ip1_jm1    +\
                                     - UFLX_ip1    )

def calc_fluxes_isjs(CFLX, QFLX, UFLX, VFLX):
    nx = CFLX.shape[0] - 2
    ny = CFLX.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        CFLX = wp(1.)/wp(12.) * (
                    VFLX_im1_jm1 + VFLX_jm1         +
           wp(2.)*( VFLX_im1     + VFLX         )   +
                    VFLX_im1_jp1 + VFLX_jp1         )

        QFLX = wp(1.)/wp(12.) * (
                    UFLX_im1_jm1 + UFLX_im1         +
           wp(2.)*( UFLX_jm1     + UFLX         )   +
                    UFLX_ip1_jm1 + UFLX_ip1         )



