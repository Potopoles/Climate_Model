#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_functions.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190510
License:            MIT

Collection of generally applicable finite difference tendency
kernels in pure python form. Can later be specified to GPU
or CPU using numba/cuda jit.
Functions are based on sigma pressure vertical coordinate.
"""
from org_namelist import wp, wp_int
from grid import nx,nxs,ny,nys,nz,nzs,nb
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





