#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_VFLX.py  
Author:             Christoph Heim (CH)
Date created:       20190511
Last modified:      20190513
License:            MIT

Computation of horizontal momentum flux in latitude
(VFLX) tendency (dVFLXdt) according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.4, page 214ff
"""
import time
import numpy as np
import cupy as cp
from math import sin, cos
from numba import cuda, njit, prange, vectorize

from namelist import (UVFLX_dif_coef,
                    i_UVFLX_main_switch,
                    i_UVFLX_hor_adv, i_UVFLX_vert_adv,
                    i_UVFLX_coriolis,
                    i_UVFLX_num_dif, i_UVFLX_pre_grad)
from constants import con_rE
from org_namelist import (wp, wp_int, wp_old)
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator

from tendency_functions import (num_dif_py, pre_grad_py,
                                interp_WWIND_UVWIND_py)
####################################################################


####################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
####################################################################
def coriolis_and_spherical_VWIND_py(
        COLP, COLP_jm1,
        UWIND, UWIND_jm1,
        UWIND_ip1, UWIND_ip1_jm1,
        corf, corf_jm1, lat_rad, lat_rad_jm1,
        dlon_rad, dlat_rad):
    """
    Calculate coriolis acceleration and spherical grid conversion
    terms for VWIND.
    """
    return(
         - con_rE*dlon_rad*dlat_rad/wp(2.)*(
          COLP_jm1 * 
        ( UWIND_jm1 + UWIND_ip1_jm1 )/wp(2.) * 
        ( corf_jm1 * con_rE *
          cos(lat_rad_jm1) +
          ( UWIND_jm1 + UWIND_ip1_jm1 )/wp(2.) * 
          sin(lat_rad_jm1) )
        + COLP  * 
        ( UWIND + UWIND_ip1 )/wp(2.) * 
        ( corf  * con_rE *
          cos(lat_rad) +
          ( UWIND + UWIND_ip1 )/wp(2.) * 
          sin(lat_rad) )
        )
    )


def add_up_tendencies_py(
            VFLX,
            VFLX_im1, VFLX_ip1, VFLX_jm1, VFLX_jp1,
            UWIND, UWIND_jm1,
            UWIND_ip1, UWIND_ip1_jm1,
            PHI, PHI_jm1,
            POTT, POTT_jm1,
            PVTF, PVTF_jm1,
            PVTFVB, PVTFVB_jm1,
            PVTFVB_jm1_kp1, PVTFVB_kp1,
            WWIND_VWIND, WWIND_VWIND_kp1,

            COLP, COLP_jm1,

            corf, corf_jm1,
            lat_rad, lat_rad_jm1,
            dlon_rad, dlat_rad,
            dxjs,

            dsigma, sigma_vb, sigma_vb_kp1):
    """
    Compute and add up all tendency contributions of VFLUX.
    """

    dVFLXdt = wp(0.)

    if i_UVFLX_main_switch:
        # HORIZONTAL ADVECTION
        if i_UVFLX_hor_adv:
            pass
        # VERTICAL ADVECTION
        if i_UVFLX_vert_adv:
            dVFLXdt = dVFLXdt + (
                (WWIND_VWIND - WWIND_VWIND_kp1 ) / dsigma)
        # CORIOLIS AND SPHERICAL GRID CONVERSION
        if i_UVFLX_coriolis:
            dVFLXdt = dVFLXdt + coriolis_and_spherical_VWIND(
                COLP, COLP_jm1,
                UWIND, UWIND_jm1,
                UWIND_ip1, UWIND_ip1_jm1,
                corf, corf_jm1, lat_rad, lat_rad_jm1,
                dlon_rad, dlat_rad)
        # PRESSURE GRADIENT
        if i_UVFLX_pre_grad:
            dVFLXdt = dVFLXdt + pre_grad(
                PHI, PHI_jm1, COLP, COLP_jm1,
                POTT, POTT_jm1,
                PVTF, PVTF_jm1,
                PVTFVB, PVTFVB_jm1,
                PVTFVB_jm1_kp1, PVTFVB_kp1,
                dsigma, sigma_vb, sigma_vb_kp1,
                dxjs)
        # NUMERICAL HORIZONTAL DIFUSION
        if i_UVFLX_num_dif and (UVFLX_dif_coef > wp(0.)):
            dVFLXdt = dVFLXdt + num_dif(
                VFLX, VFLX_im1, VFLX_ip1,
                VFLX_jm1, VFLX_jp1,
                UVFLX_dif_coef)

    return(dVFLXdt)






####################################################################
### SPECIALIZE FOR GPU
####################################################################
num_dif = njit(num_dif_py, device=True, inline=True)
pre_grad = njit(pre_grad_py, device=True, inline=True)
add_up_tendencies = njit(add_up_tendencies_py, device=True,
                        inline=True)
coriolis_and_spherical_VWIND = njit(coriolis_and_spherical_VWIND_py,
                        device=True, inline=True)
interp_WWIND_UVWIND = njit(interp_WWIND_UVWIND_py, device=True,
                        inline=True)


def launch_cuda_vert_adv_kernel(WWIND_VWIND, VWIND, WWIND,
                                COLP_NEW, A, dsigma):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < nys+nb:
        WWIND_VWIND[i  ,j  ,k  ] = \
            interp_WWIND_UVWIND(
                VWIND     [i  ,j  ,k  ], VWIND     [i  ,j  ,k-1],
                WWIND     [i  ,j  ,k  ], WWIND     [i  ,j-1,k  ],
                WWIND     [i-1,j  ,k  ], WWIND     [i+1,j  ,k  ],
                WWIND     [i-1,j-1,k  ], WWIND     [i+1,j-1,k  ], 
                COLP_NEW  [i  ,j  ,0  ], COLP_NEW  [i  ,j-1,0  ],
                COLP_NEW  [i-1,j  ,0  ], COLP_NEW  [i+1,j  ,0  ],
                COLP_NEW  [i-1,j-1,0  ], COLP_NEW  [i+1,j-1,0  ], 
                A         [i  ,j  ,0  ], A         [i  ,j-1,0  ],
                A         [i-1,j  ,0  ], A         [i+1,j  ,0  ],
                A         [i-1,j-1,0  ], A         [i+1,j-1,0  ], 
                dsigma    [0  ,0  ,k  ], dsigma    [0  ,0  ,k-1],
                False, i, nx, k)

VFLX_vert_adv_gpu = cuda.jit(cuda_kernel_decorator(
                    launch_cuda_vert_adv_kernel))(
                    launch_cuda_vert_adv_kernel)



def launch_cuda_main_kernel(dVFLXdt, VFLX,
                    UWIND, 
                    PHI, COLP, POTT,
                    PVTF, PVTFVB, WWIND_VWIND,
                    corf, lat_rad,
                    dlon_rad, dlat_rad,
                    dxjs,
                    dsigma, sigma_vb):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < nys+nb:
        dVFLXdt[i  ,j  ,k] = \
            add_up_tendencies(
            # 3D
            VFLX    [i  ,j  ,k  ],
            VFLX    [i-1,j  ,k  ], VFLX    [i+1,j  ,k  ],
            VFLX    [i  ,j-1,k  ], VFLX    [i  ,j+1,k  ],
            UWIND   [i  ,j  ,k  ], UWIND   [i  ,j-1,k  ],
            UWIND   [i+1,j  ,k  ], UWIND   [i+1,j-1,k  ],
            PHI     [i  ,j  ,k  ], PHI     [i  ,j-1,k  ],
            POTT    [i  ,j  ,k  ], POTT    [i  ,j-1,k  ],
            PVTF    [i  ,j  ,k  ], PVTF    [i  ,j-1,k  ],
            PVTFVB  [i  ,j  ,k  ], PVTFVB  [i  ,j-1,k  ],
            PVTFVB  [i  ,j-1,k+1], PVTFVB  [i  ,j  ,k+1],
            WWIND_VWIND[i  ,j  ,k  ], WWIND_VWIND[i  ,j  ,k+1],
            # 2D
            COLP    [i  ,j  ,0  ], COLP    [i  ,j-1,0  ],
            # GR horizontal
            corf    [i  ,j  ,0  ], corf    [i  ,j-1,0  ],
            lat_rad [i  ,j  ,0  ], lat_rad [i  ,j-1,0  ],
            dlon_rad[i  ,j  ,0  ], dlat_rad[i  ,j  ,0  ],
            dxjs    [i  ,j  ,0  ],
            # GR vertical
            dsigma  [0  ,0  ,k  ], sigma_vb[0  ,0  ,k  ],
            sigma_vb[0  ,0  ,k+1])


VFLX_tendency_gpu = cuda.jit(cuda_kernel_decorator(
                            launch_cuda_main_kernel))(
                            launch_cuda_main_kernel)


####################################################################
### SPECIALIZE FOR CPU
####################################################################
### SPECIALIZE FOR CPU
####################################################################
num_dif = njit(num_dif_py)
pre_grad = njit(pre_grad_py)
add_up_tendencies = njit(add_up_tendencies_py)
coriolis_and_spherical_VWIND = njit(coriolis_and_spherical_VWIND_py)
interp_WWIND_UVWIND = njit(interp_WWIND_UVWIND_py)


def launch_numba_cpu_vert_adv(WWIND_VWIND, VWIND, WWIND,
                                COLP_NEW, A, dsigma):
    for i in prange(nb,nx+nb):
        for j in range(nb,nys+nb):
            for k in range(wp_int(0),nz):
                WWIND_VWIND[i  ,j  ,k  ] = \
                    interp_WWIND_UVWIND(
                VWIND     [i  ,j  ,k  ], VWIND     [i  ,j  ,k-1],
                WWIND     [i  ,j  ,k  ], WWIND     [i  ,j-1,k  ],
                WWIND     [i-1,j  ,k  ], WWIND     [i+1,j  ,k  ],
                WWIND     [i-1,j-1,k  ], WWIND     [i+1,j-1,k  ], 
                COLP_NEW  [i  ,j  ,0  ], COLP_NEW  [i  ,j-1,0  ],
                COLP_NEW  [i-1,j  ,0  ], COLP_NEW  [i+1,j  ,0  ],
                COLP_NEW  [i-1,j-1,0  ], COLP_NEW  [i+1,j-1,0  ], 
                A         [i  ,j  ,0  ], A         [i  ,j-1,0  ],
                A         [i-1,j  ,0  ], A         [i+1,j  ,0  ],
                A         [i-1,j-1,0  ], A         [i+1,j-1,0  ], 
                dsigma    [0  ,0  ,k  ], dsigma    [0  ,0  ,k-1],
                False, i, nx, k)

VFLX_vert_adv_cpu = njit(parallel=True)(launch_numba_cpu_vert_adv)


def launch_numba_cpu_main(dVFLXdt, VFLX,
                    UWIND, 
                    PHI, COLP, POTT,
                    PVTF, PVTFVB, WWIND_VWIND,
                    corf, lat_rad,
                    dlon_rad, dlat_rad,
                    dxjs,
                    dsigma, sigma_vb):

    for i in prange(nb,nx+nb):
        for j in range(nb,nys+nb):
            for k in range(wp_int(0),nz):
                dVFLXdt[i  ,j  ,k] = \
                    add_up_tendencies(
            # 3D
            VFLX    [i  ,j  ,k  ],
            VFLX    [i-1,j  ,k  ], VFLX    [i+1,j  ,k  ],
            VFLX    [i  ,j-1,k  ], VFLX    [i  ,j+1,k  ],
            UWIND   [i  ,j  ,k  ], UWIND   [i  ,j-1,k  ],
            UWIND   [i+1,j  ,k  ], UWIND   [i+1,j-1,k  ],
            PHI     [i  ,j  ,k  ], PHI     [i  ,j-1,k  ],
            POTT    [i  ,j  ,k  ], POTT    [i  ,j-1,k  ],
            PVTF    [i  ,j  ,k  ], PVTF    [i  ,j-1,k  ],
            PVTFVB  [i  ,j  ,k  ], PVTFVB  [i  ,j-1,k  ],
            PVTFVB  [i  ,j-1,k+1], PVTFVB  [i  ,j  ,k+1],
            WWIND_VWIND[i  ,j  ,k  ], WWIND_VWIND[i  ,j  ,k+1],
            # 2D
            COLP    [i  ,j  ,0  ], COLP    [i  ,j-1,0  ],
            # GR horizontal
            corf    [i  ,j  ,0  ], corf    [i  ,j-1,0  ],
            lat_rad [i  ,j  ,0  ], lat_rad [i  ,j-1,0  ],
            dlon_rad[i  ,j  ,0  ], dlat_rad[i  ,j  ,0  ],
            dxjs    [i  ,j  ,0  ],
            # GR vertical
            dsigma  [0  ,0  ,k  ], sigma_vb[0  ,0  ,k  ],
            sigma_vb[0  ,0  ,k+1])

VFLX_tendency_cpu = njit(parallel=True)(launch_numba_cpu_main)
