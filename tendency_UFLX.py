#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
####################################################################
File name:          tendency_UFLX.py  
Author:             Christoph Heim (CH)
Date created:       20190510
Last modified:      20190521
License:            MIT

Computation of horizontal momentum flux in longitude
(UFLX) tendency (dUFLXdt) according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.4, page 214ff
####################################################################
"""
import time
import numpy as np
import cupy as cp
from math import sin, cos
from numba import cuda, njit, prange, vectorize

from namelist import (i_UVFLX_main_switch,
                    i_UVFLX_hor_adv, i_UVFLX_vert_adv,
                    i_UVFLX_coriolis,
                    i_UVFLX_num_dif, i_UVFLX_pre_grad)
from org_namelist import UVFLX_dif_coef
from constants import con_rE
from org_namelist import (wp, wp_int, wp_old)
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator

from tendency_functions import (num_dif_py, pre_grad_py)
####################################################################


####################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
####################################################################
def coriolis_and_spherical_UWIND_py(
        COLP, COLP_im1,
        VWIND, VWIND_im1,
        VWIND_jp1, VWIND_im1_jp1,
        UWIND, UWIND_im1,
        UWIND_ip1,
        corf_is, lat_is_rad,
        dlon_rad, dlat_rad):
    """
    Calculate coriolis acceleration and spherical grid conversion
    terms for UWIND.
    """
    return(
        con_rE*dlon_rad*dlat_rad/wp(2.)*(
          COLP_im1 * 
        ( VWIND_im1 + VWIND_im1_jp1 )/wp(2.) * 
        ( corf_is * con_rE *
          cos(lat_is_rad) + 
          ( UWIND_im1 + UWIND )/wp(2.) * 
          sin(lat_is_rad) )
        + COLP  * 
        ( VWIND + VWIND_jp1 )/wp(2.) * 
        ( corf_is * con_rE * 
          cos(lat_is_rad) + 
          ( UWIND + UWIND_ip1 )/wp(2.) * 
          sin(lat_is_rad) )
        )
    )


def add_up_tendencies_py(
            UFLX,
            UFLX_im1, UFLX_ip1, UFLX_jm1, UFLX_jp1,
            VWIND, VWIND_im1,
            VWIND_jp1, VWIND_im1_jp1,
            UWIND, UWIND_im1,
            UWIND_ip1,
            PHI, PHI_im1,
            POTT, POTT_im1,
            PVTF, PVTF_im1,
            PVTFVB, PVTFVB_im1, 
            PVTFVB_im1_kp1, PVTFVB_kp1,
            WWIND_UWIND, WWIND_UWIND_kp1,

            COLP, COLP_im1,

            corf_is,
            lat_is_rad,
            dlon_rad, dlat_rad,
            dyis,

            dsigma, sigma_vb, sigma_vb_kp1):
    """
    Compute and add up all tendency contributions of UFLUX.
    """

    dUFLXdt = wp(0.)

    if i_UVFLX_main_switch:
        # HORIZONTAL ADVECTION
        if i_UVFLX_hor_adv:
            pass
        # VERTICAL ADVECTION
        if i_UVFLX_vert_adv:
            dUFLXdt = dUFLXdt + (
                (WWIND_UWIND - WWIND_UWIND_kp1 ) / dsigma)
        # CORIOLIS AND SPHERICAL GRID CONVERSION
        if i_UVFLX_coriolis:
            dUFLXdt = dUFLXdt + coriolis_and_spherical_UWIND(
                COLP, COLP_im1,
                VWIND, VWIND_im1,
                VWIND_jp1, VWIND_im1_jp1,
                UWIND, UWIND_im1,
                UWIND_ip1,
                corf_is, lat_is_rad,
                dlon_rad, dlat_rad)
        # PRESSURE GRADIENT
        if i_UVFLX_pre_grad:
            dUFLXdt = dUFLXdt + pre_grad(
                PHI, PHI_im1, COLP, COLP_im1,
                POTT, POTT_im1,
                PVTF, PVTF_im1,
                PVTFVB, PVTFVB_im1,
                PVTFVB_im1_kp1, PVTFVB_kp1,
                dsigma, sigma_vb, sigma_vb_kp1,
                dyis)
        # NUMERICAL HORIZONTAL DIFUSION
        if i_UVFLX_num_dif and (UVFLX_dif_coef > wp(0.)):
            dUFLXdt = dUFLXdt + num_dif(
                UFLX, UFLX_im1, UFLX_ip1,
                UFLX_jm1, UFLX_jp1,
                UVFLX_dif_coef)

    return(dUFLXdt)






####################################################################
### SPECIALIZE FOR GPU
####################################################################
num_dif = njit(num_dif_py, device=True, inline=True)
pre_grad = njit(pre_grad_py, device=True, inline=True)
add_up_tendencies = njit(add_up_tendencies_py, device=True,
                        inline=True)
coriolis_and_spherical_UWIND = njit(coriolis_and_spherical_UWIND_py,
                        device=True, inline=True)

def launch_cuda_main_kernel(dUFLXdt, UFLX,
                        UWIND, VWIND, 
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        corf_is, lat_is_rad,
                        dlon_rad, dlat_rad,
                        dyis,
                        dsigma, sigma_vb):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nxs+nb and j >= nb and j < ny+nb:
        dUFLXdt[i  ,j  ,k] = \
            add_up_tendencies(
            # 3D
            UFLX       [i  ,j  ,k],
            UFLX       [i-1,j  ,k  ], UFLX       [i+1,j  ,k  ],
            UFLX       [i  ,j-1,k  ], UFLX       [i  ,j+1,k  ],
            VWIND      [i  ,j  ,k  ], VWIND      [i-1,j  ,k  ],
            VWIND      [i  ,j+1,k  ], VWIND      [i-1,j+1,k  ],
            UWIND      [i  ,j  ,k  ], UWIND      [i-1,j  ,k  ],
            UWIND      [i+1,j  ,k  ], 
            PHI        [i  ,j  ,k  ], PHI        [i-1,j  ,k  ],
            POTT       [i  ,j  ,k  ], POTT       [i-1,j  ,k  ],
            PVTF       [i  ,j  ,k  ], PVTF       [i-1,j  ,k  ],
            PVTFVB     [i  ,j  ,k  ], PVTFVB     [i-1,j  ,k  ],
            PVTFVB     [i-1,j  ,k+1], PVTFVB     [i  ,j  ,k+1],
            WWIND_UWIND[i  ,j  ,k  ], WWIND_UWIND[i  ,j  ,k+1],
            # 2D
            COLP       [i  ,j  ,0  ], COLP       [i-1,j  ,0  ],
            # GR horizontal
            corf_is    [i  ,j  ,0  ], lat_is_rad [i  ,j  ,0  ],
            dlon_rad   [i  ,j  ,0  ], dlat_rad   [i  ,j  ,0  ],
            dyis       [i  ,j  ,0  ],
            # GR vertical
            dsigma     [0  ,0  ,k  ], sigma_vb   [0  ,0  ,k  ],
            sigma_vb   [0  ,0  ,k+1])

UFLX_tendency_gpu = cuda.jit(cuda_kernel_decorator(
                            launch_cuda_main_kernel))(
                            launch_cuda_main_kernel)

####################################################################


####################################################################
### SPECIALIZE FOR CPU
####################################################################
num_dif = njit(num_dif_py)
pre_grad = njit(pre_grad_py)
add_up_tendencies = njit(add_up_tendencies_py)
coriolis_and_spherical_UWIND = njit(coriolis_and_spherical_UWIND_py)

def launch_numba_cpu_main(dUFLXdt, UFLX,
                        UWIND, VWIND, 
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        corf_is, lat_is_rad,
                        dlon_rad, dlat_rad,
                        dyis,
                        dsigma, sigma_vb):

    for i in prange(nb,nxs+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(0),nz):
                dUFLXdt[i  ,j  ,k] = \
                    add_up_tendencies(
            # 3D
            UFLX[i  ,j  ,k],
            UFLX       [i-1,j  ,k  ], UFLX       [i+1,j  ,k  ],
            UFLX       [i  ,j-1,k  ], UFLX       [i  ,j+1,k  ],
            VWIND      [i  ,j  ,k  ], VWIND      [i-1,j  ,k  ],
            VWIND      [i  ,j+1,k  ], VWIND      [i-1,j+1,k  ],
            UWIND      [i  ,j  ,k  ], UWIND      [i-1,j  ,k  ],
            UWIND      [i+1,j  ,k  ], 
            PHI        [i  ,j  ,k  ], PHI        [i-1,j  ,k  ],
            POTT       [i  ,j  ,k  ], POTT       [i-1,j  ,k  ],
            PVTF       [i  ,j  ,k  ], PVTF       [i-1,j  ,k  ],
            PVTFVB     [i  ,j  ,k  ], PVTFVB     [i-1,j  ,k  ],
            PVTFVB     [i-1,j  ,k+1], PVTFVB     [i  ,j  ,k+1],
            WWIND_UWIND[i  ,j  ,k  ], WWIND_UWIND[i  ,j  ,k+1],
            # 2D
            COLP       [i  ,j  ,0  ], COLP       [i-1,j  ,0  ],
            # GR horizontal
            corf_is    [i  ,j  ,0  ], lat_is_rad [i  ,j  ,0  ],
            dlon_rad   [i  ,j  ,0  ], dlat_rad   [i  ,j  ,0  ],
            dyis       [i  ,j  ,0  ],
            # GR vertical
            dsigma     [0  ,0  ,k  ], sigma_vb   [0  ,0  ,k  ],
            sigma_vb   [0  ,0  ,k+1])

UFLX_tendency_cpu = njit(parallel=True)(launch_numba_cpu_main)
