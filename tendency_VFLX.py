#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_VFLX.py  
Author:             Christoph Heim (CH)
Date created:       20190511
Last modified:      20190511
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
from numba import cuda, njit, prange, vectorize

from namelist import (UVFLX_dif_coef,
                    i_UVFLX_main_switch,
                    i_UVFLX_hor_adv, i_UVFLX_vert_adv,
                    i_UVFLX_coriolis,
                    i_UVFLX_num_dif, i_UVFLX_pre_grad)
from org_namelist import (wp, wp_int, wp_old)
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator

from tendency_functions import (num_dif_py, pre_grad_py)
####################################################################


####################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
####################################################################
def add_up_tendencies_py(
            VFLX, VFLX_im1, VFLX_ip1, VFLX_jm1, VFLX_jp1,
            PHI, PHI_jm1, COLP, COLP_jm1,
            POTT, POTT_jm1,
            PVTF, PVTF_jm1,
            PVTFVB, PVTFVB_jm1,
            PVTFVB_jm1_kp1, PVTFVB_kp1,
            dsigma, sigma_vb, sigma_vb_kp1,
            dxjs):

    dVFLXdt = wp(0.)

    if i_UVFLX_main_switch:
        # HORIZONTAL ADVECTION
        if i_UVFLX_hor_adv:
            pass
            #dPOTTdt = dPOTTdt + hor_adv(
            #    POTT,
            #    POTT_im1, POTT_ip1,
            #    POTT_jm1, POTT_jp1,
            #    VFLX, VFLX_ip1,
            #    VFLX, VFLX_jp1,
            #    A)
        # VERTICAL ADVECTION
        if i_UVFLX_vert_adv:
            pass
            #dPOTTdt = dPOTTdt + vert_adv(
            #    POTTVB, POTTVB_kp1,
            #    WWIND, WWIND_kp1,
            #    COLP_NEW, dsigma, k)
        # CORIOLIS AND SPHERICAL GRID CONVERSION
        if i_UVFLX_coriolis:
            pass
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
add_up_tendencies = njit(add_up_tendencies_py, device=True, inline=True)

def launch_cuda_kernel(dVFLXdt, VFLX, PHI, COLP, POTT,
                        PVTF, PVTFVB, dsigma, sigma_vb, dxjs):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVFLXdt[i  ,j  ,k] = \
            add_up_tendencies(VFLX[i  ,j  ,k],
            VFLX    [i-1,j  ,k  ], VFLX    [i+1,j  ,k  ],
            VFLX    [i  ,j-1,k  ], VFLX    [i  ,j+1,k  ],
            PHI     [i  ,j  ,k  ], PHI     [i  ,j-1,k  ],
            COLP    [i  ,j  ,0  ], COLP    [i  ,j-1,0  ],
            POTT    [i  ,j  ,k  ], POTT    [i  ,j-1,k  ],
            PVTF    [i  ,j  ,k  ], PVTF    [i  ,j-1,k  ],
            PVTFVB  [i  ,j  ,k  ], PVTFVB  [i  ,j-1,k  ],
            PVTFVB  [i  ,j-1,k+1], PVTFVB  [i  ,j  ,k+1],
            dsigma  [0  ,0  ,k  ], sigma_vb[0  ,0  ,k  ],
            sigma_vb[0  ,0  ,k+1], dxjs    [i  ,j  ,k  ])

VFLX_tendency_gpu = cuda.jit(cuda_kernel_decorator(launch_cuda_kernel))\
                            (launch_cuda_kernel)



####################################################################
### SPECIALIZE FOR CPU
####################################################################
num_dif = njit(num_dif_py)
pre_grad = njit(pre_grad_py)
add_up_tendencies = njit(add_up_tendencies_py)

def launch_numba_cpu(dVFLXdt, VFLX):

    for i in prange(nb,nx+nb):
        for j in range(nb,nys+nb):
            for k in range(wp_int(0),nz):

                dVFLXdt[i  ,j  ,k] = \
                    add_up_tendencies(VFLX[i  ,j  ,k],
                    VFLX[i-1,j  ,k], VFLX[i+1,j  ,k],
                    VFLX[i  ,j-1,k], VFLX[i  ,j+1,k])


VFLX_tendency_cpu = njit(parallel=True)(launch_numba_cpu)





