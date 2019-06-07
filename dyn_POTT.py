#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190604
License:            MIT

Computation of potential virtual temperature (POTT) tendency
(dPOTTdt) according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.4, page 213

HISTORY
- 20190604: CH  First implementation.
###############################################################################
"""
import time
import numpy as np
from numba import cuda, njit, prange, vectorize

from namelist import (i_POTT_main_switch,
                      i_POTT_hor_adv, i_POTT_vert_adv, i_POTT_num_dif)
from io_read_namelist import (i_POTT_radiation, i_POTT_microphys,
                              wp, wp_int, gpu_enable)
from main_grid import nx,nxs,ny,nys,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator

from dyn_functions import (hor_adv_py, vert_adv_py, 
                            num_dif_pw_py)
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################
def radiation_py(dPOTTdt_RAD, COLP):
    return(dPOTTdt_RAD * COLP)


def microphysics():
    raise NotImplementedError()
#    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                        dPOTTdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries




def add_up_tendencies_py(
            POTT, POTT_im1, POTT_ip1, POTT_jm1, POTT_jp1,
            UFLX, UFLX_ip1, VFLX, VFLX_jp1,
            COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1,
            POTTVB, POTTVB_kp1, WWIND, WWIND_kp1,
            COLP_NEW, dPOTTdt_RAD, A, dsigma, POTT_dif_coef, k):

    dPOTTdt = wp(0.)

    if i_POTT_main_switch:
        # HORIZONTAL ADVECTION
        if i_POTT_hor_adv:
            dPOTTdt = dPOTTdt + hor_adv(
                POTT,
                POTT_im1, POTT_ip1,
                POTT_jm1, POTT_jp1,
                UFLX, UFLX_ip1,
                VFLX, VFLX_jp1,
                A)
        # VERTICAL ADVECTION
        if i_POTT_vert_adv:
            dPOTTdt = dPOTTdt + vert_adv(
                POTTVB, POTTVB_kp1,
                WWIND, WWIND_kp1,
                COLP_NEW, dsigma, k)
        # NUMERICAL HORIZONTAL DIFUSION
        if i_POTT_num_dif and (POTT_dif_coef > wp(0.)):
            dPOTTdt = dPOTTdt + num_dif(
                POTT, POTT_im1, POTT_ip1,
                POTT_jm1, POTT_jp1,
                COLP, COLP_im1, COLP_ip1,
                COLP_jm1, COLP_jp1,
                POTT_dif_coef)
        # RADIATION
        if i_POTT_radiation:
            dPOTTdt = dPOTTdt + radiation(dPOTTdt_RAD, COLP)

    return(dPOTTdt)






###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
hor_adv = njit(hor_adv_py, device=True, inline=True)
num_dif = njit(num_dif_pw_py, device=True, inline=True)
vert_adv = njit(vert_adv_py, device=True, inline=True)
radiation = njit(radiation_py, device=True, inline=True)
add_up_tendencies = njit(add_up_tendencies_py, device=True, inline=True)

def launch_cuda_kernel(A, dsigma, POTT_dif_coef,
                        dPOTTdt, POTT, UFLX, VFLX, COLP,
                         POTTVB, WWIND, COLP_NEW, dPOTTdt_RAD):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dPOTTdt[i  ,j  ,k] = \
            add_up_tendencies(POTT[i  ,j  ,k],
            POTT[i-1,j  ,k], POTT[i+1,j  ,k],
            POTT[i  ,j-1,k], POTT[i  ,j+1,k],
            UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
            VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
            COLP[i  ,j  ,0],
            COLP[i-1,j  ,0], COLP[i+1,j  ,0],
            COLP[i  ,j-1,0], COLP[i  ,j+1,0],
            POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
            WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
            COLP_NEW[i  ,j  ,0], dPOTTdt_RAD[i  ,j  ,k],
            A[i  ,j  ,0],
            dsigma[0  ,0  ,k], POTT_dif_coef[0  ,0  ,k], k)



if gpu_enable:
    POTT_tendency_gpu = cuda.jit(cuda_kernel_decorator(launch_cuda_kernel))\
                                (launch_cuda_kernel)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################
hor_adv = njit(hor_adv_py)
vert_adv = njit(vert_adv_py)
num_dif = njit(num_dif_pw_py)
radiation = njit(radiation_py)
add_up_tendencies = njit(add_up_tendencies_py)

def launch_numba_cpu(A, dsigma, POTT_dif_coef,
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW, dPOTTdt_RAD):

    for i in prange(nb,nx+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(0),nz):
                dPOTTdt[i  ,j  ,k] = \
                    add_up_tendencies(POTT[i  ,j  ,k],
                        POTT[i-1,j  ,k], POTT[i+1,j  ,k],
                        POTT[i  ,j-1,k], POTT[i  ,j+1,k],
                        UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
                        VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
                        COLP[i  ,j  ,0],
                        COLP[i-1,j  ,0], COLP[i+1,j  ,0],
                        COLP[i  ,j-1,0], COLP[i  ,j+1,0],
                        POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
                        WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
                        COLP_NEW[i  ,j  ,0], dPOTTdt_RAD[i  ,j  ,k],
                        A[i  ,j  ,0],
                        dsigma[0  ,0  ,k], POTT_dif_coef[0  ,0  ,k], k)


POTT_tendency_cpu = njit(parallel=True)(launch_numba_cpu)





