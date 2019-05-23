#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_POTT.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190523
License:            MIT

Computation of potential virtual temperature (POTT) tendency
(dPOTTdt) according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.4, page 213
"""
import time
import numpy as np
import cupy as cp
from numba import cuda, njit, prange, vectorize

from namelist import (POTT_dif_coef,
                    i_POTT_main_switch,
                    i_POTT_radiation, i_POTT_microphys,
                    i_POTT_hor_adv, i_POTT_vert_adv, i_POTT_num_dif)
from org_namelist import (wp, wp_int, wp_old)
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator

from tendency_functions import (hor_adv_py, vert_adv_py, 
                                num_dif_pw_py)
####################################################################


####################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
####################################################################
def radiation():
    raise NotImplementedError()
#    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                        dPOTTdt_RAD[i-1,j-1,k]*COLP[i,j] # TODO add boundaries


def microphysics():
    raise NotImplementedError()
#    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                        dPOTTdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries




def add_up_tendencies_py(
            POTT, POTT_im1, POTT_ip1, POTT_jm1, POTT_jp1,
            UFLX, UFLX_ip1, VFLX, VFLX_jp1,
            COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1, A,
            POTTVB, POTTVB_kp1, WWIND, WWIND_kp1,
            COLP_NEW, dsigma, k):

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

    return(dPOTTdt)






####################################################################
### SPECIALIZE FOR GPU
####################################################################
hor_adv = njit(hor_adv_py, device=True, inline=True)
num_dif = njit(num_dif_pw_py, device=True, inline=True)
vert_adv = njit(vert_adv_py, device=True, inline=True)
add_up_tendencies = njit(add_up_tendencies_py, device=True, inline=True)

def launch_cuda_kernel(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                     POTTVB, WWIND, COLP_NEW, dsigma):

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
            COLP[i  ,j-1,0], COLP[i  ,j+1,0], A[i  ,j  ,0],
            POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
            WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
            COLP_NEW[i  ,j  ,0], dsigma[0  ,0  ,k], k)


POTT_tendency_gpu = cuda.jit(cuda_kernel_decorator(launch_cuda_kernel))\
                            (launch_cuda_kernel)



####################################################################
### SPECIALIZE FOR CPU
####################################################################
hor_adv = njit(hor_adv_py)
vert_adv = njit(vert_adv_py)
num_dif = njit(num_dif_pw_py)
add_up_tendencies = njit(add_up_tendencies_py)

def launch_numba_cpu(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                 POTTVB, WWIND, COLP_NEW, dsigma):

    for k in prange(wp_int(0),nz):
        for i in range(nb,nx+nb):
            for j in range(nb,ny+nb):

                dPOTTdt[i  ,j  ,k] = \
                    add_up_tendencies(POTT[i  ,j  ,k],
                        POTT[i-1,j  ,k], POTT[i+1,j  ,k],
                        POTT[i  ,j-1,k], POTT[i  ,j+1,k],
                        UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
                        VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
                        COLP[i  ,j  ,0],
                        COLP[i-1,j  ,0], COLP[i+1,j  ,0],
                        COLP[i  ,j-1,0], COLP[i  ,j+1,0], A[i  ,j  ,0],
                        POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
                        WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
                        COLP_NEW[i  ,j  ,0], dsigma[0  ,0  ,k], k)


POTT_tendency_cpu = njit(parallel=True)(launch_numba_cpu)





####################################################################
### DEBUGGING EXAMPLE
####################################################################
def run(comp_mode):

    from org_tendencies import TendencyFactory

    if comp_mode in [2,0]:
        import cupy as cp
    else:
        import numpy as cp


    i = slice(nb,nx+nb)
    j = slice(nb,ny+nb)


    COLP        = cp.full( ( nx +2*nb, ny +2*nb,   1      ), 
                             np.nan,   dtype=wp)
    A = cp.full_like(COLP, np.nan)
    COLP_NEW = cp.full_like(COLP, np.nan)
    POTT        = cp.full( ( nx +2*nb, ny +2*nb, nz  ), 
                        np.nan, dtype=wp)
    dPOTTdt = cp.full_like(POTT, np.nan)
    UFLX        = cp.full( ( nxs +2*nb, ny +2*nb, nz  ), 
                        np.nan, dtype=wp)
    VFLX        = cp.full( ( nx +2*nb, nys +2*nb, nz  ), 
                        np.nan, dtype=wp)

    POTTVB        = cp.full( ( nx +2*nb, ny +2*nb, nzs  ), 
                        np.nan, dtype=wp)
    WWIND = cp.full_like(POTTVB, np.nan)


    dsigma        = cp.full( ( 1, 1,   nz      ), 
                             np.nan,   dtype=wp)


    A[:] = 1.
    COLP[:] = 1.
    COLP_NEW[:] = 2.
    UFLX[:] = 3.
    for k in range(0,nz):
        for i in range(1,100):
            UFLX[i,:,k] += i/(k+1)
    VFLX[:] = 3.
    POTT[:] = 2.
    for k in range(0,nz):
        for i in range(100,200):
            POTT[i,:,k] += i/(k+1)

    POTTVB[:] = 2.
    WWIND[:] = 1.
    for k in range(0,nz):
        for i in range(300,200):
            POTTVB[i,:,k] += i/(k+1)
            WWIND[i,:,k] += 20*i/(k+1) - k
    dsigma[:] = 7.


    if comp_mode == 2:
        print('gpu')
        Tendencies = TendencyFactory(target='GPU')

        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        t0 = time.time()
        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        print(time.time() - t0)

        print(np.mean(cp.asnumpy(dPOTTdt[i,j,:])))
        print()

    elif comp_mode == 1:
        print('numba_par')
        Tendencies = TendencyFactory(target='CPU')

        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        t0 = time.time()
        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        print(time.time() - t0)
        print(np.mean(dPOTTdt[i,j,:]))
        print()



if __name__ == '__main__':
    
    run(2)
    run(1)




