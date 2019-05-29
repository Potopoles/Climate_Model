#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          dyn_diagnostics.py  
Author:             Christoph Heim
Date created:       20190528
Last modified:      20190529
License:            MIT

Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.6, page 221ff
###############################################################################
"""
import time
import numpy as np
import cupy as cp
from numba import cuda, njit, prange
from math import pow

from namelist import (pair_top)
from org_namelist import (wp, wp_int, wp_old)
from constants import con_g, con_Rd, con_kappa, con_cp
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################







###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
def diag_PVTF_gpu(COLP, PVTF, PVTFVB, sigma_vb):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb and k < nz:
        
        pairvb_km12 = pair_top + sigma_vb[0,0,k  ] * COLP[i,j,0]
        pairvb_kp12 = pair_top + sigma_vb[0,0,k+1] * COLP[i,j,0]
        
        PVTF[i,j,k] = wp(1.)/(wp(1.)+con_kappa) * (
                pow( pairvb_kp12/wp(100000.) , con_kappa ) * pairvb_kp12 -
                pow( pairvb_km12/wp(100000.) , con_kappa ) * pairvb_km12
                        ) /( pairvb_kp12 - pairvb_km12 )

        PVTFVB[i,j,k] = pow( pairvb_km12/wp(100000.) , con_kappa )
        if k == nz-1:
            PVTFVB[i,j,k+1] = pow( pairvb_kp12/wp(100000.) , con_kappa )
diag_PVTF_gpu = cuda.jit(cuda_kernel_decorator(
                        diag_PVTF_gpu))(diag_PVTF_gpu)


def diag_PHI_gpu(PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb and k < nzs:
        kiter = nzs-1
        if k == kiter:
            PHIVB[i,j,k] = HSURF[i,j,0]*con_g
        kiter = kiter - 1
        cuda.syncthreads()

        while kiter >= 0:
            if k == kiter:
                PHI  [i,j,k] = PHIVB[i,j,k+1] - con_cp* (
                                POTT[i,j,k] * (   PVTF  [i,j,k  ]
                                                - PVTFVB[i,j,k+1] ) )
                PHIVB[i,j,k] = PHI  [i,j,k  ] - con_cp * (
                                POTT[i,j,k] * (   PVTFVB[i,j,k  ]
                                                - PVTF  [i,j,k  ] ) )

            kiter = kiter - 1
            cuda.syncthreads()
diag_PHI_gpu = cuda.jit(cuda_kernel_decorator(diag_PHI_gpu))(diag_PHI_gpu)



def diag_POTTVB_gpu(POTTVB, POTT, PVTF, PVTFVB):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb and k < nzs:
        if k > 0 and k < nzs-1:
            POTTVB[i,j,k] =   (
                    +   (PVTFVB[i,j,k] - PVTF  [i,j,k-1]) * POTT[i,j,k-1]
                    +   (PVTF  [i,j,k] - PVTFVB[i,j,k  ]) * POTT[i,j,k  ]
                            ) / (PVTF[i,j,k] - PVTF[i,j,k-1])
            if k == 1:
                # extrapolate model top POTTVB
                POTTVB[i,j,k-1] = POTT[i,j,k-1] - ( 
                                        POTTVB[i,j,k] - POTT[i,j,k-1] )
            elif k == nzs-2:
                # extrapolate model bottom POTTVB
                POTTVB[i,j,k+1] = POTT[i,j,k  ] - (
                                        POTTVB[i,j,k] - POTT[i,j,k  ] )
diag_POTTVB_gpu = cuda.jit(cuda_kernel_decorator(
                           diag_POTTVB_gpu))(diag_POTTVB_gpu)


###############################################################################
### SPECIALIZE FOR CPU
###############################################################################
def diag_PVTF_cpu(COLP, PVTF, PVTFVB, sigma_vb):

    for i in prange(nb,nx+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(0),nz):
        
                pairvb_km12 = pair_top + sigma_vb[0,0,k  ] * COLP[i,j,0]
                pairvb_kp12 = pair_top + sigma_vb[0,0,k+1] * COLP[i,j,0]
                
                PVTF[i,j,k] = wp(1.)/(wp(1.)+con_kappa) * (
                        pow( pairvb_kp12/wp(100000.) , con_kappa ) * pairvb_kp12 -
                        pow( pairvb_km12/wp(100000.) , con_kappa ) * pairvb_km12
                                ) /( pairvb_kp12 - pairvb_km12 )

                PVTFVB[i,j,k] = pow( pairvb_km12/wp(100000.) , con_kappa )
                if k == nz-1:
                    PVTFVB[i,j,k+1] = pow( pairvb_kp12/wp(100000.) , con_kappa )
diag_PVTF_cpu = njit(parallel=True)(diag_PVTF_cpu)



def diag_PHI_cpu(PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF):
    for i in prange(nb,nx+nb):
        for j in range(nb,ny+nb):
            k = nzs-1
            PHIVB[i,j,k] = HSURF[i,j,0]*con_g
            k = k - 1
            while k >= 0:
                PHI  [i,j,k] = PHIVB[i,j,k+1] - con_cp* (
                                POTT[i,j,k] * (   PVTF  [i,j,k  ]
                                                - PVTFVB[i,j,k+1] ) )
                PHIVB[i,j,k] = PHI  [i,j,k  ] - con_cp * (
                                POTT[i,j,k] * (   PVTFVB[i,j,k  ]
                                                - PVTF  [i,j,k  ] ) )
                k = k - 1
diag_PHI_cpu = njit(parallel=True)(diag_PHI_cpu)




def diag_POTTVB_cpu(POTTVB, POTT, PVTF, PVTFVB):
    for i in prange(nb,nx+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(1),nzs-1):
                POTTVB[i,j,k] =   (
                        +   (PVTFVB[i,j,k] - PVTF  [i,j,k-1]) * POTT[i,j,k-1]
                        +   (PVTF  [i,j,k] - PVTFVB[i,j,k  ]) * POTT[i,j,k  ]
                                ) / (PVTF[i,j,k] - PVTF[i,j,k-1])
                if k == 1:
                    # extrapolate model top POTTVB
                    POTTVB[i,j,k-1] = POTT[i,j,k-1] - ( 
                                            POTTVB[i,j,k] - POTT[i,j,k-1] )
                elif k == nzs-2:
                    # extrapolate model bottom POTTVB
                    POTTVB[i,j,k+1] = POTT[i,j,k  ] - (
                                            POTTVB[i,j,k] - POTT[i,j,k  ] )
diag_POTTVB_cpu = njit(parallel=True)(diag_POTTVB_cpu)
