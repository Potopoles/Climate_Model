#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          dyn_diagnostics.py  
Author:             Christoph Heim (CH)
Date created:       20190528
Last modified:      20190528
License:            MIT

Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.X, page XXX
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

#from dyn_functions import ()
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################







###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
def diag_PVTF_gpu(COLP, PVTF, PVTFVB, sigma_vb):


    nx = PVTF.shape[0] - 2
    ny = PVTF.shape[1] - 2
    nz = PVTF.shape[2]
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        
        pairvb_km12 = pair_top + sigma_vb[0,0,k  ] * COLP[i,j,0]
        pairvb_kp12 = pair_top + sigma_vb[0,0,k+1] * COLP[i,j,0]
        
        PVTF[i,j,k] = 1./(1.+con_kappa) * \
                    ( pow( pairvb_kp12/100000. , con_kappa ) * pairvb_kp12 - \
                      pow( pairvb_km12/100000. , con_kappa ) * pairvb_km12 ) / \
                    ( pairvb_kp12 - pairvb_km12 )

        PVTFVB[i,j,k] = pow( pairvb_km12/100000. , con_kappa )
        if k == nz-1:
            PVTFVB[i,j,k+1] = pow( pairvb_kp12/100000. , con_kappa )

    cuda.syncthreads()
diag_PVTF_gpu = cuda.jit(cuda_kernel_decorator(diag_PVTF_gpu))(diag_PVTF_gpu)


def diag_PHI_gpu(PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF):
    nx  = PHIVB.shape[0] - 2
    ny  = PHIVB.shape[1] - 2
    nzs = PHIVB.shape[2]
    i, j, ks = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        kiter = nzs-1
        if ks == kiter:
            PHIVB[i,j,ks] = HSURF[i,j,0]*con_g
        kiter = kiter - 1
        cuda.syncthreads()

        while kiter >= 0:
            if ks == kiter:
                PHI  [i,j,ks] = PHIVB[i,j,ks+1] - con_cp*  \
                                        ( POTT[i,j,ks] * (   PVTF  [i,j,ks  ] \
                                                           - PVTFVB[i,j,ks+1] ) )
                PHIVB[i,j,ks] = PHI  [i,j,ks  ] - con_cp * \
                                        ( POTT[i,j,ks] * (   PVTFVB[i,j,ks  ] \
                                                           - PVTF  [i,j,ks  ] ) )

            kiter = kiter - 1
            cuda.syncthreads()
diag_PHI_gpu = cuda.jit(cuda_kernel_decorator(diag_PHI_gpu))(diag_PHI_gpu)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################

#diag_PVTF_cpu = njit(parallel=True)(diag_PVTF_cpu)





