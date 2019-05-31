#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          misc_boundaries.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190531
License:            MIT

Functions to implement lateral boundary conditions for both GPU and CPU.
###############################################################################
"""
import numpy as np
from numba import njit, cuda

from io_read_namelist import wp
from misc_gpu_functions import cuda_kernel_decorator
from grid import nx,nxs,ny,nys,nz,nzs,nb
###############################################################################

def exchange_BC_cpu(FIELD):

    fnx,fny,fnz = FIELD.shape

    # zonal boundaries
    if fnx == nxs+2*nb: # staggered in x
        FIELD[0,:,:] = FIELD[nxs-1,:,:] 
        FIELD[nxs,:,:] = FIELD[1,:,:] 
    else:     # unstaggered in x
        FIELD[0,:,:] = FIELD[nx,:,:] 
        FIELD[nx+1,:,:] = FIELD[1,:,:] 

    # meridional boundaries
    if fny == nys+2*nb: # staggered in y
        for j in [0,1,nys,nys+1]:
            FIELD[:,j,:] = wp(0.)
    else:     # unstaggered in y
        FIELD[:,0,:] = FIELD[:,1,:] 
        FIELD[:,ny+1,:] = FIELD[:,ny,:] 
exchange_BC_cpu = njit(parallel=True)(exchange_BC_cpu)




def exchange_BC_gpu(FIELD):

    fnx,fny,fnz = FIELD.shape

    i, j, k = cuda.grid(3)

    if k < fnz:
        # zonal boundaries
        if fnx == nxs+2*nb: # staggered in x
            if i == 0:
                FIELD[i,j,k] = FIELD[nxs-1,j,k] 
            elif i == nxs:
                FIELD[i,j,k] = FIELD[1,j,k] 

        else:     # unstaggered in x
            if i == 0:
                FIELD[i,j,k] = FIELD[nx,j,k] 
            elif i == nx+1:
                FIELD[i,j,k] = FIELD[1,j,k] 

        # meridional boundaries
        if fny == nys+2*nb: # staggered in y
            if j == 0 or j == 1 or j == nys or j == nys+1:
                FIELD[i,j,k] = 0.

        else:     # unstaggered in y
            if j == 0:
                FIELD[i,j,k] = FIELD[i,1,k] 
            elif j == ny+1:
                FIELD[i,j,k] = FIELD[i,ny,k] 
exchange_BC_gpu = cuda.jit(cuda_kernel_decorator(
                            exchange_BC_gpu))(exchange_BC_gpu)

