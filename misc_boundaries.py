#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190607
License:            MIT

Functions to implement lateral boundary conditions for both GPU and CPU.
###############################################################################
"""
import numpy as np
from numba import njit, cuda

from io_read_namelist import wp, gpu_enable
from main_grid import nx,nxs,ny,nys,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator
###############################################################################

def exchange_BC_cpu(FIELD):

    fnx,fny,fnz = FIELD.shape

    # zonal boundaries
    if fnx == nxs+2*nb: # staggered in x
        FIELD[0,:,:] = FIELD[nxs-1,:,:] 
        FIELD[nxs,:,:] = FIELD[1,:,:] 
        FIELD[nxs+1,:,:] = FIELD[2,:,:] 
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

    if i < fnx and j < fny and k < fnz:
        ## zonal boundaries
        if fnx == nxs+2*nb: # staggered in x
            if i == 0:
                FIELD[i,j,k] = FIELD[nxs-1,j,k] 
            elif i == nxs:
                FIELD[i,j,k] = FIELD[1,j,k] 
            elif i == nxs+1:
                FIELD[i,j,k] = FIELD[2,j,k] 

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

if gpu_enable:
    exchange_BC_gpu = cuda.jit(cuda_kernel_decorator(
                        exchange_BC_gpu))(exchange_BC_gpu)

