#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          GPU.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190528
License:            MIT

Functionality not yet defined.
Should contain:
- GPU helper functions
###############################################################################
"""
from numba import cuda
from inspect import signature

from org_namelist import wp_str
from grid import nx,nxs,ny,nys,nz,nzs,nb
###############################################################################


def cuda_kernel_decorator(function, non_3D={}):
    arguments = list(signature(function).parameters)
    n_input_args = len(arguments)
    decorator = 'void('
    for i in range(n_input_args):
        if arguments[i] not in non_3D.keys():
            decorator += wp_str + '[:,:,:],'
        else:
            decorator += non_3D[arguments[i]] + ','
    decorator = decorator[:-1]
    decorator += ')'
    return(decorator)


def set_equal(set_FIELD, get_FIELD):
    fnx,fny,fnz = set_FIELD.shape
    i, j, k = cuda.grid(3)
    if i < fnx and j < fny and k < fnz:
        set_FIELD[i,j,k] = get_FIELD[i,j,k] 
set_equal = cuda.jit(cuda_kernel_decorator(set_equal))(set_equal)


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




