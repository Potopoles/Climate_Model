#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          misc_gpu_functions.py  
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190531
License:            MIT

Helper functions for usage on GPU.
###############################################################################
"""
from numba import cuda
from inspect import signature

from io_read_namelist import wp_str
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


