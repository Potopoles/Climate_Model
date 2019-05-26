#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          CPU.py  
Author:             Christoph Heim (CH)
Date created:       20190526
Last modified:      20190526
License:            MIT

Functionality not yet defined.
Should contain:
- CPU helper functions
###############################################################################
"""
from numba import jit
from inspect import signature

from org_namelist import wp
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

exchange_BC_cpu = jit(parallel=True)(exchange_BC_cpu)

