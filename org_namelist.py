#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          org_namelist.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190509
License:            MIT

Load namelist and process variables if necessary such that
they can be imported form here in other files.
"""
import numba
import numpy as np
from namelist import (working_precision,
                    UVFLX_dif_coef, POTT_dif_coef, COLP_dif_coef,
                    comp_mode)
####################################################################

####################################################################
# COMPUTATION
####################################################################
if working_precision == 'float32':
    wp_str = 'float32'
    wp = np.float32
    wp_numba = numba.float32
    wp_int = np.int32
    # TODO
    wp_old = 'float32'
elif working_precision == 'float64':
    wp_str = 'float64'
    wp = np.float64
    wp_numba = numba.float64
    wp_int = np.int32
    # TODO
    wp_old = 'float64'


if comp_mode == 2:
    gpu_enable = True
else:
    gpu_enable = False


# names of host and device
HOST = 'host'
DEVICE = 'device'


UVFLX_dif_coef = wp(UVFLX_dif_coef)
POTT_dif_coef = wp(POTT_dif_coef)
if COLP_dif_coef > 0:
    raise NotImplementedError('no pressure difusion in gpu implemented')
