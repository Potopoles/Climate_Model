#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190701
License:            MIT

Load namelist and process variables if necessary such that
they can be imported form here in other files.
###############################################################################
"""
import numba
import numpy as np
from namelist import (working_precision,
                    UVFLX_dif_coef, POTT_dif_coef, COLP_dif_coef,
                    moist_dif_coef,
                    i_comp_mode, nb, lon0_deg, lon1_deg,
                    pair_top, i_time_stepping, nzsoil,
                    i_radiation, i_surface_scheme, i_microphysics,
                    i_POTT_radiation, i_POTT_microphys,
                    i_moist_microphys,
                    i_COLP_main_switch, i_UVFLX_main_switch)
###############################################################################

###############################################################################
# GRID
###############################################################################
if nb > 1:
    raise NotImplementedError('nb > 1 not implemented everywhere.. :-(')
if lon0_deg != 0 or lon1_deg != 360:
    raise NotImplementedError('In x direction only periodic boundaries '+
                            'implemented.')
if nzsoil > 1:
    raise NotImplementedError('nzsoil > 1 not yet implemented')

###############################################################################
# COMPUTATION
###############################################################################
if i_time_stepping == 'RK4':
    raise NotImplementedError('Runge-Kutta 4th order not yet implemented')

# working precision wp
wp_int = np.int32
wp_bool = 'boolean'
if working_precision == 'float32':
    wp_str = 'float32'
    wp = np.float32
    wp_numba = numba.float32
elif working_precision == 'float64':
    wp_str = 'float64'
    wp = np.float64
    wp_numba = numba.float64

# GPU settings
if i_comp_mode == 2:
    gpu_enable = True
else:
    gpu_enable = False

# names of host and device
CPU = 'CPU'
GPU = 'GPU'

if i_POTT_radiation and not i_radiation:
    i_POTT_radiation = 0
if i_POTT_microphys and not i_microphysics:
    i_POTT_microphys = 0
if i_moist_microphys and not i_microphysics:
    i_moist_microphys = 0

#if i_UVFLX_main_switch and not i_COLP_main_switch:
#    raise ValueError('i_UVFLX_main_switch can only be 1 if i_COLP_main_switch'+
#                    ' == 1.')

###############################################################################
# PHYSICAL PROPERTIES
###############################################################################
# diffusion coefficients
UVFLX_dif_coef = wp(UVFLX_dif_coef)
POTT_dif_coef = wp(POTT_dif_coef)
if COLP_dif_coef > 0:
    raise NotImplementedError('no pressure difusion in gpu implemented')
moist_dif_coef = wp(moist_dif_coef)

# model top pressure 
pair_top = wp(pair_top)


if i_radiation and not i_surface_scheme:
    raise ValueError('i_radiation = 1 requires i_surface_scheme = 1')
