#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190610
Last modified:      20190610
License:            MIT

Collection of meteorological functions.
###############################################################################
"""
from math import exp

from io_read_namelist import wp, wp_int
###############################################################################

def calc_specific_humidity_py(T, RH, p):
    """
    inputs:
        T: temperature in K
        RH: relative humidity in %
        p: air pressure in Pa
    output:
        q: specific humidity in kg/kg 
    """
    T -= wp(273.15)
    # pressure function
    f_p = wp(1.0016) + wp(3.15E-6)*p/wp(100) - wp(0.074)/(p/wp(100))
    # saturation vapor pressure [Pa]
    esw = wp(100)*f_p*wp(6.112)*exp((wp(17.62)*T)/(wp(243.12) + T))
    # specific humidity [g/kg]
    q = RH/wp(100)/p*wp(0.622)*esw
    return(q)

def calc_virtual_temperature_py(T, qv):
    """
    Compute virtual temperature.
    inputs:
        T: temperature in K
        qv: specific humidity in kg/kg
    outputs:
        T_v: virtual temperature in K
    """
    epsilon = wp(0.622)
    # TODO: do not assume mixing_ratio = qv
    mixing_ratio = qv
    T_v = T * (wp(1.) + mixing_ratio / epsilon) / ( wp(1.) + mixing_ratio ) 
    return(T_v)

