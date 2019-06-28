#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190531
License:            MIT

Set all constants.
###############################################################################
"""
from io_read_namelist import wp
###############################################################################
# DYNAMICS
con_g = wp(9.81)
con_rE = wp(6371000)
con_omega = wp(7.292115E-5)
con_Rd = wp(287.058)
con_cp = wp(1005 )
con_kappa = wp(con_Rd/con_cp)

# MOISTURE
# latent heat of evaporation [J kg-1]
con_Lh = wp(2264E3) 


# RADIATION
solar_constant_0 = wp(1365)
con_h = wp(6.6256E-34) # J*s Planck's constant
con_c = wp(2.9979E8) # m/s Speed of light
con_kb = wp(1.38E-23) # J/K Boltzmann's constant
