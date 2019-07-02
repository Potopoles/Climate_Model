#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190701
Last modified:      20190701
License:            MIT

Namelist for surface scheme.
###############################################################################
"""
import numpy as np
from io_read_namelist import wp
###############################################################################
# constant values
depth_soil      = wp(4   ) 
depth_ocean     = wp(20  )
cp_soil         = wp(2000) 
cp_ocean        = wp(4184)
rho_soil        = wp(3000)
rho_water       = wp(1000)

# initial values (kg water)
moisture_ocean = wp(np.nan)
moisture_soil = wp(10.)
max_moisture_soil = wp(2)*moisture_soil
desert_moisture_thresh = wp(0.1)*moisture_soil

land_evap_resist = wp(0.3)

# bulk transfer coefficient for momentum [-]
DRAGCM = wp(0.01)
# bulk transfer coefficient for heat and moisture [-]
DRAGCH = wp(0.005)
