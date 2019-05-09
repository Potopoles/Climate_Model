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
import numpy as np
from namelist import (working_precision)

####################################################################
# COMPUTATION
####################################################################
if working_precision == 'float32':
    wp_3D = 'float32[:,:,:]'
    wp = np.float32
    wp_int = np.int32
    # TODO
    wp_old = 'float32'
elif working_precision == 'float64':
    wp_3D = 'float64[:,:,:]'
    wp = np.float64
    wp_int = np.int64
    # TODO
    wp_old = 'float64'
