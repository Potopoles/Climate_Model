#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          benchmark.py  
Author:             Christoph Heim (CH)
Date created:       20190521
Last modified:      20190521
License:            MIT

Script for testing speed
"""
#import time
#from datetime import timedelta
#import numpy as np
#
from grid import Grid
from fields import initialize_fields, CPU_Fields, GPU_Fields
from namelist import comp_mode
####################################################################

####################################################################
# CREATE MODEL GRID
####################################################################
# main grid
GR = Grid()
# optional subgrids for domain decomposition (not completly implemented)
#GR, subgrids = create_subgrids(GR, njobs)
subgrids = {} # option

####################################################################
# CREATE MODEL FIELDS
####################################################################
CF = CPU_Fields(GR, subgrids)
CF, RAD, SURF, MIC, TURB = initialize_fields(GR, subgrids, CF)
if comp_mode == 2:
    GF = GPU_Fields(GR, subgrids, CF)
else:
    GF = None



quit()
