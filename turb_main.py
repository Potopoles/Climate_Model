#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190607
Last modified:      20190616
License:            MIT

Main script to organize turbulent transport.
###############################################################################
"""
import numpy as np
from numba import cuda

from io_read_namelist import wp, GPU, CPU, gpu_enable
from main_grid import tpb, bpg
if gpu_enable:
    from turb_compute import compute_turbulence_gpu
from turb_compute import compute_turbulence_cpu
from misc_utilities import function_input_fields
###############################################################################

###############################################################################
# NAMELIST
###############################################################################

# constant values


class Turbulence:

    def __init__(self, GR, target):

        self.target = target
        self.fields_main = function_input_fields(self.compute_turbulence)


    def compute_turbulence(self, GR, KMOM, KHEAT, PHIVB, HSURF, PHI, QV,
                            WINDX, WINDY, POTTVB, POTT):
        if self.target == GPU:
            compute_turbulence_gpu[bpg, tpb](
                            KMOM, KHEAT, PHIVB, HSURF, PHI, QV,
                            WINDX, WINDY, POTTVB, POTT)

        elif self.target == CPU:
            compute_turbulence_cpu(
                            KMOM, KHEAT, PHIVB, HSURF, PHI, QV,
                            WINDX, WINDY, POTTVB, POTT)
    
        



