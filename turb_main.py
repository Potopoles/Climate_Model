#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190607
Last modified:      20190610
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
###############################################################################

###############################################################################
# NAMELIST
###############################################################################

# constant values


class Turbulence:

    fields_main = ['PHI', 'PHIVB', 'HSURF', 'RHO', 'RHOVB', 'COLP',
                   'QV', 'dQVdt_TURB', 'KHEAT', 'SQVFLX']


    def __init__(self, GR, target):

        self.target = target


    def compute_turbulence(self, GR, PHI, PHIVB, HSURF,
                           RHO, RHOVB, COLP,
                           QV, dQVdt_TURB, KHEAT, SQVFLX):
        if self.target == GPU:
            compute_turbulence_gpu[bpg, tpb](
                            PHI, PHIVB, HSURF, RHO, RHOVB, COLP,
                            QV, dQVdt_TURB, KHEAT, SQVFLX)


        elif self.target == CPU:
            pass
            #advance_timestep_srfc_cpu(SOILTEMP,
            #                       LWFLXNET, SWFLXNET, SOILCP,
            #                       SOILRHO, SOILDEPTH, OCEANMASK,
            #                       SURFALBEDSW, SURFALBEDLW, GR.dt)
        




