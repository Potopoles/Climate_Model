#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190701
License:            MIT

Functions to write restart files and load model from restart files or initial
condition (IC).
###############################################################################
"""
import numpy as np
import os
import pickle

from namelist import (i_comp_mode, i_radiation, i_surface_scheme, njobs_rad,
                    i_turbulence, i_microphysics, i_moist_main_switch)
from io_read_namelist import wp, gpu_enable, GPU, CPU
###############################################################################


def write_restart(GR, F):

    print('###########################################')
    print('###########################################')
    print('WRITE RESTART')
    print('###########################################')
    print('###########################################')
 
    filename = '../restart/'+str(GR.dlat_deg).zfill(2) + '_' +\
                            str(GR.dlon_deg).zfill(2) + '_' +\
                            str(GR.nz).zfill(3)+'.pkl'

    ## set values for certain variables
    #RAD.done = 1 # make sure async radiation starts to run after loading

    # temporarily remove unpicklable GR objects for GPU 
    #if gpu_enable:
    grf_gpu = GR.GRF[GPU]
    fields_gpu = F.device
    del GR.GRF[GPU]
    del F.device

    out = {}
    out['GR'] = GR
    out['F'] = F
    with open(filename, 'wb') as f:
        pickle.dump(out, f)

    # restore unpicklable GR objects for GPU 
    #if gpu_enable:
    GR.GRF[GPU] = grf_gpu
    F.device = fields_gpu



def load_restart_grid(dlat_deg, dlon_deg, nz):
    filename = '../restart/'+str(dlat_deg).zfill(2) + '_' +\
                            str(dlon_deg).zfill(2) + '_' +\
                            str(nz).zfill(3)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
        GR = inp['GR']
        return(GR)
    else:
        raise ValueError('Restart File does not exist.')

def load_existing_fields(GR, directory='restart'):
    filename = os.path.join('..',directory,
                            str(GR.dlat_deg).zfill(2) + '_' +\
                            str(GR.dlon_deg).zfill(2) + '_' +\
                            str(GR.nz).zfill(3)+'.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    F = inp['F']

    # SPECIFIC NAMELIST ADJUSTMENTS THAT CAN HAVE BEEN CHANGED:
    if gpu_enable:
        target=GPU
    else:
        target=CPU

    ## SURFACE
    if i_surface_scheme:
        F.SURF.target = target

    ## TURBULENCE 
    if i_turbulence:
        F.TURB.target = target

    ## MOISTURE & MICROPHYSICS
    if i_moist_main_switch or i_microphysics:
        F.MIC.target = target

    if i_radiation:
        F.RAD.done = 1
        F.RAD.njobs_rad = njobs_rad
    return(F)


