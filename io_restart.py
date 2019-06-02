#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190602
License:            MIT

Functions to write restart files and load model from restart files.
###############################################################################
"""
import numpy as np
import os
import pickle

from namelist import i_comp_mode, i_radiation, njobs_rad
from io_read_namelist import wp, gpu_enable, GPU
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
    if gpu_enable:
        grf_gpu = GR.GRF[GPU]
        fields_gpu = F.device
        fields_cpu = F.host
        del GR.GRF[GPU]
        del F.device


    out = {}
    out['GR'] = GR
    out['F'] = F
    #out['RAD'] = RAD
    #out['SOIL'] = SOIL
    #out['MIC'] = MIC
    #out['TURB'] = TURB
    with open(filename, 'wb') as f:
        pickle.dump(out, f)

    # restore unpicklable GR objects for GPU 
    #if i_comp_mode == 2:
    if gpu_enable:
        GR.GRF[GPU] = grf_gpu
        F.device = fields_gpu

        #GR.stream      =   stream     
        #GR.zonal       =   zonal   
        #GR.zonals      =   zonals  
        #GR.zonalvb     =   zonalvb 
        #GR.merid       =   merid   
        #GR.merids      =   merids  
        #GR.meridvb     =   meridvb 
        #GR.Ad          =   Ad         
        #GR.dxjsd       =   dxjsd      
        #GR.corfd       =   corfd      
        #GR.corf_isd    =   corf_isd   
        #GR.lat_radd    =   lat_radd   
        #GR.latis_radd  =   latis_radd 
        #GR.dsigmad     =   dsigmad    
        #GR.sigma_vbd   =   sigma_vbd  


def load_restart_grid(dlat_deg, dlon_deg, nz):
    filename = '../restart/'+str(dlat_deg).zfill(2) + '_' +\
                            str(dlon_deg).zfill(2) + '_' +\
                            str(nz).zfill(3)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    else:
        raise ValueError('Restart File does not exist.')
    GR = inp['GR']
    return(GR)

def load_restart_fields(GR):
    filename = '../restart/'+str(GR.dlat_deg).zfill(2) + '_' +\
                            str(GR.dlon_deg).zfill(2) + '_' +\
                            str(GR.nz).zfill(3)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    F = inp['F']
    #RAD = inp['RAD']
    if i_radiation:
        F.RAD.done = 1
        F.RAD.njobs_rad = njobs_rad
    #SOIL = inp['SOIL'] 
    #MIC = inp['MIC'] 
    #TURB = inp['TURB'] 
    #return(F, RAD, SOIL, MIC, TURB)
    return(F)


