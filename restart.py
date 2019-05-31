#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          restart.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190531
License:            MIT

Functions to write restart files and load model from restart files.
###############################################################################
"""
import numpy as np
import os
import pickle

from namelist import n_topo_smooth, tau_topo_smooth, comp_mode
from org_namelist import wp, pair_top
#from radiation.namelist_radiation import njobs_rad
from constants import con_g, con_Rd, con_kappa, con_cp
from boundaries import exchange_BC_rigid_y, exchange_BC_periodic_x
from boundaries import exchange_BC
###############################################################################


def write_restart(GR, CF, RAD, SOIL, MIC, TURB):

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
    if hasattr(GR, 'stream'):

        stream      =   GR.stream     
        zonal       =   GR.zonal   
        zonals      =   GR.zonals  
        zonalvb     =   GR.zonalvb 
        merid       =   GR.merid   
        merids      =   GR.merids  
        meridvb     =   GR.meridvb 
        Ad          =   GR.Ad         
        dxjsd       =   GR.dxjsd      
        corfd       =   GR.corfd      
        corf_isd    =   GR.corf_isd   
        lat_radd    =   GR.lat_radd   
        latis_radd  =   GR.latis_radd 
        dsigmad     =   GR.dsigmad    
        sigma_vbd   =   GR.sigma_vbd  

        del GR.stream     
        del GR.zonal   
        del GR.zonals  
        del GR.zonalvb 
        del GR.merid   
        del GR.merids  
        del GR.meridvb 
        del GR.Ad         
        del GR.dxjsd      
        del GR.corfd      
        del GR.corf_isd   
        del GR.lat_radd   
        del GR.latis_radd 
        del GR.dsigmad    
        del GR.sigma_vbd  

    out = {}
    out['GR'] = GR
    out['CF'] = CF
    out['RAD'] = RAD
    out['SOIL'] = SOIL
    out['MIC'] = MIC
    out['TURB'] = TURB
    with open(filename, 'wb') as f:
        pickle.dump(out, f)

    # restore unpicklable GR objects for GPU 
    if comp_mode == 2:
        GR.stream      =   stream     
        GR.zonal       =   zonal   
        GR.zonals      =   zonals  
        GR.zonalvb     =   zonalvb 
        GR.merid       =   merid   
        GR.merids      =   merids  
        GR.meridvb     =   meridvb 
        GR.Ad          =   Ad         
        GR.dxjsd       =   dxjsd      
        GR.corfd       =   corfd      
        GR.corf_isd    =   corf_isd   
        GR.lat_radd    =   lat_radd   
        GR.latis_radd  =   latis_radd 
        GR.dsigmad     =   dsigmad    
        GR.sigma_vbd   =   sigma_vbd  


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
    CF = inp['CF']
    RAD = inp['RAD']
    RAD.done = 1
    RAD.njobs_rad = njobs_rad
    SOIL = inp['SOIL'] 
    MIC = inp['MIC'] 
    TURB = inp['TURB'] 
    return(CF, RAD, SOIL, MIC, TURB)


