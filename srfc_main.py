#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190604
License:            MIT

Main script of surface scheme.
Currently the surface scheme is VERY simple.
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import i_comp_mode, i_radiation, i_microphysics
from io_read_namelist import wp, GPU, CPU, gpu_enable
from main_grid import tpb_2D, bpg
from srfc_timestep import advance_timestep_srfc_cpu
if gpu_enable:
    from srfc_timestep import advance_timestep_srfc_gpu
###############################################################################

###############################################################################
# NAMELIST
###############################################################################

# constant values
depth_soil      = wp(4   ) 
depth_ocean     = wp(50  )
con_cp_soil     = wp(2000) 
con_cp_ocean    = wp(4184)
rho_soil        = wp(3000)
rho_water       = wp(1000)

# initial values (kg/kg air equivalent)
moisture_ocean = wp(np.nan)
moisture_soil = wp(10.0)
evapity_thresh = wp(10.)

class Surface:

    fields_timestep = ['SOILTEMP', 'LWFLXNET', 'SWFLXNET', 'SOILCP',
                       'SOILRHO', 'SOILDEPTH', 'OCEANMASK',
                       'SURFALBEDSW', 'SURFALBEDLW']


    def __init__(self, GR, F, target):

        F.set(self.initial_conditions(GR, **F.get(F.field_groups[F.SRFC_FIELDS])))

        self.target = target
        


    def initial_conditions(self, GR, **fields):
        fields['OCEANMASK']  [fields['HSURF'][GR.ii,GR.jj,0]  > 100]  = 0
        fields['OCEANMASK']  [fields['HSURF'][GR.ii,GR.jj,0] <= 100] = 1
        
        fields['SOILDEPTH']  [fields['OCEANMASK'] == 0] = depth_soil
        fields['SOILDEPTH']  [fields['OCEANMASK'] == 1] = depth_ocean

        fields['SOILCP']     [fields['OCEANMASK'] == 0] = con_cp_soil
        fields['SOILCP']     [fields['OCEANMASK'] == 1] = con_cp_ocean

        fields['SOILRHO']    [fields['OCEANMASK'] == 0] = rho_soil
        fields['SOILRHO']    [fields['OCEANMASK'] == 1] = rho_water

        fields['SOILTEMP']   [:,:,0]                    = (295 - 
                                    np.sin(GR.lat_rad[GR.ii,GR.jj,0])**2*25)

        fields['SOILMOIST']  [fields['OCEANMASK'] == 0] = moisture_soil
        fields['SOILMOIST']  [fields['OCEANMASK'] == 1] = moisture_ocean

        fields['SOILEVAPITY'][fields['OCEANMASK'] == 0]  = 0.
        fields['SOILEVAPITY'][fields['OCEANMASK'] == 1]  = 1.

        self.calc_albedo(fields['SURFALBEDSW'], fields['SURFALBEDLW'],
                         fields['SOILTEMP'], fields['SOILMOIST'],
                         fields['OCEANMASK'])

        return(fields)



    def advance_timestep(self, GR, SOILTEMP, LWFLXNET, SWFLXNET,
                        SOILCP, SOILRHO, SOILDEPTH, OCEANMASK,
                        SURFALBEDSW, SURFALBEDLW):

        if self.target == GPU:
            advance_timestep_srfc_gpu[bpg, tpb_2D](SOILTEMP,
                                   LWFLXNET, SWFLXNET, SOILCP,
                                   SOILRHO, SOILDEPTH, OCEANMASK,
                                   SURFALBEDSW, SURFALBEDLW, GR.dt)
            #raise NotImplementedError()

        elif self.target == CPU:
            advance_timestep_srfc_cpu(SOILTEMP,
                                   LWFLXNET, SWFLXNET, SOILCP,
                                   SOILRHO, SOILDEPTH, OCEANMASK,
                                   SURFALBEDSW, SURFALBEDLW, GR.dt)
            #calc_albedo(SURFALBEDSW, SURFALBEDLW, SOILTEMP,
            #            SOILMOIST, OCEANMASK)



        #    # calc evaporation capacity
        #    CF.SOILEVAPITY[CF.OCEANMASK == 0] = \
        #             np.minimum(np.maximum(0, CF.SOILMOIST[CF.OCEANMASK == 0] \
        #                                                / evapity_thresh), 1)


        #    calc_evaporation_capacity_gpu[GR.riddim_xy_in, GR.blockdim_xy, GR.stream]\
        #                        (GF.SOILEVAPITY, GF.SOILMOIST, GF.OCEANMASK, GF.SOILTEMP)



    # TODO IS USED FOR INITIALIZATION ONLY. CHANGE TO NORMAL FUNCTION?
    def calc_albedo(self, SURFALBEDSW, SURFALBEDLW, SOILTEMP,
                    SOILMOIST, OCEANMASK):
        ## ALBEDO
        # forest
        SURFALBEDSW[:] = wp(0.2)
        SURFALBEDLW[:] = wp(0.0)
        # ocean
        SURFALBEDSW[OCEANMASK == 1] = wp(0.05)
        SURFALBEDLW[OCEANMASK == 1] = wp(0.00)
        # ice (land and sea)
        #SURFALBEDSW[(SOILTEMP[:,:,0] <= 273.15) & (SOILMOIST[:,:,0] > 10)] = 0.6
        #SURFALBEDLW[(SOILTEMP[:,:,0] <= 273.15) & (SOILMOIST[:,:,0] > 10)] = 0.2
        SURFALBEDSW[(SOILTEMP[:,:,0] <= wp(273.15))] = wp(0.6)
        SURFALBEDLW[(SOILTEMP[:,:,0] <= wp(273.15))] = wp(0.2)




