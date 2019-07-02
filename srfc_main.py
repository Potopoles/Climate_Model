#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190629
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
from srfc_namelist import (depth_soil, depth_ocean, cp_soil, cp_ocean,
                            rho_soil, rho_water, moisture_soil,
                            moisture_ocean, DRAGCM, DRAGCH)
from srfc_timestep import advance_timestep_srfc_cpu
if gpu_enable:
    from srfc_timestep import advance_timestep_srfc_gpu
from srfc_timestep import advance_timestep_srfc_cpu
from misc_utilities import function_input_fields
###############################################################################


class Surface:

    def __init__(self, GR, F, target):

        F.set(self.initial_conditions(GR, 
            **F.get(F.field_groups[F.SRFC_FIELDS], target=CPU)),
             target=CPU)

        self.target = target
        self.fields_timestep = function_input_fields(self.advance_timestep)
        


    def initial_conditions(self, GR, **fields):
        fields['OCEANMASK']  [fields['HSURF']  > 100]  = 0
        fields['OCEANMASK']  [fields['HSURF'] <= 100] = 1
        
        fields['SOILDEPTH']  [fields['OCEANMASK'] == 0] = depth_soil
        fields['SOILDEPTH']  [fields['OCEANMASK'] == 1] = depth_ocean

        fields['SOILCP']     [fields['OCEANMASK'] == 0] = cp_soil
        fields['SOILCP']     [fields['OCEANMASK'] == 1] = cp_ocean

        fields['SOILRHO']    [fields['OCEANMASK'] == 0] = rho_soil
        fields['SOILRHO']    [fields['OCEANMASK'] == 1] = rho_water

        fields['SOILTEMP']   [:,:,0] = (295 - np.sin(GR.lat_rad[:,:,0])**2*30)
        fields['SOILTEMP'] -= fields['HSURF']*wp(0.0100)

        fields['SOILMOIST']  [fields['OCEANMASK'] == 0] = moisture_soil
        fields['SOILMOIST']  [fields['OCEANMASK'] == 1] = moisture_ocean

        fields['SURFALBEDSW'][fields['OCEANMASK'] == 1]  = 0.05
        fields['SURFALBEDLW'][fields['OCEANMASK'] == 1]  = 0.00
        fields['SURFALBEDSW'][fields['OCEANMASK'] == 0]  = 0.20
        fields['SURFALBEDLW'][fields['OCEANMASK'] == 0]  = 0.00
        fields['SURFALBEDSW'][
                        fields['SOILTEMP'][:,:,0] <= wp(273.15)]  = 0.6
        fields['SURFALBEDLW'][
                        fields['SOILTEMP'][:,:,0] <= wp(273.15)]  = 0.0

        return(fields)



    def advance_timestep(self, GR, GRF, SOILTEMP, SOILMOIST,
                        LWFLXNET, SWFLXNET,
                        SOILCP, SOILRHO, SOILDEPTH, OCEANMASK,
                        SURFALBEDSW, SURFALBEDLW, TAIR, QV, WIND,
                        RHO, PSURF, COLP, WINDX, WINDY,
                        SMOMXFLX, SMOMYFLX, SSHFLX, SLHFLX,
                        RAIN):

        if self.target == GPU:
            advance_timestep_srfc_gpu[bpg, tpb_2D](SOILTEMP, SOILMOIST,
                                   LWFLXNET, SWFLXNET, SOILCP,
                                   SOILRHO, SOILDEPTH, OCEANMASK,
                                   SURFALBEDSW, SURFALBEDLW,
                                   TAIR, QV, WIND, RHO, PSURF, COLP,
                                   SMOMXFLX, SMOMYFLX, SSHFLX, SLHFLX,
                                   WINDX, WINDY, RAIN, DRAGCM, DRAGCH,
                                   GRF['A'], GR.dt)


        elif self.target == CPU:
            advance_timestep_srfc_cpu(SOILTEMP, SOILMOIST,
                                   LWFLXNET, SWFLXNET, SOILCP,
                                   SOILRHO, SOILDEPTH, OCEANMASK,
                                   SURFALBEDSW, SURFALBEDLW,
                                   TAIR, QV, WIND, RHO, PSURF, COLP,
                                   SMOMXFLX, SMOMYFLX, SSHFLX, SLHFLX,
                                   WINDX, WINDY, RAIN, DRAGCM, DRAGCH,
                                   GRF['A'], GR.dt)


        #    # calc evaporation capacity
        #    CF.SOILEVAPITY[CF.OCEANMASK == 0] = \
        #             np.minimum(np.maximum(0, CF.SOILMOIST[CF.OCEANMASK == 0] \
        #                                                / evapity_thresh), 1)


        #    calc_evaporation_capacity_gpu[GR.riddim_xy_in, GR.blockdim_xy, GR.stream]\
        #                        (GF.SOILEVAPITY, GF.SOILMOIST, GF.OCEANMASK, GF.SOILTEMP)



