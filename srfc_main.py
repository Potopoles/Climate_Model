#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190601
License:            MIT

Main script of surface scheme.
Currently the surface scheme is VERY simple.
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import i_comp_mode, i_radiation, i_microphysics
from io_read_namelist import wp, GPU, CPU
from main_grid import tpb_2D, bpg
from srfc_timestep import advance_timestep_srfc_gpu, advance_timestep_srfc_cpu
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

    fields_timestep = ['SOILTEMP']


    def __init__(self, GR, F, target):

        ##############################################################################
        # SET INITIAL VALUES
        ##############################################################################
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



        ##############################################################################
    def advance_timestep(self, GR, SOILTEMP):

        if self.target == GPU:
            advance_timestep_srfc_gpu[bpg, tpb_2D](SOILTEMP, GR.dt)

        elif self.target == CPU:
            advance_timestep_srfc_cpu(SOILTEMP, GR.dt)

        #if i_comp_mode == 1:
        #    #dSOILTEMPdt = np.zeros( (GR.nx, GR.ny) , dtype=wp)

        #    if i_radiation > 0:
        #        dSOILTEMPdt = (CF.LWFLXNET[:,:,GR.nzs-1] + CF.SWFLXNET[:,:,GR.nzs-1])/ \
        #                        (CF.SOILCP * CF.SOILRHO * CF.SOILDEPTH)

        #    if i_microphysics > 0:
        #        dSOILTEMPdt = dSOILTEMPdt - ( MIC.surf_evap_flx * MIC.lh_cond_water ) / \
        #                                    (CF.SOILCP * CF.SOILRHO * CF.SOILDEPTH)

        #    CF.SOILTEMP[:,:,0] = CF.SOILTEMP[:,:,0] + GR.dt * dSOILTEMPdt

        #    self.calc_albedo(GR, CF)

        #    # calc evaporation capacity
        #    CF.SOILEVAPITY[CF.OCEANMASK == 0] = \
        #             np.minimum(np.maximum(0, CF.SOILMOIST[CF.OCEANMASK == 0] \
        #                                                / evapity_thresh), 1)


        #elif i_comp_mode == 2:

        #    #dSOILTEMPdt = cuda.device_array( (GR.nx, GR.ny, 1), dtype=GF.SOILTEMP.dtype)

        #    soil_temperature_euler_forward_gpu[GR.griddim_xy_in, GR.blockdim_xy, GR.stream]\
        #                        (dSOILTEMPdt, GF.SOILTEMP, GF.LWFLXNET, GF.SWFLXNET,
        #                        GF.SOILCP, GF.SOILRHO, GF.SOILDEPTH, GR.dt)
        #    if i_microphysics > 0:
        #        raise NotImplementedError()

        #    calc_albedo_gpu[GR.main_griddim_xy_in, GR.blockdim_xy, GR.stream]\
        #                        (GF.SURFALBEDSW, GF.SURFALBEDLW, GF.OCEANMASK, GF.SOILTEMP)

        #    calc_evaporation_capacity_gpu[GR.riddim_xy_in, GR.blockdim_xy, GR.stream]\
        #                        (GF.SOILEVAPITY, GF.SOILMOIST, GF.OCEANMASK, GF.SOILTEMP)



        ##############################################################################
    def calc_albedo(self, SURFALBEDSW, SURFALBEDLW, SOILTEMP, SOILMOIST, OCEANMASK):
        ## ALBEDO
        # forest
        SURFALBEDSW[:] = 0.2
        SURFALBEDLW[:] = 0.0
        # ocean
        SURFALBEDSW[OCEANMASK == 1] = 0.05
        SURFALBEDLW[OCEANMASK == 1] = 0.00
        # ice (land and sea)
        SURFALBEDSW[(SOILTEMP[:,:,0] <= 273.15) & (SOILMOIST[:,:,0] > 10)] = 0.6
        SURFALBEDLW[(SOILTEMP[:,:,0] <= 273.15) & (SOILMOIST[:,:,0] > 10)] = 0.2




