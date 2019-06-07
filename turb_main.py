#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190607
Last modified:      20190607 
License:            MIT

Main script to organize turbulent transport.
###############################################################################
"""
import numpy as np
from numba import cuda

#from namelist import i_comp_mode
#from io_read_namelist import wp, GPU, CPU, gpu_enable
#from main_grid import tpb_2D, bpg
#from srfc_timestep import advance_timestep_srfc_cpu
#if gpu_enable:
#    from srfc_timestep import advance_timestep_srfc_gpu
###############################################################################

###############################################################################
# NAMELIST
###############################################################################

# constant values

# initial values (kg/kg air equivalent)

class Surface:

    fields_timestep = []


    def __init__(self, GR, F, target):

        self.target = target
        




    def advance_timestep(self, GR ):

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




