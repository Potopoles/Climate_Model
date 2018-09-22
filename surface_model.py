import numpy as np
import time
from namelist import wp, comp_mode, i_radiation, i_microphysics
from surface_model_cuda import soil_temperature_euler_forward_gpu,\
                              calc_albedo_gpu, calc_evaporation_capacity_gpu
from numba import cuda
if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np

####################################################################
# NAMELIST
####################################################################

nz_soil = 1

# constant values
depth_soil = 4
depth_ocean = 50
con_cp_soil = 2000 
con_cp_ocean = 4184
rho_soil = 3000
rho_water = 1000

# initial values (kg/kg air equivalent)
moisture_ocean = np.nan
moisture_soil = 10.0
evapity_thresh = 10

class surface:



    def __init__(self, GR, CF):

        ##############################################################################
        # SET INITIAL VALUES
        ##############################################################################

        CF.OCEANMASK  [CF.HSURF[GR.iijj] > 100]  = 0
        CF.OCEANMASK  [CF.HSURF[GR.iijj] <= 100] = 1

        CF.SOILDEPTH  [CF.OCEANMASK == 0] = depth_soil
        CF.SOILDEPTH  [CF.OCEANMASK == 1] = depth_ocean

        CF.SOILCP     [CF.OCEANMASK == 0] = con_cp_soil
        CF.SOILCP     [CF.OCEANMASK == 1] = con_cp_ocean

        CF.SOILRHO    [CF.OCEANMASK == 0] = rho_soil
        CF.SOILRHO    [CF.OCEANMASK == 1] = rho_water

        CF.SOILTEMP   [:,:,0] = 295 - np.sin(GR.lat_rad[GR.iijj])**2*25

        CF.SOILMOIST  [CF.OCEANMASK == 0] = moisture_soil
        CF.SOILMOIST  [CF.OCEANMASK == 1] = moisture_ocean

        CF.SOILEVAPITY[CF.OCEANMASK == 0]  = 0.
        CF.SOILEVAPITY[CF.OCEANMASK == 1]  = 1.

        self.calc_albedo(GR, CF)

        CF.RAINRATE = 0.
        CF.ACCRAIN = 0.




    def advance_timestep(self, GR, CF, GF, RAD):

        if comp_mode in [0,1]:
            dSOILTEMPdt = np.zeros( (GR.nx, GR.ny) , dtype=wp_np)

            if i_radiation > 0:
                dSOILTEMPdt = (CF.LWFLXNET[:,:,GR.nzs-1] + CF.SWFLXNET[:,:,GR.nzs-1])/ \
                                (CF.SOILCP * CF.SOILRHO * CF.SOILDEPTH)

            if i_microphysics > 0:
                dSOILTEMPdt = dSOILTEMPdt - ( MIC.surf_evap_flx * MIC.lh_cond_water ) / \
                                            (CF.SOILCP * CF.SOILRHO * CF.SOILDEPTH)

            CF.SOILTEMP[:,:,0] = CF.SOILTEMP[:,:,0] + GR.dt * dSOILTEMPdt

            self.calc_albedo(GR, CF)

            # calc evaporation capacity
            CF.SOILEVAPITY[CF.OCEANMASK == 0] = \
                     np.minimum(np.maximum(0, CF.SOILMOIST[CF.OCEANMASK == 0] \
                                                        / evapity_thresh), 1)


        elif comp_mode == 2:

            dSOILTEMPdt = cuda.device_array( (GR.nx, GR.ny, 1), dtype=GF.SOILTEMP.dtype)

            soil_temperature_euler_forward_gpu[GR.griddim_xy_in, GR.blockdim_xy, GR.stream]\
                                (dSOILTEMPdt, GF.SOILTEMP, GF.LWFLXNET, GF.SWFLXNET,
                                GF.SOILCP, GF.SOILRHO, GF.SOILDEPTH, GR.dt)
            if i_microphysics > 0:
                raise NotImplementedError()

            calc_albedo_gpu[GR.griddim_xy_in, GR.blockdim_xy, GR.stream]\
                                (GF.SURFALBEDSW, GF.SURFALBEDLW, GF.OCEANMASK, GF.SOILTEMP)

            calc_evaporation_capacity_gpu[GR.griddim_xy_in, GR.blockdim_xy, GR.stream]\
                                (GF.SOILEVAPITY, GF.SOILMOIST, GF.OCEANMASK, GF.SOILTEMP)





    def calc_albedo(self, GR, CF):
        # forest
        CF.SURFALBEDSW[:] = 0.2
        CF.SURFALBEDLW[:] = 0.2
        # ocean
        CF.SURFALBEDSW[CF.OCEANMASK == 1] = 0.05
        CF.SURFALBEDLW[CF.OCEANMASK == 1] = 0.05
        # ice (land and sea)
        CF.SURFALBEDSW[CF.SOILTEMP[:,:,0] <= 273.15] = 0.7
        CF.SURFALBEDLW[CF.SOILTEMP[:,:,0] <= 273.15] = 0.3

