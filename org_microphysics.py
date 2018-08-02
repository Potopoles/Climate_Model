import numpy as np
import scipy
import time
import matplotlib.pyplot as plt


class microphysics:
    
    surf_moisture_flux_constant = 1E-7
    qv_moisture_conversion = 10

    def __init__(self, GR, i_microphysics):
        print('Prepare Microphysics')

        self.i_microphysics = i_microphysics 

        # temperature tendency due to microphysics
        self.dPOTTdt_MIC = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)
        # total qv tendency
        self.dQVdt = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)
        # qv tendency due to microphysics
        self.dQVdt_MIC = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)

        if self.i_microphysics >= 1:
            # specific water vapor content
            self.QV = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
            self.QV[:] = 0
            #self.QV[3:7,3:7,range(0,GR.nz)] = 5 - 5*np.exp(-0.8*np.arange(0,GR.nz))
            ## specific cloud liquid water content
            #self.QC =np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
            

    def calc_microphysics(self, GR, WIND, SOIL):
        t_start = time.time()

        # SURFACE MOISTURE UPTAKE
        self.dQVdt_MIC[:] = 0
        self.dQVdt_MIC[:,:,-1] = WIND[:,:,-1][GR.iijj] * self.surf_moisture_flux_constant * \
                                    np.abs((SOIL.MOIST - self.QV[:,:,-1][GR.iijj]))



        t_end = time.time()
        GR.mic_comp_time += t_end - t_start
