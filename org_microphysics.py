import numpy as np
import scipy
import time
import matplotlib.pyplot as plt


class microphysics:
    
    surf_moisture_flux_constant = 1E-6
    qv_to_qc_rate = 1E-3

    def __init__(self, GR, i_microphysics):
        print('Prepare Microphysics')

        self.i_microphysics = i_microphysics 

        # temperature tendency due to microphysics
        self.dPOTTdt_MIC = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)
        ## total qv tendency
        #self.dQVdt = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)
        # qv tendency due to microphysics
        self.dQVdt_MIC = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)
        # qc tendency due to microphysics
        self.dQCdt_MIC = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)

        self.RH = np.zeros( ( GR.nx, GR.ny, GR.nz ) )

        if self.i_microphysics >= 1:
            # specific water vapor content
            self.QV = np.zeros( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ) ) 
            #self.QV[3:7,3:7,range(0,GR.nz)] = 0.005 - 0.005*np.exp(-0.8*np.arange(0,GR.nz))
            ## specific cloud liquid water content
            self.QC = np.zeros( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ) ) 


    def relative_humidity(self, GR, TAIR, QV, PAIR):
        e_satw = 610.94 * np.exp( (17.625 * (TAIR[GR.iijj] - 273.15)) / (TAIR[GR.iijj] - 273.15 + 243.04) )
        rh = QV[GR.iijj] * PAIR[GR.iijj] / (0.622 * e_satw)
        qv_sat = (0.622 * e_satw) / PAIR[GR.iijj]
        qv_diff = QV[GR.iijj] - qv_sat
        qv_diff[rh <= 1] = 0
        return(rh, qv_diff)
            

    def calc_microphysics(self, GR, WIND, SOIL, TAIR, PAIR):
        t_start = time.time()

        # relative humidity
        self.RH, qv_diff = self.relative_humidity(GR, TAIR, self.QV, PAIR)

        self.dQCdt_MIC[:] = + qv_diff * self.qv_to_qc_rate
        self.dQVdt_MIC[:] = - qv_diff * self.qv_to_qc_rate
        #self.dPOTTdt_MIC[:] = qv_diff / 
        # SURFACE MOISTURE UPTAKE
        self.dQVdt_MIC[:,:,-1] = self.dQVdt_MIC[:,:,-1] + WIND[:,:,-1][GR.iijj] * self.surf_moisture_flux_constant * \
                                    np.abs((SOIL.MOIST - self.QV[:,:,-1][GR.iijj]))





        t_end = time.time()
        GR.mic_comp_time += t_end - t_start
