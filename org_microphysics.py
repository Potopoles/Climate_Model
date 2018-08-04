import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
from constants import con_g, con_cp
from boundaries import exchange_BC


class microphysics:
    
    RH_init = 0.6
    surf_moisture_flux_constant = 1E-9
    qv_to_qc_rate = 1E-3
    qc_to_qr_rate = 1E-5

    lh_cond_water = 2260 # J / (kg)

    def __init__(self, GR, i_microphysics, TAIR, PAIR):
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

        self.surf_evap_flx = np.full( ( GR.nx, GR.ny ), np.nan)

        if self.i_microphysics >= 1:
            # specific water vapor content
            self.QV = np.zeros( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ) ) 
            e_satw = 610.94 * np.exp( (17.625 * (TAIR[GR.iijj] - 273.15)) / (TAIR[GR.iijj] - 273.15 + 243.04) )
            self.QV[GR.iijj] = self.RH_init * (0.622 * e_satw) / PAIR[GR.iijj]
            self.QV = exchange_BC(GR, self.QV)
            #self.QV[:,:,range(0,GR.nz)] = 0.003 - 0.003*(np.exp(-0.1*np.arange(0,GR.nz))-0.01)
            ## specific cloud liquid water content
            self.QC = np.zeros( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ) ) 


    def relative_humidity(self, GR, TAIR, QV, PAIR):
        e_satw = 610.94 * np.exp( (17.625 * (TAIR[GR.iijj] - 273.15)) / (TAIR[GR.iijj] - 273.15 + 243.04) )
        rh = QV[GR.iijj] * PAIR[GR.iijj] / (0.622 * e_satw)
        qv_sat = (0.622 * e_satw) / PAIR[GR.iijj]
        qv_diff = QV[GR.iijj] - qv_sat
        qv_diff[rh <= 1] = 0
        return(rh, qv_diff)
            

    def calc_microphysics(self, GR, WIND, SOIL, TAIR, PAIR, RHO, PHIVB):
        t_start = time.time()

        ALTVB = PHIVB / con_g
        dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]

        # relative humidity
        self.RH, qv_diff = self.relative_humidity(GR, TAIR, self.QV, PAIR)

        # rain
        rain_rate = self.qc_to_qr_rate * self.QC[GR.iijj]

        self.dQVdt_MIC[:] = - qv_diff * self.qv_to_qc_rate
        self.dQCdt_MIC[:] = + qv_diff * self.qv_to_qc_rate - rain_rate

        self.dPOTTdt_MIC = qv_diff * self.qv_to_qc_rate * self.lh_cond_water / con_cp

        # surface moisture uptake
        QV_surf_flux = np.maximum((1 - self.RH[:,:,-1]),0) * WIND[:,:,-1][GR.iijj] * \
                                    self.surf_moisture_flux_constant * SOIL.EVAPITY
        self.surf_evap_flx[:,:] = QV_surf_flux * RHO[:,:,-1][GR.iijj] * dz[:,:,-1] # kg_w/m^2/s
        #print(np.max(self.surf_evap_flx))
        total_evaporation = self.surf_evap_flx * GR.dt

        self.dQVdt_MIC[:,:,-1] = self.dQVdt_MIC[:,:,-1] + QV_surf_flux

        SOIL.RAINRATE = np.sum(rain_rate*RHO[GR.iijj]*dz,2) # mm/s
        SOIL.ACCRAIN = SOIL.ACCRAIN + SOIL.RAINRATE*GR.dt
        SOIL.MOIST = SOIL.MOIST - total_evaporation + SOIL.RAINRATE*GR.dt
        SOIL.MOIST[SOIL.MOIST < 0] = 0
        SOIL.MOIST[SOIL.MOIST > 20] = 20

        t_end = time.time()
        GR.mic_comp_time += t_end - t_start
