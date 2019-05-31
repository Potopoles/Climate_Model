import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
from constants import con_g, con_cp
from boundaries import exchange_BC
from org_namelist import wp


class microphysics:
    
    RH_init = 0.6
    surf_moisture_flux_constant = 1E-9
    qv_to_qc_rate = 1E-3
    qc_to_qr_rate = 1E-5

    lh_cond_water = 2260 # J / (kg)

    def __init__(self, GR, F, i_microphysics, TAIR, PAIR):
        print('Prepare Microphysics')

        self.i_microphysics = i_microphysics 

        self.RH = np.zeros( ( GR.nx, GR.ny, GR.nz ), dtype=wp)

        self.surf_evap_flx = np.full( ( GR.nx, GR.ny ), np.nan, dtype=wp)

        #if self.i_microphysics >= 1:
        # specific water vapor content
        e_satw = 610.94 * np.exp( (17.625 * (TAIR[GR.iijj] - 273.15)) / \
                (TAIR[GR.iijj] - 273.15 + 243.04) )
        F.QV[GR.iijj] = self.RH_init * (0.622 * e_satw) / PAIR[GR.iijj]
        F.QV = exchange_BC(GR, F.QV)
        #F.QV[:,:,range(0,GR.nz)] = 0.003 - 0.003*(np.exp(-0.1*np.arange(0,GR.nz))-0.01)
        ## specific cloud liquid water content


    def relative_humidity(self, GR, TAIR, QV, PAIR):
        e_satw = 610.94 * np.exp( (17.625 * (TAIR[GR.iijj] - 273.15)) / (TAIR[GR.iijj] - 273.15 + 243.04) )
        rh = QV[GR.iijj] * PAIR[GR.iijj] / (0.622 * e_satw)
        qv_sat = (0.622 * e_satw) / PAIR[GR.iijj]
        qv_diff = QV[GR.iijj] - qv_sat
        qv_diff[rh <= 1] = 0
        return(rh, qv_diff)
            

    def calc_microphysics(self, GR, WIND, SOIL, TAIR, PAIR, RHO, PHIVB):

        ALTVB = PHIVB / con_g
        dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]

        # relative humidity
        self.RH, qv_diff = self.relative_humidity(GR, TAIR, F.QV, PAIR)

        # rain
        rain_rate = self.qc_to_qr_rate * F.QC[GR.iijj]

        F.dQVdt_MIC[:] = - qv_diff * self.qv_to_qc_rate
        F.dQCdt_MIC[:] = + qv_diff * self.qv_to_qc_rate - rain_rate

        F.dPOTTdt_MIC = qv_diff * self.qv_to_qc_rate * self.lh_cond_water / con_cp

        # surface moisture uptake
        QV_surf_flux = np.maximum((1 - self.RH[:,:,-1]),0) * WIND[:,:,-1][GR.iijj] * \
                                    self.surf_moisture_flux_constant * SOIL.EVAPITY
        self.surf_evap_flx[:,:] = QV_surf_flux * RHO[:,:,-1][GR.iijj] * dz[:,:,-1] # kg_w/m^2/s
        #print(np.max(self.surf_evap_flx))
        total_evaporation = self.surf_evap_flx * GR.dt

        F.dQVdt_MIC[:,:,-1] = F.dQVdt_MIC[:,:,-1] + QV_surf_flux

        SOIL.RAINRATE = np.sum(rain_rate*RHO[GR.iijj]*dz,2) # mm/s
        SOIL.ACCRAIN = SOIL.ACCRAIN + SOIL.RAINRATE*GR.dt
        SOIL.MOIST = SOIL.MOIST - total_evaporation + SOIL.RAINRATE*GR.dt
        SOIL.MOIST[SOIL.MOIST < 0] = 0
        SOIL.MOIST[SOIL.MOIST > 20] = 20

