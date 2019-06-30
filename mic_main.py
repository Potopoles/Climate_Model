#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190630
Last modified:      20190630
License:            MIT

Main script of microphysics.
###############################################################################
"""

from io_read_namelist import wp, GPU, CPU, gpu_enable
from main_grid import nx,ny,nz,nzs,nb
from main_grid import tpb, bpg
from misc_meteo_utilities import calc_specific_humidity_py
from misc_utilities import function_input_fields
if gpu_enable:
    from mic_compute import compute_microphysics_gpu
###############################################################################

###############################################################################
# NAMELIST
###############################################################################
#RH_init = wp(60) # [%]
#qv_to_qc_rate = 1E-3
#qc_to_qr_rate = 1E-5

class Microphysics:


    def __init__(self, GR, F, target):

        self.target = target

        #self.fields_init = function_input_fields(self.initial_conditions)
        #self.initial_conditions(GR, **F.get(self.fields_init, target=CPU))

        self.fields_main = function_input_fields(self.compute_microhpysics)


        ##if self.i_microphysics >= 1:
        ## specific water vapor content
        #e_satw = 610.94 * np.exp( (17.625 * (TAIR[GR.iijj] - 273.15)) / \
        #        (TAIR[GR.iijj] - 273.15 + 243.04) )
        #F.QV[GR.iijj] = self.RH_init * (0.622 * e_satw) / PAIR[GR.iijj]
        #F.QV = exchange_BC(GR, F.QV)
        ##F.QV[:,:,range(0,GR.nz)] = 0.003 - 0.003*(np.exp(-0.1*np.arange(0,GR.nz))-0.01)
        ### specific cloud liquid water content

    def compute_microhpysics(self, GR, QV, QC, QR, POTT, TAIR,
                            PAIR, RHO, dPOTTdt_MIC):
        if self.target == GPU:
            compute_microphysics_gpu[bpg, tpb](QV, QC, QR, POTT, TAIR,
                                                PAIR, RHO,
                                                dPOTTdt_MIC, GR.dt)

        #elif self.target == CPU:
        #    compute_microphysics_cpu(QV, QC, TAIR)


    #def initial_conditions(self, GR, TAIR, PAIR, QV):
    #    for i in range(0,nx+2*nb):
    #        for j in range(0,ny+2*nb):
    #            for k in range(0,nz):
    #                QV[i,j,k] = calc_specific_humidity_py(TAIR[i,j,k],
    #                                                RH_init, PAIR[i,j,k])
            

    #def calc_microphysics(self, GR, WIND, SOIL, TAIR, PAIR, RHO, PHIVB):

    #    ALTVB = PHIVB / con_g
    #    dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]

    #    # relative humidity
    #    self.RH, qv_diff = self.relative_humidity(GR, TAIR, F.QV, PAIR)

    #    # rain
    #    rain_rate = self.qc_to_qr_rate * F.QC[GR.iijj]

    #    F.dQVdt_MIC[:] = - qv_diff * self.qv_to_qc_rate
    #    F.dQCdt_MIC[:] = + qv_diff * self.qv_to_qc_rate - rain_rate

    #    F.dPOTTdt_MIC = qv_diff * self.qv_to_qc_rate * self.lh_cond_water / con_cp

    #    # surface moisture uptake
    #    QV_surf_flux = np.maximum((1 - self.RH[:,:,-1]),0) * WIND[:,:,-1][GR.iijj] * \
    #                                self.surf_moisture_flux_constant * SOIL.EVAPITY
    #    self.surf_evap_flx[:,:] = QV_surf_flux * RHO[:,:,-1][GR.iijj] * dz[:,:,-1] # kg_w/m^2/s
    #    #print(np.max(self.surf_evap_flx))
    #    total_evaporation = self.surf_evap_flx * GR.dt

    #    F.dQVdt_MIC[:,:,-1] = F.dQVdt_MIC[:,:,-1] + QV_surf_flux

    #    SOIL.RAINRATE = np.sum(rain_rate*RHO[GR.iijj]*dz,2) # mm/s
    #    SOIL.ACCRAIN = SOIL.ACCRAIN + SOIL.RAINRATE*GR.dt
    #    SOIL.MOIST = SOIL.MOIST - total_evaporation + SOIL.RAINRATE*GR.dt
    #    SOIL.MOIST[SOIL.MOIST < 0] = 0
    #    SOIL.MOIST[SOIL.MOIST > 20] = 20

