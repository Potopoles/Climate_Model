#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190602
License:            MIT

Main script of radiation scheme.
###############################################################################
"""
import time, scipy
import numpy as np
#import matplotlib.pyplot as plt
import multiprocessing as mp

from namelist import (pseudo_rad_inpRate, pseudo_rad_outRate,
                      rad_nth_hour, i_async_radiation, planck_n_lw_bins,
                      njobs_rad)
from io_constants import con_cp, con_g
from io_read_namelist import wp, CPU, GPU
from rad_shortwave import (org_shortwave, rad_solar_zenith_angle,
                           calc_current_solar_constant)
from rad_longwave import org_longwave
###############################################################################

class Radiation:

    fields_radiation = ['PHIVB', 'SOLZEN', 'MYSUN', 'SWINTOA', 'TAIR',
                        'RHO', 'SOILTEMP', 'SURFALBEDLW', 'SURFALBEDSW',
                        'QC', 'LWFLXDO', 'LWFLXUP', 'SWDIFFLXDO',
                        'SWDIRFLXDO', 'SWFLXUP', 'SWFLXDO',
                        'LWFLXNET', 'SWFLXNET', 'SWFLXDIV', 'LWFLXDIV',
                        'TOTFLXDIV', 'dPOTTdt_RAD'] 

    def __init__(self, GR):
        print('Prepare Radiation')

        self.done = 0

        self.i_async_radiation = i_async_radiation

        self.njobs_rad = njobs_rad

        self.rad_nth_hour = rad_nth_hour 
        self.rad_nth_ts = int(self.rad_nth_hour * 3600/GR.dt)

        ## solar zenith angle
        #self.SOLZEN = np.full( ( GR.nx, GR.ny ), np.nan)
        #
        ## cos solar zenith angle
        #self.MYSUN = np.full( ( GR.nx, GR.ny ), np.nan)

        ## incoming shortwave at TOA
        #self.SWINTOA = np.full( ( GR.nx, GR.ny ), np.nan)

        #self.LWFLXUP =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        #self.LWFLXDO =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        ##self.LWFLXNET =     np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        #self.SWDIFFLXDO =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        #self.SWDIRFLXDO =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        #self.SWFLXUP =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        #self.SWFLXDO =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        ##self.SWFLXNET =     np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
        #self.LWFLXDIV =     np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)
        #self.SWFLXDIV =     np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)
        #self.TOTFLXDIV =    np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)


        # Planck emission calculations
        nu0 = 50.
        nu1 = 2500.
        nnu = planck_n_lw_bins
        nu0 = nu0*100 # 1/m
        nu1 = nu1*100 # 1/m
        dnu = (nu1 - nu0)/(nnu)
        #dnu = 5000 # 1/m
        nus = np.arange(nu0,nu1+1,dnu).astype(wp)
        nus_center = nus[:-1]+dnu/2
        lambdas = 1./nus
        self.planck_lambdas_center = 1./nus_center
        self.planck_dlambdas = np.diff(lambdas)


    def calc_radiation(self, GR, F):
        
        t_start = time.time()
        GR.timer.start('rad')
        sim_t_start = GR.sim_time_sec
        

        #if GR.ts % self.rad_nth_ts == 0:
        print('###########################################')
        print('RADIATION START')
        self.simple_radiation(GR, **F.get(self.fields_radiation, target=CPU))
        #self.simple_radiation_par(GR, CF)

        self.done = 1
        t_end = time.time()
        GR.timer.stop('rad')
        sim_t_end = GR.sim_time_sec

            
        print('###########################################')
        print('RADIATION DONE')
        print('took ' + str(round(t_end-t_start,0)) + ' seconds.')
        if self.i_async_radiation:
            print('took ' + str(round((sim_t_end-sim_t_start)/3600,1)) + 
                    ' simulated hours.')
        print('###########################################')





    def simple_radiation(self, GR, PHIVB, SOLZEN, MYSUN, SWINTOA, TAIR,
                        RHO, SOILTEMP, SURFALBEDLW, SURFALBEDSW, QC, LWFLXDO,
                        LWFLXUP, SWDIFFLXDO, SWDIRFLXDO, SWFLXUP, SWFLXDO,
                        LWFLXNET, SWFLXNET, SWFLXDIV, LWFLXDIV, TOTFLXDIV,
                        dPOTTdt_RAD):

        #GR.timer.start('prep')
        ALTVB = PHIVB / con_g
        dz = ALTVB[:,:,:-1][GR.ii,GR.jj] -  ALTVB[:,:,1:][GR.ii,GR.jj]

        SOLZEN = rad_solar_zenith_angle(GR, SOLZEN)
        MYSUN = np.cos(SOLZEN)
        MYSUN[MYSUN < 0] = 0
        self.solar_constant = calc_current_solar_constant(GR) 
        SWINTOA = self.solar_constant * np.cos(SOLZEN)
        #GR.timer.stop('prep')

        GR.timer.start('lw')
        for i in range(0,GR.nx):
            i_ref = i+GR.nb
            for j in range(0,GR.ny):
                j_ref = j+GR.nb

                # LONGWAVE
                # toon et al 1989 method
                #down_diffuse, up_diffuse = \
                #                    org_longwave(GR, dz[i,j],
                #                                TAIRVB[i_ref,j_ref,:], RHO[i_ref,j_ref], \
                #                                SOIL.TSOIL[i,j,0], SOIL.ALBEDOLW[i,j])
                # self-manufactured method
                down_diffuse, up_diffuse = \
                            org_longwave(GR, GR.nz, GR.nzs, dz[i,j],
                                        TAIR[i_ref,j_ref,:],    RHO[i_ref,j_ref],
                                        SOILTEMP[i,j,0],        SURFALBEDLW[i,j,0],
                                        QC[i,j,:],
                                        self.planck_lambdas_center,
                                        self.planck_dlambdas)

                LWFLXDO[i,j,:] = - down_diffuse
                LWFLXUP[i,j,:] =   up_diffuse

        LWFLXNET[:] = LWFLXDO[:] - LWFLXUP[:] 
        GR.timer.stop('lw')

        GR.timer.start('sw')
        for i in range(0,GR.nx):
            i_ref = i+GR.nb
            for j in range(0,GR.ny):
                j_ref = j+GR.nb

                # SHORTWAVE
                if MYSUN[i,j] > 0:

                    # toon et al 1989 method
                    down_diffuse, up_diffuse, down_direct = \
                                    org_shortwave(GR.nz, GR.nzs, dz[i,j],
                                                self.solar_constant,
                                                RHO[i_ref,j_ref],
                                                SWINTOA[i,j,0],
                                                MYSUN[i,j,0],
                                                SURFALBEDSW[i,j,0],
                                                QC[i,j,:])

                    SWDIFFLXDO[i,j,:] = - down_diffuse
                    SWDIRFLXDO[i,j,:] = - down_direct
                    SWFLXUP   [i,j,:] = up_diffuse
                    SWFLXDO   [i,j,:] = - down_diffuse - down_direct
                else:
                    SWDIFFLXDO[i,j,:] = 0
                    SWDIRFLXDO[i,j,:] = 0
                    SWFLXUP   [i,j,:] = 0
                    SWFLXDO   [i,j,:] = 0

        SWFLXNET[:] = SWFLXDO[:] - SWFLXUP[:] 
        GR.timer.stop('sw')

        #GR.timer.start('finish')
        for k in range(0,GR.nz):

            SWFLXDIV[:,:,k] = ( SWFLXNET[:,:,k] - SWFLXNET[:,:,k+1] ) \
                               / dz[:,:,k]
            LWFLXDIV[:,:,k] = ( LWFLXNET[:,:,k] - LWFLXNET[:,:,k+1] ) \
                               / dz[:,:,k]
            TOTFLXDIV[:,:,k] = SWFLXDIV[:,:,k] + LWFLXDIV[:,:,k]
            
            dPOTTdt_RAD[GR.ii,GR.jj,k] = 1/(con_cp * RHO[:,:,k][GR.ii,GR.jj]) * \
                                                TOTFLXDIV[:,:,k]
        #GR.timer.stop('finish')








    def simple_radiation_par(self, GR, CF):
        ALTVB = CF.PHIVB / con_g
        dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]

        self.SOLZEN = rad_solar_zenith_angle(GR, self.SOLZEN)
        self.MYSUN = np.cos(self.SOLZEN)
        self.MYSUN[self.MYSUN < 0] = 0
        self.solar_constant = calc_current_solar_constant(GR) 
        self.SWINTOA = self.solar_constant * np.cos(self.SOLZEN)


        ij_all = np.full( (GR.nx*GR.ny, 2), np.int)
        ij_ref_all = np.full( (GR.nx*GR.ny, 2), np.int)

        #ii = np.arange(3,6).astype(np.int)
        ii = np.tile(np.arange(0,GR.nx).astype(np.int),GR.ny)
        jj = np.repeat(np.arange(0,GR.ny).astype(np.int),GR.nx)
        ii_ref = np.tile(np.arange(0,GR.nx).astype(np.int),GR.ny)+1
        jj_ref = np.repeat(np.arange(0,GR.ny).astype(np.int),GR.nx)+1

        #t1 = time.time()
        p = mp.Pool(processes=self.njobs_rad)
        
        #print(type(TAIR))
        #print(type(RHO))
        #print(type(PHIVB))
        #quit()

        input = [(GR.nz, GR.nzs, GR.kk, CF.TAIR[ii_ref[c],jj_ref[c],:],
                CF.RHO[ii_ref[c],jj_ref[c],:], CF.PHIVB[ii[c],jj[c],:],
                CF.SOILTEMP[ii[c],jj[c],0], CF.SURFALBEDLW[ii[c],jj[c]],
                CF.SURFALBEDSW[ii[c],jj[c]], CF.QC[ii[c],jj[c],:],
                self.SWINTOA[ii[c],jj[c]], self.MYSUN[ii[c],jj[c]],
                dz[ii[c],jj[c],:], self.solar_constant, self.planck_lambdas_center,
                self.planck_dlambdas) for c in range(0,len(ii))]

        result = p.starmap(calc_par, input)
        p.close()
        p.join()
        #t2 = time.time()
        #print(len(result))
        #print(result[len(ii)-1][0])
        #print(t2 - t1)
        #quit()

        for c in range(0,len(ii)):
            i = ii[c]
            j = jj[c]
            self.LWFLXDO[i,j,:]     = result[c][0]
            self.LWFLXUP[i,j,:]     = result[c][1]
            self.SWDIFFLXDO[i,j,:]  = result[c][2]
            self.SWDIRFLXDO[i,j,:]  = result[c][3]
            self.SWFLXUP[i,j,:]     = result[c][4]
            self.SWFLXDO[i,j,:]     = result[c][5]
            CF.LWFLXNET[i,j,:]      = result[c][6]
            CF.SWFLXNET[i,j,:]      = result[c][7]
            self.LWFLXDIV[i,j,:]    = result[c][8]
            self.SWFLXDIV[i,j,:]    = result[c][9]
            self.TOTFLXDIV[i,j,:]   = result[c][10]
            CF.dPOTTdt_RAD[i,j,:]   = result[c][11]
        




def calc_par(nz, nzs, kk, TAIR, RHO, PHIVB, TSOIL, ALBEDOLW, ALBEDOSW, QC,
            SWINTOA, MYSUN, dz, solar_constant, planck_lambdas_center,
            planck_dlambdas):


        down_diffuse, up_diffuse = \
                            org_longwave(nz, nzs, dz, TAIR, RHO, TSOIL, ALBEDOLW, QC,
                                         planck_lambdas_center, planck_dlambdas)


        LWFLXDO = - down_diffuse
        LWFLXUP =   up_diffuse

        # SHORTWAVE
        if MYSUN > 0:

            # toon et al 1989 method
            down_diffuse, up_diffuse, down_direct = \
                                org_shortwave(nz, nzs, dz, solar_constant, RHO, SWINTOA,
                                            MYSUN, ALBEDOSW, QC)

            SWDIFFLXDO = - down_diffuse
            SWDIRFLXDO = - down_direct
            SWFLXUP    = up_diffuse
            SWFLXDO    = - down_diffuse - down_direct
        else:
            SWDIFFLXDO = np.zeros(nzs)
            SWDIRFLXDO = np.zeros(nzs) 
            SWFLXUP    = np.zeros(nzs) 
            SWFLXDO    = np.zeros(nzs) 

        LWFLXNET = LWFLXDO - LWFLXUP 
        SWFLXNET = SWFLXDO - SWFLXUP 

        #print(LWFLXNET)
        LWFLXDIV = ( LWFLXNET[kk] - LWFLXNET[kk+1] ) / dz[kk]
        SWFLXDIV = ( SWFLXNET[kk] - SWFLXNET[kk+1] ) / dz[kk]
        TOTFLXDIV = SWFLXDIV + LWFLXDIV
        dPOTTdt_RAD = 1/(con_cp * RHO) * TOTFLXDIV


        result = (LWFLXDO, LWFLXUP, SWDIFFLXDO, SWDIRFLXDO, SWFLXUP, SWFLXDO,
                LWFLXNET, SWFLXNET,
                LWFLXDIV, SWFLXDIV, TOTFLXDIV, dPOTTdt_RAD)
        return(result)

