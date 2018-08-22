import numpy as np
import scipy
import time
from radiation.namelist_radiation import \
        pseudo_rad_inpRate, pseudo_rad_outRate, \
        rad_nth_hour 
from constants import con_cp, con_g
from radiation.shortwave import rad_solar_zenith_angle, \
       calc_current_solar_constant
from radiation.longwave import org_longwave
from radiation.shortwave import org_shortwave
import matplotlib.pyplot as plt

import multiprocessing as mp
from radiation.namelist_radiation import njobs_rad


def calc_par(GR, TAIR, RHO, PHIVB, TSOIL, ALBEDOLW, ALBEDOSW, QC,
            SWINTOA, MYSUN, dz, solar_constant):


        down_diffuse, up_diffuse = \
                            org_longwave(GR, dz, TAIR, RHO, TSOIL, ALBEDOLW, QC)

        LWFLXDO = - down_diffuse
        LWFLXUP =   up_diffuse

        # SHORTWAVE
        if MYSUN > 0:

            # toon et al 1989 method
            down_diffuse, up_diffuse, down_direct = \
                                org_shortwave(GR, dz, solar_constant, RHO, SWINTOA,
                                            MYSUN, ALBEDOSW, QC)

            SWDIFFLXDO = - down_diffuse
            SWDIRFLXDO = - down_direct
            SWFLXUP    = up_diffuse
            SWFLXDO    = - down_diffuse - down_direct
        else:
            SWDIFFLXDO = np.zeros(GR.nzs)
            SWDIRFLXDO = np.zeros(GR.nzs) 
            SWFLXUP    = np.zeros(GR.nzs) 
            SWFLXDO    = np.zeros(GR.nzs) 

        LWFLXNET = LWFLXDO - LWFLXUP 
        SWFLXNET = SWFLXDO - SWFLXUP 

        #print(LWFLXNET)
        LWFLXDIV = ( LWFLXNET[GR.kk] - LWFLXNET[GR.kk+1] ) / dz[GR.kk]
        SWFLXDIV = ( SWFLXNET[GR.kk] - SWFLXNET[GR.kk+1] ) / dz[GR.kk]
        TOTFLXDIV = SWFLXDIV + LWFLXDIV
        dPOTTdt_RAD = 1/(con_cp * RHO) * TOTFLXDIV


        result = (LWFLXDO, LWFLXUP, SWDIFFLXDO, SWDIRFLXDO, SWFLXUP, SWFLXDO,
                LWFLXNET, SWFLXNET,
                LWFLXDIV, SWFLXDIV, TOTFLXDIV, dPOTTdt_RAD)
        return(result)




class radiation:

    def __init__(self, GR, i_radiation):
        print('Prepare Radiation')

        self.i_radiation = i_radiation


        self.rad_nth_hour = rad_nth_hour 
        self.rad_nth_ts = int(self.rad_nth_hour * 3600/GR.dt)

        # temperature tendency due to radiation
        self.dPOTTdt_RAD = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan)

        if self.i_radiation >= 1:
            # solar zenith angle
            self.SOLZEN = np.full( ( GR.nx, GR.ny ), np.nan)
            
            # cos solar zenith angle
            self.MYSUN = np.full( ( GR.nx, GR.ny ), np.nan)

            # incoming shortwave at TOA
            self.SWINTOA = np.full( ( GR.nx, GR.ny ), np.nan)

        if self.i_radiation >= 2:
            self.LWFLXUP =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.LWFLXDO =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.LWFLXNET =     np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWDIFFLXDO =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWDIRFLXDO =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWFLXUP =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWFLXDO =      np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWFLXNET =     np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.LWFLXDIV =     np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)
            self.SWFLXDIV =     np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)
            self.TOTFLXDIV =    np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)



    def calc_radiation(self, GR, TAIR, TAIRVB, RHO, PHIVB, SOIL, MIC):


        if self.i_radiation == 1:
            self.pseudo_constant_radiation(GR, TAIR)
        elif self.i_radiation == 2:
            self.pseudo_varying_radiation(GR, TAIR)
        elif self.i_radiation == 3:
            if GR.ts % self.rad_nth_ts == 0:
                print('calculate radiation')
                #self.simple_radiation(GR, TAIR, RHO, PHIVB, SOIL, MIC)
                self.simple_radiation_par(GR, TAIR, RHO, PHIVB, SOIL, MIC)
        elif self.i_radiation == 0:
            pass
        else:
            raise NotImplementedError()







    def simple_radiation_par(self, GR, TAIR, RHO, PHIVB, SOIL, MIC):
        ALTVB = PHIVB / con_g
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
        p = mp.Pool(processes=njobs_rad)

        input = [(GR, TAIR[ii_ref[c],jj_ref[c],:], RHO[ii_ref[c],jj_ref[c],:],
                PHIVB[ii[c],jj[c],:],
                SOIL.TSOIL[ii[c],jj[c],0], SOIL.ALBEDOLW[ii[c],jj[c]],
                SOIL.ALBEDOSW[ii[c],jj[c]], MIC.QC[ii[c],jj[c],:],
                self.SWINTOA[ii[c],jj[c]], self.MYSUN[ii[c],jj[c]],
                dz[ii[c],jj[c],:], self.solar_constant) for c in range(0,len(ii))]

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
            self.LWFLXNET[i,j,:]    = result[c][6]
            self.SWFLXNET[i,j,:]    = result[c][7]
            self.LWFLXDIV[i,j,:]    = result[c][8]
            self.SWFLXDIV[i,j,:]    = result[c][9]
            self.TOTFLXDIV[i,j,:]   = result[c][10]
            self.dPOTTdt_RAD[i,j,:] = result[c][11]
        

















    def simple_radiation(self, GR, TAIR, RHO, PHIVB, SOIL, MIC):
        ALTVB = PHIVB / con_g
        dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]

        self.SOLZEN = rad_solar_zenith_angle(GR, self.SOLZEN)
        self.MYSUN = np.cos(self.SOLZEN)
        self.MYSUN[self.MYSUN < 0] = 0
        self.solar_constant = calc_current_solar_constant(GR) 
        self.SWINTOA = self.solar_constant * np.cos(self.SOLZEN)

        for i in range(0,GR.nx):
            #i = 10
            i_ref = i+GR.nb
            for j in range(0,GR.ny):
                #j = 10
                j_ref = j+GR.nb


                # LONGWAVE
                # toon et al 1989 method
                #down_diffuse, up_diffuse = \
                #                    org_longwave(GR, dz[i,j],
                #                                TAIRVB[i_ref,j_ref,:], RHO[i_ref,j_ref], \
                #                                SOIL.TSOIL[i,j,0], SOIL.ALBEDOLW[i,j])
                # self-manufactured method
                down_diffuse, up_diffuse = \
                                    org_longwave(GR, dz[i,j],
                                                TAIR[i_ref,j_ref,:], RHO[i_ref,j_ref], \
                                                SOIL.TSOIL[i,j,0], SOIL.ALBEDOLW[i,j],
                                                MIC.QC[i,j,:])

                self.LWFLXDO[i,j,:] = - down_diffuse
                self.LWFLXUP[i,j,:] =   up_diffuse

                # SHORTWAVE
                if self.MYSUN[i,j] > 0:

                    # toon et al 1989 method
                    down_diffuse, up_diffuse, down_direct = \
                                        org_shortwave(GR, dz[i,j], self.solar_constant,
                                                    RHO[i_ref,j_ref],
                                                    self.SWINTOA[i,j],
                                                    self.MYSUN[i,j],
                                                    SOIL.ALBEDOSW[i,j],
                                                    MIC.QC[i,j,:])

                    self.SWDIFFLXDO[i,j,:] = - down_diffuse
                    self.SWDIRFLXDO[i,j,:] = - down_direct
                    self.SWFLXUP   [i,j,:] = up_diffuse
                    self.SWFLXDO   [i,j,:] = - down_diffuse - down_direct
                else:
                    self.SWDIFFLXDO[i,j,:] = 0
                    self.SWDIRFLXDO[i,j,:] = 0
                    self.SWFLXUP   [i,j,:] = 0
                    self.SWFLXDO   [i,j,:] = 0


        self.LWFLXNET = self.LWFLXDO - self.LWFLXUP 
        self.SWFLXNET = self.SWFLXDO - self.SWFLXUP 

        for k in range(0,GR.nz):

            self.SWFLXDIV[:,:,k] = ( self.SWFLXNET[:,:,k] - self.SWFLXNET[:,:,k+1] ) \
                                    / dz[:,:,k]
            self.LWFLXDIV[:,:,k] = ( self.LWFLXNET[:,:,k] - self.LWFLXNET[:,:,k+1] ) \
                                    / dz[:,:,k]
            self.TOTFLXDIV[:,:,k] = self.SWFLXDIV[:,:,k] + self.LWFLXDIV[:,:,k]
            
            self.dPOTTdt_RAD[:,:,k] = 1/(con_cp * RHO[:,:,k][GR.iijj]) * \
                                                self.TOTFLXDIV[:,:,k]


    def pseudo_varying_radiation(self, GR, TAIR):
        self.SOLZEN = rad_solar_zenith_angle(GR, self.SOLZEN)
        self.SWINTOA = incoming_SW_TOA(GR, self.SWINTOA, self.SOLZEN) 

        for k in range(0,GR.nz):
            self.dPOTTdt_RAD[:,:,k] = \
                                -  pseudo_rad_outRate*TAIR[:,:,k][GR.iijj]**1/2  \
                                +  pseudo_rad_inpRate*self.SWINTOA/1000


    def pseudo_constant_radiation(self, GR, TAIR):
        for k in range(0,GR.nz):
            self.dPOTTdt_RAD[:,:,k] = \
                                -  pseudo_rad_outRate*TAIR[:,:,k][GR.iijj]**1  \
                                +  pseudo_rad_inpRate*np.cos(GR.lat_rad[GR.iijj])


