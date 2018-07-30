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
            #self.SOLZEN = rad_solar_zenith_angle(GR, self.SOLZEN)
            
            # cos solar zenith angle
            self.MYSUN = np.full( ( GR.nx, GR.ny ), np.nan)

            # incoming shortwave at TOA
            self.SWINTOA = np.full( ( GR.nx, GR.ny ), np.nan)
            #self.SWINTOA = incoming_SW_TOA(GR, self.SWINTOA, self.SOLZEN) 

        if self.i_radiation >= 2:
            # two-stream method fluxes and divergences
            self.LWFLXUP =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.LWFLXDO =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.LWFLXNET =  np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWFLXUP =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWFLXDO =   np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.SWFLXNET =  np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan)
            self.LWFLXDIV =  np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)
            self.SWFLXDIV =  np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)
            self.TOTFLXDIV = np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan)



    def calc_radiation(self, GR, POTT, TAIR, RHO, PHIVB, SOIL):

        t_start = time.time()

        if self.i_radiation == 1:
            self.pseudo_constant_radiation(GR, POTT)
        elif self.i_radiation == 2:
            self.pseudo_varying_radiation(GR, POTT)
        elif self.i_radiation == 3:
            if GR.ts % self.rad_nth_ts == 0:
                print('calculate radiation')
                self.simple_radiation(GR, POTT, TAIR, RHO, PHIVB, SOIL)
        elif self.i_radiation == 0:
            pass
        else:
            raise NotImplementedError()

        t_end = time.time()
        GR.rad_comp_time += t_end - t_start


    def simple_radiation(self, GR, POTT, TAIR, RHO, PHIVB, SOIL):
        ALTVB = PHIVB / con_g
        dz = ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj]

        self.SOLZEN = rad_solar_zenith_angle(GR, self.SOLZEN)
        self.MYSUN = np.cos(self.SOLZEN)
        self.MYSUN[self.MYSUN < 0] = 0
        self.solar_constant = calc_current_solar_constant(GR) 
        #self.SWINTOA = self.solar_constant * np.maximum(np.cos(self.SOLZEN), 0)
        self.SWINTOA = self.solar_constant * np.cos(self.SOLZEN)

        #import matplotlib.pyplot as plt
        ##plt.contourf(self.SOLZEN.T)
        ##self.MYSUN[self.MYSUN < 0] = 0
        #plt.contourf(self.MYSUN.T)
        #plt.colorbar()
        #plt.show()
        #quit()
       
        for i in range(0,GR.nx):
            #i = 10
            i_ref = i+GR.nb
            for j in range(0,GR.ny):
                #j = 10
                j_ref = j+GR.nb


                # LONGWAVE
                self.LWFLXUP[i,j,:], self.LWFLXDO[i,j,:] = \
                                    org_longwave(GR, dz[i,j],
                                                TAIR[i_ref,j_ref,:], RHO[i_ref,j_ref], \
                                                SOIL.TSOIL[i,j,0], SOIL.ALBEDO[i,j])

                # SHORTWAVE
                if self.MYSUN[i,j] > 0:

                    self.SWFLXUP[i,j,:], self.SWFLXDO[i,j,:] = \
                                        org_shortwave(GR, dz[i,j], self.solar_constant,
                                                    RHO[i_ref,j_ref],
                                                    self.SWINTOA[i,j],
                                                    self.MYSUN[i,j],
                                                    SOIL.ALBEDO[i,j])
                else:
                    self.SWFLXUP [i,j,:] = 0
                    self.SWFLXDO [i,j,:] = 0

        print(np.max(self.SWFLXUP[:,:,-1]))
        print(np.max(self.SWFLXDO[:,:,-1]))
        #quit()

        self.LWFLXNET = self.LWFLXDO - self.LWFLXUP 
        self.SWFLXNET = self.SWFLXDO - self.SWFLXUP 

        i = 6
        j = 5
        plt.plot(self.SWFLXUP[i,j,:], ALTVB[i,j,:])
        plt.plot(self.SWFLXDO[i,j,:], ALTVB[i,j,:])
        plt.plot(self.SWFLXNET[i,j,:], ALTVB[i,j,:])
        plt.axvline(x=0, color='k')
        plt.show()
        quit()

        for k in range(0,GR.nz):

            self.SWFLXDIV[:,:,k] = ( self.SWFLXNET[:,:,k] - self.SWFLXNET[:,:,k+1] ) \
                                    / dz[:,:,k]
            self.LWFLXDIV[:,:,k] = ( self.LWFLXNET[:,:,k] - self.LWFLXNET[:,:,k+1] ) \
                                    / dz[:,:,k]
            self.TOTFLXDIV[:,:,k] = self.SWFLXDIV[:,:,k] + self.LWFLXDIV[:,:,k]
            
            self.dPOTTdt_RAD[:,:,k] = 1/(con_cp * RHO[:,:,k][GR.iijj]) * \
                                                self.TOTFLXDIV[:,:,k]


    def pseudo_varying_radiation(self, GR, POTT):
        self.SOLZEN = rad_solar_zenith_angle(GR, self.SOLZEN)
        self.SWINTOA = incoming_SW_TOA(GR, self.SWINTOA, self.SOLZEN) 

        for k in range(0,GR.nz):
            self.dPOTTdt_RAD[:,:,k] = \
                                -  pseudo_rad_outRate*POTT[:,:,k][GR.iijj]**1/2  \
                                +  pseudo_rad_inpRate*self.SWINTOA/1000


    def pseudo_constant_radiation(self, GR, POTT):
        for k in range(0,GR.nz):
            self.dPOTTdt_RAD[:,:,k] = \
                                -  pseudo_rad_outRate*POTT[:,:,k][GR.iijj]**1  \
                                +  pseudo_rad_inpRate*np.cos(GR.lat_rad[GR.iijj])


