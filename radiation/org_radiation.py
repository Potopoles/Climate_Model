import numpy as np
import scipy
import time
from radiation.namelist_radiation import \
        pseudo_rad_inpRate, pseudo_rad_outRate, \
        rad_nth_hour 
from constants import con_cp, con_g
from radiation.shortwave import rad_solar_zenith_angle, \
       calc_current_solar_constant, rad_calc_SW_RTE_matrix 
from radiation.longwave import rad_calc_LW_RTE_matrix, \
        calc_planck_intensity


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
        self.solar_constant = calc_current_solar_constant(GR) 
        self.SWINTOA = self.solar_constant * np.maximum(np.cos(self.SOLZEN), 0)
       
        for i in range(0,GR.nx):
            #i = 10
            i_ref = i+GR.nb
            for j in range(0,GR.ny):
                #j = 10
                j_ref = j+GR.nb

                # LONGWAVE
                nu0 = 50
                nu1 = 2500
                taus = np.zeros(GR.nzs)
                taus[0] = 0
                taus[1] = 0.114 + taus[0]
                taus[2] = 1.272 + taus[1] 
                taus[3] = 4.543 + taus[2]
                dtau = np.diff(taus)
                gamma1 = np.zeros(GR.nz)
                gamma1[0] = 1.73 
                gamma1[1] = 1.73
                gamma1[2] = 1.73
                gamma2 = np.zeros(GR.nz)
                gamma2[0] = 0.12E-9 
                gamma2[1] = 0.01E-9 
                gamma2[2] = 0.003E-9 
                emissivity_surface = 0.9
                albedo_surface_LW = 0.126
                # single scattering albedo
                omega_s = 0

                B_air = 2*np.pi * (1 - omega_s) * \
                        calc_planck_intensity(GR, nu0, nu1, TAIR[i_ref,j_ref,:])
                B_surf = emissivity_surface * np.pi * \
                        calc_planck_intensity(GR, nu0, nu1, SOIL.TSOIL[i,j,0])

                A_mat, g_vec = rad_calc_LW_RTE_matrix(GR, dtau, gamma1, gamma2,
                                    B_air, B_surf, albedo_surface_LW)
                fluxes = scipy.sparse.linalg.spsolve(A_mat, g_vec)

                self.LWFLXUP [i,j,:] = fluxes[range(1,len(fluxes),2)]
                self.LWFLXDO [i,j,:] = fluxes[range(0,len(fluxes),2)]


                # SHORTWAVE
                if self.SWINTOA[i,j] > 0:

                    taus = np.zeros(GR.nzs)
                    taus[0] = 0
                    taus[1] = 0.056
                    #taus[2] = 0.3375
                    taus[2] = 0.25
                    #taus[3] = 1.2242
                    taus[3] = 0.9
                    dtau = np.diff(taus)
                    tau = taus[:-1] + dtau/2
                    gamma1 = np.zeros(GR.nz)
                    gamma1[0] = 1.73 
                    gamma1[1] = 1.73
                    gamma1[2] = 1.73
                    gamma2 = np.zeros(GR.nz)
                    gamma2[0] = 0.163E-5 
                    gamma2[1] = 0.034E-5 
                    gamma2[2] = 0.011E-5 
                    gamma3 = np.zeros(GR.nz)
                    gamma3[0] = 0.5
                    gamma3[1] = 0.5
                    gamma3[2] = 0.5
                    albedo_surface_SW = 0.126
                    # single scattering albedo
                    omega_s = 0

                    surf_reflected_SW = albedo_surface_SW * self.SWINTOA[i,j] * \
                                            np.exp(-taus[GR.nzs-1]/self.MYSUN[i,j])
                    sw_dir = omega_s * self.solar_constant * np.exp(-tau/self.MYSUN[i,j])

                    A_mat, g_vec = rad_calc_SW_RTE_matrix(GR, dtau, gamma1, gamma2, gamma3,
                                            self.SWINTOA[i,j], surf_reflected_SW,
                                            albedo_surface_SW, sw_dir)
                    fluxes = scipy.sparse.linalg.spsolve(A_mat, g_vec)

                    self.SWFLXUP [i,j,:] = fluxes[range(1,len(fluxes),2)]
                    self.SWFLXDO [i,j,:] = fluxes[range(0,len(fluxes),2)]
                else:
                    self.SWFLXUP [i,j,:] = 0
                    self.SWFLXDO [i,j,:] = 0


        self.LWFLXNET = self.LWFLXDO - self.LWFLXUP 
        self.SWFLXNET = self.SWFLXDO - self.SWFLXUP 
        #i = 1
        #j = 1
        #print(self.LWFLXNET[i,j,:])
        #print(self.SWFLXNET[i,j,:])
        #quit()


        for k in range(0,GR.nz):

            self.SWFLXDIV[:,:,k] = ( self.SWFLXNET[:,:,k] - self.SWFLXNET[:,:,k+1] ) \
                                    / dz[:,:,k]
            self.LWFLXDIV[:,:,k] = ( self.LWFLXNET[:,:,k] - self.LWFLXNET[:,:,k+1] ) \
                                    / dz[:,:,k]
            self.TOTFLXDIV[:,:,k] = self.SWFLXDIV[:,:,k] + self.LWFLXDIV[:,:,k]
            
            self.dPOTTdt_RAD[:,:,k] = 1/(con_cp * RHO[:,:,k][GR.iijj]) * \
                                                self.TOTFLXDIV[:,:,k]

        #i = 1
        #j = 1
        #print(self.dPOTTdt_RAD[i,j,:])
        #quit()


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


