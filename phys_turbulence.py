import numpy as np
#import scipy
import time
from namelist import pTop
#import matplotlib.pyplot as plt
from constants import con_Rd, con_g, con_kappa
#from boundaries import exchange_BC
from org_namelist import wp


class turbulence:
    
    def __init__(self, GR, i_turbulence):
        print('Prepare Turbulence')

        self.i_turbulence = i_turbulence 

        #self.RHO   = np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan, dtype=wp)
        self.RHOVB = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp)
        self.K_h   = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp) 
        self.K_h[:,:,1:GR.nzs-1] = 0.1

        self.dzs = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp)
        self.dz = np.full( ( GR.nx, GR.ny, GR.nz ), np.nan, dtype=wp)

    def diag_rho(self, GR, COLP, POTT, PVTF, POTTVB, PVTFVB):
        PAIR = 100000*np.power(PVTF[GR.iijj], 1/con_kappa)
        for k in range(0,GR.nz):
            tair = POTT[:,:,k][GR.iijj] / PVTF[:,:,k][GR.iijj]
            self.RHO[:,:,k] = PAIR[:,:,k] / (con_Rd * tair)

        for ks in range(0,GR.nzs):
            pairvb = pTop + COLP[GR.iijj] * GR.sigma_vb[ks]
            tairvb = POTTVB[:,:,ks][GR.iijj] / PVTFVB[:,:,ks][GR.iijj]
            self.RHOVB[:,:,ks] = pairvb / (con_Rd * tairvb)

        #print(np.mean(np.mean(self.RHO,0),0))
        #print(np.mean(np.mean(self.RHOVB,0),0))
        #quit()

    def diag_dz(self, GR, PHI, PHIVB):
        ALTVB = PHIVB / con_g
        self.dz[:,:,:] = - (ALTVB[:,:,:-1][GR.iijj] -  ALTVB[:,:,1:][GR.iijj])
        ALT = PHI / con_g
        self.dzs[:,:,1:GR.nzs-1] = -(ALT[:,:,:-1][GR.iijj] -  ALT[:,:,1:][GR.iijj])

        #print(np.mean(np.mean(self.dz,0),0))
        #print(np.mean(np.mean(self.dzs,0),0))
        #quit()


    def turbulent_flux_divergence(self, GR, VAR):

        dvardz = np.diff(VAR[GR.iijj], axis=2) / self.dzs[:,:,1:GR.nzs-1]

        FLUXVB = np.zeros( ( GR.nx, GR.ny, GR.nzs ) , dtype=wp) 
        FLUXVB[:,:,1:GR.nzs-1] = self.RHOVB[:,:,1:GR.nzs-1] * \
                                self.K_h[:,:,1:GR.nzs-1] * dvardz

        FLUXDIV = np.diff(FLUXVB, axis=2) / (self.dz * self.RHO)

        return(FLUXDIV)

