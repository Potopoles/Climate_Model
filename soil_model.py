import numpy as np


class soil:

    height_soil = 2
    con_cp_soil = 2000 
    rho_soil = 3000


    def __init__(self, GR, HSURF):

        self.nz_soil = 1

        self.HSURF = HSURF
        
        self.TSOIL = np.full( ( GR.nx, GR.ny, self.nz_soil ), np.nan)
        self.TSOIL[:,:,0] = 284
        self.TSOIL[:,:,0] = 254

        # mask: ocean = 1, other = 0
        self.OCEANMSK = np.zeros( ( GR.nx, GR.ny ), np.int )
        self.OCEANMSK[self.HSURF[GR.iijj] <= 100] = 1

        self.ALBEDO = np.full( ( GR.nx, GR.ny ), np.nan) 
        self.ALBEDO = self.calc_albedo(GR, self.ALBEDO, self.TSOIL, self.OCEANMSK)


    def advance_timestep(self, GR, RAD):


        dTSURFdt = (RAD.LWFLXNET[:,:,GR.nzs-1] + RAD.SWFLXNET[:,:,GR.nzs-1])/ \
                        (self.con_cp_soil * self.rho_soil * self.height_soil)

        self.TSOIL[:,:,0] = self.TSOIL[:,:,0] + GR.dt * dTSURFdt

        self.ALBEDO = self.calc_albedo(GR, self.ALBEDO, self.TSOIL, self.OCEANMSK)


    def calc_albedo(self, GR, ALBEDO, TSOIL, OCEANMSK):
        # forest
        ALBEDO[:] = 0.2
        # ocean
        ALBEDO[OCEANMSK == 1] = 0.05
        # land and sea ice
        ALBEDO[TSOIL[:,:,0] <= 273.15] = 0.8

        #ii = 1:GR.nx
        #jj = 1:GR.ny

        #for i in range(0,3):
        #    ALBEDO[ii,jj] = ALBEDO[ii,jj] + 0.1*\
        #                                (ALBEDO[ii-1,jj] + ALBEDO[ii+1,jj] + \
        #                                ALBEDO[ii,jj+1] + ALBEDO[ii,jj+1] - \
        #                                4*ALBEDO[ii,jj]) 

        return(ALBEDO)
