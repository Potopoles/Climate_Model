import numpy as np


class soil:

    height_soil = 2
    con_cp_soil = 2000 
    rho_soil = 3000


    def __init__(self, GR, HSURF):

        self.nz_soil = 1

        self.HSURF = HSURF
        
        self.TSOIL = np.full( ( GR.nx, GR.ny, self.nz_soil ), np.nan)
        self.TSOIL[:,:,0] = 286
        #self.TSOIL[:,:,0] = 254

        # mask: ocean = 1, other = 0
        self.OCEANMSK = np.zeros( ( GR.nx, GR.ny ), np.int )
        self.OCEANMSK[self.HSURF[GR.iijj] <= 100] = 1

        self.ALBEDOSW = np.full( ( GR.nx, GR.ny ), np.nan) 
        self.ALBEDOLW = np.full( ( GR.nx, GR.ny ), np.nan) 
        self.ALBEDOSW, self.ALBEDOLW = self.calc_albedo(GR, self.ALBEDOSW, self.ALBEDOLW, self.TSOIL, self.OCEANMSK)


    def advance_timestep(self, GR, RAD):


        dTSURFdt = (RAD.LWFLXNET[:,:,GR.nzs-1] + RAD.SWFLXNET[:,:,GR.nzs-1])/ \
                        (self.con_cp_soil * self.rho_soil * self.height_soil)

        self.TSOIL[:,:,0] = self.TSOIL[:,:,0] + GR.dt * dTSURFdt

        self.ALBEDOSW, self.ALBEDOLW = self.calc_albedo(GR, self.ALBEDOSW, self.ALBEDOLW, self.TSOIL, self.OCEANMSK)


    def calc_albedo(self, GR, ALBEDOSW, ALBEDOLW, TSOIL, OCEANMSK):
        # forest
        ALBEDOSW[:] = 0.2
        ALBEDOLW[:] = 0.2
        # ocean
        ALBEDOSW[OCEANMSK == 1] = 0.05
        ALBEDOLW[OCEANMSK == 1] = 0.05
        # land and sea ice
        ALBEDOSW[TSOIL[:,:,0] <= 273.15] = 0.7
        ALBEDOLW[TSOIL[:,:,0] <= 273.15] = 0.3

        return(ALBEDOSW, ALBEDOLW)
