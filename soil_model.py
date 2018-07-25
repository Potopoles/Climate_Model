import numpy as np


class soil:

    height_soil = 2
    con_cp_soil = 2000 
    rho_soil = 3000

    def __init__(self, GR, HSURF):

        self.nz_soil = 1
        
        self.TSOIL = np.full( ( GR.nx, GR.ny, self.nz_soil ), np.nan)
        self.TSOIL[:,:,0] = 285


    def advance_timestep(self, GR, RAD):


        dTSURFdt = (RAD.LWFLXNET[:,:,GR.nzs-1] + RAD.SWFLXNET[:,:,GR.nzs-1])/ \
                        (self.con_cp_soil * self.rho_soil * self.height_soil)

        self.TSOIL[:,:,0] = self.TSOIL[:,:,0] + GR.dt * dTSURFdt


