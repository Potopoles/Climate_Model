import numpy as np


class soil:

    # constant values
    depth_soil = 4
    depth_ocean = 50
    con_cp_soil = 2000 
    con_cp_ocean = 4184
    rho_soil = 3000
    rho_water = 1000

    # initial values (kg/kg air equivalent)
    moisture_ocean = 10*1E-3
    moisture_soil = 1*1E-3

    def __init__(self, GR, HSURF):

        self.nz_soil = 1

        self.HSURF = HSURF
        

        # CONSTANT FIELDS 
        # mask: ocean = 1, other = 0
        self.OCEANMSK = np.zeros( ( GR.nx, GR.ny ), np.int )
        self.OCEANMSK[self.HSURF[GR.iijj] <= 100] = 1

        self.DEPTH = np.full( ( GR.nx, GR.ny ), self.depth_soil )
        self.DEPTH[self.OCEANMSK == 1] = self.depth_ocean

        self.CP = np.full( ( GR.nx, GR.ny ), self.con_cp_soil )
        self.CP[self.OCEANMSK == 1] = self.con_cp_ocean

        self.RHO = np.full( ( GR.nx, GR.ny ), self.rho_soil )
        self.RHO[self.OCEANMSK == 1] = self.rho_water


        # DYNAMIC FIELDS
        self.TSOIL = np.full( ( GR.nx, GR.ny, self.nz_soil ), np.nan)
        self.TSOIL[:,:,0] = 295 - np.sin(GR.lat_rad[GR.iijj])**2*30

        self.MOIST = np.full( ( GR.nx, GR.ny ), self.moisture_soil )
        self.MOIST[self.OCEANMSK == 1] = self.moisture_ocean

        self.ALBEDOSW = np.full( ( GR.nx, GR.ny ), np.nan) 
        self.ALBEDOLW = np.full( ( GR.nx, GR.ny ), np.nan) 
        self.ALBEDOSW, self.ALBEDOLW = self.calc_albedo(GR, self.ALBEDOSW, self.ALBEDOLW, 
                                                            self.TSOIL, self.OCEANMSK)



    def advance_timestep(self, GR, RAD):


        if RAD.i_radiation > 0:
            dTSURFdt = (RAD.LWFLXNET[:,:,GR.nzs-1] + RAD.SWFLXNET[:,:,GR.nzs-1])/ \
                            (self.CP * self.RHO * self.DEPTH)

            self.TSOIL[:,:,0] = self.TSOIL[:,:,0] + GR.dt * dTSURFdt

            self.ALBEDOSW, self.ALBEDOLW = self.calc_albedo(GR, self.ALBEDOSW, self.ALBEDOLW,
                                                            self.TSOIL, self.OCEANMSK)


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
