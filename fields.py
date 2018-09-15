import numpy as np
import time
from namelist import *
from boundaries import exchange_BC
from IO import load_topo, load_restart_fields, load_profile
from diagnostics import diagnose_secondary_fields, diagnose_POTTVB_jacobson
from geopotential import diag_geopotential_jacobson, diag_pvt_factor
from jacobson import diagnose_fields_jacobson
from radiation.org_radiation import radiation
from soil_model import soil
from org_microphysics import microphysics
from org_turbulence import turbulence
from constants import con_g, con_Rd, con_kappa, con_cp

from numba import cuda

##############################################################################
# ORDER OF FIELDS
# pressure fields
# flux fields
# velocity fields
# temperature fields
# primary diagnostic fields (relevant for dynamics)
# secondary diagnostic fields (not relevant for dynamics)
# constant fields
# radiation fields
# microphysics fields

class CPU_Fields:
    
    def __init__(self, GR, subgrids):

        ##############################################################################
        # 2D FIELDS
        # pressure fields
        self.COLP_OLD    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        self.COLP        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        self.COLP_NEW    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        self.dCOLPdt     = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        # secondary diagnostic fields
        self.PSURF       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        # constant fields
        self.HSURF       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), np.nan)
        self.OCEANMSK    = np.full( ( GR.nx         , GR.ny                  ), np.nan)

        ##############################################################################
        # 3D FIELDS
        # flux fields
        self.UFLX        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.dUFLXdt     = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.VFLX        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.dVFLXdt     = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.FLXDIV      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.BFLX        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.CFLX        = np.full( ( GR.nxs+2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.DFLX        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.EFLX        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.RFLX        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.QFLX        = np.full( ( GR.nxs+2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.SFLX        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.TFLX        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        # velocity fields
        self.UWIND_OLD   = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.UWIND       = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.VWIND_OLD   = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.VWIND       = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), np.nan)
        self.WIND        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.WWIND       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        # temperature fields
        self.POTT_OLD    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.POTT        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.dPOTTdt     = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.POTTVB      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        self.TAIR        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.TAIRVB      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        # primary diagnostic fields
        self.PHI         = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.PHIVB       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        self.PVTF        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.PVTFVB      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), np.nan)
        # secondary diagnostic fields
        self.PAIR        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        self.RHO         = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan)
        # radiation fields
        self.dPOTTdt_RAD = np.full( ( GR.nx         , GR.ny         , GR.nz  ), np.nan)
        # microphysics fields
        self.QV_OLD      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
        self.QV          = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
        self.dQVdt       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
        self.QC_OLD      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
        self.QC          = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
        self.dQCdt       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), np.nan) 
        self.dQVdt_MIC   = np.full( ( GR.nx         , GR.ny         , GR.nz  ), np.nan)
        self.dQCdt_MIC   = np.full( ( GR.nx         , GR.ny         , GR.nz  ), np.nan)
        self.dPOTTdt_MIC = np.full( ( GR.nx         , GR.ny         , GR.nz  ), np.nan)


class GPU_Fields:

    def __init__(self, GR, subgrids, CF):

        t_start = time.time()

        ##############################################################################
        # 2D FIELDS
        # pressure fields
        self.COLP_OLD         = cuda.to_device(CF.COLP_OLD,     GR.stream)
        self.COLP             = cuda.to_device(CF.COLP,         GR.stream)
        self.COLP_NEW         = cuda.to_device(CF.COLP_NEW,     GR.stream)
        self.dCOLPdt          = cuda.to_device(CF.dCOLPdt,      GR.stream)
        # secondary diagnostic fields
        self.PSURF            = cuda.to_device(CF.PSURF,        GR.stream)
        # constant fields
        self.HSURF            = cuda.to_device(CF.HSURF,        GR.stream)
        #self.OCEANMSK         = cuda.to_device(CF.HSURF,        GR.stream) 

        ##############################################################################
        # 3D FIELDS
        # flux fields
        self.UFLX             = cuda.to_device(CF.UFLX,         GR.stream)
        self.dUFLXdt          = cuda.to_device(CF.dUFLXdt,      GR.stream)
        self.VFLX             = cuda.to_device(CF.VFLX,         GR.stream)
        self.dVFLXdt          = cuda.to_device(CF.dVFLXdt,      GR.stream)
        self.FLXDIV           = cuda.to_device(CF.FLXDIV,       GR.stream)
        self.BFLX             = cuda.to_device(CF.BFLX,         GR.stream)
        self.CFLX             = cuda.to_device(CF.CFLX,         GR.stream)
        self.DFLX             = cuda.to_device(CF.DFLX,         GR.stream)
        self.EFLX             = cuda.to_device(CF.EFLX,         GR.stream)
        self.RFLX             = cuda.to_device(CF.RFLX,         GR.stream)
        self.QFLX             = cuda.to_device(CF.QFLX,         GR.stream)
        self.SFLX             = cuda.to_device(CF.SFLX,         GR.stream)
        self.TFLX             = cuda.to_device(CF.TFLX,         GR.stream)
        # velocity fields
        self.UWIND_OLD        = cuda.to_device(CF.UWIND_OLD,    GR.stream)
        self.UWIND            = cuda.to_device(CF.UWIND,        GR.stream)
        self.VWIND_OLD        = cuda.to_device(CF.VWIND_OLD,    GR.stream)
        self.VWIND            = cuda.to_device(CF.VWIND,        GR.stream)
        self.WIND             = cuda.to_device(CF.WIND,         GR.stream)
        self.WWIND            = cuda.to_device(CF.WWIND,        GR.stream)
        # temperature fields
        self.POTT_OLD         = cuda.to_device(CF.POTT_OLD,     GR.stream)
        self.POTT             = cuda.to_device(CF.POTT,         GR.stream)
        self.dPOTTdt          = cuda.to_device(CF.dPOTTdt,      GR.stream)
        self.POTTVB           = cuda.to_device(CF.POTTVB,       GR.stream)
        self.TAIR             = cuda.to_device(CF.TAIR,         GR.stream)
        self.TAIRVB           = cuda.to_device(CF.TAIRVB,       GR.stream)
        # primary diagnostic fields
        self.PHI              = cuda.to_device(CF.PHI,          GR.stream)
        self.PHIVB            = cuda.to_device(CF.PHIVB,        GR.stream)
        self.PVTF             = cuda.to_device(CF.PVTF,         GR.stream)
        self.PVTFVB           = cuda.to_device(CF.PVTFVB,       GR.stream)
        # secondary diagnostic fields
        self.PAIR             = cuda.to_device(CF.PAIR,         GR.stream)
        self.RHO              = cuda.to_device(CF.RHO,          GR.stream)
        # radiation fields
        self.dPOTTdt_RAD      = cuda.to_device(CF.dPOTTdt_RAD,  GR.stream) 
        # microphysics fields
        self.QV_OLD           = cuda.to_device(CF.QV_OLD,       GR.stream) 
        self.QV               = cuda.to_device(CF.QV,           GR.stream) 
        self.dQVdt            = cuda.to_device(CF.dQVdt,        GR.stream) 
        self.QC_OLD           = cuda.to_device(CF.QC_OLD,       GR.stream) 
        self.QC               = cuda.to_device(CF.QC,           GR.stream) 
        self.dQCdt            = cuda.to_device(CF.dQCdt,        GR.stream) 
        self.dQVdt_MIC        = cuda.to_device(CF.dQVdt_MIC,    GR.stream)
        self.dQCdt_MIC        = cuda.to_device(CF.dQCdt_MIC,    GR.stream)
        self.dPOTTdt_MIC      = cuda.to_device(CF.dPOTTdt_MIC,  GR.stream)

        t_end = time.time()
        GR.copy_time += t_end - t_start

    def copy_fields_to_host(self, GR):
        self.COLP             .to_host(GR.stream)
        self.PAIR             .to_host(GR.stream)
        self.PHI              .to_host(GR.stream)
        self.PHIVB            .to_host(GR.stream)
        self.UWIND            .to_host(GR.stream)
        self.VWIND            .to_host(GR.stream)
        self.WIND             .to_host(GR.stream)
        self.WWIND            .to_host(GR.stream) 
        self.POTT             .to_host(GR.stream)
        self.TAIR             .to_host(GR.stream)
        self.RHO              .to_host(GR.stream)
        self.PVTFVB           .to_host(GR.stream)
        self.QV_OLD           .to_host(GR.stream)
        self.QV               .to_host(GR.stream)
        self.dQVdt            .to_host(GR.stream)
        self.QC_OLD           .to_host(GR.stream)
        self.QC               .to_host(GR.stream)
        self.dQCdt            .to_host(GR.stream)
        self.dQCdt_MIC        .to_host(GR.stream)
        self.dPOTTdt_RAD      .to_host(GR.stream)
        self.dPOTTdt_MIC      .to_host(GR.stream)

        GR.stream.synchronize()


def initialize_fields(GR, subgrids, F):
    if i_load_from_restart:
        raise NotImplementedError()
        #COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND, \
        #UFLX, VFLX, \
        #HSURF, POTT, TAIR, TAIRVB, RHO, \
        #POTTVB, PVTF, PVTFVB, \
        #RAD, SOIL, MIC, TURB = load_restart_fields(GR)
    else:

        ####################################################################
        # SET INITIAL FIELD VALUES
        ####################################################################

        # need to have non-nan-values because values from half-level 0 and GR.nzs
        # are used in calculation.
        F.POTTVB[:] = 0
        F.WWIND[:] = 0

        #  LOAD TOPOGRAPHY (HSURF)
        F.HSURF = load_topo(GR, F.HSURF) 
        if not i_use_topo:
            F.HSURF[GR.iijj] = 0.
            F.HSURF = exchange_BC(GR, F.HSURF)

        # INITIALIZE PROFILE
        GR, F.COLP, F.PSURF, F.POTT, F.TAIR \
                = load_profile(GR, subgrids, F.COLP, F.HSURF, F.PSURF, F.PVTF, \
                                F.PVTFVB, F.POTT, F.TAIR)

        # LOAD PROFILE TO GPU
        if comp_mode == 2:
            GR.dsigmad       = cuda.to_device(GR.dsigma, GR.stream)
            GR.sigma_vbd     = cuda.to_device(GR.sigma_vb, GR.stream)

        # INITIAL CONDITIONS
        F.COLP = gaussian2D(GR, F.COLP, COLP_gaussian_pert, np.pi*3/4, 0, \
                            gaussian_dlon, gaussian_dlat)
        F.COLP = random2D(GR, F.COLP, COLP_random_pert)

        for k in range(0,GR.nz):
            F.UWIND[:,:,k][GR.iisjj] = u0   
            F.UWIND[:,:,k] = gaussian2D(GR, F.UWIND[:,:,k], UWIND_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            F.UWIND[:,:,k] = random2D(GR, F.UWIND[:,:,k], UWIND_random_pert)
            F.VWIND[:,:,k][GR.iijjs] = v0
            F.VWIND[:,:,k] = gaussian2D(GR, F.VWIND[:,:,k], VWIND_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            F.VWIND[:,:,k] = random2D(GR, F.VWIND[:,:,k], VWIND_random_pert)

            F.POTT[:,:,k] = gaussian2D(GR, F.POTT[:,:,k], POTT_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            F.POTT[:,:,k] = random2D(GR, F.POTT[:,:,k], POTT_random_pert)

        # BOUNDARY EXCHANGE OF INITIAL CONDITIONS
        F.COLP  = exchange_BC(GR, F.COLP)
        F.UWIND  = exchange_BC(GR, F.UWIND)
        F.VWIND  = exchange_BC(GR, F.VWIND)
        F.POTT  = exchange_BC(GR, F.POTT)

        # PRIMARY DIAGNOSTIC FIELDS
        diagnose_fields_jacobson(GR, F)

        # SECONDARY DIAGNOSTIC FIELDS
        F.PAIR, F.TAIR, F.TAIRVB, F.RHO, F.WIND = \
                diagnose_secondary_fields(GR, F.COLP, F.PAIR, F.PHI, F.POTT, F.POTTVB,
                                        F.TAIR, F.TAIRVB, F.RHO,\
                                        F.PVTF, F.PVTFVB, F.UWIND, F.VWIND, F.WIND)

        ####################################################################
        # INITIALIZE PROCESSES
        ####################################################################

        # MOISTURE & MICROPHYSICS
        if i_microphysics:
            MIC = microphysics(GR, F, i_microphysics, F.TAIR, F.PAIR) 
        else:
            MIC = None

        # RADIATION
        if i_radiation:
            RAD = radiation(GR, i_radiation, F.dPOTTdt_RAD)
            RAD.calc_radiation(GR, F.TAIR, F.TAIRVB, F.RHO, F.PHIVB, SOIL, MIC)
        else:
            RAD = None

        # TURBULENCE 
        if i_turbulence:
            raise NotImplementedError('Baustelle')
            TURB = turbulence(GR, i_turbulence) 
        else:
            TURB = None

        ####################################################################
        # INITIALIZE NON-ATMOSPHERIC COMPONENTS
        ####################################################################

        # SOIL MODEL
        if i_soil:
            SOIL = soil(GR, F.HSURF)
        else:
            SOIL = None



    return(RAD, SOIL, MIC, TURB)



def random2D(GR, FIELD, pert):
    FIELD = FIELD + pert*np.random.rand(FIELD.shape[0], FIELD.shape[1])
    return(FIELD)

def gaussian2D(GR, FIELD, pert, lon0_rad, lat0_rad, lonSig_rad, latSig_rad):

    dimx,dimy = FIELD.shape

    if (dimy == GR.nys+2*GR.nb): # staggered in y 
        selinds = GR.iijjs
        perturb = pert*np.exp( \
                - np.power(GR.lon_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
                - np.power(GR.latjs_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )
    elif (dimx == GR.nxs+2*GR.nb): # staggered in x 
        selinds = GR.iisjj
        perturb = pert*np.exp( \
                - np.power(GR.lonis_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
                - np.power(GR.lat_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )
    else: # unstaggered in y and x 
        selinds = GR.iijj
        perturb = pert*np.exp( \
                - np.power(GR.lon_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
                - np.power(GR.lat_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )

    FIELD[selinds] = FIELD[selinds] + perturb

    return(FIELD)


