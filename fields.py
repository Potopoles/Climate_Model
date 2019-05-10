import numpy as np
import time
from namelist import *
from org_namelist import wp
from boundaries import exchange_BC
from IO import load_topo, load_restart_fields, load_profile
from diagnostics import diagnose_secondary_fields, diagnose_POTTVB_jacobson
from geopotential import diag_geopotential_jacobson, diag_pvt_factor
from jacobson import diagnose_fields_jacobson
from radiation.org_radiation import radiation
from surface_model import surface
from grid import tpbh, tpbv, tpbvs
from org_microphysics import microphysics
from org_turbulence import turbulence
from constants import con_g, con_Rd, con_kappa, con_cp
from surface_model import nz_soil
from numba import cuda

##############################################################################
# ORDER OF FIELDS
# pressure fields
# flux fields
# velocity fields
# temperature fields
# primary diagnostic fields (relevant for dynamics)
# secondary diagnostic fields (not relevant for dynamics)

class CPU_Fields:
    
    def __init__(self, GR, subgrids):

        ##############################################################################
        # 2D FIELDS
        ##############################################################################

        # pressure fields
        ##############################################################################
        self.COLP_OLD    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ),
                                    np.nan, dtype=wp)
        self.COLP        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), 
                                    np.nan, dtype=wp)
        self.COLP_NEW    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ),
                                    np.nan, dtype=wp)
        self.dCOLPdt     = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), 
                                    np.nan, dtype=wp)
        # secondary diagnostic fields
        ##############################################################################
        self.PSURF       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), 
                                    np.nan, dtype=wp)

        # constant fields
        ##############################################################################
        self.HSURF       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb         ), 
                                    np.nan, dtype=wp)

        ##############################################################################
        # 3D FIELDS
        ##############################################################################

        # flux fields
        ##############################################################################
        self.UFLX        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.dUFLXdt     = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.VFLX        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.dVFLXdt     = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.FLXDIV      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.BFLX        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.CFLX        = np.full( ( GR.nxs+2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.DFLX        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.EFLX        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.RFLX        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.QFLX        = np.full( ( GR.nxs+2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.SFLX        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.TFLX        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.WWIND_UWIND = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        self.WWIND_VWIND = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)

        # velocity field
        ##############################################################################
        self.UWIND_OLD   = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.UWIND       = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.VWIND_OLD   = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.VWIND       = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.WIND        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.WWIND       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)

        # temperature fields
        ##############################################################################
        self.POTT_OLD    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.POTT        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.dPOTTdt     = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.POTTVB      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        self.TAIR        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.TAIRVB      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)

        # primary diagnostic fields
        ##############################################################################
        self.PHI         = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.PHIVB       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        self.PVTF        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.PVTFVB      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)

        # secondary diagnostic fields
        ##############################################################################
        self.PAIR        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        self.RHO         = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)

        ##############################################################################
        # SURFACE
        ##############################################################################
        if i_surface: 
            self.OCEANMASK   = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp)
            self.SOILDEPTH   = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp)
            self.SOILCP      = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp)
            self.SOILRHO     = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp)
            self.SOILTEMP    = np.full( ( GR.nx, GR.ny, nz_soil ), np.nan, dtype=wp)
            self.SOILMOIST   = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp)
            self.SOILEVAPITY = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp)
            self.SURFALBEDSW = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp) 
            self.SURFALBEDLW = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp) 
            self.RAINRATE    = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp) 
            self.ACCRAIN     = np.full( ( GR.nx, GR.ny          ), np.nan, dtype=wp) 

        ##############################################################################
        # RADIATION
        ##############################################################################
        self.dPOTTdt_RAD = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)
        self.LWFLXNET    = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp)
        self.SWFLXNET    = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp)

        ##############################################################################
        # MICROPHYSICS
        ##############################################################################
        self.QV_OLD      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp) 
        self.QV          = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp) 
        self.dQVdt       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp) 
        self.QC_OLD      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp) 
        self.QC          = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp) 
        self.dQCdt       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp) 
        self.dQVdt_MIC   = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)
        self.dQCdt_MIC   = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)
        self.dPOTTdt_MIC = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)


class GPU_Fields:

    def __init__(self, GR, subgrids, CF):

        self.CF = CF

        if not hasattr(GR, 'stream'):
            GR.stream = cuda.stream()

            if tpbh > 1:
                raise NotImplementedError('tpbh > 1 not yet possible see below')
            elif tpbv != GR.nz:
                raise NotImplementedError('tpbv != nz not yet possible see below')
            GR.blockdim      = (tpbh, tpbh, tpbv)
            GR.blockdim_ks   = (tpbh, tpbh, tpbv+1)
            GR.blockdim_xy   = (tpbh, tpbh, 1)
            GR.griddim       = ((GR.nx +2*GR.nb)//GR.blockdim[0], \
                                  (GR.ny +2*GR.nb)//GR.blockdim[1], \
                                   GR.nz //GR.blockdim[2])
            GR.griddim_is    = ((GR.nxs+2*GR.nb)//GR.blockdim[0], \
                                  (GR.ny +2*GR.nb)//GR.blockdim[1], \
                                   GR.nz //GR.blockdim[2])
            GR.griddim_js    = ((GR.nx +2*GR.nb)//GR.blockdim[0], \
                                  (GR.nys+2*GR.nb)//GR.blockdim[1], \
                                   GR.nz //GR.blockdim[2])
            GR.griddim_is_js = ((GR.nxs+2*GR.nb)//GR.blockdim[0], \
                                  (GR.nys+2*GR.nb)//GR.blockdim[1], \
                                   GR.nz //GR.blockdim[2])
            GR.griddim_ks    = ((GR.nx +2*GR.nb)//GR.blockdim_ks[0], \
                                  (GR.ny +2*GR.nb)//GR.blockdim_ks[1], \
                                   GR.nzs//GR.blockdim_ks[2])
            GR.griddim_is_ks = ((GR.nxs+2*GR.nb)//GR.blockdim[0], \
                                  (GR.ny +2*GR.nb)//GR.blockdim[1], \
                                   GR.nzs//GR.blockdim_ks[2])
            GR.griddim_js_ks = ((GR.nx +2*GR.nb)//GR.blockdim[0], \
                                  (GR.nys+2*GR.nb)//GR.blockdim[1], \
                                   GR.nzs//GR.blockdim_ks[2])
            GR.griddim_xy    = ((GR.nx +2*GR.nb)//GR.blockdim_xy[0], \
                                  (GR.ny +2*GR.nb)//GR.blockdim_xy[1], \
                                   1       //GR.blockdim_xy[2])
            GR.griddim_xy_in = ((GR.nx         )//GR.blockdim_xy[0], \
                                  (GR.ny         )//GR.blockdim_xy[1], \
                                   1       //GR.blockdim_xy[2])

            zonal   = np.zeros((2,GR.ny +2*GR.nb   ,GR.nz  ), dtype=wp)
            zonals  = np.zeros((2,GR.nys+2*GR.nb   ,GR.nz  ), dtype=wp)
            zonalvb = np.zeros((2,GR.ny +2*GR.nb   ,GR.nz+1), dtype=wp)
            merid   = np.zeros((  GR.nx +2*GR.nb,2 ,GR.nz  ), dtype=wp)
            merids  = np.zeros((  GR.nxs+2*GR.nb,2 ,GR.nz  ), dtype=wp)
            meridvb = np.zeros((  GR.nx +2*GR.nb,2 ,GR.nz+1), dtype=wp)

            GR.zonal   = cuda.to_device(zonal,  GR.stream)
            GR.zonals  = cuda.to_device(zonals, GR.stream)
            GR.zonalvb = cuda.to_device(zonalvb, GR.stream)
            GR.merid   = cuda.to_device(merid,  GR.stream)
            GR.merids  = cuda.to_device(merids, GR.stream)
            GR.meridvb = cuda.to_device(meridvb, GR.stream)

            GR.Ad            = cuda.to_device(GR.A, GR.stream)
            GR.dxjsd         = cuda.to_device(GR.dxjs, GR.stream)
            GR.corfd         = cuda.to_device(GR.corf, GR.stream)
            GR.corf_isd      = cuda.to_device(GR.corf_is, GR.stream)
            GR.lat_radd      = cuda.to_device(GR.lat_rad, GR.stream)
            GR.latis_radd    = cuda.to_device(GR.latis_rad, GR.stream)
            GR.dsigmad       = cuda.to_device(GR.dsigma, GR.stream)
            GR.sigma_vbd     = cuda.to_device(GR.sigma_vb, GR.stream)

        t_start = time.time()

        ##############################################################################
        # 2D FIELDS
        ##############################################################################

        # pressure fields
        ##############################################################################
        self.COLP_OLD         = cuda.to_device(CF.COLP_OLD,     GR.stream)
        self.COLP             = cuda.to_device(CF.COLP,         GR.stream)
        self.COLP_NEW         = cuda.to_device(CF.COLP_NEW,     GR.stream)
        self.dCOLPdt          = cuda.to_device(CF.dCOLPdt,      GR.stream)

        # secondary diagnostic fields
        ##############################################################################
        self.PSURF            = cuda.to_device(CF.PSURF,        GR.stream)

        # constant fields
        ##############################################################################
        self.HSURF            = cuda.to_device(CF.HSURF,        GR.stream)
        self.OCEANMSK         = cuda.to_device(CF.HSURF,        GR.stream) 

        ##############################################################################
        # 3D FIELDS
        ##############################################################################

        # flux fields
        ##############################################################################
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
        self.WWIND_UWIND      = cuda.to_device(CF.WWIND_UWIND,  GR.stream) 
        self.WWIND_VWIND      = cuda.to_device(CF.WWIND_VWIND,  GR.stream) 

        # velocity fields
        ##############################################################################
        self.UWIND_OLD        = cuda.to_device(CF.UWIND_OLD,    GR.stream)
        self.UWIND            = cuda.to_device(CF.UWIND,        GR.stream)
        self.VWIND_OLD        = cuda.to_device(CF.VWIND_OLD,    GR.stream)
        self.VWIND            = cuda.to_device(CF.VWIND,        GR.stream)
        self.WIND             = cuda.to_device(CF.WIND,         GR.stream)
        self.WWIND            = cuda.to_device(CF.WWIND,        GR.stream)

        # temperature fields
        ##############################################################################
        self.POTT_OLD         = cuda.to_device(CF.POTT_OLD,     GR.stream)
        self.POTT             = cuda.to_device(CF.POTT,         GR.stream)
        self.dPOTTdt          = cuda.to_device(CF.dPOTTdt,      GR.stream)
        self.POTTVB           = cuda.to_device(CF.POTTVB,       GR.stream)
        self.TAIR             = cuda.to_device(CF.TAIR,         GR.stream)
        self.TAIRVB           = cuda.to_device(CF.TAIRVB,       GR.stream)

        # primary diagnostic fields
        ##############################################################################
        self.PHI              = cuda.to_device(CF.PHI,          GR.stream)
        self.PHIVB            = cuda.to_device(CF.PHIVB,        GR.stream)
        self.PVTF             = cuda.to_device(CF.PVTF,         GR.stream)
        self.PVTFVB           = cuda.to_device(CF.PVTFVB,       GR.stream)

        # secondary diagnostic fields
        ##############################################################################
        self.PAIR             = cuda.to_device(CF.PAIR,         GR.stream)
        self.RHO              = cuda.to_device(CF.RHO,          GR.stream)

        ##############################################################################
        # SURFACE
        ##############################################################################
        if i_surface: 
            self.OCEANMASK    = cuda.to_device(CF.OCEANMASK,    GR.stream) 
            self.SOILDEPTH    = cuda.to_device(CF.SOILDEPTH,    GR.stream) 
            self.SOILCP       = cuda.to_device(CF.SOILCP,       GR.stream) 
            self.SOILRHO      = cuda.to_device(CF.SOILRHO,      GR.stream) 
            self.SOILTEMP     = cuda.to_device(CF.SOILTEMP,     GR.stream) 
            self.SOILMOIST    = cuda.to_device(CF.SOILMOIST,    GR.stream) 
            self.SOILEVAPITY  = cuda.to_device(CF.SOILEVAPITY,  GR.stream) 
            self.SURFALBEDSW  = cuda.to_device(CF.SURFALBEDSW,  GR.stream) 
            self.SURFALBEDLW  = cuda.to_device(CF.SURFALBEDLW,  GR.stream) 
            self.RAINRATE     = cuda.to_device(CF.RAINRATE,     GR.stream) 
            self.ACCRAIN      = cuda.to_device(CF.RAINRATE,     GR.stream) 

        ##############################################################################
        # RADIATION
        ##############################################################################
        self.dPOTTdt_RAD      = cuda.to_device(CF.dPOTTdt_RAD,  GR.stream) 
        self.SWFLXNET         = cuda.to_device(CF.SWFLXNET,     GR.stream) 
        self.LWFLXNET         = cuda.to_device(CF.LWFLXNET,     GR.stream) 

        ##############################################################################
        # MICROPHYSICS
        ##############################################################################
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



    def copy_stepDiag_fields_to_host(self, GR):
        t_start = time.time()
        #self.COLP              .to_host(GR.stream)
        #self.WIND              .to_host(GR.stream) 
        #self.POTT              .to_host(GR.stream) 
        #GR.stream.synchronize()

        self.CF.COLP          =    self.COLP.copy_to_host()
        self.CF.WIND          =    self.WIND.copy_to_host() 
        self.CF.POTT          =    self.POTT.copy_to_host() 


        t_end = time.time()
        GR.copy_time += t_end - t_start


    def copy_radiation_fields_to_device(self, GR, CF):
        t_start = time.time()

        self.dPOTTdt_RAD      = cuda.to_device(CF.dPOTTdt_RAD,  GR.stream) 
        self.SWFLXNET         = cuda.to_device(CF.SWFLXNET,     GR.stream) 
        self.LWFLXNET         = cuda.to_device(CF.LWFLXNET,     GR.stream) 

        GR.stream.synchronize()

        t_end = time.time()
        GR.copy_time += t_end - t_start

    def copy_radiation_fields_to_host(self, GR):
        t_start = time.time()
        self.RHO               .to_host(GR.stream)
        self.TAIR              .to_host(GR.stream)
        self.PHIVB             .to_host(GR.stream) 
        self.SOILTEMP          .to_host(GR.stream) 
        self.SURFALBEDLW       .to_host(GR.stream) 
        self.SURFALBEDSW       .to_host(GR.stream) 
        self.QC                .to_host(GR.stream) 

        GR.stream.synchronize()

        t_end = time.time()
        GR.copy_time += t_end - t_start


    def copy_all_fields_to_host(self, GR):
        t_start = time.time()

        # TODO: NEW STYLE (???) MAYBE USE GPU CLASS INSTEAD
        self.CF.COLP          =    self.COLP.copy_to_host()
        self.CF.WIND          =    self.WIND.copy_to_host() 
        self.CF.POTT          =    self.POTT.copy_to_host() 
        self.CF.COLP_NEW      =    self.COLP_NEW.copy_to_host() 

        self.COLP_OLD          .to_host(GR.stream)
        #self.COLP              .to_host(GR.stream)
        #self.COLP_NEW          .to_host(GR.stream)
        self.dCOLPdt           .to_host(GR.stream)
        self.PSURF             .to_host(GR.stream)
        self.HSURF             .to_host(GR.stream) 
        self.OCEANMSK          .to_host(GR.stream) 

        self.UWIND_OLD         .to_host(GR.stream) 
        self.UWIND             .to_host(GR.stream) 
        self.VWIND_OLD         .to_host(GR.stream) 
        self.VWIND             .to_host(GR.stream) 
        #self.WIND              .to_host(GR.stream) 
        self.WWIND             .to_host(GR.stream) 
        self.POTT_OLD          .to_host(GR.stream) 
        #self.POTT              .to_host(GR.stream) 
        self.dPOTTdt           .to_host(GR.stream) 
        self.POTTVB            .to_host(GR.stream) 
        self.TAIR              .to_host(GR.stream) 
        self.TAIRVB            .to_host(GR.stream) 
        self.PHI               .to_host(GR.stream)
        self.PHIVB             .to_host(GR.stream)
        self.PVTF              .to_host(GR.stream)
        self.PVTFVB            .to_host(GR.stream)
        self.PAIR              .to_host(GR.stream)
        self.RHO               .to_host(GR.stream)
        self.dPOTTdt_RAD       .to_host(GR.stream) 
        self.QV_OLD            .to_host(GR.stream) 
        self.QV                .to_host(GR.stream) 
        self.dQVdt             .to_host(GR.stream) 
        self.QC_OLD            .to_host(GR.stream) 
        self.QC                .to_host(GR.stream) 
        self.dQCdt             .to_host(GR.stream) 
        self.dQVdt_MIC         .to_host(GR.stream)
        self.dQCdt_MIC         .to_host(GR.stream)
        self.dPOTTdt_MIC       .to_host(GR.stream)
        ##############################################################################
        # SURFACE
        ##############################################################################
        if i_surface: 
            self.OCEANMASK     .to_host(GR.stream) 
            self.SOILDEPTH     .to_host(GR.stream) 
            self.SOILCP        .to_host(GR.stream) 
            self.SOILRHO       .to_host(GR.stream) 
            self.SOILTEMP      .to_host(GR.stream) 
            self.SOILMOIST     .to_host(GR.stream) 
            self.SOILEVAPITY   .to_host(GR.stream) 
            self.SURFALBEDSW   .to_host(GR.stream) 
            self.SURFALBEDLW   .to_host(GR.stream) 
            self.RAINRATE      .to_host(GR.stream) 
            self.ACCRAIN       .to_host(GR.stream) 

        GR.stream.synchronize()

        t_end = time.time()
        GR.copy_time += t_end - t_start




def initialize_fields(GR, subgrids, CF):
    if i_load_from_restart:
        CF, RAD, SURF, MIC, TURB = load_restart_fields(GR)
    else:

        ####################################################################
        # SET INITIAL FIELD VALUES
        ####################################################################

        # need to have non-nan-values because values from half-level 0 and GR.nzs
        # are used in calculation.
        CF.POTTVB[:] = 0
        CF.WWIND[:] = 0

        #  LOAD TOPOGRAPHY (HSURF)
        CF.HSURF = load_topo(GR, CF.HSURF) 
        if not i_use_topo:
            CF.HSURF[GR.iijj] = 0.
            CF.HSURF = exchange_BC(GR, CF.HSURF)

        # INITIALIZE PROFILE
        GR, CF.COLP, CF.PSURF, CF.POTT, CF.TAIR \
                = load_profile(GR, subgrids, CF.COLP, CF.HSURF, CF.PSURF, CF.PVTF, \
                                CF.PVTFVB, CF.POTT, CF.TAIR)


        # INITIAL CONDITIONS
        CF.COLP = gaussian2D(GR, CF.COLP, COLP_gaussian_pert, np.pi*3/4, 0, \
                            gaussian_dlon, gaussian_dlat)
        CF.COLP = random2D(GR, CF.COLP, COLP_random_pert)

        for k in range(0,GR.nz):
            CF.UWIND[:,:,k][GR.iisjj] = u0   
            CF.UWIND[:,:,k] = gaussian2D(GR, CF.UWIND[:,:,k], UWIND_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            CF.UWIND[:,:,k] = random2D(GR, CF.UWIND[:,:,k], UWIND_random_pert)
            CF.VWIND[:,:,k][GR.iijjs] = v0
            CF.VWIND[:,:,k] = gaussian2D(GR, CF.VWIND[:,:,k], VWIND_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            CF.VWIND[:,:,k] = random2D(GR, CF.VWIND[:,:,k], VWIND_random_pert)

            CF.POTT[:,:,k] = gaussian2D(GR, CF.POTT[:,:,k], POTT_gaussian_pert, \
                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
            CF.POTT[:,:,k] = random2D(GR, CF.POTT[:,:,k], POTT_random_pert)

        # BOUNDARY EXCHANGE OF INITIAL CONDITIONS
        CF.COLP  = exchange_BC(GR, CF.COLP)
        CF.UWIND  = exchange_BC(GR, CF.UWIND)
        CF.VWIND  = exchange_BC(GR, CF.VWIND)
        CF.POTT  = exchange_BC(GR, CF.POTT)

        # PRIMARY DIAGNOSTIC FIELDS
        diagnose_fields_jacobson(GR, CF, on_host=True)

        # SECONDARY DIAGNOSTIC FIELDS
        CF.PAIR, CF.TAIR, CF.TAIRVB, CF.RHO, CF.WIND = \
                diagnose_secondary_fields(GR, CF.COLP, CF.PAIR, CF.PHI, CF.POTT, CF.POTTVB,
                                        CF.TAIR, CF.TAIRVB, CF.RHO,\
                                        CF.PVTF, CF.PVTFVB, CF.UWIND, CF.VWIND, CF.WIND)

        ####################################################################
        # INITIALIZE NON-ATMOSPHERIC COMPONENTS
        ####################################################################

        # SURF MODEL
        if i_surface:
            SURF = surface(GR, CF)
        else:
            SURF = None

        ####################################################################
        # INITIALIZE PROCESSES
        ####################################################################

        # MOISTURE & MICROPHYSICS
        if i_microphysics:
            MIC = microphysics(GR, CF, i_microphysics, CF.TAIR, CF.PAIR) 
        else:
            MIC = None

        # RADIATION
        if i_radiation:
            if SURF is None:
                raise ValueError('Soil model must be used for i_radiation > 0')
            RAD = radiation(GR, i_radiation)
            rad_njobs_orig = RAD.njobs_rad
            RAD.njobs_rad = 4
            #t_start = time.time()
            RAD.calc_radiation(GR, CF)
            #t_end = time.time()
            #GR.rad_comp_time += t_end - t_start
            RAD.njobs_rad = rad_njobs_orig
        else:
            RAD = None

        # TURBULENCE 
        if i_turbulence:
            raise NotImplementedError('Baustelle')
            TURB = turbulence(GR, i_turbulence) 
        else:
            TURB = None




    return(CF, RAD, SURF, MIC, TURB)



def random2D(GR, FIELD, pert):
    FIELD[:] = FIELD[:] + pert*np.random.rand(FIELD.shape[0], FIELD.shape[1])
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


