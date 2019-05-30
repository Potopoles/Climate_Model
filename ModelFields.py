#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          ModelFields.py  
Author:             Christoph Heim
Date created:       20190525
Last modified:      20190526
License:            MIT

Setup and store model fields. Have each field in memory (for CPU)
and if GPU enabled also on GPU.
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import (i_surface_scheme)
from org_namelist import wp
from initial_conditions import initialize_fields
#from boundaries import exchange_BC
#from IO import load_topo, load_restart_fields, load_profile
#from IO import load_topo, set_up_profile
#from diagnostics import diagnose_secondary_fields, diagnose_POTTVB_jacobson
#from geopotential import diag_geopotential_jacobson, diag_pvt_factor
#from radiation.org_radiation import radiation
#from surface_model import surface
#from grid import tpbh, tpbv, tpbvs
#from org_microphysics import microphysics
#from org_turbulence import turbulence
#from constants import con_g, con_Rd, con_kappa, con_cp
#from surface_model import nz_soil
#from numba import cuda





class ModelFields:


    # names of field groups
    ALL_FIELDS = 'all_fields'

    def old_to_new(self, OF, host=True):
        for field_name in self.field_groups[self.ALL_FIELDS]:
            if host:
                exec('self.host[field_name] = OF.' + field_name)
            else:
                exec('self.device[field_name] = OF.' + field_name)


    def new_to_old(self, OF, host=True):
        for field_name in self.field_groups[self.ALL_FIELDS]:
            if host:
                exec('OF.' + field_name + ' = self.host[field_name]')
            else:
                exec('OF.' + field_name + ' = self.device[field_name]')


    
    def __init__(self, GR, gpu_enable, old_fields):

        self.GR = GR
        self.gpu_enable = gpu_enable

        self.host   = {}
        self.device = {}

        self.allocate_fields(GR)
        self.set_field_groups()


        field_names = ['POTTVB', 'WWIND', 'HSURF',
                        'COLP', 'PSURF', 'PVTF', 'PVTFVB',
                        'POTT', 'TAIR', 'TAIRVB', 'PAIR', 
                        'UWIND', 'VWIND', 'WIND', 'RHO',
                        'PHI', 'PHIVB']
        self.set(initialize_fields(GR, **self.get(field_names)))

        #self.old_to_new(old_fields)

        if self.gpu_enable:
            self.copy_host_to_device(field_group=self.ALL_FIELDS)
        #######################################################################


    def get(self, field_names, target='host'):
        field_dict = {}
        if target == 'host':
            for field_name in field_names:
                field_dict[field_name] = self.host[field_name]
        elif target == 'device':
            for field_name in field_names:
                field_dict[field_name] = self.device[field_name]
        return(field_dict)


    def set(self, field_dict, target='host'):
        if target == 'host':
            for field_name,field in field_dict.items():
                self.host[field_name] = field
        elif target == 'device':
            for field_name,field in field_dict.items():
                self.device[field_name] = field


    def copy_host_to_device(self, field_group):
        self.GR.timer.start('copy')
        for field_name in self.field_groups[field_group]:
            self.device[field_name] = cuda.to_device(self.host[field_name]) 
        self.GR.timer.stop('copy')


    def copy_device_to_host(self, field_group):
        self.GR.timer.start('copy')
        for field_name in self.field_groups[field_group]:
            self.host[field_name] = self.device[field_name].to_host()
        self.GR.timer.stop('copy')

    def set_field_groups(self):

        self.field_groups = {
            self.ALL_FIELDS:        self.host.keys(),
        }






    def allocate_fields(self, GR):
        """
        1 DYNAMICAL CORE FIELDS
            1.1 pressure fields
            1.2 flux fields
            1.3 velocity fields
            1.4 temperature fields
            1.5 additional diagnostic fields
            1.6 constant fields
        2 SURFACE FIELDS
        3 RADIATION FIELDS
        4 MOISTURE FIELDS
        """
        f = {}
        #######################################################################
        #######################################################################
        ### 1 DYNAMICAL CORE FIELDS
        #######################################################################
        #######################################################################

        #######################################################################
        # 1.1 pressure fields 
        #######################################################################

        # column pressure [Pa]
        f['COLP']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, 1      ), 
                                    np.nan, dtype=wp)
        # column pressure [Pa] last time step
        f['COLP_OLD']    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, 1      ),
                                    np.nan, dtype=wp)
        # column pressure [Pa] next time step
        f['COLP_NEW']    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, 1      ),
                                    np.nan, dtype=wp)
        # derivative of column pressure with respect to time [Pa s-1]
        f['dCOLPdt']     = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, 1      ), 
                                    np.nan, dtype=wp)
        # air pressure [Pa]
        f['PAIR']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # surface pressure [Pa]
        f['PSURF']       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, 1      ), 
                                    np.nan, dtype=wp)

        #######################################################################
        # 1.2 flux fields 
        #######################################################################

        # momentum flux in x direction 
        f['UFLX']        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # change of momentum flux in x direction with time
        f['dUFLXdt']     = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # momentum flux in y direction 
        f['VFLX']        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # change of momentum flux in y direction with time
        f['dVFLXdt']     = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # divergence of horizontal momentum fluxes (UFLX and VFLX)
        f['FLXDIV']      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # auxiliary momentum fluxes for UFLX/VFLX tendency computation
        f['BFLX']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['CFLX']        = np.full( ( GR.nxs+2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['DFLX']        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['EFLX']        = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['RFLX']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['QFLX']        = np.full( ( GR.nxs+2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['SFLX']        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['TFLX']        = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)

        #######################################################################
        # 1.3 velocity fields
        #######################################################################

        # wind speed in x direction [m/s]
        f['UWIND']       = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # wind speed in x direction [m/s] last time step
        f['UWIND_OLD']   = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # wind speed in y direction [m/s]
        f['VWIND']       = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # wind speed in y direction [m/s] last time step
        f['VWIND_OLD']   = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # horizontal wind speed [m/s]
        f['WIND']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # vertical wind speed [sigma/s]
        f['WWIND']       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        # product of UWIND and WWIND (auxiliary field)
        f['WWIND_UWIND'] = np.full( ( GR.nxs+2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        # product of VWIND and WWIND (auxiliary field)
        f['WWIND_VWIND'] = np.full( ( GR.nx +2*GR.nb, GR.nys+2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)

        #######################################################################
        # 1.4 temperature fields
        #######################################################################

        # virtual potential temperature [K] 
        f['POTT']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # virtual potential temperature [K] last timestep
        f['POTT_OLD']    = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # change of virtual potential temperature with time [K s-1]
        f['dPOTTdt']     = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # virtual potential temperature at vertical borders
        f['POTTVB']      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        # air temperature [K]
        f['TAIR']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # air temperature [K] at vertical borders
        f['TAIRVB']      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        # virtual potential temperature factor
        f['PVTF']        = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # virtual potential temperature factor at vertical borders
        f['PVTFVB']      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)

        #######################################################################
        # 1.5 additional diagnostic fields
        #######################################################################

        # geopotential
        f['PHI']         = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        # geopotential at vertical borders
        f['PHIVB']       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nzs ), 
                            np.nan, dtype=wp)
        # air density [kg m-2]
        f['RHO']         = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)

        #######################################################################
        # 1.6 constant fields
        #######################################################################

        # surface elevation [m]
        f['HSURF']       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, 1      ), 
                                    np.nan, dtype=wp)



        #######################################################################
        #######################################################################
        ### 2 SURFACE FIELDS 
        #######################################################################
        #######################################################################
        # TODO: Add comments
        if i_surface_scheme: 
            self.OCEANMASK   = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SOILDEPTH   = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SOILCP      = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SOILRHO     = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SOILTEMP    = np.full( ( GR.nx, GR.ny, nz_soil), np.nan, dtype=wp)
            self.SOILMOIST   = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SOILEVAPITY = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SURFALBEDSW = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.SURFALBEDLW = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.RAINRATE    = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)
            self.ACCRAIN     = np.full( ( GR.nx, GR.ny, 1      ), np.nan, dtype=wp)



        #######################################################################
        #######################################################################
        ### 3 RADIATION FIELDS
        #######################################################################
        #######################################################################
        # TODO: Add comments
        self.dPOTTdt_RAD = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                                    np.nan, dtype=wp)
        self.LWFLXNET    = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp)
        self.SWFLXNET    = np.full( ( GR.nx, GR.ny, GR.nzs ), np.nan, dtype=wp)

        #######################################################################
        #######################################################################
        ### 4 MOISTURE FIELDS
        #######################################################################
        #######################################################################
        # TODO: Add comments
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

        self.host = f





    def copy_stepDiag_fields_to_host(self, GR):
        GR.timer.start('copy')
        #self.COLP              .to_host(GR.stream)
        #self.WIND              .to_host(GR.stream) 
        #self.POTT              .to_host(GR.stream) 
        #GR.stream.synchronize()

        self.CF.COLP          =    self.COLP.copy_to_host()
        self.CF.WIND          =    self.WIND.copy_to_host() 
        self.CF.POTT          =    self.POTT.copy_to_host() 

        GR.timer.stop('copy')


    def copy_radiation_fields_to_device(self, GR, CF):
        GR.timer.start('copy')

        self.dPOTTdt_RAD      = cuda.to_device(CF.dPOTTdt_RAD,  GR.stream) 
        self.SWFLXNET         = cuda.to_device(CF.SWFLXNET,     GR.stream) 
        self.LWFLXNET         = cuda.to_device(CF.LWFLXNET,     GR.stream) 

        GR.stream.synchronize()

        GR.timer.stop('copy')

    def copy_radiation_fields_to_host(self, GR):
        GR.timer.start('copy')
        self.RHO               .to_host(GR.stream)
        self.TAIR              .to_host(GR.stream)
        self.PHIVB             .to_host(GR.stream) 
        self.SOILTEMP          .to_host(GR.stream) 
        self.SURFALBEDLW       .to_host(GR.stream) 
        self.SURFALBEDSW       .to_host(GR.stream) 
        self.QC                .to_host(GR.stream) 

        GR.stream.synchronize()

        GR.timer.stop('copy')


    def copy_all_fields_to_host(self, GR):
        GR.timer.start('copy')

        # TODO: NEW STYLE (???) MAYBE USE GPU CLASS INSTEAD
        self.CF.COLP          =    self.COLP.copy_to_host()
        self.CF.WIND          =    self.WIND.copy_to_host() 
        self.CF.POTT          =    self.POTT.copy_to_host() 
        self.CF.COLP_NEW      =    self.COLP_NEW.copy_to_host() 
        self.CF.dCOLPdt      =    self.dCOLPdt.copy_to_host() 
        self.CF.COLP_OLD      =    self.COLP_OLD.copy_to_host() 
        self.CF.HSURF       =    self.HSURF.copy_to_host() 

        #self.COLP_OLD          .to_host(GR.stream)
        #self.COLP              .to_host(GR.stream)
        #self.COLP_NEW          .to_host(GR.stream)
        #self.dCOLPdt           .to_host(GR.stream)
        self.PSURF             .to_host(GR.stream)
        #self.HSURF             .to_host(GR.stream) 
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

        GR.timer.stop('copy')




##def initialize_fields(GR, subgrids, CF):
##    if i_load_from_restart:
##        CF, RAD, SURF, MIC, TURB = load_restart_fields(GR)
##    else:
##
##        ####################################################################
##        # SET INITIAL FIELD VALUES
##        ####################################################################
##
##        # need to have non-nan-values because values from half-level 0 and GR.nzs
##        # are used in calculation.
##        CF.POTTVB[:] = 0
##        CF.WWIND[:] = 0
##
##        #  LOAD TOPOGRAPHY (HSURF)
##        CF.HSURF = load_topo(GR, CF.HSURF) 
##        if not i_use_topo:
##            CF.HSURF[GR.iijj] = 0.
##            CF.HSURF = exchange_BC(GR, CF.HSURF)
##
##        # INITIALIZE PROFILE
##        GR, CF.COLP, CF.PSURF, CF.POTT, CF.TAIR \
##                = load_profile(GR, subgrids, CF.COLP, CF.HSURF, CF.PSURF, CF.PVTF, \
##                                CF.PVTFVB, CF.POTT, CF.TAIR)
##
##
##        # INITIAL CONDITIONS
##        CF.COLP = gaussian2D(GR, CF.COLP, COLP_gaussian_pert, np.pi*3/4, 0, \
##                            gaussian_dlon, gaussian_dlat)
##        CF.COLP = random2D(GR, CF.COLP, COLP_random_pert)
##
##        for k in range(0,GR.nz):
##            CF.UWIND[:,:,k][GR.iisjj] = u0   
##            CF.UWIND[:,:,k] = gaussian2D(GR, CF.UWIND[:,:,k], UWIND_gaussian_pert, \
##                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
##            CF.UWIND[:,:,k] = random2D(GR, CF.UWIND[:,:,k], UWIND_random_pert)
##            CF.VWIND[:,:,k][GR.iijjs] = v0
##            CF.VWIND[:,:,k] = gaussian2D(GR, CF.VWIND[:,:,k], VWIND_gaussian_pert, \
##                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
##            CF.VWIND[:,:,k] = random2D(GR, CF.VWIND[:,:,k], VWIND_random_pert)
##
##            CF.POTT[:,:,k] = gaussian2D(GR, CF.POTT[:,:,k], POTT_gaussian_pert, \
##                            np.pi*3/4, 0, gaussian_dlon, gaussian_dlat)
##            CF.POTT[:,:,k] = random2D(GR, CF.POTT[:,:,k], POTT_random_pert)
##
##        # BOUNDARY EXCHANGE OF INITIAL CONDITIONS
##        CF.COLP  = exchange_BC(GR, CF.COLP)
##        CF.UWIND  = exchange_BC(GR, CF.UWIND)
##        CF.VWIND  = exchange_BC(GR, CF.VWIND)
##        CF.POTT  = exchange_BC(GR, CF.POTT)
##
##        # PRIMARY DIAGNOSTIC FIELDS
##        #diagnose_fields_jacobson(GR, CF, on_host=True)
##        diagnose_fields_initializaiton(GR, CF)
##
##        # SECONDARY DIAGNOSTIC FIELDS
##        CF.PAIR, CF.TAIR, CF.TAIRVB, CF.RHO, CF.WIND = \
##                diagnose_secondary_fields(GR, CF.COLP, CF.PAIR, CF.PHI, CF.POTT, CF.POTTVB,
##                                        CF.TAIR, CF.TAIRVB, CF.RHO,\
##                                        CF.PVTF, CF.PVTFVB, CF.UWIND, CF.VWIND, CF.WIND)
##
##        ####################################################################
##        # INITIALIZE NON-ATMOSPHERIC COMPONENTS
##        ####################################################################
##
##        # SURF MODEL
##        if i_surface:
##            SURF = surface(GR, CF)
##        else:
##            SURF = None
##
##        ####################################################################
##        # INITIALIZE PROCESSES
##        ####################################################################
##
##        # MOISTURE & MICROPHYSICS
##        if i_microphysics:
##            MIC = microphysics(GR, CF, i_microphysics, CF.TAIR, CF.PAIR) 
##        else:
##            MIC = None
##
##        # RADIATION
##        if i_radiation:
##            if SURF is None:
##                raise ValueError('Soil model must be used for i_radiation > 0')
##            RAD = radiation(GR, i_radiation)
##            rad_njobs_orig = RAD.njobs_rad
##            RAD.njobs_rad = 4
##            #t_start = time.time()
##            RAD.calc_radiation(GR, CF)
##            #t_end = time.time()
##            #GR.rad_comp_time += t_end - t_start
##            RAD.njobs_rad = rad_njobs_orig
##        else:
##            RAD = None
##
##        # TURBULENCE 
##        if i_turbulence:
##            raise NotImplementedError('Baustelle')
##            TURB = turbulence(GR, i_turbulence) 
##        else:
##            TURB = None
##
##
##
##
##    return(CF, RAD, SURF, MIC, TURB)








