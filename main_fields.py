#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190525
Last modified:      20190601
License:            MIT

Setup and store model fields. Have each field in memory (for CPU)
and if GPU enabled also on GPU.
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import (i_surface_scheme, nz_soil,
                      i_radiation, i_load_from_restart)
from io_read_namelist import wp, wp_int, CPU, GPU
from io_initial_conditions import initialize_fields
from io_restart import load_restart_fields
from rad_main import Radiation
from srfc_main import Surface
###############################################################################

class ModelFields:

    # names of field groups
    ALL_FIELDS          = 'all_fields'
    PRINT_DIAG_FIELDS   = 'print_diag_fields'
    NC_OUT_DIAG_FIELDS  = 'nc_out_diag_fields'
    SRFC_FIELDS         = 'srfc_fields'
    RAD_TO_DEVICE_FIELDS= 'rad_to_device_fields'
    RAD_TO_HOST_FIELDS  = 'rad_to_device_fields'

    
    def __init__(self, GR, gpu_enable):

        if i_load_from_restart:
            loaded_F = load_restart_fields(GR)
            self.__dict__ = loaded_F.__dict__
            self.device = {}
            if self.gpu_enable:
                self.copy_host_to_device(field_group=self.ALL_FIELDS)
        else:

            # TODO remove GR from F.
            self.GR = GR
            self.gpu_enable = gpu_enable

            self.host   = {}
            self.device = {}

            self.allocate_fields(GR)
            self.set_field_groups()


            ###################################################################
            ## INITIALIZE FIELDS
            ###################################################################
            ## DYNAMICS
            fields_to_init = ['POTTVB', 'WWIND', 'HSURF',
                            'COLP', 'PSURF', 'PVTF', 'PVTFVB',
                            'POTT', 'TAIR', 'TAIRVB', 'PAIR', 
                            'UWIND', 'VWIND', 'WIND', 'RHO',
                            'PHI', 'PHIVB']
            self.set(initialize_fields(GR, **self.get(fields_to_init)))

            ###################################################################
            ## INITIALIZE PROCESSES
            ###################################################################

            ## SURFACE
            if i_surface_scheme:
                if gpu_enable:
                    self.SURF = Surface(GR, self, target=GPU)
                else:
                    self.SURF = Surface(GR, self, target=CPU)
            else:
                self.SURF = None

            ## RADIATION
            if i_radiation:
                RAD = Radiation(GR)
                rad_njobs_orig = RAD.njobs_rad
                RAD.njobs_rad = 4
                RAD.calc_radiation(GR, self)
                RAD.njobs_rad = rad_njobs_orig
                self.RAD = RAD

            ## MOISTURE & MICROPHYSICS
            #if i_microphysics:
            #    MIC = microphysics(GR, CF, i_microphysics, CF.TAIR, CF.PAIR) 
            #else:
            #    MIC = None

            ## TURBULENCE 
            #if i_turbulence:
            #    raise NotImplementedError('Baustelle')
            #    TURB = turbulence(GR, i_turbulence) 
            #else:
            #    TURB = None

            

            if self.gpu_enable:
                self.copy_host_to_device(field_group=self.ALL_FIELDS)
            ###################################################################

    def set_field_groups(self):

        self.field_groups = {
            self.ALL_FIELDS:            list(self.host.keys()),
            self.PRINT_DIAG_FIELDS:     ['COLP', 'WIND', 'POTT'],
            self.NC_OUT_DIAG_FIELDS:    ['UWIND', 'VWIND' ,'WWIND' ,'POTT',
                                         'COLP', 'PVTF' ,'PVTFVB', 'PHI',
                                         'PHIVB', 'RHO', 'QV', 'QC'],
            self.SRFC_FIELDS:           ['HSURF', 'OCEANMASK', 'SOILDEPTH',
                                         'SOILCP', 'SOILRHO',
                                         'SOILTEMP', 'dSOILTEMPdt', 'SOILMOIST',
                                         'SOILEVAPITY',
                                         'SURFALBEDSW', 'SURFALBEDLW',
                                         'RAINRATE', 'ACCRAIN'],
            #self.RAD_TO_DEVICE_FIELDS:  ['dPOTTdt_RAD', 'SWFLXNET', 'LWFLXNET'],
            #self.RAD_TO_HOST_FIELDS:    ['RHO', 'TAIR', 'PHIVB', 'SOILTEMP',
            #                             'SURFALBEDLW', 'SURFALBEDSW', 'QC'],


        }


    def get(self, field_names, target=CPU):
        field_dict = {}
        if target == CPU:
            for field_name in field_names:
                field_dict[field_name] = self.host[field_name]
        elif target == GPU:
            for field_name in field_names:
                field_dict[field_name] = self.device[field_name]
        return(field_dict)


    def set(self, field_dict, target=CPU):
        if target == CPU:
            for field_name,field in field_dict.items():
                self.host[field_name] = field
        elif target == GPU:
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
            self.device[field_name].copy_to_host(self.host[field_name])
        self.GR.timer.stop('copy')


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
        if i_surface_scheme: 
            # 1 if grid point is ocean [-]
            f['OCEANMASK']   = np.full( ( GR.nx, GR.ny, 1      ),
                                        0, dtype=wp_int)
            # depth of soil [m]
            f['SOILDEPTH']   = np.full( ( GR.nx, GR.ny, 1      ),
                                        np.nan, dtype=wp)
            # heat capcity of soil [J kg-1 K-1]
            f['SOILCP']      = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # density of soil [kg m-3]
            f['SOILRHO']     = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # soil temperature [K]
            f['SOILTEMP']    = np.full( ( GR.nx, GR.ny, nz_soil), 
                                        np.nan, dtype=wp)
            # change of soil temperature with time [K s-1]
            f['dSOILTEMPdt'] = np.full( ( GR.nx, GR.ny, nz_soil), 
                                        np.nan, dtype=wp)
            # moisture content of soil [kg m-2]
            f['SOILMOIST']   = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # evaporation likliness of soil [-]
            f['SOILEVAPITY'] = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # surface albedo for shortwave radiation [-]
            f['SURFALBEDSW'] = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # surface albedo for longwave radiation [-]
            f['SURFALBEDLW'] = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # rain rate at surface [kg m-2 s-1]
            f['RAINRATE']    = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            # accumulated rain during simulation at surface [kg m-2]
            f['ACCRAIN']     = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)



        #######################################################################
        #######################################################################
        ### 3 RADIATION FIELDS
        #######################################################################
        #######################################################################
        # potential temperature change due to radiation [K s-1]
        f['dPOTTdt_RAD']     = np.full(
                                    ( GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nz  ), 
                                        np.nan, dtype=wp)
        # net longwave flux (direction?) [W m-2]
        f['LWFLXNET']        = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        # net shortwave flux (direction?) [W m-2]
        f['SWFLXNET']        = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)

        # solar zenith angle
        f['SOLZEN']          = np.full( ( GR.nx, GR.ny, 1      ),
                                        np.nan, dtype=wp)
        
        # cos solar zenith angle
        f['MYSUN']           = np.full( ( GR.nx, GR.ny, 1      ),
                                        np.nan, dtype=wp)

        # incoming shortwave at TOA
        f['SWINTOA']         = np.full( ( GR.nx, GR.ny, 1      ),
                                        np.nan, dtype=wp)

        f['LWFLXUP']         = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        f['LWFLXDO']         = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        f['SWDIFFLXDO']      = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        f['SWDIRFLXDO']      = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        f['SWFLXUP']         = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        f['SWFLXDO']         = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        f['LWFLXDIV']        = np.full( ( GR.nx, GR.ny, GR.nz  ),
                                        np.nan, dtype=wp)
        f['SWFLXDIV']        = np.full( ( GR.nx, GR.ny, GR.nz  ),
                                        np.nan, dtype=wp)
        f['TOTFLXDIV']       = np.full( ( GR.nx, GR.ny, GR.nz  ), np.nan, dtype=wp)

        #######################################################################
        #######################################################################
        ### 4 MOISTURE FIELDS
        #######################################################################
        #######################################################################
        # TODO: Add comments
        f['QV_OLD']      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['QV']          = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['dQVdt']       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['QC_OLD']      = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['QC']          = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['dQCdt']       = np.full( ( GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz  ), 
                            np.nan, dtype=wp)
        f['dQVdt_MIC']   = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)
        f['dQCdt_MIC']   = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)
        f['dPOTTdt_MIC'] = np.full( ( GR.nx         , GR.ny         , GR.nz  ), 
                            np.nan, dtype=wp)

        self.host = f

