#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190525
Last modified:      20190531
License:            MIT

Setup and store model fields. Have each field in memory (for CPU)
and if GPU enabled also on GPU.
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import i_surface_scheme
from io_read_namelist import wp, CPU, GPU
from io_initial_conditions import initialize_fields
from radiation.org_radiation import Radiation
###############################################################################

class ModelFields:

    # names of field groups
    ALL_FIELDS          = 'all_fields'
    PRINT_DIAG_FIELDS   = 'print_diag_fields'
    NC_OUT_DIAG_FIELDS  = 'nc_out_diag_fields'

    
    def __init__(self, GR, gpu_enable):

        self.GR = GR
        self.gpu_enable = gpu_enable

        self.host   = {}
        self.device = {}

        self.allocate_fields(GR)
        self.set_field_groups()


        fields_to_init = ['POTTVB', 'WWIND', 'HSURF',
                        'COLP', 'PSURF', 'PVTF', 'PVTFVB',
                        'POTT', 'TAIR', 'TAIRVB', 'PAIR', 
                        'UWIND', 'VWIND', 'WIND', 'RHO',
                        'PHI', 'PHIVB']
        self.set(initialize_fields(GR, **self.get(fields_to_init)))


        ## RADIATION
        #if i_radiation:
        #    if SURF is None:
        #        raise ValueError('Soil model must be used for i_radiation > 0')
        #    RAD = radiation(GR, i_radiation)
        #    rad_njobs_orig = RAD.njobs_rad
        #    RAD.njobs_rad = 4
        #    RAD.calc_radiation(GR, CF)
        #    RAD.njobs_rad = rad_njobs_orig
        

        if self.gpu_enable:
            self.copy_host_to_device(field_group=self.ALL_FIELDS)
        #######################################################################

    def set_field_groups(self):

        self.field_groups = {
            self.ALL_FIELDS:            self.host.keys(),
            self.PRINT_DIAG_FIELDS:     ['COLP', 'WIND', 'POTT'],
            self.NC_OUT_DIAG_FIELDS:    ['UWIND', 'VWIND' ,'WWIND' ,'POTT',
                                         'COLP', 'PVTF' ,'PVTFVB', 'PHI',
                                         'PHIVB', 'RHO', 'QV', 'QC']
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
        # TODO: Add comments
        if i_surface_scheme: 
            f['OCEANMASK']   = np.full( ( GR.nx, GR.ny, 1      ),
                                        np.nan, dtype=wp)
            f['SOILDEPTH']   = np.full( ( GR.nx, GR.ny, 1      ),
                                        np.nan, dtype=wp)
            f['SOILCP']      = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['SOILRHO']     = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['SOILTEMP']    = np.full( ( GR.nx, GR.ny, nz_soil), 
                                        np.nan, dtype=wp)
            f['SOILMOIST']   = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['SOILEVAPITY'] = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['SURFALBEDSW'] = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['SURFALBEDLW'] = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['RAINRATE']    = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)
            f['ACCRAIN']     = np.full( ( GR.nx, GR.ny, 1      ), 
                                        np.nan, dtype=wp)



        #######################################################################
        #######################################################################
        ### 3 RADIATION FIELDS
        #######################################################################
        #######################################################################
        # potential temperature change due to radiation [K s-1]
        f['dPOTTdt_RAD']     = np.full( ( GR.nx, GR.ny, GR.nz  ), 
                                        np.nan, dtype=wp)
        # net longwave flux (direction?) [W m-2]
        f['LWFLXNET']        = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)
        # net shortwave flux (direction?) [W m-2]
        f['SWFLXNET']        = np.full( ( GR.nx, GR.ny, GR.nzs ),
                                        np.nan, dtype=wp)

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




#def copy_radiation_fields_to_device(self, GR, CF):
#    GR.timer.start('copy')

#    self.dPOTTdt_RAD      = cuda.to_device(CF.dPOTTdt_RAD,  GR.stream) 
#    self.SWFLXNET         = cuda.to_device(CF.SWFLXNET,     GR.stream) 
#    self.LWFLXNET         = cuda.to_device(CF.LWFLXNET,     GR.stream) 

#    GR.stream.synchronize()

#    GR.timer.stop('copy')

#def copy_radiation_fields_to_host(self, GR):
#    GR.timer.start('copy')
#    self.RHO               .to_host(GR.stream)
#    self.TAIR              .to_host(GR.stream)
#    self.PHIVB             .to_host(GR.stream) 
#    self.SOILTEMP          .to_host(GR.stream) 
#    self.SURFALBEDLW       .to_host(GR.stream) 
#    self.SURFALBEDSW       .to_host(GR.stream) 
#    self.QC                .to_host(GR.stream) 

#    GR.stream.synchronize()

#    GR.timer.stop('copy')

