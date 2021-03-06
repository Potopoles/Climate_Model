#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190525
Last modified:      20190701
License:            MIT

Setup and store model fields. Have each field in memory (for CPU)
and if GPU enabled also on GPU.
###############################################################################
"""
import numpy as np
from numba import cuda
from scipy.interpolate import interp2d

from namelist import (i_surface_scheme, nzsoil, i_turbulence,
                      i_radiation, i_load_from_restart, i_load_from_IC,
                      i_moist_main_switch, i_microphysics,)
from io_read_namelist import wp, wp_int, CPU, GPU
from io_initial_conditions import initialize_fields
from io_restart import load_existing_fields 
from srfc_main import Surface
from turb_main import Turbulence
from mic_main import Microphysics
from rad_main import Radiation
###############################################################################

class ModelFields:

    # names of field groups
    ALL_FIELDS          = 'all_fields'
    PRINT_DIAG_FIELDS   = 'print_diag_fields'
    NC_OUT_DIAG_FIELDS  = 'nc_out_diag_fields'
    SRFC_FIELDS         = 'srfc_fields'
    RAD_TO_DEVICE       = 'rad_to_device_fields'
    RAD_TO_HOST         = 'rad_to_host_fields'

    
    def __init__(self, GR, gpu_enable):

        if i_load_from_restart:
            loaded_F = load_existing_fields(GR, directory='restart')
            self.__dict__ = loaded_F.__dict__
            self.device = {}
            if self.gpu_enable:
                self.copy_host_to_device(GR, field_group=self.ALL_FIELDS)
        else:

            self.gpu_enable = gpu_enable

            self.host   = {}
            self.device = {}

            self.host, self.fdict = allocate_fields(GR)
            self.set_field_groups()


            ###################################################################
            ## DEFAULT INITIAL CONDITIONS
            ###################################################################
            ## DYNAMICS
            fields_to_init = ['POTTVB', 'WWIND', 'HSURF',
                            'COLP', 'PSURF', 'PVTF', 'PVTFVB',
                            'POTT', 'TAIR', 'TAIRVB', 'PAIR', 
                            'UWIND', 'VWIND', 'WIND', 'RHO',
                            'PHI', 'PHIVB', 'QV', 'QC']
            self.set(initialize_fields(GR, **self.get(fields_to_init)))

            if gpu_enable:
                target=GPU
            else:
                target=CPU

            ## SURFACE
            if i_surface_scheme:
                self.SURF = Surface(GR, self, target=target)
            else:
                self.SURF = None

            ## TURBULENCE 
            if i_turbulence:
                self.TURB = Turbulence(GR, target=target) 
            else:
                self.TURB = None

            ## MOISTURE & MICROPHYSICS
            if i_moist_main_switch or i_microphysics:
                self.MIC = Microphysics(GR, self, target=target) 
            else:
                self.MIC = None

            ## RADIATION
            if i_radiation:
                RAD = Radiation(GR)
                rad_njobs_orig = RAD.njobs_rad
                RAD.njobs_rad = 4
                RAD.njobs_rad = rad_njobs_orig
                self.RAD = RAD
                self.RAD.calc_radiation(GR, self)


            ###################################################################
            ## LOAD INITIAL CONDITIONS
            ###################################################################
            non_IC_fields = ['HSURF', 'OCEANMASK', 'ACCRAIN']
            
            if i_load_from_IC:
                IC_F = load_existing_fields(GR.IC, directory='IC')
                #### exceptions
                IC_F.host['SOILMOIST'][
                    np.isnan(IC_F.host['SOILMOIST'])] = wp(0.)
                #### exceptions
                for field_name, fd in self.fdict.items():
                    if field_name in non_IC_fields:
                        continue
                    ii = GR.ii
                    jj = GR.jj
                    ii_IC = GR.IC.ii
                    jj_IC = GR.IC.jj
                    if fd['stgx']:
                        ii = GR.iis
                        ii_IC = GR.IC.iis
                        x = GR.lon_is_rad[ii,GR.nb,0]
                        y = GR.lat_is_rad[GR.nb,jj,0]
                        x_IC = GR.IC.lon_is_rad[ii_IC,GR.nb,0]
                        y_IC = GR.IC.lat_is_rad[GR.nb,jj_IC,0]
                    elif fd['stgy']:
                        jj = GR.jjs
                        jj_IC = GR.IC.jjs
                        x = GR.lon_js_rad[ii,GR.nb,0]
                        y = GR.lat_js_rad[GR.nb,jj,0]
                        x_IC = GR.IC.lon_js_rad[ii_IC,GR.nb,0]
                        y_IC = GR.IC.lat_js_rad[GR.nb,jj_IC,0]
                    else:
                        x = GR.lon_rad[ii,GR.nb,0]
                        y = GR.lat_rad[GR.nb,jj,0]
                        x_IC = GR.IC.lon_rad[ii_IC,GR.nb,0]
                        y_IC = GR.IC.lat_rad[GR.nb,jj_IC,0]

                    for k in range(fd['dimz']):
                        layer_IC = IC_F.host[field_name][ii_IC,jj_IC,k].squeeze()
                        intrp = interp2d(y_IC.squeeze(),
                                        x_IC.squeeze(),layer_IC)
                        layer = intrp(y.squeeze(), x.squeeze()) 
                        self.host[field_name][ii,jj,k] = layer

                    GR.exchange_BC(self.host[field_name])

                #### exceptions
                self.host['SOILMOIST'][self.host['OCEANMASK']==1] = np.nan
                #### exceptions


            # if necessary, copy fields to gpu
            if self.gpu_enable:
                self.copy_host_to_device(GR, field_group=self.ALL_FIELDS)
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
                                         'SOILTEMP', 'SOILMOIST',
                                         'SURFALBEDSW', 'SURFALBEDLW'],
            self.RAD_TO_DEVICE:         ['SOLZEN', 'MYSUN', 'SWINTOA',
                                        'LWFLXDO', 'LWFLXUP', 'SWDIFFLXDO',
                                        'SWDIRFLXDO', 'SWFLXUP', 'SWFLXDO',
                                        'LWFLXNET', 'SWFLXNET', 'SWFLXDIV',
                                        'LWFLXDIV', 'TOTFLXDIV', 'dPOTTdt_RAD'],
            self.RAD_TO_HOST:           ['RHO', 'TAIR', 'PHIVB', 'SOILTEMP',
                                         'SURFALBEDLW', 'SURFALBEDSW', 'QC'],

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


    def copy_host_to_device(self, GR, field_group):
        GR.timer.start('copy')
        for field_name in self.field_groups[field_group]:
            self.device[field_name] = cuda.to_device(self.host[field_name]) 
        GR.timer.stop('copy')


    def copy_device_to_host(self, GR, field_group):
        GR.timer.start('copy')
        for field_name in self.field_groups[field_group]:
            self.device[field_name].copy_to_host(self.host[field_name])
        GR.timer.stop('copy')


def allocate_fields(GR):
    """
    1 DYNAMICAL CORE FIELDS
        1.1 pressure fields
        1.2 flux fields
        1.3 velocity fields
        1.4 temperature fields
        1.5 additional diagnostic fields
        1.6 constant fields
    2 SURFACE FIELDS
    3 TURBULENCE FIELDS
    4 RADIATION FIELDS
    5 MOISTURE FIELDS
    """
    f = {}

    fdict = {
    #######################################################################
    #######################################################################
    ### 1 DYNAMICAL CORE FIELDS
    #######################################################################
    #######################################################################

    #######################################################################
    # 1.1 pressure fields 
    #######################################################################
    # column pressure [Pa]
    'COLP':         {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # column pressure [Pa] last time step
    'COLP_OLD':     {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # column pressure [Pa] next time step
    'COLP_NEW':     {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # derivative of column pressure with respect to time [Pa s-1]
    'dCOLPdt':      {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface pressure [Pa]
    'PSURF':        {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # air pressure [Pa]
    'PAIR':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # air pressure [Pa] at vertical borders
    'PAIRVB':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},

    #######################################################################
    # 1.2 flux fields 
    #######################################################################
    # momentum flux in x direction 
    'UFLX':         {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # change of momentum flux in x direction with time
    'dUFLXdt':      {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # momentum flux in y direction 
    'VFLX':         {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    # change of momentum flux in y direction with time
    'dVFLXdt':      {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    # divergence of horizontal momentum fluxes (UFLX and VFLX)
    'FLXDIV':       {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # auxiliary momentum fluxes for UFLX/VFLX tendency computation
    'BFLX':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    'CFLX':         {'stgx':1,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    'DFLX':         {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    'EFLX':         {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    'RFLX':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    'QFLX':         {'stgx':1,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    'SFLX':         {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    'TFLX':         {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},

    #######################################################################
    # 1.3 velocity fields
    #######################################################################
    # wind speed in x direction [m/s]
    'UWIND':        {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # wind speed in x direction [m/s] last time step
    'UWIND_OLD':    {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # wind speed in y direction [m/s]
    'VWIND':        {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    # wind speed in y direction [m/s] last time step
    'VWIND_OLD':    {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    # horizontal wind speed [m/s]
    'WIND':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # horizontal wind speed x component (but unstaggered) [m/s]
    'WINDX':        {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # horizontal wind speed y component (but unstaggered) [m/s]
    'WINDY':        {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # vertical wind speed [sigma/s]
    'WWIND':        {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # product of UWIND and WWIND (auxiliary field)
    'WWIND_UWIND':  {'stgx':1,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # product of VWIND and WWIND (auxiliary field)
    'WWIND_VWIND':  {'stgx':0,'stgy':1,'dimz':GR.nzs,'dtype':wp},
    # product of dUWINDdz and KMOM (auxiliary field)
    'KMOM_dUWINDdz':{'stgx':1,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # product of dVWINDdz and KMOM (auxiliary field)
    'KMOM_dVWINDdz':{'stgx':0,'stgy':1,'dimz':GR.nzs,'dtype':wp},

    #######################################################################
    # 1.4 temperature fields
    #######################################################################
    # virtual potential temperature [K] 
    'POTT':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # virtual potential temperature [K] last timestep
    'POTT_OLD':     {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # change of virtual potential temperature with time [K s-1]
    'dPOTTdt':      {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # virtual potential temperature at vertical borders
    'POTTVB':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # air temperature [K]
    'TAIR':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # air temperature [K] at vertical borders
    'TAIRVB':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # virtual potential temperature factor
    'PVTF':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # virtual potential temperature factor at vertical borders
    'PVTFVB':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},

    #######################################################################
    # 1.5 additional diagnostic fields
    #######################################################################
    # geopotential
    'PHI':          {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # geopotential at vertical borders
    'PHIVB':        {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # air density [kg m-2]
    'RHO':          {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # air density at vertical borders [kg m-2]
    'RHOVB':        {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},

    #######################################################################
    # 1.6 constant fields
    #######################################################################
    # surface elevation [m]
    'HSURF':         {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},


    #######################################################################
    #######################################################################
    ### 2 SURFACE FIELDS 
    #######################################################################
    #######################################################################
    # 1 if grid point is ocean [-]
    'OCEANMASK':     {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp_int},
    # depth of soil [m]
    'SOILDEPTH':     {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # heat capcity of soil [J kg-1 K-1]
    'SOILCP':        {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # density of soil [kg m-3]
    'SOILRHO':       {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # soil temperature [K]
    'SOILTEMP':      {'stgx':0,'stgy':0,'dimz':nzsoil,'dtype':wp},
    # moisture content of soil [kg m-2]
    'SOILMOIST':     {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    ## evaporation likliness of soil [-]
    #'SOILEVAPITY':   {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface roughness length [m]
    'SURFZ0':        {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface albedo for shortwave radiation [-]
    'SURFALBEDSW':   {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface albedo for longwave radiation [-]
    'SURFALBEDLW':   {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # rain at surface during time step [kg m-2]
    'RAIN':          {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # rain rate at surface [kg m-2 s-1]
    'RAINRATE':      {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # accumulated rain during simulation at surface [kg m-2]
    'ACCRAIN':       {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface momentum flux in x direction
    # (pointing towards atmosphere) [???]
    'SMOMXFLX':      {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface momentum flux in y direction
    # (pointing towards atmosphere) [???]
    'SMOMYFLX':      {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface sensible heat flux (pointing towards atmosphere) [W m-2]
    'SSHFLX':        {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # surface latent heat flux (pointing towards atmosphere) [W m-2]
    'SLHFLX':        {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},


    #######################################################################
    #######################################################################
    ### 3 TURBULENCE FIELDS
    #######################################################################
    #######################################################################
    # UFLX change due to turbulent fluxes [??]
    'dUFLXdt_TURB':  {'stgx':1,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # VFLX change due to turbulent fluxes [??]
    'dVFLXdt_TURB':  {'stgx':0,'stgy':1,'dimz':GR.nz ,'dtype':wp},
    # potential temperature change due to turbulent fluxes [K s-1]
    'dPOTTdt_TURB':  {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # QV change due to turbulence [kg kg-1 s-1]
    'dQVdt_TURB':    {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # QC change due to turbulence [kg kg-1 s-1]
    'dQCdt_TURB':    {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # turbulenc exchange coefficient for heat and moisture 
    'KHEAT':         {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # turbulent exchange coefficient for momentum
    'KMOM':          {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},


    #######################################################################
    #######################################################################
    ### 4 RADIATION FIELDS
    #######################################################################
    #######################################################################
    # net shortwave flux pointing into surface [W m-2]
    'SWFLXNET':      {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # net longwave flux pointing into surface [W m-2]
    'LWFLXNET':      {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # potential temperature change due to radiation [K s-1]
    'dPOTTdt_RAD':   {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # solar zenith angle
    'SOLZEN':        {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # cos solar zenith angle
    'MYSUN':         {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # incoming shortwave at TOA
    'SWINTOA':       {'stgx':0,'stgy':0,'dimz':1     ,'dtype':wp},
    # upward component of longwave flux [W m-2]
    'LWFLXUP':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # downward component of longwave flux [W m-2]
    'LWFLXDO':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # diffuse downward component of shortwave flux [W m-2]
    'SWDIFFLXDO':    {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # direct downward component of shortwave flux [W m-2]
    'SWDIRFLXDO':    {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # upward component of shortwave flux [W m-2]
    'SWFLXUP':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # downward component of shortwave flux [W m-2]
    'SWFLXDO':       {'stgx':0,'stgy':0,'dimz':GR.nzs,'dtype':wp},
    # longwave flux divergence [W m-2]
    'LWFLXDIV':      {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # shortwave flux divergence [W m-2]
    'SWFLXDIV':      {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # total flux divergence (longwave + shortwave) [W m-2]
    'TOTFLXDIV':     {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},


    #######################################################################
    #######################################################################
    ### 5 MOISTURE FIELDS
    #######################################################################
    #######################################################################
    # specific water vapor content [kg kg-1]
    'QV':            {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # specific water vapor content [kg kg-1] last time level
    'QV_OLD':        {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # change of specific water vapor content with time [kg kg-1 s-1]
    'dQVdt':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # specific cloud water content [kg kg-1]
    'QC':            {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # specific cloud water content [kg kg-1] last time level
    'QC_OLD':        {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # specific rain water content [kg kg-1]
    'QR':            {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # change of specific cloud water content with time [kg kg-1 s-1]
    'dQCdt':         {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # change of QV due to microphysics [kg kg-1 s-1]
    'dQVdt_MIC':     {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # change of QC due to microphysics [kg kg-1 s-1]
    'dQCdt_MIC':     {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    # potential temperature change due to microphysics [K s-1]
    'dPOTTdt_MIC':   {'stgx':0,'stgy':0,'dimz':GR.nz ,'dtype':wp},
    }

    for key,set in fdict.items():
        dimx = GR.nx + 2*GR.nb
        if set['stgx']:
            dimx += 1
        dimy = GR.ny + 2*GR.nb
        if set['stgy']:
            dimy += 1
        dimz = set['dimz']
        f[key] = np.full( ( dimx, dimy, dimz ), np.nan, dtype=set['dtype'] )

    return(f, fdict)
