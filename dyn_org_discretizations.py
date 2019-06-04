#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190604
License:            MIT

SPATIAL DISCRETIZATION
----------------------
Organise the computation of all RHS of equations to compute 
time tendencies for prognostica variables of dynamical core:
- virtual potential temperature POTT
- horizontal momentum fluxes UFLX and VFLX
- continuity equation resulting in column pressure COLP tendency and
  vertical wind speed WWIND

Organise the computation of all diagnostic functions:
- primary diagnostics containing geopotential PHI, virtual potential
  temperature factor at mass point PVTF and at vertical borders (PVTFVB)
    
   TIME DISCRETIZATION
----------------------
Organise the computation of one Euler forward time step.

           DIAGNOSTICS
----------------------
Organise the computation of diagnostics.

Differentiate between computation targets GPU and CPU.
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import (i_UVFLX_hor_adv, i_UVFLX_vert_adv)
from io_read_namelist import CPU, GPU, gpu_enable
from main_grid import (nx,nxs,ny,nys,nz,nzs,nb,
                 tpb, tpb_ks, bpg, tpb_sc, bpg_sc)

from misc_boundaries import exchange_BC_cpu
from dyn_continuity import continuity_cpu
from dyn_POTT import POTT_tendency_cpu
from dyn_UVFLX_prepare import UVFLX_prep_adv_cpu
from dyn_UFLX import UFLX_tendency_cpu
from dyn_VFLX import VFLX_tendency_cpu

from dyn_diagnostics import (diag_PVTF_cpu, diag_PHI_cpu,
                             diag_POTTVB_cpu, diag_secondary_cpu)
from dyn_timestep import make_timestep_cpu
if gpu_enable:
    from misc_boundaries import exchange_BC_gpu
    from dyn_continuity import continuity_gpu
    from dyn_POTT import POTT_tendency_gpu

    from dyn_diagnostics import (diag_PVTF_gpu, diag_PHI_gpu,
                                 diag_POTTVB_gpu, diag_secondary_gpu)
    from dyn_timestep import make_timestep_gpu
    from dyn_UVFLX_prepare import UVFLX_prep_adv_gpu
    from dyn_UFLX import UFLX_tendency_gpu
    from dyn_VFLX import VFLX_tendency_gpu
###############################################################################



class TendencyFactory:
    """
    """
    
    def __init__(self, target):
        """
        """
        self.fields_continuity = ['UFLX', 'VFLX', 'FLXDIV',
                                  'UWIND', 'VWIND', 'WWIND',
                                  'COLP', 'dCOLPdt', 'COLP_NEW', 'COLP_OLD']
        self.fields_temperature = ['dPOTTdt', 'POTT', 'UFLX', 'VFLX',
                             'COLP', 'POTTVB', 'WWIND', 'COLP_NEW',
                             'dPOTTdt_RAD']
        self.fields_momentum = ['dUFLXdt', 'dVFLXdt',
                        'UWIND', 'VWIND', 'WWIND',
                        'UFLX', 'VFLX',
                        'CFLX', 'QFLX', 'DFLX', 'EFLX',
                        'SFLX', 'TFLX', 'BFLX', 'RFLX',
                        'PHI', 'COLP', 'COLP_NEW', 'POTT',
                        'PVTF', 'PVTFVB',
                        'WWIND_UWIND', 'WWIND_VWIND']

        self.target = target


    def continuity(self, GR, GRF, UFLX, VFLX, FLXDIV,
                    UWIND, VWIND, WWIND,
                    COLP, dCOLPdt, COLP_NEW, COLP_OLD):
        """
        """

        if self.target == GPU:
            continuity_gpu[bpg_sc, tpb_sc](UFLX, VFLX, FLXDIV,
                    UWIND, VWIND, WWIND,
                    COLP, dCOLPdt, COLP_NEW, COLP_OLD,
                    GRF['dyis'], GRF['dxjs'],
                    GRF['dsigma'], GRF['sigma_vb'],
                    GRF['A'], GR.dt)
            exchange_BC_gpu[bpg, tpb](UFLX)
            exchange_BC_gpu[bpg, tpb](VFLX)
            exchange_BC_gpu[bpg, tpb](WWIND)
            exchange_BC_gpu[bpg, tpb](COLP_NEW)

        elif self.target == CPU:
            continuity_cpu(UFLX, VFLX, FLXDIV,
                    UWIND, VWIND, WWIND,
                    COLP, dCOLPdt, COLP_NEW, COLP_OLD,
                    GRF['dyis'], GRF['dxjs'],
                    GRF['dsigma'], GRF['sigma_vb'],
                    GRF['A'], GR.dt)
            exchange_BC_cpu(UFLX)
            exchange_BC_cpu(VFLX)
            exchange_BC_cpu(WWIND)
            exchange_BC_cpu(COLP_NEW)


    def temperature(self, GRF,
                    dPOTTdt, POTT, UFLX, VFLX,
                    COLP, POTTVB, WWIND, COLP_NEW, dPOTTdt_RAD):
        """
        """
        if self.target == GPU:
            POTT_tendency_gpu[bpg, tpb](GRF['A'], GRF['dsigma'],
                    GRF['POTT_dif_coef'],
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW, dPOTTdt_RAD)
        elif self.target == CPU:
            POTT_tendency_cpu(GRF['A'], GRF['dsigma'],
                    GRF['POTT_dif_coef'],
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW, dPOTTdt_RAD)


    def momentum(self, GRF,
                    dUFLXdt, dVFLXdt,
                    UWIND, VWIND, WWIND,
                    UFLX, VFLX,
                    CFLX, QFLX, DFLX, EFLX,
                    SFLX, TFLX, BFLX, RFLX,
                    PHI, COLP, COLP_NEW, POTT,
                    PVTF, PVTFVB,
                    WWIND_UWIND, WWIND_VWIND):
        """
        """

        if self.target == GPU:
            # PREPARE ADVECTIVE FLUXES
            if i_UVFLX_hor_adv or i_UVFLX_vert_adv:
                UVFLX_prep_adv_gpu[bpg, tpb_ks](
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP_NEW, GRF['A'], GRF['dsigma'])

            # UFLX
            UFLX_tendency_gpu[bpg, tpb](
                        dUFLXdt, UFLX, UWIND, VWIND,
                        BFLX, CFLX, DFLX, EFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        GRF['corf_is'], GRF['lat_is_rad'],
                        GRF['dlon_rad'], GRF['dlat_rad'],
                        GRF['dyis'],
                        GRF['dsigma'], GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])

            # VFLX
            VFLX_tendency_gpu[bpg, tpb](
                        dVFLXdt, VFLX, UWIND, VWIND,
                        RFLX, SFLX, TFLX, QFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,
                        GRF['corf'],        GRF['lat_rad'],
                        GRF['dlon_rad'],    GRF['dlat_rad'],
                        GRF['dxjs'], 
                        GRF['dsigma'],      GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])


        elif self.target == CPU:
            # PREPARE ADVECTIVE FLUXES
            if i_UVFLX_hor_adv or i_UVFLX_vert_adv:
                UVFLX_prep_adv_cpu(
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP_NEW, GRF['A'], GRF['dsigma'])
            # UFLX
            UFLX_tendency_cpu(
                        dUFLXdt, UFLX, UWIND, VWIND,
                        BFLX, CFLX, DFLX, EFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        GRF['corf_is'], GRF['lat_is_rad'],
                        GRF['dlon_rad'], GRF['dlat_rad'],
                        GRF['dyis'],
                        GRF['dsigma'], GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])

            # VFLX
            VFLX_tendency_cpu(
                        dVFLXdt, VFLX, UWIND, VWIND,
                        RFLX, SFLX, TFLX, QFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,
                        GRF['corf'],        GRF['lat_rad'],
                        GRF['dlon_rad'],    GRF['dlat_rad'],
                        GRF['dxjs'], 
                        GRF['dsigma'],      GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])



class DiagnosticsFactory:
    """
    """
    
    def __init__(self, target):
        """
        """
        self.fields_primary_diag = ['COLP', 'PVTF', 'PVTFVB',
                                  'PHI', 'PHIVB', 'POTT', 'POTTVB',
                                  'HSURF']
        self.fields_secondary_diag = ['POTTVB', 'TAIRVB', 'PVTFVB',
                       'COLP', 'PAIR', 'PHI', 'POTT',
                       'TAIR', 'RHO', 'PVTF',
                       'UWIND', 'VWIND', 'WIND']

        self.target = target

    def primary_diag(self, GRF,
                        COLP, PVTF, PVTFVB, 
                        PHI, PHIVB, POTT, POTTVB, HSURF):

        if self.target == GPU:

            diag_PVTF_gpu[bpg, tpb](COLP, PVTF, PVTFVB, GRF['sigma_vb'])
            diag_PHI_gpu[bpg, tpb_ks] (PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 

            #exchange_BC_gpu[bpg, tpb](PVTF)
            #exchange_BC_gpu[bpg, tpb_ks](PVTFVB)
            #exchange_BC_gpu[bpg, tpb](PHI)

            diag_POTTVB_gpu[bpg, tpb_ks](POTTVB, POTT, PVTF, PVTFVB)

            #TURB.diag_rho(GR, COLP, POTT, PVTF, POTTVB, PVTFVB)
            #TURB.diag_dz(GR, PHI, PHIVB)

        elif self.target == CPU:

            diag_PVTF_cpu(COLP, PVTF, PVTFVB, GRF['sigma_vb'])
            diag_PHI_cpu(PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 

            diag_POTTVB_cpu(POTTVB, POTT, PVTF, PVTFVB)


    def secondary_diag(self,
                       POTTVB, TAIRVB, PVTFVB, 
                       COLP, PAIR, PHI, POTT, 
                       TAIR, RHO, PVTF,
                       UWIND, VWIND, WIND):

        if self.target == GPU:

            diag_secondary_gpu[bpg, tpb_ks](POTTVB, TAIRVB, PVTFVB, 
                                        COLP, PAIR, PHI, POTT, 
                                        TAIR, RHO, PVTF,
                                        UWIND, VWIND, WIND)

        elif self.target == CPU:

            diag_secondary_cpu(POTTVB, TAIRVB, PVTFVB, 
                                        COLP, PAIR, PHI, POTT, 
                                        TAIR, RHO, PVTF,
                                        UWIND, VWIND, WIND)



class PrognosticsFactory:
    def __init__(self, target):
        """
        """
        self.fields_prognostic = ['UWIND_OLD', 'UWIND', 'VWIND_OLD',
                    'VWIND', 'COLP_OLD', 'COLP', 'POTT_OLD', 'POTT',
                    'dUFLXdt', 'dVFLXdt', 'dPOTTdt']

        self.target = target


    def euler_forward(self, GR, GRF, UWIND_OLD, UWIND, VWIND_OLD,
                    VWIND, COLP_OLD, COLP, POTT_OLD, POTT,
                    dUFLXdt, dVFLXdt, dPOTTdt):
        """
        """
        if self.target == GPU:

            make_timestep_gpu[bpg, tpb](COLP, COLP_OLD,
                      POTT, POTT_OLD, dPOTTdt,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt, GRF['A'], GR.dt)
            exchange_BC_gpu[bpg, tpb](POTT)
            exchange_BC_gpu[bpg, tpb](VWIND)
            exchange_BC_gpu[bpg, tpb](UWIND)

        elif self.target == CPU:

            make_timestep_cpu(COLP, COLP_OLD,
                      POTT, POTT_OLD, dPOTTdt,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt, GRF['A'], GR.dt)
            exchange_BC_cpu(POTT)
            exchange_BC_cpu(VWIND)
            exchange_BC_cpu(UWIND)

