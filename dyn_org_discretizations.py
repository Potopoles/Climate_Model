#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190616
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

HISTORY
- 20190604  : Created (CH) 
20190609    : Added moisture variables QV and QC (CH)
###############################################################################
"""
import numpy as np
from numba import cuda

from namelist import (i_UVFLX_hor_adv, i_UVFLX_vert_adv,
                      i_UVFLX_vert_turb, i_moist_main_switch)
from io_read_namelist import CPU, GPU, gpu_enable
from main_grid import (nx,nxs,ny,nys,nz,nzs,nb,
                 tpb, tpb_ks, bpg, tpb_sc, bpg_sc)

from misc_boundaries import exchange_BC_cpu
from misc_utilities import function_input_fields
from dyn_continuity import continuity_cpu
from dyn_UVFLX_prepare import UVFLX_prep_adv_cpu
from dyn_UFLX import UFLX_tendency_cpu
from dyn_VFLX import VFLX_tendency_cpu
from dyn_POTT import POTT_tendency_cpu
from dyn_moist import moist_tendency_cpu

from dyn_diagnostics import (diag_PVTF_cpu, diag_PHI_cpu,
                             diag_POTTVB_cpu, diag_secondary_cpu)
from dyn_timestep import make_timestep_cpu
if gpu_enable:
    from misc_boundaries import exchange_BC_gpu
    from dyn_continuity import continuity_gpu
    from dyn_UVFLX_prepare import UVFLX_prep_adv_gpu
    from dyn_UFLX import UFLX_tendency_gpu
    from dyn_VFLX import VFLX_tendency_gpu
    from dyn_POTT import POTT_tendency_gpu
    from dyn_moist import moist_tendency_gpu

    from dyn_diagnostics import (diag_PVTF_gpu, diag_PHI_gpu,
                                 diag_POTTVB_gpu, diag_secondary_gpu)
    from dyn_timestep import make_timestep_gpu
###############################################################################



class TendencyFactory:
    """
    """
    
    def __init__(self, target):
        """
        """
        self.target = target
        self.fields_continuity = function_input_fields(self.continuity)
        self.fields_momentum = function_input_fields(self.momentum)
        self.fields_temperature = function_input_fields(self.temperature)
        self.fields_moisture = function_input_fields(self.moisture)


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



    def momentum(self, GRF,
                 dUFLXdt, dVFLXdt,
                 UWIND, VWIND, WWIND,
                 UFLX, VFLX,
                 CFLX, QFLX, DFLX, EFLX,
                 SFLX, TFLX, BFLX, RFLX,
                 PHI, PHIVB, COLP, COLP_NEW, POTT,
                 PVTF, PVTFVB,
                 WWIND_UWIND, WWIND_VWIND,

                 KMOM_dUWINDdz, KMOM_dVWINDdz,
                 KMOM, RHOVB, RHO,

                 dUFLXdt_TURB, dVFLXdt_TURB,
                 SMOMXFLX, SMOMYFLX):
        """
        """
        if self.target == GPU:

            #TODO why is this necessary?
            exchange_BC_gpu[bpg, tpb](KMOM)

            # PREPARE ADVECTIVE FLUXES
            if i_UVFLX_hor_adv or i_UVFLX_vert_adv or i_UVFLX_vert_turb:
                UVFLX_prep_adv_gpu[bpg, tpb_ks](
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP, COLP_NEW,
                            KMOM_dUWINDdz, KMOM_dVWINDdz,
                            KMOM, PHI, RHOVB,
                            GRF['A'], GRF['dsigma'])


            exchange_BC_gpu[bpg, tpb](KMOM_dUWINDdz)
            exchange_BC_gpu[bpg, tpb](KMOM_dVWINDdz)
            # TODO: Can remove this after surface is run on full domain
            # including nb grid points
            exchange_BC_gpu[bpg, tpb](SMOMXFLX)
            exchange_BC_gpu[bpg, tpb](SMOMYFLX)

            # UFLX
            UFLX_tendency_gpu[bpg, tpb](
                        dUFLXdt, UFLX, UWIND, VWIND,
                        BFLX, CFLX, DFLX, EFLX,
                        PHI, PHIVB, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,

                        KMOM_dUWINDdz, RHO,
                        dUFLXdt_TURB, SMOMXFLX,

                        GRF['corf_is'], GRF['lat_is_rad'],
                        GRF['dlon_rad'], GRF['dlat_rad'],
                        GRF['dyis'],
                        GRF['dsigma'], GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])

            #show = np.isnan(np.asarray(dUFLXdt))
            #print(np.sum(show))

            # VFLX
            VFLX_tendency_gpu[bpg, tpb](
                        dVFLXdt, VFLX, UWIND, VWIND,
                        RFLX, SFLX, TFLX, QFLX,
                        PHI, PHIVB, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,

                        KMOM_dVWINDdz, RHO,
                        dVFLXdt_TURB, SMOMYFLX,

                        GRF['corf'],        GRF['lat_rad'],
                        GRF['dlon_rad'],    GRF['dlat_rad'],
                        GRF['dxjs'], 
                        GRF['dsigma'],      GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])

            #show = np.isnan(np.asarray(dVFLXdt))
            #print(np.sum(show))


        elif self.target == CPU:

            #TODO why is this necessary?
            exchange_BC_cpu(KMOM)

            # PREPARE ADVECTIVE FLUXES
            if i_UVFLX_hor_adv or i_UVFLX_vert_adv or i_UVFLX_vert_turb:
                UVFLX_prep_adv_cpu(
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP, COLP_NEW,
                            KMOM_dUWINDdz, KMOM_dVWINDdz,
                            KMOM, PHI, RHOVB,
                            GRF['A'], GRF['dsigma'])

            exchange_BC_cpu(KMOM_dUWINDdz)
            exchange_BC_cpu(KMOM_dVWINDdz)
            # TODO: Can remove this after surface is run on full domain
            # including nb grid points
            exchange_BC_cpu(SMOMXFLX)
            exchange_BC_cpu(SMOMYFLX)

            #import matplotlib.pyplot as plt
            #k = 10
            ##var = KMOM_dUWINDdz
            #var = KMOM_dVWINDdz
            ##var = RHO
            #plt.contourf(np.asarray(var)[:,:,k].T)
            #print(np.asarray(var)[:,:,k])
            #print(np.sum(np.isnan(np.asarray(var))))
            #plt.colorbar()
            #plt.show()
            #quit()

            #show = np.isnan(np.asarray(KMOM_dUWINDdz))
            #print(np.sum(show))
            #quit()

            #show = np.isnan(np.asarray(dUFLXdt))
            #print(np.sum(show))

            # UFLX
            UFLX_tendency_cpu(
                        dUFLXdt, UFLX, UWIND, VWIND,
                        BFLX, CFLX, DFLX, EFLX,
                        PHI, PHIVB, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,

                        KMOM_dUWINDdz, RHO,
                        dUFLXdt_TURB, SMOMXFLX,

                        GRF['corf_is'], GRF['lat_is_rad'],
                        GRF['dlon_rad'], GRF['dlat_rad'],
                        GRF['dyis'],
                        GRF['dsigma'], GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])

            #show = np.isnan(np.asarray(dUFLXdt))
            #print(np.sum(show))

            # VFLX
            VFLX_tendency_cpu(
                        dVFLXdt, VFLX, UWIND, VWIND,
                        RFLX, SFLX, TFLX, QFLX,
                        PHI, PHIVB, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,

                        KMOM_dVWINDdz, RHO,
                        dVFLXdt_TURB, SMOMYFLX,

                        GRF['corf'],        GRF['lat_rad'],
                        GRF['dlon_rad'],    GRF['dlat_rad'],
                        GRF['dxjs'], 
                        GRF['dsigma'],      GRF['sigma_vb'],
                        GRF['UVFLX_dif_coef'])

            #show = np.isnan(np.asarray(dVFLXdt))
            #print(np.sum(show))


    def temperature(self, GRF,
                    dPOTTdt, POTT, UFLX, VFLX,
                    COLP, POTTVB, WWIND, COLP_NEW,
                    dPOTTdt_TURB, dPOTTdt_RAD):
        """
        """
        if self.target == GPU:
            POTT_tendency_gpu[bpg, tpb](GRF['A'], GRF['dsigma'],
                    GRF['POTT_dif_coef'],
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW,
                    dPOTTdt_TURB, dPOTTdt_RAD)
        elif self.target == CPU:
            POTT_tendency_cpu(GRF['A'], GRF['dsigma'],
                    GRF['POTT_dif_coef'],
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW,
                    dPOTTdt_TURB, dPOTTdt_RAD)



    def moisture(self, GRF,
                dQVdt, dQVdt_TURB, QV, dQCdt, QC, UFLX, VFLX,
                COLP, WWIND, COLP_NEW,
                PHI, PHIVB, KHEAT, RHO, RHOVB, SQVFLX):
        """
        """
        if self.target == GPU:
            moist_tendency_gpu[bpg, tpb](GRF['A'], GRF['dsigma'],
                    GRF['moist_dif_coef'],
                    dQVdt, dQVdt_TURB, QV, dQCdt, QC, UFLX, VFLX, COLP,
                    WWIND, COLP_NEW,
                    PHI, PHIVB, KHEAT, RHO, RHOVB, SQVFLX)
        elif self.target == CPU:
            moist_tendency_cpu(GRF['A'], GRF['dsigma'],
                    GRF['moist_dif_coef'],
                    dQVdt, dQVdt_TURB, QV, dQCdt, QC, UFLX, VFLX, COLP,
                    WWIND, COLP_NEW,
                    PHI, PHIVB, KHEAT, RHO, RHOVB, SQVFLX)




class DiagnosticsFactory:
    """
    """
    
    def __init__(self, target):
        """
        """
        self.target = target
        self.fields_primary_diag = function_input_fields(self.primary_diag)
        self.fields_secondary_diag = function_input_fields(self.secondary_diag)

    def primary_diag(self, GRF,
                        COLP, PVTF, PVTFVB, 
                        PHI, PHIVB, POTT, POTTVB, HSURF):

        if self.target == GPU:

            diag_PVTF_gpu[bpg, tpb](COLP, PVTF, PVTFVB, GRF['sigma_vb'])
            diag_PHI_gpu[bpg, tpb_ks] (PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 

            diag_POTTVB_gpu[bpg, tpb_ks](POTTVB, POTT, PVTF, PVTFVB)

        elif self.target == CPU:

            diag_PVTF_cpu(COLP, PVTF, PVTFVB, GRF['sigma_vb'])
            diag_PHI_cpu(PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 

            diag_POTTVB_cpu(POTTVB, POTT, PVTF, PVTFVB)


    def secondary_diag(self,
                       POTTVB, TAIRVB, PVTFVB, 
                       COLP, PAIR, PAIRVB, PHI, POTT, 
                       TAIR, RHO, RHOVB, PVTF,
                       UWIND, VWIND, WINDX, WINDY, WIND):

        if self.target == GPU:

            diag_secondary_gpu[bpg, tpb_ks](POTTVB, TAIRVB, PVTFVB, 
                                        COLP, PAIR, PAIRVB, PHI, POTT, 
                                        TAIR, RHO, RHOVB, PVTF,
                                        UWIND, VWIND, WINDX, WINDY, WIND)

        elif self.target == CPU:

            diag_secondary_cpu(POTTVB, TAIRVB, PVTFVB, 
                                        COLP, PAIR, PAIRVB, PHI, POTT, 
                                        TAIR, RHO, RHOVB, PVTF,
                                        UWIND, VWIND, WINDX, WINDY, WIND)



class PrognosticsFactory:
    def __init__(self, target):
        """
        """
        self.target = target
        self.fields_prognostic = function_input_fields(self.euler_forward)


    def euler_forward(self, GR, GRF, UWIND_OLD, UWIND, VWIND_OLD,
                    VWIND, COLP_OLD, COLP, POTT_OLD, POTT,
                    QV, QV_OLD, QC, QC_OLD,
                    dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt):
        """
        """
        if self.target == GPU:

            make_timestep_gpu[bpg, tpb](COLP, COLP_OLD,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt,
                      POTT, POTT_OLD, dPOTTdt,
                      QV, QV_OLD, dQVdt,
                      QC, QC_OLD, dQCdt, GRF['A'], GR.dt)
            exchange_BC_gpu[bpg, tpb](POTT)
            exchange_BC_gpu[bpg, tpb](VWIND)
            exchange_BC_gpu[bpg, tpb](UWIND)
            if i_moist_main_switch:
                exchange_BC_gpu[bpg, tpb](QV)
                exchange_BC_gpu[bpg, tpb](QC)

        elif self.target == CPU:

            make_timestep_cpu(COLP, COLP_OLD,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt,
                      POTT, POTT_OLD, dPOTTdt,
                      QV, QV_OLD, dQVdt,
                      QC, QC_OLD, dQCdt, GRF['A'], GR.dt)
            exchange_BC_cpu(POTT)
            exchange_BC_cpu(VWIND)
            exchange_BC_cpu(UWIND)
            if i_moist_main_switch:
                exchange_BC_cpu(QV)
                exchange_BC_cpu(QC)

