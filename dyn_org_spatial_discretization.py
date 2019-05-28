#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          dyn_org_spatial_discretization.py  
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190528
License:            MIT

Organise the computation of all tendencies in dynamical core:
- virtual potential temperature POTT
- horizontal momentum fluxes UFLX and VFLX
- continuity equation resulting in column pressure COLP tendency and
  vertical wind speed WWIND

Organise the computation of all diagnostic functions:
- 

Differentiate between computation targets GPU and CPU.
###############################################################################
"""
import math
import numpy as np
from numba import cuda

from namelist import (i_UVFLX_hor_adv, i_UVFLX_vert_adv)
from org_namelist import HOST, DEVICE
from grid import nx,nxs,ny,nys,nz,nzs,nb
from grid import tpb, tpb_ks, bpg, tpb_sc, bpg_sc
from GPU import exchange_BC_gpu
from CPU import exchange_BC_cpu
from dyn_continuity import (continuity_gpu, continuity_cpu)
from dyn_POTT import POTT_tendency_gpu, POTT_tendency_cpu
from dyn_UVFLX_prepare import (UVFLX_prep_adv_gpu, UVFLX_prep_adv_cpu)
from dyn_UFLX import (UFLX_tendency_gpu, UFLX_tendency_cpu)
from dyn_VFLX import (VFLX_tendency_gpu, VFLX_tendency_cpu)
from dyn_diagnostics import (diag_PVTF_gpu,
                             diag_PHI_gpu)
###############################################################################





class TendencyFactory:
    """
    """
    
    def __init__(self):
        """
        """
        self.fields_continuity = ['UFLX', 'VFLX', 'FLXDIV',
                                  'UWIND', 'VWIND', 'WWIND',
                                  'COLP', 'dCOLPdt', 'COLP_NEW', 'COLP_OLD']
        self.fields_temperature = ['dPOTTdt', 'POTT', 'UFLX', 'VFLX',
                             'COLP', 'POTTVB', 'WWIND', 'COLP_NEW']
        self.fields_momentum = ['dUFLXdt', 'dVFLXdt',
                        'UWIND', 'VWIND', 'WWIND',
                        'UFLX', 'VFLX',
                        'CFLX', 'QFLX', 'DFLX', 'EFLX',
                        'SFLX', 'TFLX', 'BFLX', 'RFLX',
                        'PHI', 'COLP', 'COLP_NEW', 'POTT',
                        'PVTF', 'PVTFVB',
                        'WWIND_UWIND', 'WWIND_VWIND']


    def continuity(self, target, GR, UFLX, VFLX, FLXDIV,
                    UWIND, VWIND, WWIND,
                    COLP, dCOLPdt, COLP_NEW, COLP_OLD):
        """
        """

        if target == DEVICE:
            continuity_gpu[bpg_sc, tpb_sc](UFLX, VFLX, FLXDIV,
                    UWIND, VWIND, WWIND,
                    COLP, dCOLPdt, COLP_NEW, COLP_OLD,
                    GR.dyisd, GR.dxjsd, GR.dsigmad, GR.sigma_vbd,
                    GR.Ad, GR.dt)
            exchange_BC_gpu[bpg, tpb](UFLX)
            exchange_BC_gpu[bpg, tpb](VFLX)
            exchange_BC_gpu[bpg, tpb](WWIND)
            exchange_BC_gpu[bpg, tpb](COLP_NEW)

        elif target == HOST:
            continuity_cpu(UFLX, VFLX, FLXDIV,
                    UWIND, VWIND, WWIND,
                    COLP, dCOLPdt, COLP_NEW, COLP_OLD,
                    GR.dyis, GR.dxjs, GR.dsigma, GR.sigma_vb,
                    GR.A, GR.dt)
            exchange_BC_cpu(UFLX)
            exchange_BC_cpu(VFLX)
            exchange_BC_cpu(WWIND)
            exchange_BC_cpu(COLP_NEW)


    def temperature(self, target, GR,
                            dPOTTdt, POTT, UFLX, VFLX,
                            COLP, POTTVB, WWIND, COLP_NEW):
        """
        """
        if target == DEVICE:
            POTT_tendency_gpu[bpg, tpb](GR.Ad, GR.dsigmad,
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW)
        elif target == HOST:
            POTT_tendency_cpu(GR.A, GR.dsigma,
                    dPOTTdt, POTT, UFLX, VFLX, COLP,
                    POTTVB, WWIND, COLP_NEW)


    def momentum(self, target, GR,
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

        if target == DEVICE:
            # PREPARE ADVECTIVE FLUXES
            if i_UVFLX_hor_adv or i_UVFLX_vert_adv:
                UVFLX_prep_adv_gpu[bpg, tpb_ks](
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP_NEW, GR.Ad, GR.dsigmad)

            # UFLX
            UFLX_tendency_gpu[bpg, tpb](
                        dUFLXdt, UFLX, UWIND, VWIND,
                        BFLX, CFLX, DFLX, EFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        GR.corf_isd, GR.lat_is_radd,
                        GR.dlon_radd, GR.dlat_radd,
                        GR.dyisd,
                        GR.dsigmad, GR.sigma_vbd)

            # VFLX
            VFLX_tendency_gpu[bpg, tpb](
                        dVFLXdt, VFLX, UWIND, VWIND,
                        RFLX, SFLX, TFLX, QFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,
                        GR.corfd,       GR.lat_radd,
                        GR.dlon_radd,   GR.dlat_radd,
                        GR.dxjsd, 
                        GR.dsigmad,     GR.sigma_vbd)


        elif target == HOST:
            # PREPARE ADVECTIVE FLUXES
            if i_UVFLX_hor_adv or i_UVFLX_vert_adv:
                UVFLX_prep_adv_cpu(
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP_NEW, GR.A, GR.dsigma)
            # UFLX
            UFLX_tendency_cpu(
                        dUFLXdt, UFLX, UWIND, VWIND,
                        BFLX, CFLX, DFLX, EFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        GR.corf_is,     GR.lat_is_rad,
                        GR.dlon_rad,    GR.dlat_rad,
                        GR.dyis,
                        GR.dsigma,      GR.sigma_vb)

            # VFLX
            VFLX_tendency_cpu(
                        dVFLXdt, VFLX, UWIND, VWIND,
                        RFLX, SFLX, TFLX, QFLX,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,
                        GR.corf,        GR.lat_rad,
                        GR.dlon_rad,    GR.dlat_rad,
                        GR.dxjs, 
                        GR.dsigma,      GR.sigma_vb)





from geopotential_cuda import get_geopotential, diag_pvt_factor
class DiagnosticsFactory:
    """
    """
    
    def __init__(self):
        """
        """
        self.fields_primary_diag = ['COLP', 'PVTF', 'PVTFVB',
                                  'PHI', 'PHIVB', 'POTT',
                                  'HSURF']

    def primary_diag(self, target, GR_OLD, GR,
                        COLP, PVTF, PVTFVB, 
                        PHI, PHIVB, POTT, HSURF):

        if target == DEVICE:
            #diag_PVTF_gpu[GR_OLD.griddim, GR_OLD.blockdim] \
            #                    (COLP, PVTF, PVTFVB, GR.sigma_vbd)
            diag_pvt_factor[GR_OLD.griddim, GR_OLD.blockdim, GR_OLD.stream] \
                                (COLP, PVTF, PVTFVB, GR_OLD.sigma_vbd)
            GR_OLD.stream.synchronize()

            #diag_PHI_gpu[GR_OLD.griddim_ks, GR_OLD.blockdim_ks] \
            #                   (PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 
            get_geopotential[GR_OLD.griddim_ks, GR_OLD.blockdim_ks, GR_OLD.stream] \
                               (PHI, PHIVB, PVTF, PVTFVB, POTT, HSURF) 
            GR_OLD.stream.synchronize()

            #exchange_BC_gpu[bpg, tpb](PVTF)
            #exchange_BC_gpu[bpg, tpb](PVTFVB)
            #exchange_BC_gpu[bpg, tpb](PHI)
            #PVTF  = exchange_BC_gpu(PVTF, GR_OLD.zonal, GR_OLD.merid,   \
            #                        GR_OLD.griddim, GR_OLD.blockdim, GR_OLD.stream)
            PVTFVB  = exchange_BC_gpu(PVTFVB, GR_OLD.zonalvb, GR_OLD.meridvb,   \
                                    GR_OLD.griddim_ks, GR_OLD.blockdim_ks, GR_OLD.stream)
            PHI  = exchange_BC_gpu(PHI, GR_OLD.zonal, GR_OLD.merid,   \
                                    GR_OLD.griddim, GR_OLD.blockdim, GR_OLD.stream)
            GR_OLD.stream.synchronize()
