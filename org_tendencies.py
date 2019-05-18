#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          org_tendencies.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190513
License:            MIT

Organise the computation of all tendencies in dynamical core:
- potential temperature POTT

Differentiate between computation targets:
- GPU
- CPU
"""
import numpy as np
from numba import cuda

from tendency_POTT import POTT_tendency_gpu, POTT_tendency_cpu
from tendency_UVFLX_prepare import (UVFLX_vert_adv_gpu,
                            UVFLX_vert_adv_cpu)
from tendency_UFLX import (UFLX_tendency_gpu, UFLX_tendency_cpu)
from tendency_VFLX import (VFLX_tendency_gpu, VFLX_tendency_cpu)
from namelist import i_UVFLX_vert_adv
from grid import tpb, tpb_ks, bpg
####################################################################



class TendencyFactory:
    """
    Depending on computation target, calls the right function
    to calculate the tendencies.
    """
    
    def __init__(self, target):
        """
        INPUT:
        - target: either 'CPU' or 'GPU'
        """
        self.target = target


    def POTT_tendency(self, dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                            POTTVB, WWIND, COLP_NEW, dsigma):
        if self.target == 'GPU':
            POTT_tendency_gpu[bpg, tpb](
                    dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                    POTTVB, WWIND, COLP_NEW, dsigma)
            cuda.synchronize()
        elif self.target == 'CPU':
            POTT_tendency_cpu(
                    dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                    POTTVB, WWIND, COLP_NEW, dsigma)
        return(dPOTTdt)


    def UVFLX_tendency(self, dUFLXdt, dVFLXdt,
                        UWIND, VWIND, WWIND,
                        UFLX, VFLX,
                        CFLX, QFLX, DFLX, EFLX,
                        SFLX, TFLX, BFLX, RFLX,
                        PHI, COLP, COLP_NEW, POTT,
                        PVTF, PVTFVB,
                        WWIND_UWIND, WWIND_VWIND,
                        A, corf_is, corf,
                        lat_rad, lat_is_rad,
                        dlon_rad, dlat_rad,
                        dyis, dxjs,
                        dsigma, sigma_vb):

        if self.target == 'GPU':
            # PREPARE
            if i_UVFLX_vert_adv:
                UVFLX_vert_adv_gpu[bpg, tpb_ks](
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX,
                            CFLX, QFLX, DFLX, EFLX,
                            SFLX, TFLX, BFLX, RFLX,
                            COLP_NEW, A, dsigma)
                cuda.synchronize()

            # UFLX
            UFLX_tendency_gpu[bpg, tpb](
                        dUFLXdt, UFLX, UWIND, VWIND,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        corf_is, lat_is_rad,
                        dlon_rad, dlat_rad,
                        dyis,
                        dsigma, sigma_vb)
            cuda.synchronize()

            # VFLX
            VFLX_tendency_gpu[bpg, tpb](
                        dVFLXdt, VFLX, UWIND,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,
                        corf, lat_rad,
                        dlon_rad, dlat_rad,
                        dxjs, 
                        dsigma, sigma_vb)
            cuda.synchronize()


        elif self.target == 'CPU':
            # PREPARE
            if i_UVFLX_vert_adv:
                UVFLX_vert_adv_cpu(
                            WWIND_UWIND, WWIND_VWIND,
                            UWIND, VWIND, WWIND,
                            COLP_NEW, A, dsigma)
            # UFLX
            UFLX_tendency_cpu(
                        dUFLXdt, UFLX, UWIND, VWIND,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        corf_is, lat_is_rad,
                        dlon_rad, dlat_rad,
                        dyis,
                        dsigma, sigma_vb)

            # VFLX
            VFLX_tendency_cpu(
                        dVFLXdt, VFLX, UWIND,
                        PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_VWIND,
                        corf, lat_rad,
                        dlon_rad, dlat_rad,
                        dxjs, 
                        dsigma, sigma_vb)

        return(dUFLXdt, dVFLXdt)
