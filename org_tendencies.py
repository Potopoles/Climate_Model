#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          org_tendencies.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190512
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
from tendency_UFLX import (UFLX_tendency_gpu, UFLX_tendency_cpu,
                           UFLX_vert_adv_gpu)
from tendency_VFLX import VFLX_tendency_gpu, VFLX_tendency_cpu
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
                        PHI, COLP, COLP_NEW, POTT,
                        PVTF, PVTFVB,
                        WWIND_UWIND, WWIND_VWIND,
                        A, dsigma, sigma_vb,
                        dyis, dxjs):
        if self.target == 'GPU':
            # UFLX
            if i_UVFLX_vert_adv:
                UFLX_vert_adv_gpu[bpg, tpb_ks](
                            WWIND_UWIND, UWIND, WWIND,
                            COLP_NEW, A, dsigma)
                cuda.synchronize()
                print(np.asarray(WWIND_UWIND))
                quit()
            UFLX_tendency_gpu[bpg, tpb](
                        dUFLXdt, UFLX, PHI, COLP, POTT,
                        PVTF, PVTFVB, WWIND_UWIND,
                        dsigma, sigma_vb, dyis)
            cuda.synchronize()

            # VFLX
            VFLX_tendency_gpu[bpg, tpb](
                        dVFLXdt, VFLX, PHI, COLP, POTT,
                        PVTF, PVTFVB, dsigma, sigma_vb, dxjs)
            cuda.synchronize()

        elif self.target == 'CPU':
            raise NotImplementedError()
        return(dUFLXdt, dVFLXdt)
