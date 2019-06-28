#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190609
Last modified:      20190628
License:            MIT

Computation of moisture variables (QV, QC) tendencies
(dQVdt, dQCdt) according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.3, page 212

HISTORY
20190609: CH  First implementation.
###############################################################################
"""
import numpy as np
from numba import cuda, njit, prange

from namelist import (i_moist_main_switch, i_moist_hor_adv, i_moist_vert_adv,
                      i_moist_vert_turb, i_moist_num_dif)
from io_read_namelist import (i_moist_microphys,
                              wp, wp_int, gpu_enable)
from io_constants import con_Lh
from main_grid import nx,nxs,ny,nys,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator

from dyn_functions import (hor_adv_py, vert_adv_py, turb_flux_tendency_py,
                           num_dif_pw_py, comp_VARVB_log_py)
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################
#def turbulence_py(dQVdt_TURB, COLP):
#    return(dQVdt_TURB * COLP)


def microphysics_py():
    raise NotImplementedError()
#    dQVdt[i,j,k] = dQVdt[i,j,k] + \
#                        dQVdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries

def add_up_tendencies_py(
            QV, QV_im1, QV_ip1, QV_jm1, QV_jp1,
            QV_km1, QV_kp1,
            QC, QC_im1, QC_ip1, QC_jm1, QC_jp1,
            QC_km1, QC_kp1,
            UFLX, UFLX_ip1, VFLX, VFLX_jp1,
            COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1,
            WWIND, WWIND_kp1,
            COLP_NEW, 

            PHI, PHI_kp1, PHI_km1, PHIVB, PHIVB_kp1, 
            KHEAT, KHEAT_kp1, 
            RHO, RHOVB, RHOVB_kp1, SLHFLX,

            A, dsigma, moist_dif_coef, k):

    dQVdt = wp(0.)
    dQCdt = wp(0.)

    if i_moist_main_switch:
        # HORIZONTAL ADVECTION
        if i_moist_hor_adv:
            dQVdt = dQVdt + hor_adv(
                QV,
                QV_im1, QV_ip1,
                QV_jm1, QV_jp1,
                UFLX, UFLX_ip1,
                VFLX, VFLX_jp1,
                A)
            dQCdt = dQCdt + hor_adv(
                QC,
                QC_im1, QC_ip1,
                QC_jm1, QC_jp1,
                UFLX, UFLX_ip1,
                VFLX, VFLX_jp1,
                A)
        # VERTICAL ADVECTION
        if i_moist_vert_adv:
            QVVB = comp_VARVB_log(QV, QV_km1)
            QVVB_kp1 = comp_VARVB_log(QV_kp1, QV)
            dQVdt = dQVdt + vert_adv(
                QVVB, QVVB_kp1,
                WWIND, WWIND_kp1,
                COLP_NEW, dsigma, k)
            QCVB = comp_VARVB_log(QC, QC_km1)
            QCVB_kp1 = comp_VARVB_log(QC_kp1, QC)
            dQCdt = dQCdt + vert_adv(
                QCVB, QCVB_kp1,
                WWIND, WWIND_kp1,
                COLP_NEW, dsigma, k)
        # VERTICAL TURBULENT TRANSPORT
        if i_moist_vert_turb:
            surf_flux = SLHFLX / con_Lh
            dQVdt_TURB = turb_flux_tendency(
                    PHI, PHI_kp1, PHI_km1, PHIVB, PHIVB_kp1, 
                    QV, QV_kp1, QV_km1, KHEAT, KHEAT_kp1, 
                    RHO, RHOVB, RHOVB_kp1, COLP, surf_flux, k)
            dQVdt = dQVdt + dQVdt_TURB

        # NUMERICAL HORIZONTAL DIFUSION
        if i_moist_num_dif and (moist_dif_coef > wp(0.)):
            dQVdt = dQVdt + num_dif(
                QV, QV_im1, QV_ip1,
                QV_jm1, QV_jp1,
                COLP, COLP_im1, COLP_ip1,
                COLP_jm1, COLP_jp1,
                moist_dif_coef)
            dQCdt = dQCdt + num_dif(
                QC, QC_im1, QC_ip1,
                QC_jm1, QC_jp1,
                COLP, COLP_im1, COLP_ip1,
                COLP_jm1, COLP_jp1,
                moist_dif_coef)

    return(dQVdt, dQVdt_TURB, dQCdt)






###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
hor_adv         = njit(hor_adv_py, device=True, inline=True)
num_dif         = njit(num_dif_pw_py, device=True, inline=True)
comp_VARVB_log  = njit(comp_VARVB_log_py, device=True, inline=True)
vert_adv        = njit(vert_adv_py, device=True, inline=True)
turb_flux_tendency = njit(turb_flux_tendency_py, device=True, inline=True)
add_up_tendencies = njit(add_up_tendencies_py, device=True, inline=True)


def launch_cuda_kernel(A, dsigma, moist_dif_coef,
                       dQVdt, dQVdt_TURB, QV, dQCdt, QC, UFLX, VFLX, COLP,
                       WWIND, COLP_NEW,
                       PHI, PHIVB, KHEAT, RHO, RHOVB, SLHFLX):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dQVdt[i  ,j  ,k], dQVdt_TURB[i  ,j  ,k], dQCdt[i  ,j  ,k] = \
            add_up_tendencies(
            QV          [i  ,j  ,k  ],
            QV          [i-1,j  ,k  ], QV         [i+1,j  ,k  ],
            QV          [i  ,j-1,k  ], QV         [i  ,j+1,k  ],
            QV          [i  ,j  ,k-1], QV         [i  ,j  ,k+1],
            QC          [i  ,j  ,k  ],
            QC          [i-1,j  ,k  ], QC         [i+1,j  ,k  ],
            QC          [i  ,j-1,k  ], QC         [i  ,j+1,k  ],
            QC          [i  ,j  ,k-1], QC         [i  ,j  ,k+1],
            UFLX        [i  ,j  ,k  ], UFLX       [i+1,j  ,k  ],
            VFLX        [i  ,j  ,k  ], VFLX       [i  ,j+1,k  ],
            COLP        [i  ,j  ,0  ],
            COLP        [i-1,j  ,0  ], COLP       [i+1,j  ,0  ],
            COLP        [i  ,j-1,0  ], COLP       [i  ,j+1,0  ],
            WWIND       [i  ,j  ,k  ], WWIND      [i  ,j  ,k+1],
            COLP_NEW    [i  ,j  ,0  ],

            PHI         [i  ,j  ,k  ], PHI        [i  ,j  ,k+1],
            PHI         [i  ,j  ,k-1], PHIVB      [i  ,j  ,k  ],
            PHIVB       [i  ,j  ,k+1], 
            KHEAT       [i  ,j  ,k  ], KHEAT      [i  ,j  ,k+1],
            RHO         [i  ,j  ,k  ], RHOVB      [i  ,j  ,k  ],
            RHOVB       [i  ,j  ,k+1], SLHFLX     [i  ,j  ,0  ],

            A[i  ,j  ,0],
            dsigma[0  ,0  ,k], moist_dif_coef[0  ,0  ,k],
            k)



if gpu_enable:
    moist_tendency_gpu = cuda.jit(cuda_kernel_decorator(launch_cuda_kernel))\
                                (launch_cuda_kernel)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################
hor_adv         = njit(hor_adv_py)
comp_VARVB_log  = njit(comp_VARVB_log_py)
vert_adv        = njit(vert_adv_py)
num_dif         = njit(num_dif_pw_py)
turb_flux_tendency = njit(turb_flux_tendency_py)
add_up_tendencies = njit(add_up_tendencies_py)

def launch_numba_cpu(A, dsigma, moist_dif_coef,
                    dQVdt, dQVdt_TURB, QV, dQCdt, QC, UFLX, VFLX, COLP,
                    WWIND, COLP_NEW,
                    PHI, PHIVB, KHEAT, RHO, RHOVB, SLHFLX):

    for i in prange(nb,nx+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(0),nz):
                dQVdt[i  ,j  ,k], dQVdt_TURB[i  ,j  ,k], dQCdt[i  ,j  ,k] = \
                    add_up_tendencies(
                        QV          [i  ,j  ,k  ],
                        QV          [i-1,j  ,k  ], QV       [i+1,j  ,k  ],
                        QV          [i  ,j-1,k  ], QV       [i  ,j+1,k  ],
                        QV          [i  ,j  ,k-1], QV       [i  ,j  ,k+1],
                        QC          [i  ,j  ,k  ],
                        QC          [i-1,j  ,k  ], QC       [i+1,j  ,k  ],
                        QC          [i  ,j-1,k  ], QC       [i  ,j+1,k  ],
                        QC          [i  ,j  ,k-1], QC       [i  ,j  ,k+1],
                        UFLX        [i  ,j  ,k  ], UFLX     [i+1,j  ,k  ],
                        VFLX        [i  ,j  ,k  ], VFLX     [i  ,j+1,k  ],
                        COLP        [i  ,j  ,0  ],
                        COLP        [i-1,j  ,0  ], COLP     [i+1,j  ,0  ],
                        COLP        [i  ,j-1,0  ], COLP     [i  ,j+1,0  ],
                        WWIND       [i  ,j  ,k  ], WWIND    [i  ,j  ,k+1],
                        COLP_NEW    [i  ,j  ,0  ],

                        PHI         [i  ,j  ,k  ], PHI      [i  ,j  ,k+1],
                        PHI         [i  ,j  ,k-1], PHIVB    [i  ,j  ,k  ],
                        PHIVB       [i  ,j  ,k+1], 
                        KHEAT       [i  ,j  ,k  ], KHEAT    [i  ,j  ,k+1],
                        RHO         [i  ,j  ,k  ], RHOVB    [i  ,j  ,k  ],
                        RHOVB       [i  ,j  ,k+1], SLHFLX   [i  ,j  ,0  ],

                        A           [i  ,j  ,0  ],
                        dsigma      [0  ,0  ,k  ], moist_dif_coef[0  ,0  ,k],
                        k)






moist_tendency_cpu = njit(parallel=True)(launch_numba_cpu)





