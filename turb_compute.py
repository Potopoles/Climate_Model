#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190609
Last modified:      20190616
License:            MIT

Computation steps for turbulence for GPU and CPU

HISTORY
#- 20190609: CH  First implementation.
###############################################################################
"""
from math import exp, sqrt
from numba import cuda, njit, prange

from io_read_namelist import (wp, wp_int, wp_str, gpu_enable)
from io_constants import con_g
from main_grid import nx,nxs,ny,nys,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator
from misc_meteo_utilities import calc_virtual_temperature_py
from dyn_functions import comp_VARVB_log_py
###############################################################################

## CONSTANT VALUES
# critical bulk richardson number (value from theory from gradient richardson
# number is 0.25 but due to bulk approximation take higher value.
# (source: Wikipedia article on bulk richardson number.)
#Ri_c = wp(0.25)
Ri_c = wp(1.0)
# free atmosphericm mixing length [m]
free_mix_len = wp(200.)
# von Kármán constant
con_k = wp(0.35)
# Prandtl number for air
con_Pr = wp(0.72)

# minimum vertical wind difference to set manually
min_wind_diff = wp(0.0001)
# minimum value for KMOM
min_KMOM = wp(0.000001)
# maximum value for KMOM
#max_KMOM = wp(0.01) 
max_KMOM = wp(0.015)


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################

def bulk_richardson_py(QV_k, POTT_k,
                       POTT_km05,   POTT_kp05,
                       ALT_km05,    ALT_kp05,
                       WINDX_km05,  WINDX_kp05,
                       WINDY_km05,  WINDY_kp05):
    POTT_v_k = calc_virtual_temperature(POTT_k, QV_k)
    Ri_b_k = ( ( con_g / POTT_v_k * (POTT_km05 - POTT_kp05) *
               (ALT_km05 - ALT_kp05) ) /
             ( (WINDX_km05 - WINDX_kp05) ** wp(2.) + 
               (WINDY_km05 - WINDY_kp05) ** wp(2.) ) )
    return(Ri_b_k)


def compute_K_coefs_py(HGT_k,
                       WINDX_km05,  WINDX_kp05,
                       WINDY_km05,  WINDY_kp05,
                       ALT_km05,    ALT_kp05,
                       QV_km05,     QV_kp05,
                       POTT_k,
                       POTT_km05,      POTT_kp05):
    # compute mixing length
    mix_len = con_k * HGT_k / (wp(1.) + con_k * HGT_k / free_mix_len )

    # bulk richardson number
    QV_k = comp_VARVB_log(QV_kp05, QV_km05)
    Ri_b_k = bulk_richardson(QV_k, POTT_k,
                           POTT_km05,   POTT_kp05,
                           ALT_km05,    ALT_kp05,
                           WINDX_km05,  WINDX_kp05,
                           WINDY_km05,  WINDY_kp05) 

    # vertical wind shear term
    shear_term = sqrt( ( (WINDX_km05 - WINDX_kp05) /
                         (ALT_km05 - ALT_kp05)     ) ** wp(2.) +
                       ( (WINDY_km05 - WINDY_kp05) /
                         (ALT_km05 - ALT_kp05)     ) ** wp(2.) )

    # mixing coefficient for momentum
    KMOM_k = mix_len ** wp(2.) * shear_term * (Ri_c - Ri_b_k) / Ri_c

    # negative values possible in above formula but implies zero.
    # also most often the values are way too big ?? TODO
    if KMOM_k < min_KMOM:
        KMOM_k = min_KMOM
    if KMOM_k > max_KMOM:
        KMOM_k = max_KMOM

    # mixing coefficient for heat and moisture
    KHEAT_k = KMOM_k / con_Pr

    ## TODO
    #KHEAT_k = wp(1.)

    return(KMOM_k, KHEAT_k)


def run_all_py(PHI_k,       HSURF,
               PHI_km05,    PHI_kp05,
               QV_km05,     QV_kp05,
               WINDX_km05,  WINDX_kp05, 
               WINDY_km05,  WINDY_kp05, 
               POTT_k,
               POTT_km05,   POTT_kp05,
               k):
    # make sure no invalid values possible
    if WINDX_km05 == WINDX_kp05:
        WINDX_km05 += min_wind_diff
    if WINDY_km05 == WINDY_kp05:
        WINDY_km05 += min_wind_diff

    ## compute altitudes
    ALT_k       = PHI_k     / con_g
    ALT_km05    = PHI_km05  / con_g
    ALT_kp05    = PHI_kp05  / con_g

    # compute heights
    HGT_k       = ALT_k - HSURF

    # compute K coefficients
    KMOM_k, KHEAT_k  = compute_K_coefs(HGT_k,
                                       WINDX_km05,  WINDX_kp05,
                                       WINDY_km05,  WINDY_kp05,
                                       ALT_km05,    ALT_kp05,
                                       QV_km05,     QV_kp05,
                                       POTT_k,  
                                       POTT_km05,   POTT_kp05)
    
    return(KMOM_k, KHEAT_k)






###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
calc_virtual_temperature = njit(calc_virtual_temperature_py, device=True,
                                inline=True)
comp_VARVB_log  = njit(comp_VARVB_log_py, device=True, inline=True)
bulk_richardson = njit(bulk_richardson_py, device=True, inline=True)
compute_K_coefs = njit(compute_K_coefs_py, device=True, inline=True)
run_all     = njit(run_all_py, device=True, inline=True)

def launch_cuda_kernel(KMOM, KHEAT, PHIVB, HSURF, PHI, QV,
                       WINDX, WINDY, POTTVB, POTT):

    i, j, k = cuda.grid(3)
    if i < nx+2*nb and j < ny+2*nb and k > 0 and k < nzs-1:
        KMOM[i  ,j  ,k], KHEAT[i  ,j  ,k] = run_all(
                PHIVB       [i  ,j  ,k  ], HSURF      [i  ,j  ,0  ], 
                PHI         [i  ,j  ,k-1], PHI        [i  ,j  ,k  ],
                QV          [i  ,j  ,k-1], QV         [i  ,j  ,k  ],
                WINDX       [i  ,j  ,k-1], WINDX      [i  ,j  ,k  ],
                WINDY       [i  ,j  ,k-1], WINDY      [i  ,j  ,k  ],
                POTTVB      [i  ,j  ,k  ], 
                POTT        [i  ,j  ,k-1], POTT       [i  ,j  ,k  ],
                k)




if gpu_enable:
    compute_turbulence_gpu = cuda.jit(cuda_kernel_decorator(
                            launch_cuda_kernel))(launch_cuda_kernel)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################
calc_virtual_temperature = njit(calc_virtual_temperature_py)
comp_VARVB_log           = njit(comp_VARVB_log_py)
bulk_richardson          = njit(bulk_richardson_py)
compute_K_coefs          = njit(compute_K_coefs_py)
run_all                  = njit(run_all_py)

def launch_numba_cpu(KMOM, KHEAT, PHIVB, HSURF, PHI, QV,
                       WINDX, WINDY, POTTVB, POTT):

    for i in prange(0,nx+2*nb):
        for j in range(0,ny+2*nb):
            for k in range(wp_int(1),nzs-1):
                KMOM[i  ,j  ,k], KHEAT[i  ,j  ,k] = run_all(
                    PHIVB       [i  ,j  ,k  ], HSURF      [i  ,j  ,0  ], 
                    PHI         [i  ,j  ,k-1], PHI        [i  ,j  ,k  ],
                    QV          [i  ,j  ,k-1], QV         [i  ,j  ,k  ],
                    WINDX       [i  ,j  ,k-1], WINDX      [i  ,j  ,k  ],
                    WINDY       [i  ,j  ,k-1], WINDY      [i  ,j  ,k  ],
                    POTTVB      [i  ,j  ,k  ], 
                    POTT        [i  ,j  ,k-1], POTT       [i  ,j  ,k  ],
                    k)


compute_turbulence_cpu = njit(parallel=True)(launch_numba_cpu)





