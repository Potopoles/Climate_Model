#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190609
Last modified:      20190610
License:            MIT

Computation steps for turbulence for GPU and CPU

HISTORY
#- 20190609: CH  First implementation.
###############################################################################
"""
from math import exp
from numba import cuda, njit, prange

from io_read_namelist import (wp, wp_int, wp_str, gpu_enable)
from io_constants import con_g
from main_grid import nx,nxs,ny,nys,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator
from mix_meteo_utilities import calc_virtual_temperature
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################
def turb_flux_tendency_py(ALT, ALT_kp1, ALT_km1, ALTVB, ALTVB_kp1, 
                VAR, VAR_kp1, VAR_km1, KVAR, KVAR_kp1,
                RHO, RHOVB, RHOVB_kp1, COLP, surf_flux_VAR, k):
    """
    Vertical turbulent transport. 
    """

    if k == wp_int(0):
        dVARdt_TURB = (
                (
                + wp(0.)
                - ( ( VAR     - VAR_kp1 ) / ( ALT     - ALT_kp1 )
                    * RHOVB_kp1 * KVAR_kp1 )
                ) / ( ( ALTVB - ALTVB_kp1 ) * RHO )
        )
    elif k == nz-1:
        dVARdt_TURB = (
                (
                + ( ( VAR_km1 - VAR     ) / ( ALT_km1 - ALT     )
                    * RHOVB     * KVAR     )
                + surf_flux_VAR
                ) / ( ( ALTVB - ALTVB_kp1 ) * RHO )
        )
    else:
        dVARdt_TURB = (
                (
                + ( ( VAR_km1 - VAR     ) / ( ALT_km1 - ALT     )
                    * RHOVB     * KVAR     )
                - ( ( VAR     - VAR_kp1 ) / ( ALT     - ALT_kp1 )
                    * RHOVB_kp1 * KVAR_kp1 )
                ) / ( ( ALTVB - ALTVB_kp1 ) * RHO )
        )
    return(dVARdt_TURB)

def bulk_richardson_py():
    #TODO implement calculation
    #T_v = calc_virtual_temperature_py(TAIR, QV)
    #Ri_b = con_g / 
    Ri_b = wp(0.15)
    return(Ri_b)

def compute_K_coefs_py(HGTVB, UWIND_km1, UWIND, VWIND_km1, VWIND_k,
                       ALT_km1, ALT, con_k, free_mix_len):
    # compute mixing length
    mix_len = con_k * HGTVB / (wp(1.) + con_k * HGTVB / free_mix_len )

    Ri_c = wp(0.25)
    Ri_b = bulk_richardson() 

    KMOM = mix_len ** 2 * shear_term * (Ri_c - Ri_b) / Ri_c

    if KHEAT < wp(0.):
        KHEAT = wp(0.)

    #KHEAT = wp(0.1)

    return(KHEAT)


def run_all_py(PHI, PHI_kp1, PHI_km1, PHIVB, PHIVB_kp1, HSURF,
                QV, QV_kp1, QV_km1, KHEAT, KHEAT_kp1,
                RHO, RHOVB, RHOVB_kp1, COLP, SQVFLX,
                k):

    # constant values
    # free atmosphericm mixing length [m]
    free_mix_len = wp(200.)
    # von Kármán constant
    con_k = wp(0.35)

    # compute altitudes
    ALT         = PHI       / con_g
    ALT_kp1     = PHI_kp1   / con_g
    ALT_km1     = PHI_km1   / con_g
    ALTVB       = PHIVB     / con_g
    ALTVB_kp1   = PHIVB_kp1 / con_g

    # compute heights
    HGTVB       = ALTVB - HSURF
    HGTVB_kp1   = ALTVB_kp1 - HSURF


    # calculate K coefficients
    #value_srf = wp(0.1)
    #vert_reduce = wp(2)
    #KHEAT       = value_srf * exp( - vert_reduce * 
    #                               (nzs - k - wp_int(1))/nzs )
    #KHEAT_kp1   = value_srf * exp( - vert_reduce * 
    #                               (nzs - k            )/nzs ) 
    KHEAT       = compute_K_coefs(HGTVB, con_k, free_mix_len)
    KHEAT_kp1   = compute_K_coefs(HGTVB_kp1, con_k, free_mix_len)

    # calculate turbulent flux divergences
    dQVdt_TURB  = turb_flux_tendency(
            ALT, ALT_kp1, ALT_km1, ALTVB, ALTVB_kp1, 
            QV, QV_kp1, QV_km1, KHEAT, KHEAT_kp1, 
            RHO, RHOVB, RHOVB_kp1, COLP, SQVFLX, k)

    return(KHEAT, dQVdt_TURB)






###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
calc_virtual_temperature = njit(calc_virtual_temperature_py, device=True,
                                inline=True)
turb_flux_tendency = njit(turb_flux_tendency_py, device=True, inline=True)
compute_K_coefs = njit(compute_K_coefs_py, device=True, inline=True)
run_all     = njit(run_all_py, device=True, inline=True)

def launch_cuda_kernel(PHI, PHIVB, HSURF, RHO, RHOVB, COLP,
                       QV, dQVdt_TURB, KHEAT, SQVFLX):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        KHEAT[i  ,j  ,k], dQVdt_TURB[i,j,k] = run_all(
                PHI         [i  ,j  ,k  ], PHI        [i  ,j  ,k+1],
                PHI         [i  ,j  ,k-1],
                PHIVB       [i  ,j  ,k  ], PHIVB      [i  ,j  ,k+1],
                HSURF       [i  ,j  ,0  ], 
                QV          [i  ,j  ,k  ], QV         [i  ,j  ,k+1],
                QV          [i  ,j  ,k-1], 
                KHEAT       [i  ,j  ,k  ], KHEAT      [i  ,j  ,k+1],
                RHO         [i  ,j  ,k  ],
                RHOVB       [i  ,j  ,k  ], RHOVB      [i  ,j  ,k+1],
                COLP        [i  ,j  ,0  ], SQVFLX     [i  ,j  ,0  ],
                k)




if gpu_enable:
    compute_turbulence_gpu = cuda.jit(cuda_kernel_decorator(
                            launch_cuda_kernel,
                            non_3D={'mixing_length':wp_str}
                            ))(launch_cuda_kernel)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################

#def launch_numba_cpu(A, dsigma, POTT_dif_coef,
#                    dPOTTdt, POTT, UFLX, VFLX, COLP,
#                    POTTVB, WWIND, COLP_NEW, dPOTTdt_RAD):
#
#    for i in prange(nb,nx+nb):
#        for j in range(nb,ny+nb):
#            for k in range(wp_int(0),nz):
#                dPOTTdt[i  ,j  ,k] = \
#                    add_up_tendencies(POTT[i  ,j  ,k],
#                        POTT[i-1,j  ,k], POTT[i+1,j  ,k],
#                        POTT[i  ,j-1,k], POTT[i  ,j+1,k],
#                        UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
#                        VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
#                        COLP[i  ,j  ,0],
#                        COLP[i-1,j  ,0], COLP[i+1,j  ,0],
#                        COLP[i  ,j-1,0], COLP[i  ,j+1,0],
#                        POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
#                        WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
#                        COLP_NEW[i  ,j  ,0], dPOTTdt_RAD[i  ,j  ,k],
#                        A[i  ,j  ,0],
#                        dsigma[0  ,0  ,k], POTT_dif_coef[0  ,0  ,k], k)
#
#
#POTT_tendency_cpu = njit(parallel=True)(launch_numba_cpu)





