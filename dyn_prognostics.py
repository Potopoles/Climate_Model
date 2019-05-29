#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          dyn_prognostics.py  
Author:             Christoph Heim
Date created:       20190529
Last modified:      20190529
License:            MIT

Functions for prognostic step for both CPU and GPU.
###############################################################################
"""
import time
import numpy as np
import cupy as cp
from numba import cuda, njit, prange
from math import pow

from namelist import (pair_top)
from org_namelist import (wp, wp_str, wp_int)
from constants import con_g, con_Rd, con_kappa, con_cp
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################
def euler_forward_pw_py(VAR, dVARdt, COLP, COLP_OLD, dt):
    """
    Advance time step euler forward with pressure weighting.
    """
    return( VAR * COLP_OLD/COLP + dt*dVARdt/COLP )

def interp_COLPA_js_py(COLP, COLP_jm1, COLP_im1, COLP_ip1,
                       COLP_jm1_ip1, COLP_jm1_im1,
                       A, A_jm1, A_im1, A_ip1,
                       A_jm1_ip1, A_jm1_im1):
    """
    """
    return( wp(1.)/wp(8.)*(
                    COLP_jm1_ip1    * A_jm1_ip1 +
                    COLP_ip1        * A_ip1     +
           wp(2.) * COLP_jm1        * A_jm1     +
           wp(2.) * COLP            * A         +
                    COLP_jm1_im1    * A_jm1_im1 +
                    COLP_im1        * A_im1     ) )

def interp_COLPA_is_py(COLP, COLP_im1, COLP_jm1, COLP_jp1,
                       COLP_im1_jp1, COLP_im1_jm1,
                       A, A_im1, A_jm1, A_jp1,
                       A_im1_jp1, A_im1_jm1, j):
    """
    """
    if j == 1:
        return( wp(1.)/wp(4.)*(
                        COLP_im1_jp1    * A_im1_jp1 +
                        COLP_jp1        * A_jp1     +
                        COLP_im1        * A_im1     +
                        COLP            * A         ) )
    elif j == ny:
        return( wp(1.)/wp(4.)*(
                        COLP_im1_jm1    * A_im1_jm1 +
                        COLP_jm1        * A_jm1     +
                        COLP_im1        * A_im1     +
                        COLP            * A         ) )
    else:
        return( wp(1.)/wp(8.)*(
                        COLP_im1_jp1    * A_im1_jp1 +
                        COLP_jp1        * A_jp1     +
               wp(2.) * COLP_im1        * A_im1     +
               wp(2.) * COLP            * A         +
                        COLP_im1_jm1    * A_im1_jm1 +
                        COLP_jm1        * A_jm1     ) )



def run_all_py( COLP,           
                COLP_im1,           COLP_ip1,       
                COLP_jm1,           COLP_jp1,
                COLP_im1_jm1,       COLP_im1_jp1,
                COLP_ip1_jm1,       COLP_ip1_jp1,   
                COLP_OLD,           
                COLP_OLD_im1,       COLP_OLD_ip1,       
                COLP_OLD_jm1,       COLP_OLD_jp1,
                COLP_OLD_im1_jm1,   COLP_OLD_im1_jp1,
                COLP_OLD_ip1_jm1,   COLP_OLD_ip1_jp1,
                POTT_OLD, dPOTTdt,
                UWIND_OLD, dUFLXdt,
                VWIND_OLD, dVFLXdt,
                A,           
                A_im1,              A_ip1,       
                A_jm1,              A_jp1,
                A_im1_jm1,          A_im1_jp1,
                A_ip1_jm1,          A_ip1_jp1,
                dt, j):
    """
    """

    # POTT
    POTT = euler_forward_pw(POTT_OLD, dPOTTdt, COLP, COLP_OLD, dt)

    ## UWIND
    COLPA_is     = interp_COLPA_is(COLP, COLP_im1, COLP_jm1, COLP_jp1,
                               COLP_im1_jp1,   COLP_im1_jm1,
                               A,    A_im1,    A_jm1,    A_jp1,
                               A_im1_jp1,      A_im1_jm1, j)
    COLPA_OLD_is = interp_COLPA_is(COLP_OLD, COLP_OLD_im1,
                               COLP_OLD_jm1, COLP_OLD_jp1,
                               COLP_OLD_im1_jp1,   COLP_OLD_im1_jm1,
                               A,    A_im1,    A_jm1,    A_jp1,
                               A_im1_jp1,      A_im1_jm1, j)
    UWIND = euler_forward_pw(UWIND_OLD, dUFLXdt, COLPA_is, COLPA_OLD_is, dt)
    
    # VWIND
    COLPA_js     = interp_COLPA_js(COLP, COLP_jm1, COLP_im1, COLP_ip1,
                                   COLP_ip1_jm1,   COLP_im1_jm1,
                                   A,    A_jm1,    A_im1,    A_ip1,
                                   A_ip1_jm1,      A_im1_jm1)
    COLPA_OLD_js = interp_COLPA_js(COLP_OLD, COLP_OLD_jm1,
                                   COLP_OLD_im1,   COLP_OLD_ip1,
                                   COLP_OLD_ip1_jm1,   COLP_OLD_im1_jm1,
                                   A,    A_jm1,    A_im1,    A_ip1,
                                   A_ip1_jm1,      A_im1_jm1)
    VWIND = euler_forward_pw(VWIND_OLD, dVFLXdt, COLPA_js, COLPA_OLD_js, dt)

    
    return(POTT, UWIND, VWIND)







###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
euler_forward_pw = njit(euler_forward_pw_py, device=True, inline=True)
interp_COLPA_js = njit(interp_COLPA_js_py, device=True, inline=True)
interp_COLPA_is = njit(interp_COLPA_is_py, device=True, inline=True)
run_all = njit(run_all_py, device=True, inline=True)

def make_timestep_gpu(COLP, COLP_OLD,
                      POTT, POTT_OLD, dPOTTdt,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt,
                      A, dt):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nxs+nb and j >= nb and j < nys+nb:
        POTT[i,j,k], UWIND[i,j,k], VWIND[i,j,k] = run_all(
                COLP     [i  ,j  ,0  ], 
                COLP     [i-1,j  ,0  ], COLP     [i+1,j  ,0  ],       
                COLP     [i  ,j-1,0  ], COLP     [i  ,j+1,0  ],
                COLP     [i-1,j-1,0  ], COLP     [i-1,j+1,0  ],
                COLP     [i+1,j-1,0  ], COLP     [i+1,j+1,0  ],   
                COLP_OLD [i  ,j  ,0  ], 
                COLP_OLD [i-1,j  ,0  ], COLP_OLD [i+1,j  ,0  ],       
                COLP_OLD [i  ,j-1,0  ], COLP_OLD [i  ,j+1,0  ],
                COLP_OLD [i-1,j-1,0  ], COLP_OLD [i-1,j+1,0  ],
                COLP_OLD [i+1,j-1,0  ], COLP_OLD [i+1,j+1,0  ],   

                POTT_OLD [i  ,j  ,k  ], dPOTTdt  [i  ,j  ,k  ],
                UWIND_OLD[i  ,j  ,k  ], dUFLXdt  [i  ,j  ,k  ],
                VWIND_OLD[i  ,j  ,k  ], dVFLXdt  [i  ,j  ,k  ],

                A        [i  ,j  ,0  ], 
                A        [i-1,j  ,0  ], A        [i+1,j  ,0  ],       
                A        [i  ,j-1,0  ], A        [i  ,j+1,0  ],
                A        [i-1,j-1,0  ], A        [i-1,j+1,0  ],
                A        [i+1,j-1,0  ], A        [i+1,j+1,0  ],   
                dt, j)

make_timestep_gpu = cuda.jit(cuda_kernel_decorator(make_timestep_gpu,
                        non_3D={'dt':wp_str}))(make_timestep_gpu)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################

#diag_PVTF_cpu = njit(parallel=True)(diag_PVTF_cpu)





