#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190529
Last modified:      20190609
License:            MIT

Functions for prognostic step for both CPU and GPU.

HISTORY
- 20190529  : Created (CH) 
20190609    : Added moisture variables QV and QC (CH)
###############################################################################
"""
import time
import numpy as np
from numba import cuda, njit, prange
from math import pow

from namelist import (pair_top)
from io_read_namelist import (wp, wp_str, wp_int, gpu_enable)
from io_constants import con_g, con_Rd, con_kappa, con_cp
from main_grid import nx,nxs,ny,nys,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator
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










###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
def run_all_gpu(COLP,           
                COLP_im1,           COLP_ip1,       
                COLP_jm1,           COLP_jp1,
                COLP_im1_jm1,       COLP_im1_jp1,
                COLP_ip1_jm1,       COLP_ip1_jp1,   
                COLP_OLD,           
                COLP_OLD_im1,       COLP_OLD_ip1,       
                COLP_OLD_jm1,       COLP_OLD_jp1,
                COLP_OLD_im1_jm1,   COLP_OLD_im1_jp1,
                COLP_OLD_ip1_jm1,   COLP_OLD_ip1_jp1,
                UWIND_OLD, dUFLXdt,
                VWIND_OLD, dVFLXdt,
                POTT_OLD, dPOTTdt,
                QV_OLD, dQVdt,
                QC_OLD, dQCdt,
                A,           
                A_im1,              A_ip1,       
                A_jm1,              A_jp1,
                A_im1_jm1,          A_im1_jp1,
                A_ip1_jm1,          A_ip1_jp1,
                dt, i, j):
    """
    """

    ## UWIND
    if j < ny+nb:
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
    
    ## VWIND
    if i < nx+nb:
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
    if i < nx+nb and j < ny+nb:
        ## POTT
        POTT = euler_forward_pw(POTT_OLD, dPOTTdt, COLP, COLP_OLD, dt)
        ## QV
        QV = euler_forward_pw(QV_OLD, dQVdt, COLP, COLP_OLD, dt)
        # clip negative values
        if QV < wp(0.):
            QV = wp(0.)
        ## QC
        QC = euler_forward_pw(QC_OLD, dQCdt, COLP, COLP_OLD, dt)
        # clip negative values
        if QC < wp(0.):
            QC = wp(0.)

    return(POTT, UWIND, VWIND, QV, QC)

euler_forward_pw = njit(euler_forward_pw_py, device=True, inline=True)
interp_COLPA_js = njit(interp_COLPA_js_py, device=True, inline=True)
interp_COLPA_is = njit(interp_COLPA_is_py, device=True, inline=True)
run_all_gpu = njit(run_all_gpu, device=True, inline=True)

def make_timestep_gpu(COLP, COLP_OLD,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt,
                      POTT, POTT_OLD, dPOTTdt,
                      QV, QV_OLD, dQVdt,
                      QC, QC_OLD, dQCdt,
                      A, dt):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nxs+nb and j >= nb and j < nys+nb:
        (POTT[i,j,k], UWIND[i,j,k], VWIND[i,j,k],
        QV[i,j,k], QC[i,j,k]) = run_all_gpu(
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

                UWIND_OLD[i  ,j  ,k  ], dUFLXdt  [i  ,j  ,k  ],
                VWIND_OLD[i  ,j  ,k  ], dVFLXdt  [i  ,j  ,k  ],
                POTT_OLD [i  ,j  ,k  ], dPOTTdt  [i  ,j  ,k  ],
                QV_OLD   [i  ,j  ,k  ], dQVdt    [i  ,j  ,k  ],
                QC_OLD   [i  ,j  ,k  ], dQCdt    [i  ,j  ,k  ],

                A        [i  ,j  ,0  ], 
                A        [i-1,j  ,0  ], A        [i+1,j  ,0  ],       
                A        [i  ,j-1,0  ], A        [i  ,j+1,0  ],
                A        [i-1,j-1,0  ], A        [i-1,j+1,0  ],
                A        [i+1,j-1,0  ], A        [i+1,j+1,0  ],   
                dt, i, j)

if gpu_enable:
    make_timestep_gpu = cuda.jit(cuda_kernel_decorator(make_timestep_gpu,
                            non_3D={'dt':wp_str}))(make_timestep_gpu)



###############################################################################
### SPECIALIZE FOR CPU
###############################################################################
euler_forward_pw    = njit(euler_forward_pw_py)
interp_COLPA_js     = njit(interp_COLPA_js_py)
interp_COLPA_is     = njit(interp_COLPA_is_py)

def make_timestep_cpu(COLP, COLP_OLD,
                      UWIND, UWIND_OLD, dUFLXdt,
                      VWIND, VWIND_OLD, dVFLXdt,
                      POTT, POTT_OLD, dPOTTdt,
                      QV, QV_OLD, dQVdt,
                      QC, QC_OLD, dQCdt,
                      A, dt):

    for i in prange(nb,nxs+nb):
        for j in range(nb,nys+nb):

            COLP_           = COLP      [i  ,j  ,0  ] 
            COLP_im1        = COLP      [i-1,j  ,0  ]
            COLP_ip1        = COLP      [i+1,j  ,0  ]       
            COLP_jm1        = COLP      [i  ,j-1,0  ]
            COLP_jp1        = COLP      [i  ,j+1,0  ]
            COLP_im1_jm1    = COLP      [i-1,j-1,0  ]
            COLP_im1_jp1    = COLP      [i-1,j+1,0  ]
            COLP_ip1_jm1    = COLP      [i+1,j-1,0  ]
            COLP_ip1_jp1    = COLP      [i+1,j+1,0  ]   

            COLP_OLD_       = COLP_OLD  [i  ,j  ,0  ] 
            COLP_OLD_im1    = COLP_OLD  [i-1,j  ,0  ]
            COLP_OLD_ip1    = COLP_OLD  [i+1,j  ,0  ]       
            COLP_OLD_jm1    = COLP_OLD  [i  ,j-1,0  ]
            COLP_OLD_jp1    = COLP_OLD  [i  ,j+1,0  ]
            COLP_OLD_im1_jm1= COLP_OLD  [i-1,j-1,0  ]
            COLP_OLD_im1_jp1= COLP_OLD  [i-1,j+1,0  ]
            COLP_OLD_ip1_jm1= COLP_OLD  [i+1,j-1,0  ]
            COLP_OLD_ip1_jp1= COLP_OLD  [i+1,j+1,0  ]   

            A_              = A         [i  ,j  ,0  ] 
            A_im1           = A         [i-1,j  ,0  ]
            A_ip1           = A         [i+1,j  ,0  ]       
            A_jm1           = A         [i  ,j-1,0  ]
            A_jp1           = A         [i  ,j+1,0  ]
            A_im1_jm1       = A         [i-1,j-1,0  ]
            A_im1_jp1       = A         [i-1,j+1,0  ]
            A_ip1_jm1       = A         [i+1,j-1,0  ]
            A_ip1_jp1       = A         [i+1,j+1,0  ]   

            ## UWIND
            COLPA_is     = interp_COLPA_is(COLP_, COLP_im1, COLP_jm1, COLP_jp1,
                                       COLP_im1_jp1,   COLP_im1_jm1,
                                       A_,    A_im1,    A_jm1,    A_jp1,
                                       A_im1_jp1,      A_im1_jm1, j)
            COLPA_OLD_is = interp_COLPA_is(COLP_OLD_, COLP_OLD_im1,
                                       COLP_OLD_jm1, COLP_OLD_jp1,
                                       COLP_OLD_im1_jp1,   COLP_OLD_im1_jm1,
                                       A_,    A_im1,    A_jm1,    A_jp1,
                                       A_im1_jp1,      A_im1_jm1, j)
            
            # VWIND
            COLPA_js     = interp_COLPA_js(COLP_, COLP_jm1, COLP_im1, COLP_ip1,
                                           COLP_ip1_jm1,   COLP_im1_jm1,
                                           A_,    A_jm1,    A_im1,    A_ip1,
                                           A_ip1_jm1,      A_im1_jm1)
            COLPA_OLD_js = interp_COLPA_js(COLP_OLD_, COLP_OLD_jm1,
                                           COLP_OLD_im1,   COLP_OLD_ip1,
                                           COLP_OLD_ip1_jm1,   COLP_OLD_im1_jm1,
                                           A_,    A_jm1,    A_im1,    A_ip1,
                                           A_ip1_jm1,      A_im1_jm1)

            for k in range(wp_int(0),nz):
                UWIND_OLD_      = UWIND_OLD [i  ,j  ,k  ]
                dUFLXdt_        = dUFLXdt   [i  ,j  ,k  ]
                VWIND_OLD_      = VWIND_OLD [i  ,j  ,k  ]
                dVFLXdt_        = dVFLXdt   [i  ,j  ,k  ]
                POTT_OLD_       = POTT_OLD  [i  ,j  ,k  ]
                dPOTTdt_        = dPOTTdt   [i  ,j  ,k  ]
                QV_OLD_         = QV_OLD    [i  ,j  ,k  ]
                dQVdt_          = dQVdt     [i  ,j  ,k  ]
                QC_OLD_         = QC_OLD    [i  ,j  ,k  ]
                dQCdt_          = dQCdt     [i  ,j  ,k  ]

                UWIND[i,j,k] = euler_forward_pw(UWIND_OLD_, dUFLXdt_,
                                        COLPA_is, COLPA_OLD_is, dt)
                VWIND[i,j,k] = euler_forward_pw(VWIND_OLD_, dVFLXdt_,
                                        COLPA_js, COLPA_OLD_js, dt)
                POTT[i,j,k]  = euler_forward_pw(POTT_OLD_, dPOTTdt_,
                                        COLP_, COLP_OLD_, dt)
                QV[i,j,k]    = euler_forward_pw(QV_OLD_, dQVdt_,
                                          COLP_, COLP_OLD_, dt)
                QC[i,j,k]    = euler_forward_pw(QC_OLD_, dQCdt_,
                                        COLP_, COLP_OLD_, dt)
make_timestep_cpu = njit(make_timestep_cpu)





