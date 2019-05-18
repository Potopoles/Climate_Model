#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          tendency_UVFLX.py  
Author:             Christoph Heim (CH)
Date created:       20190518
Last modified:      20190518
License:            MIT

Prepare computation of horizontal momentum flux tendencies
for both directions (longitude dUFLXdt, latitude dVFLXdt)
according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition
Chapter 7.4, page 214ff
"""
import time
import numpy as np
import cupy as cp
from math import sin, cos
from numba import cuda, njit, prange

from org_namelist import (wp, wp_int, wp_old)
from grid import nx,nxs,ny,nys,nz,nzs,nb
from GPU import cuda_kernel_decorator

from tendency_functions import (interp_WWIND_UVWIND_py,
                                calc_momentum_fluxes_isjs_py,
                                calc_momentum_fluxes_ijs_py,
                                calc_momentum_fluxes_isj_py,
                                calc_momentum_fluxes_ij_py)
####################################################################


####################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
####################################################################



####################################################################
### SPECIALIZE FOR GPU
####################################################################
interp_WWIND_UVWIND = njit(interp_WWIND_UVWIND_py, device=True,
                        inline=True)
calc_momentum_fluxes_isjs = njit(calc_momentum_fluxes_isjs_py,
                        device=True, inline=True)
calc_momentum_fluxes_ijs = njit(calc_momentum_fluxes_ijs_py,
                        device=True, inline=True)
calc_momentum_fluxes_isj = njit(calc_momentum_fluxes_isj_py,
                        device=True, inline=True)
calc_momentum_fluxes_ij = njit(calc_momentum_fluxes_ij_py,
                        device=True, inline=True)

def launch_cuda_vert_adv_kernel(WWIND_UWIND, WWIND_VWIND,
                                UWIND, VWIND, WWIND,
                                UFLX_3D, VFLX_3D,
                                CFLX, QFLX, DFLX, EFLX,
                                SFLX, TFLX, BFLX, RFLX,
                                COLP_NEW, A, dsigma):
    i, j, k = cuda.grid(3)

    UFLX            = UFLX_3D[i  ,j  ,k  ]
    UFLX_im1        = UFLX_3D[i-1,j  ,k  ]
    UFLX_im1_jm1    = UFLX_3D[i-1,j-1,k  ]
    UFLX_im1_jp1    = UFLX_3D[i-1,j+1,k  ]
    UFLX_ip1        = UFLX_3D[i+1,j  ,k  ]
    UFLX_ip1_jm1    = UFLX_3D[i+1,j-1,k  ]
    UFLX_ip1_jp1    = UFLX_3D[i+1,j+1,k  ]
    UFLX_jm1        = UFLX_3D[i  ,j-1,k  ]
    UFLX_jp1        = UFLX_3D[i  ,j+1,k  ]

    VFLX            = VFLX_3D[i  ,j  ,k  ]
    VFLX_im1        = VFLX_3D[i-1,j  ,k  ]
    VFLX_im1_jm1    = VFLX_3D[i-1,j-1,k  ]
    VFLX_im1_jp1    = VFLX_3D[i-1,j+1,k  ]
    VFLX_ip1        = VFLX_3D[i+1,j  ,k  ]
    VFLX_ip1_jm1    = VFLX_3D[i+1,j-1,k  ]
    VFLX_ip1_jp1    = VFLX_3D[i+1,j+1,k  ]
    VFLX_jm1        = VFLX_3D[i  ,j-1,k  ]
    VFLX_jp1        = VFLX_3D[i  ,j+1,k  ]

    if i >= nb and i < nxs+nb and j >= nb and j < ny+nb:
        WWIND_UWIND[i  ,j  ,k  ] = \
            interp_WWIND_UVWIND(
            UWIND     [i  ,j  ,k  ], UWIND     [i  ,j  ,k-1],
            WWIND     [i  ,j  ,k  ], WWIND     [i-1,j  ,k  ],
            WWIND     [i  ,j-1,k  ], WWIND     [i  ,j+1,k  ],
            WWIND     [i-1,j-1,k  ], WWIND     [i-1,j+1,k  ], 
            COLP_NEW  [i  ,j  ,0  ], COLP_NEW  [i-1,j  ,0  ],
            COLP_NEW  [i  ,j-1,0  ], COLP_NEW  [i  ,j+1,0  ],
            COLP_NEW  [i-1,j-1,0  ], COLP_NEW  [i-1,j+1,0  ], 
            A         [i  ,j  ,0  ], A         [i-1,j  ,0  ],
            A         [i  ,j-1,0  ], A         [i  ,j+1,0  ],
            A         [i-1,j-1,0  ], A         [i-1,j+1,0  ], 
            dsigma    [0  ,0  ,k  ], dsigma    [0  ,0  ,k-1],
            True, j, ny, k)

    if i >= nb and i < nx+nb and j >= nb and j < nys+nb:
        WWIND_VWIND[i  ,j  ,k  ] = \
            interp_WWIND_UVWIND(
            VWIND     [i  ,j  ,k  ], VWIND     [i  ,j  ,k-1],
            WWIND     [i  ,j  ,k  ], WWIND     [i  ,j-1,k  ],
            WWIND     [i-1,j  ,k  ], WWIND     [i+1,j  ,k  ],
            WWIND     [i-1,j-1,k  ], WWIND     [i+1,j-1,k  ], 
            COLP_NEW  [i  ,j  ,0  ], COLP_NEW  [i  ,j-1,0  ],
            COLP_NEW  [i-1,j  ,0  ], COLP_NEW  [i+1,j  ,0  ],
            COLP_NEW  [i-1,j-1,0  ], COLP_NEW  [i+1,j-1,0  ], 
            A         [i  ,j  ,0  ], A         [i  ,j-1,0  ],
            A         [i-1,j  ,0  ], A         [i+1,j  ,0  ],
            A         [i-1,j-1,0  ], A         [i+1,j-1,0  ], 
            dsigma    [0  ,0  ,k  ], dsigma    [0  ,0  ,k-1],
            False, i, nx, k)

    if i >= nb and i < nxs+nb and j >= nb and j < nys+nb:
        CFLX[i,j,k],QFLX[i,j,k] = calc_momentum_fluxes_isjs(
                                    UFLX, UFLX_im1,
                                    UFLX_im1_jm1, UFLX_im1_jp1,
                                    UFLX_ip1, UFLX_ip1_jm1,
                                    UFLX_ip1_jp1, UFLX_jm1,
                                    UFLX_jp1,
                                    VFLX, VFLX_im1,
                                    VFLX_im1_jm1, VFLX_im1_jp1,
                                    VFLX_ip1, VFLX_ip1_jm1,
                                    VFLX_ip1_jp1, VFLX_jm1,
                                    VFLX_jp1)


    if i >= nb and i < nx+nb and j >= nb and j < nys+nb:
        DFLX[i,j,k],EFLX[i,j,k] = calc_momentum_fluxes_ijs(
                                    UFLX, UFLX_im1,
                                    UFLX_im1_jm1, UFLX_im1_jp1,
                                    UFLX_ip1, UFLX_ip1_jm1,
                                    UFLX_ip1_jp1, UFLX_jm1,
                                    UFLX_jp1,
                                    VFLX, VFLX_im1,
                                    VFLX_im1_jm1, VFLX_im1_jp1,
                                    VFLX_ip1, VFLX_ip1_jm1,
                                    VFLX_ip1_jp1, VFLX_jm1,
                                    VFLX_jp1)


    if i >= nb and i < nxs+nb and j >= nb and j < ny+nb:
        SFLX[i,j,k],TFLX[i,j,k] = calc_momentum_fluxes_isj(
                                    UFLX, UFLX_im1,
                                    UFLX_im1_jm1, UFLX_im1_jp1,
                                    UFLX_ip1, UFLX_ip1_jm1,
                                    UFLX_ip1_jp1, UFLX_jm1,
                                    UFLX_jp1,
                                    VFLX, VFLX_im1,
                                    VFLX_im1_jm1, VFLX_im1_jp1,
                                    VFLX_ip1, VFLX_ip1_jm1,
                                    VFLX_ip1_jp1, VFLX_jm1,
                                    VFLX_jp1)


    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        BFLX[i,j,k],RFLX[i,j,k] = calc_momentum_fluxes_ij(
                                    UFLX, UFLX_im1,
                                    UFLX_im1_jm1, UFLX_im1_jp1,
                                    UFLX_ip1, UFLX_ip1_jm1,
                                    UFLX_ip1_jp1, UFLX_jm1,
                                    UFLX_jp1,
                                    VFLX, VFLX_im1,
                                    VFLX_im1_jm1, VFLX_im1_jp1,
                                    VFLX_ip1, VFLX_ip1_jm1,
                                    VFLX_ip1_jp1, VFLX_jm1,
                                    VFLX_jp1)

UVFLX_vert_adv_gpu = cuda.jit(cuda_kernel_decorator(
                    launch_cuda_vert_adv_kernel))(
                    launch_cuda_vert_adv_kernel)

####################################################################


####################################################################
### SPECIALIZE FOR CPU
####################################################################
interp_WWIND_UVWIND = njit(interp_WWIND_UVWIND_py)


def launch_numba_cpu_vert_adv(WWIND_UWIND, WWIND_VWIND,
                                UWIND, VWIND, WWIND,
                                COLP_NEW, A, dsigma):
    for i in prange(nb,nxs+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(0),nz):
                WWIND_UWIND[i  ,j  ,k  ] = \
                    interp_WWIND_UVWIND(
            UWIND     [i  ,j  ,k  ], UWIND     [i  ,j  ,k-1],
            WWIND     [i  ,j  ,k  ], WWIND     [i-1,j  ,k  ],
            WWIND     [i  ,j-1,k  ], WWIND     [i  ,j+1,k  ],
            WWIND     [i-1,j-1,k  ], WWIND     [i-1,j+1,k  ], 
            COLP_NEW  [i  ,j  ,0  ], COLP_NEW  [i-1,j  ,0  ],
            COLP_NEW  [i  ,j-1,0  ], COLP_NEW  [i  ,j+1,0  ],
            COLP_NEW  [i-1,j-1,0  ], COLP_NEW  [i-1,j+1,0  ], 
            A         [i  ,j  ,0  ], A         [i-1,j  ,0  ],
            A         [i  ,j-1,0  ], A         [i  ,j+1,0  ],
            A         [i-1,j-1,0  ], A         [i-1,j+1,0  ], 
            dsigma    [0  ,0  ,k  ], dsigma    [0  ,0  ,k-1],
            True, j, ny, k)

    for i in prange(nb,nx+nb):
        for j in range(nb,nys+nb):
            for k in range(wp_int(0),nz):
                WWIND_VWIND[i  ,j  ,k  ] = \
                    interp_WWIND_UVWIND(
            VWIND     [i  ,j  ,k  ], VWIND     [i  ,j  ,k-1],
            WWIND     [i  ,j  ,k  ], WWIND     [i  ,j-1,k  ],
            WWIND     [i-1,j  ,k  ], WWIND     [i+1,j  ,k  ],
            WWIND     [i-1,j-1,k  ], WWIND     [i+1,j-1,k  ], 
            COLP_NEW  [i  ,j  ,0  ], COLP_NEW  [i  ,j-1,0  ],
            COLP_NEW  [i-1,j  ,0  ], COLP_NEW  [i+1,j  ,0  ],
            COLP_NEW  [i-1,j-1,0  ], COLP_NEW  [i+1,j-1,0  ], 
            A         [i  ,j  ,0  ], A         [i  ,j-1,0  ],
            A         [i-1,j  ,0  ], A         [i+1,j  ,0  ],
            A         [i-1,j-1,0  ], A         [i+1,j-1,0  ], 
            dsigma    [0  ,0  ,k  ], dsigma    [0  ,0  ,k-1],
            False, i, nx, k)

UVFLX_vert_adv_cpu = njit(parallel=True)(launch_numba_cpu_vert_adv)


