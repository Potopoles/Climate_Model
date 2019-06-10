#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190601
Last modified:      20190610
License:            MIT

Time step in surface scheme.
Prognose change in soil temperature SOILTEMP
Tendencies are only implemented for 1 soil layer (nz_soil = 1)
###############################################################################
"""
from numba import cuda, njit, prange
from namelist import i_radiation#, i_microphysics
from io_read_namelist import wp, wp_str, wp_int, gpu_enable
from main_grid import nx,ny,nz,nzs,nb
if gpu_enable:
    from misc_gpu_functions import cuda_kernel_decorator
from misc_meteo_utilities import calc_specific_humidity_py 
###############################################################################


###############################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
###############################################################################
def tendency_SOILTEMP_py(LWFLXNET_srfc, SWFLXNET_srfc,
                         SOILCP, SOILRHO, SOILDEPTH):

    dSOILTEMPdt = wp(0.)

    if i_radiation:
        dSOILTEMPdt += ( (LWFLXNET_srfc + SWFLXNET_srfc) /
                         (SOILCP * SOILRHO * SOILDEPTH) )

    #dSOILTEMPdt = wp(0.0001)

    #if i_microphysics > 0:
    #    dSOILTEMPdt = dSOILTEMPdt - ( MIC.surf_evap_flx * MIC.lh_cond_water ) / \
    #                                (CF.SOILCP * CF.SOILRHO * CF.SOILDEPTH)
    return(dSOILTEMPdt)


def timestep_SOILTEMP_py(SOILTEMP, dSOILTEMPdt, dt):
    return(SOILTEMP + dt * dSOILTEMPdt)

def calc_albedo_py(OCEANMASK, SOILTEMP):
    # ocean
    if OCEANMASK:
        SURFALBEDSW = wp(0.05)
        SURFALBEDLW = wp(0.00)
    # land
    else:
        SURFALBEDSW = wp(0.2)
        SURFALBEDLW = wp(0.0)
    # ice (land and sea)
    if SOILTEMP <= wp(273.15):
        SURFALBEDSW = wp(0.6)
        SURFALBEDLW = wp(0.0)
    return(SURFALBEDSW, SURFALBEDLW)


def calc_srfc_fluxes_py(SOILTEMP, TAIR_nz, QV_nz, WIND_nz, PSURF, DRAGCH):
    # surface sensible heat flux
    SSHFLX = - DRAGCH * WIND_nz * ( TAIR_nz - SOILTEMP )
    # surface latent heat flux
    # for QV of soil assume saturation specific humidity for SOILTEMP
    SOILQV = calc_specific_humidity(SOILTEMP, wp(100.), PSURF)
    SQVFLX = - DRAGCH * WIND_nz * ( QV_nz - SOILQV )
    return(SSHFLX, SQVFLX)


def run_full_timestep_py(SOILTEMP, LWFLXNET_srfc, SWFLXNET_srfc,
                         SOILCP, SOILRHO, SOILDEPTH, OCEANMASK,
                         TAIR_nz, QV_nz, WIND_nz, PSURF, DRAGCH, dt):

    # comute surface fluxes
    SSHFLX, SQVFLX = calc_srfc_fluxes(SOILTEMP, TAIR_nz, QV_nz,
                                      WIND_nz, PSURF, DRAGCH)
    # soil temperature change
    dSOILTEMPdt     = tendency_SOILTEMP(LWFLXNET_srfc, SWFLXNET_srfc,
                                        SOILCP, SOILRHO, SOILDEPTH)
    SOILTEMP        = timestep_SOILTEMP(SOILTEMP, dSOILTEMPdt, dt)
    # update surface albedo
    SURFALBEDSW, SURFALBEDLW = calc_albedo(OCEANMASK, SOILTEMP)
    
    return(SOILTEMP, SURFALBEDSW, SURFALBEDLW, SSHFLX, SQVFLX)





#@jit([wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:,:]  '], target='gpu')
#def calc_evaporation_capacity_gpu(SOILEVAPITY, SOILMOIST, OCEANMASK, SOILTEMP):
#    i, j = cuda.grid(2)
#    # calc evaporation capacity
#    if OCEANMASK[i,j] == 0:
#        SOILEVAPITY[i,j] = min(max(0., SOILMOIST[i,j] / evapity_thresh), 1.)



###############################################################################
### SPECIALIZE FOR GPU
###############################################################################
tendency_SOILTEMP = njit(tendency_SOILTEMP_py, device=True, inline=True)
timestep_SOILTEMP = njit(timestep_SOILTEMP_py, device=True, inline=True)
calc_albedo       = njit(calc_albedo_py, device=True, inline=True)
calc_specific_humidity = njit(calc_specific_humidity_py,
                            device=True, inline=True)
calc_srfc_fluxes  = njit(calc_srfc_fluxes_py, device=True, inline=True)
run_full_timestep = njit(run_full_timestep_py, device=True, inline=True)

def launch_cuda_kernel(SOILTEMP, LWFLXNET, SWFLXNET, SOILCP,
                       SOILRHO, SOILDEPTH, OCEANMASK,
                       SURFALBEDSW, SURFALBEDLW,
                       TAIR, QV, WIND, PSURF, SSHFLX, SQVFLX,
                       DRAGCH, dt):

    i, j = cuda.grid(2)
    if i < nx and j < ny:
        ( SOILTEMP[i,j,0], SURFALBEDSW[i,j,0], SURFALBEDLW[i,j,0],
          SSHFLX[i+nb,j+nb,0], SQVFLX[i+nb,j+nb,0]) = run_full_timestep(
                        SOILTEMP[i,j,0],
                        LWFLXNET[i,j,nzs-1], SWFLXNET[i,j,nzs-1],
                        SOILCP[i,j,0], SOILRHO[i,j,0],
                        SOILDEPTH[i,j,0], OCEANMASK[i,j,0],
                        TAIR[i+nb,j+nb,nz-1], QV[i+nb,j+nb,nz-1],
                        WIND[i+nb,j+nb,nz-1], PSURF[i+nb,j+nb,0],
                        DRAGCH, dt)

if gpu_enable:
    advance_timestep_srfc_gpu = cuda.jit(cuda_kernel_decorator(
                    launch_cuda_kernel,
                    non_3D={'dt':wp_str,'OCEANMASK':'int32[:,:,:]',
                            'DRAGCH':wp_str}))(
                    launch_cuda_kernel)


###############################################################################
### SPECIALIZE FOR CPU
###############################################################################
tendency_SOILTEMP = njit(tendency_SOILTEMP_py)
timestep_SOILTEMP = njit(timestep_SOILTEMP_py)
calc_albedo       = njit(calc_albedo_py)
run_full_timestep = njit(run_full_timestep_py)
def launch_numba_cpu(SOILTEMP, LWFLXNET, SWFLXNET, SOILCP,
                   SOILRHO, SOILDEPTH, OCEANMASK,
                   SURFALBEDSW, SURFALBEDLW ,dt):

    for i in prange(nx):
        for j in range(ny):
            SOILTEMP[i,j,0], SURFALBEDSW[i,j,0], SURFALBEDLW[i,j,0] =\
                     run_full_timestep(SOILTEMP[i,j,0],
                          LWFLXNET[i,j,nzs-1], SWFLXNET[i,j,nzs-1],
                          SOILCP[i,j,0], SOILRHO[i,j,0],
                          SOILDEPTH[i,j,0], OCEANMASK[i,j,0], dt) 

advance_timestep_srfc_cpu = njit(parallel=True)(launch_numba_cpu) 
