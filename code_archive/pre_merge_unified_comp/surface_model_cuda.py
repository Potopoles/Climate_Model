import numpy as np
from namelist import i_radiation, i_microphysics
from org_namelist import wp_old
from numba import cuda, jit
#from surface_model import evapity_thresh 

evapity_thresh = 10.


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
      wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+\
      wp_old], target='gpu')
def soil_temperature_euler_forward_gpu(dSOILTEMPdt, SOILTEMP, LWFLXNET, SWFLXNET,
                                    SOILCP, SOILRHO, SOILDEPTH, dt):

    nx = SOILTEMP.shape[0]
    ny = SOILTEMP.shape[1]
    nzs = LWFLXNET.shape[2]

    i, j = cuda.grid(2)

    dSOILTEMPdt[i,j,0] = 0.

    if i_radiation > 0:
        dSOILTEMPdt[i,j,0] = (LWFLXNET[i,j,nzs-1] + SWFLXNET[i,j,nzs-1])/ \
                        (SOILCP[i,j] * SOILRHO[i,j] * SOILDEPTH[i,j])

    #if i_microphysics > 0:
    #    dSOILTEMPdt = dSOILTEMPdt - ( MIC.surf_evap_flx * MIC.lh_cond_water ) / \
    #                                (CF.SOILCP * CF.SOILRHO * CF.SOILDEPTH)

    SOILTEMP[i,j,0] = SOILTEMP[i,j,0] + dt * dSOILTEMPdt[i,j,0]


    cuda.syncthreads()


@jit([wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:,:]  '], target='gpu')
def calc_albedo_gpu(SURFALBEDSW, SURFALBEDLW, OCEANMASK, SOILTEMP):

    i, j = cuda.grid(2)

    # ocean
    if OCEANMASK[i,j] == 1:
        SURFALBEDSW[i,j] = 0.05
        #SURFALBEDLW[i,j] = 0.05
        SURFALBEDLW[i,j] = 0.00
    # land
    else:
        SURFALBEDSW[i,j] = 0.2
        #SURFALBEDLW[i,j] = 0.2
        SURFALBEDLW[i,j] = 0.0

    # ice (land and sea)
    if SOILTEMP[i,j,0] <= 273.15:
        SURFALBEDSW[i,j] = 0.5
        #SURFALBEDLW[i,j] = 0.3
        SURFALBEDLW[i,j] = 0.0

    cuda.syncthreads()


@jit([wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:,:]  '], target='gpu')
def calc_evaporation_capacity_gpu(SOILEVAPITY, SOILMOIST, OCEANMASK, SOILTEMP):
    i, j = cuda.grid(2)
    # calc evaporation capacity
    if OCEANMASK[i,j] == 0:
        SOILEVAPITY[i,j] = min(max(0., SOILMOIST[i,j] / evapity_thresh), 1.)
