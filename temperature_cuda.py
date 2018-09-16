import numpy as np
import time
from namelist import POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics, wp
from numba import cuda, jit

i_vert_adv  = 1
i_hor_adv   = 1
i_num_dif   = 1

@jit([wp+'[:,:,:], '+\
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:  ], '+\
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
      wp+'[:,:,:], '+wp+'[:,:,:], '+\
      wp+'[:,:  ], '+wp+'[:    ]  '
     ], target='gpu')
def temperature_tendency_jacobson_gpu(dPOTTdt, \
                                        POTT, POTTVB, COLP, COLP_NEW, \
                                        UFLX, VFLX, WWIND, \
                                        dPOTTdt_RAD, dPOTTdt_MIC, \
                                        A, dsigma):


    nx = dPOTTdt.shape[0] - 2
    ny = dPOTTdt.shape[1] - 2
    nz = dPOTTdt.shape[2]

    if i_temperature_tendency:

        i, j, k = cuda.grid(3)
        if i > 0 and i < nx+1 and j > 0 and j < ny+1:
            # HORIZONTAL ADVECTION
            if i_hor_adv:
                dPOTTdt[i,j,k] = (+ UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/2.\
                                  - UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/2.\
                                  + VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/2.\
                                  - VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/2.)\
                                  / A[i  ,j  ]

            # VERTICAL ADVECTION
            if i_vert_adv:
                if k == 0:
                    vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
                elif k == nz:
                    vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ]) / dsigma[k]
                else:
                    vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + vertAdv_POTT


            # NUMERICAL DIFUSION 
            if i_num_dif and (POTT_hor_dif_tau > 0.):
                num_dif = POTT_hor_dif_tau * \
                             (+ COLP[i-1,j  ] * POTT[i-1,j  ,k  ] \
                              + COLP[i+1,j  ] * POTT[i+1,j  ,k  ] \
                              + COLP[i  ,j-1] * POTT[i  ,j-1,k  ] \
                              + COLP[i  ,j+1] * POTT[i  ,j+1,k  ] \
                         - 4. * COLP[i  ,j  ] * POTT[i  ,j  ,k  ] )
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + num_dif

            # RADIATION 
            if i_radiation:
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
                                    dPOTTdt_RAD[i-1,j-1,k]*COLP[i,j] # TODO add boundaries
            # MICROPHYSICS
            if i_microphysics:
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
                                    dPOTTdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries

    cuda.syncthreads()


