from namelist import QV_hor_dif_tau, wp

from numba import cuda, jit
if wp == 'float64':
    from numba import float64

i_hor_adv      = 1
i_vert_adv     = 1
i_num_dif      = 1
i_microphysics = 1
i_turb         = 0



@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:  ], '+\
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
      wp+'[:,:  ], '+wp+'[:    ]'], target='gpu')
def water_vapor_tendency_gpu(dQVdt, QV, COLP, COLP_NEW, \
                            UFLX, VFLX, WWIND, dQVdt_MIC, \
                            A, dsigma):

    nx = dQVdt.shape[0] - 2
    ny = dQVdt.shape[1] - 2
    nz = dQVdt.shape[2]

    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        dQVdt[i,j,k] = 0. 

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            dQVdt[i,j,k] = (  + UFLX[i  ,j  ,k] * (QV[i-1,j  ,k] + QV[i  ,j  ,k])/2.\
                              - UFLX[i+1,j  ,k] * (QV[i  ,j  ,k] + QV[i+1,j  ,k])/2.\
                              + VFLX[i  ,j  ,k] * (QV[i  ,j-1,k] + QV[i  ,j  ,k])/2.\
                              - VFLX[i  ,j+1,k] * (QV[i  ,j  ,k] + QV[i  ,j+1,k])/2.)\
                              / A[i  ,j  ]

        # VERTICAL ADVECTION
        if i_vert_adv:
            if k == 0:
                QVkp12 = (QV[i  ,j  ,k  ] + QV[i  ,j  ,k+1]) / 2.
                vert_adv = COLP_NEW[i  ,j  ] * (\
                        - WWIND[i  ,j  ,k+1] * QVkp12 \
                                               ) / dsigma[k]
            elif k == nz:
                QVkm12 = (QV[i  ,j  ,k-1] + QV[i  ,j  ,k  ]) / 2.
                vert_adv = COLP_NEW[i  ,j  ] * (\
                        + WWIND[i  ,j  ,k  ] * QVkm12 \
                                               ) / dsigma[k]
            else:
                QVkm12 = (QV[i  ,j  ,k-1] + QV[i  ,j  ,k  ]) / 2.
                QVkp12 = (QV[i  ,j  ,k  ] + QV[i  ,j  ,k+1]) / 2.
                vert_adv = COLP_NEW[i  ,j  ] * (\
                        + WWIND[i  ,j  ,k  ] * QVkm12 \
                        - WWIND[i  ,j  ,k+1] * QVkp12 \
                                               ) / dsigma[k]
            dQVdt[i,j,k] = dQVdt[i,j,k] + vert_adv


        # NUMERICAL DIFUSION 
        if i_num_dif and (QV_hor_dif_tau > 0.):
            num_dif = QV_hor_dif_tau * \
                         (+ COLP[i-1,j  ] * QV[i-1,j  ,k  ] \
                          + COLP[i+1,j  ] * QV[i+1,j  ,k  ] \
                          + COLP[i  ,j-1] * QV[i  ,j-1,k  ] \
                          + COLP[i  ,j+1] * QV[i  ,j+1,k  ] \
                     - 4. * COLP[i  ,j  ] * QV[i  ,j  ,k  ] )
            dQVdt[i,j,k] = dQVdt[i,j,k] + num_dif

        # MICROPHYSICS
        if i_microphysics:
            dQVdt[i,j,k] = dQVdt[i,j,k] + \
                                dQVdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries

    cuda.syncthreads()




@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:  ], '+\
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
      wp+'[:,:  ], '+wp+'[:    ]'], target='gpu')
def cloud_water_tendency_gpu(dQCdt, QC, COLP, COLP_NEW, \
                            UFLX, VFLX, WWIND, dQCdt_MIC, \
                            A, dsigma):

    nx = dQCdt.shape[0] - 2
    ny = dQCdt.shape[1] - 2
    nz = dQCdt.shape[2]

    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        dQCdt[i,j,k] = 0. 

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            dQCdt[i,j,k] = (  + UFLX[i  ,j  ,k] * (QC[i-1,j  ,k] + QC[i  ,j  ,k])/2.\
                              - UFLX[i+1,j  ,k] * (QC[i  ,j  ,k] + QC[i+1,j  ,k])/2.\
                              + VFLX[i  ,j  ,k] * (QC[i  ,j-1,k] + QC[i  ,j  ,k])/2.\
                              - VFLX[i  ,j+1,k] * (QC[i  ,j  ,k] + QC[i  ,j+1,k])/2.)\
                              / A[i  ,j  ]

        # VERTICAL ADVECTION
        if i_vert_adv:
            if k == 0:
                QCkp12 = (QC[i  ,j  ,k  ] + QC[i  ,j  ,k+1]) / 2.
                vert_adv = COLP_NEW[i  ,j  ] * (\
                        - WWIND[i  ,j  ,k+1] * QCkp12 \
                                               ) / dsigma[k]
            elif k == nz:
                QCkm12 = (QC[i  ,j  ,k-1] + QC[i  ,j  ,k  ]) / 2.
                vert_adv = COLP_NEW[i  ,j  ] * (\
                        + WWIND[i  ,j  ,k  ] * QCkm12 \
                                               ) / dsigma[k]
            else:
                QCkm12 = (QC[i  ,j  ,k-1] + QC[i  ,j  ,k  ]) / 2.
                QCkp12 = (QC[i  ,j  ,k  ] + QC[i  ,j  ,k+1]) / 2.
                vert_adv = COLP_NEW[i  ,j  ] * (\
                        + WWIND[i  ,j  ,k  ] * QCkm12 \
                        - WWIND[i  ,j  ,k+1] * QCkp12 \
                                               ) / dsigma[k]
            dQCdt[i,j,k] = dQCdt[i,j,k] + vert_adv


        # NUMERICAL DIFUSION 
        if i_num_dif and (QV_hor_dif_tau > 0.):
            num_dif = QV_hor_dif_tau * \
                         (+ COLP[i-1,j  ] * QC[i-1,j  ,k  ] \
                          + COLP[i+1,j  ] * QC[i+1,j  ,k  ] \
                          + COLP[i  ,j-1] * QC[i  ,j-1,k  ] \
                          + COLP[i  ,j+1] * QC[i  ,j+1,k  ] \
                     - 4. * COLP[i  ,j  ] * QC[i  ,j  ,k  ] )
            dQCdt[i,j,k] = dQCdt[i,j,k] + num_dif

        # MICROPHYSICS
        if i_microphysics:
            dQCdt[i,j,k] = dQCdt[i,j,k] + \
                                dQCdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries

    cuda.syncthreads()

