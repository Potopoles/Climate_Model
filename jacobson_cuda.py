from namelist import  wp_old
from boundaries_cuda import exchange_BC_gpu
from numba import cuda, jit

def proceed_timestep_jacobson_gpu(GR, stream, UWIND_OLD, UWIND, VWIND_OLD, VWIND,
                    COLP_OLD, COLP, POTT_OLD, POTT, QV_OLD, QV, QC_OLD, QC,
                    dUFLXdt, dVFLXdt, dPOTTdt, dQVdt, dQCdt, A):

    # TIME STEPPING
    time_step_3D_UWIND[GR.griddim_is, GR.blockdim, stream] \
                        (UWIND, UWIND_OLD, dUFLXdt, COLP, COLP_OLD, A, GR.dt)
    time_step_3D_VWIND[GR.griddim_js, GR.blockdim, stream] \
                        (VWIND, VWIND_OLD, dVFLXdt, COLP, COLP_OLD, A, GR.dt)
    time_step_3D[GR.griddim, GR.blockdim, stream] \
                (POTT, POTT_OLD, dPOTTdt, COLP, COLP_OLD, GR.dt)
    time_step_3D[GR.griddim, GR.blockdim, stream] \
                (QV, QV_OLD, dQVdt, COLP, COLP_OLD, GR.dt)
    time_step_3D[GR.griddim, GR.blockdim, stream] \
                (QC, QC_OLD, dQCdt, COLP, COLP_OLD, GR.dt)
    stream.synchronize()

    make_equal_or_larger[GR.griddim, GR.blockdim, stream](QV, 0.)
    make_equal_or_larger[GR.griddim, GR.blockdim, stream](QC, 0.)
    stream.synchronize()

    # TODO 4 NECESSARY
    UWIND = exchange_BC_gpu(UWIND, GR.zonal, GR.merids, \
                            GR.griddim_is, GR.blockdim, stream, stagx=True)
    VWIND = exchange_BC_gpu(VWIND, GR.zonals, GR.merid, \
                            GR.griddim_js, GR.blockdim, stream, stagy=True)
    POTT  = exchange_BC_gpu(POTT, GR.zonal, GR.merid,   \
                            GR.griddim, GR.blockdim, stream)
    QV    = exchange_BC_gpu(QV, GR.zonal, GR.merid,     \
                            GR.griddim, GR.blockdim, stream)
    QC    = exchange_BC_gpu(QC, GR.zonal, GR.merid,     \
                            GR.griddim, GR.blockdim, stream)

    return(UWIND, VWIND, COLP, POTT, QV, QC)




@jit([wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old], target='gpu')
def time_step_2D(FIELD_NEW, FIELD_OLD, dFIELDdt, dt):
    nx = FIELD_NEW.shape[0] - 2
    ny = FIELD_NEW.shape[1] - 2
    i, j = cuda.grid(2)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        FIELD_NEW[i,j] = FIELD_OLD[i,j] + dt*dFIELDdt[i,j]


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+ \
      wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old], target='gpu')
def time_step_3D(FIELD_NEW, FIELD_OLD, dFIELDdt, \
                 COLP, COLP_OLD, dt):
    nx = FIELD_NEW.shape[0] - 2
    ny = FIELD_NEW.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        FIELD_NEW[i,j,k] = FIELD_OLD[i,j,k] * COLP_OLD[i,j]/COLP[i,j] + \
                             dt*dFIELDdt[i,j,k] / COLP[i,j]


@jit([wp_old+'[:,:,:], '+wp_old],target='gpu')
def make_equal_or_larger(FIELD, number):
    i, j, k = cuda.grid(3)
    if FIELD[i,j,k] < number:
        FIELD[i,j,k] = number
    cuda.syncthreads()


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+ \
      wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old], target='gpu')
def time_step_3D_UWIND(UWIND, UWIND_OLD, dUFLXdt, \
                         COLP, COLP_OLD, A, dt):

    nx = UWIND.shape[0] - 2
    ny = UWIND.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:

        if j == 1:
            COLPA_is = 1./4.*( \
                              COLP[i-1,j  ] * A[i-1,j  ] + \
                              COLP[i  ,j  ] * A[i  ,j  ] + \
                              COLP[i-1,j+1] * A[i-1,j+1] + \
                              COLP[i  ,j+1] * A[i  ,j+1]   )
            COLPA_OLD_is = 1./4.*( \
                              COLP_OLD[i-1,j  ] * A[i-1,j  ] + \
                              COLP_OLD[i  ,j  ] * A[i  ,j  ] + \
                              COLP_OLD[i-1,j+1] * A[i-1,j+1] + \
                              COLP_OLD[i  ,j+1] * A[i  ,j+1]   )
        elif j == ny:
            COLPA_is = 1./4.*( \
                              COLP[i-1,j  ] * A[i-1,j  ] + \
                              COLP[i  ,j  ] * A[i  ,j  ] + \
                              COLP[i-1,j-1] * A[i-1,j-1] + \
                              COLP[i  ,j-1] * A[i  ,j-1]   )
            COLPA_OLD_is = 1./4.*( \
                              COLP_OLD[i-1,j  ] * A[i-1,j  ] + \
                              COLP_OLD[i  ,j  ] * A[i  ,j  ] + \
                              COLP_OLD[i-1,j-1] * A[i-1,j-1] + \
                              COLP_OLD[i  ,j-1] * A[i  ,j-1]   )
        else:
            COLPA_is = 1./8.*( \
                            COLP[i-1,j+1] * A[i-1,j+1] + \
                            COLP[i  ,j+1] * A[i  ,j+1] + \
                       2. * COLP[i-1,j  ] * A[i-1,j  ] + \
                       2. * COLP[i  ,j  ] * A[i  ,j  ] + \
                            COLP[i-1,j-1] * A[i-1,j-1] + \
                            COLP[i  ,j-1] * A[i  ,j-1]   )
            COLPA_OLD_is = 1./8.*( \
                            COLP_OLD[i-1,j+1] * A[i-1,j+1] + \
                            COLP_OLD[i  ,j+1] * A[i  ,j+1] + \
                       2. * COLP_OLD[i-1,j  ] * A[i-1,j  ] + \
                       2. * COLP_OLD[i  ,j  ] * A[i  ,j  ] + \
                            COLP_OLD[i-1,j-1] * A[i-1,j-1] + \
                            COLP_OLD[i  ,j-1] * A[i  ,j-1]   )

        UWIND[i,j,k] = UWIND_OLD[i,j,k] * COLPA_OLD_is/COLPA_is + \
                             dt*dUFLXdt[i,j,k] /COLPA_is 


@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+ \
      wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old+'[:,:  ], '+wp_old], target='gpu')
def time_step_3D_VWIND(VWIND, VWIND_OLD, dVFLXdt, \
                         COLP, COLP_OLD, A, dt):

    nx = VWIND.shape[0] - 2
    ny = VWIND.shape[1] - 2
    i, j, k = cuda.grid(3)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:

        COLPA_js = 1./8.*( \
                        COLP[i+1,j-1] * A[i+1,j-1] + \
                        COLP[i+1,j  ] * A[i+1,j  ] + \
                   2. * COLP[i  ,j-1] * A[i  ,j-1] + \
                   2. * COLP[i  ,j  ] * A[i  ,j  ] + \
                        COLP[i-1,j-1] * A[i-1,j-1] + \
                        COLP[i-1,j  ] * A[i-1,j  ]   )
        COLPA_OLD_js = 1./8.*( \
                        COLP_OLD[i+1,j-1] * A[i+1,j-1] + \
                        COLP_OLD[i+1,j  ] * A[i+1,j  ] + \
                   2. * COLP_OLD[i  ,j-1] * A[i  ,j-1] + \
                   2. * COLP_OLD[i  ,j  ] * A[i  ,j  ] + \
                        COLP_OLD[i-1,j-1] * A[i-1,j-1] + \
                        COLP_OLD[i-1,j  ] * A[i-1,j  ]   )

        VWIND[i,j,k] = VWIND_OLD[i,j,k] * COLPA_OLD_js/COLPA_js + \
                             dt*dVFLXdt[i,j,k] /COLPA_js 

