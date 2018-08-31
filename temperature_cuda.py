import numpy as np
import time
from namelist import POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics, wp
from numba import cuda, jit

if wp == 'float64':
    from numba import float64


i_vert_adv  = 0
i_hor_adv   = 1
i_num_dif   = 0

#def temperature_tendency_jacobson_gpu(POTT, POTTVB, COLP, COLP_NEW, \
#                                    UFLX, VFLX, WWIND, dPOTTdt_RAD, dPOTTdt_MIC):
@jit([wp+'[:,:,:], '+\
      wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
      wp+'[:,:]'
     ], target='gpu')
def temperature_tendency_jacobson_gpu(dPOTTdt, \
                                        POTT, UFLX, VFLX, \
                                        A):


    nx = dPOTTdt.shape[0] - 2
    ny = dPOTTdt.shape[1] - 2
    nz = dPOTTdt.shape[2]

    if i_temperature_tendency:

        i, j, k = cuda.grid(3)
        if i > 0 and i < nx+1 and j > 0 and j < ny+1:
            # HORIZONTAL ADVECTION
            if i_hor_adv:
                dPOTTdt[i,j,k] = (+ UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/2. \
                                  - UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/2. \
                                  + VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/2. \
                                  - VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/2. ) \
                                 / A[i  ,j  ]

            #if i_hor_adv:
            #    dPOTTdt[:,:,k] = (+ UFLX[:,:,k][GR.iijj    ] *\
            #                         (POTT[:,:,k][GR.iijj_im1] + POTT[:,:,k][GR.iijj    ])/2 \
            #                      - UFLX[:,:,k][GR.iijj_ip1] *\
            #                         (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_ip1])/2 \
            #                      + VFLX[:,:,k][GR.iijj    ] *\
            #                         (POTT[:,:,k][GR.iijj_jm1] + POTT[:,:,k][GR.iijj    ])/2 \
            #                      - VFLX[:,:,k][GR.iijj_jp1] *\
            #                         (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_jp1])/2 \
            #                     ) / GR.A[GR.iijj]

            ## VERTICAL ADVECTION
            #if i_vert_adv:
            #    if k == 0:
            #        vertAdv_POTT = COLP_NEW[GR.iijj] * (\
            #                - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
            #                                       ) / GR.dsigma[k]
            #    elif k == GR.nz:
            #        vertAdv_POTT = COLP_NEW[GR.iijj] * (\
            #                + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
            #                                       ) / GR.dsigma[k]
            #    else:
            #        vertAdv_POTT = COLP_NEW[GR.iijj] * (\
            #                + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
            #                - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
            #                                       ) / GR.dsigma[k]

            #    dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + vertAdv_POTT


            ## NUMERICAL DIFUSION 
            #if i_num_dif and (POTT_hor_dif_tau > 0):
            #    num_diff = POTT_hor_dif_tau * \
            #                 (+ COLP[GR.iijj_im1] * POTT[:,:,k][GR.iijj_im1] \
            #                  + COLP[GR.iijj_ip1] * POTT[:,:,k][GR.iijj_ip1] \
            #                  + COLP[GR.iijj_jm1] * POTT[:,:,k][GR.iijj_jm1] \
            #                  + COLP[GR.iijj_jp1] * POTT[:,:,k][GR.iijj_jp1] \
            #                  - 4*COLP[GR.iijj] * POTT[:,:,k][GR.iijj] )
            #    dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + num_diff

            ## RADIATION 
            #if i_radiation:
            #    dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + \
            #                        dPOTTdt_RAD[:,:,k]*COLP[GR.iijj]
            #if i_microphysics:
            #    dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + \
            #                        dPOTTdt_MIC[:,:,k]*COLP[GR.iijj]

    cuda.syncthreads()






#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
#      wp+'[:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+ \
#      wp+'[:], '+ \
#      wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
##@cuda.jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'])
#def temp_cuda_BC(dPOTTdt, UFLX, VFLX, POTT,
#                COLP_NEW, WWIND, POTTVB, 
#                dsigma,
#                zonal, merid):
#    
#    nx = dPOTTdt.shape[0] - 2
#    ny = dPOTTdt.shape[1] - 2
#    nz = dPOTTdt.shape[2]
#
#    #bottom = cuda.shared.array((nxb, nz), dtype=float64)
#
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#        dPOTTdt[i,j,k] = ( UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/2. -
#                           UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/2. -
#                           VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/2. -
#                           VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/2. )
#
#        if k == 0:
#            vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
#                    - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
#        elif k == nz:
#            vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
#                    + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ]) / dsigma[k]
#        else:
#            vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
#                    + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
#                    - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
#        dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + vertAdv_POTT
#
#    cuda.syncthreads()
#
#    if i == 0:
#        zonal[0,j,k] = dPOTTdt[nx,j,k] 
#    elif i == nx+1:
#        zonal[1,j,k] = dPOTTdt[1,j,k] 
#
#    if j == 1:
#        merid[i,0,k] = dPOTTdt[i,j,k]
#    elif j == ny+1:
#        merid[i,1,k] = dPOTTdt[i,ny,k] 
#
#
#
#    #if i == nx+1:
#    #    merid[i,j,k] = dPOTTdt[1,j,k] 
#
#    #cuda.syncthreads()
#
#    #if i == 0:
#    #    dPOTTdt[i,j,k] = dPOTTdt[nx,j,k] 
#    #    dPOTTdt[i+1,j,k] = 0.
#    #if i == nx+1:
#    #    dPOTTdt[i,j,k] = dPOTTdt[1,j,k] 
#        #dPOTTdt[i,j,k] = 0.
#
#
#    #if j == 0:
#    #    dPOTTdt[i,j,k] = dPOTTdt[i,1,k] 
#    #    #dPOTTdt[i,j,k] = 0.
#    #   dPOTTdt[i,j,k] = bottom[i,k]
#    #elif j == ny+1:
#    #    dPOTTdt[i,j,k] = dPOTTdt[i,ny,k] 
#    #    #dPOTTdt[i,j,k] = 0.
#
#    cuda.syncthreads()
#
#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
#def set_BC(dPOTTdt, zonal, merid):
#
#    nx = dPOTTdt.shape[0] - 2
#    ny = dPOTTdt.shape[1] - 2
#
#    i, j, k = cuda.grid(3)
#    # TODO set correct values
#    if i == 0:
#        dPOTTdt[i,j,k] = zonal[0,j,k] 
#    elif i == nx+1:
#        dPOTTdt[i,j,k] = zonal[1,j,k] 
#
#    if j == 1:
#        dPOTTdt[i,j,k] = merid[i,0,k]
#    elif j == ny+1:
#        dPOTTdt[i,j,k] = merid[i,1,k] 
#
#
#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
##@cuda.jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'])
#def temp_cuda_noBC(dPOTTdt, UFLX, VFLX, POTT):
#    
#    nx = dPOTTdt.shape[0] - 2
#    ny = dPOTTdt.shape[1] - 2
#
#    i, j, k = cuda.grid(3)
#    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#        dPOTTdt[i,j,k] = ( UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/2. -
#                           UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/2. -
#                           VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/2. -
#                           VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/2. )
#
#    cuda.syncthreads()




#@jit(['float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:]'], target='gpu')
#def temperature_tendency_jacobson_cuda(GR, POTT, POTTVB, COLP, COLP_NEW, \
#                                    UFLX, VFLX, WWIND, dPOTTdt_RAD, dPOTTdt_MIC):
#
#    #t_start = time.time()
#
#    dPOTTdt = np.zeros( (GR.nx ,GR.ny ,GR.nz) )
#
#    if i_temperature_tendency:
#
#        for k in range(0,GR.nz):
#
#            # HORIZONTAL ADVECTION
#            if i_hor_adv:
#                dPOTTdt[:,:,k] = (+ UFLX[:,:,k][GR.iijj    ] *\
#                                     (POTT[:,:,k][GR.iijj_im1] + POTT[:,:,k][GR.iijj    ])/2 \
#                                  - UFLX[:,:,k][GR.iijj_ip1] *\
#                                     (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_ip1])/2 \
#                                  + VFLX[:,:,k][GR.iijj    ] *\
#                                     (POTT[:,:,k][GR.iijj_jm1] + POTT[:,:,k][GR.iijj    ])/2 \
#                                  - VFLX[:,:,k][GR.iijj_jp1] *\
#                                     (POTT[:,:,k][GR.iijj    ] + POTT[:,:,k][GR.iijj_jp1])/2 \
#                                 ) / GR.A[GR.iijj]
#
#            # VERTICAL ADVECTION
#            if i_vert_adv:
#                if k == 0:
#                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
#                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
#                                                   ) / GR.dsigma[k]
#                elif k == GR.nz:
#                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
#                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
#                                                   ) / GR.dsigma[k]
#                else:
#                    vertAdv_POTT = COLP_NEW[GR.iijj] * (\
#                            + WWIND[:,:,k  ][GR.iijj] * POTTVB[:,:,k  ][GR.iijj] \
#                            - WWIND[:,:,k+1][GR.iijj] * POTTVB[:,:,k+1][GR.iijj] \
#                                                   ) / GR.dsigma[k]
#
#                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + vertAdv_POTT
#
#
#            # NUMERICAL DIFUSION 
#            if i_num_dif and (POTT_hor_dif_tau > 0):
#                num_diff = POTT_hor_dif_tau * \
#                             (+ COLP[GR.iijj_im1] * POTT[:,:,k][GR.iijj_im1] \
#                              + COLP[GR.iijj_ip1] * POTT[:,:,k][GR.iijj_ip1] \
#                              + COLP[GR.iijj_jm1] * POTT[:,:,k][GR.iijj_jm1] \
#                              + COLP[GR.iijj_jp1] * POTT[:,:,k][GR.iijj_jp1] \
#                              - 4*COLP[GR.iijj] * POTT[:,:,k][GR.iijj] )
#                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + num_diff
#
#            # RADIATION 
#            if i_radiation:
#                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + \
#                                    dPOTTdt_RAD[:,:,k]*COLP[GR.iijj]
#            if i_microphysics:
#                dPOTTdt[:,:,k] = dPOTTdt[:,:,k] + \
#                                    dPOTTdt_MIC[:,:,k]*COLP[GR.iijj]
#
#    #t_end = time.time()
#    #GR.temp_comp_time += t_end - t_start










