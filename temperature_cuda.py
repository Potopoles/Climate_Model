import numpy as np
import cupy as cp
import time
from namelist import POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics, wp
from numba import cuda, njit, prange, jit, vectorize
from math import exp

i_vert_adv  = 1
i_hor_adv   = 1
i_num_dif   = 1

#@jit([wp+'[:,:,:], '+\
#  wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:  ], '+wp+'[:,:  ], '+\
#  wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:], '+\
#  wp+'[:,:,:], '+wp+'[:,:,:], '+\
#  wp+'[:,:  ], '+wp+'[:    ]  '
# ], target='gpu')
#def calc_dPOTTdt(dPOTTdt,
#            POTT, POTTVB, COLP, COLP_NEW,
#            UFLX, VFLX, WWIND,
#            dPOTTdt_RAD, dPOTTdt_MIC,
#            A, dsigma):
#
#
#    nx = dPOTTdt.shape[0] - 2
#    ny = dPOTTdt.shape[1] - 2
#    nz = dPOTTdt.shape[2]
#
#    if i_temperature_tendency:
#
#        i, j, k = cuda.grid(3)
#        if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#            # HORIZONTAL ADVECTION
#            if i_hor_adv:
#                dPOTTdt[i,j,k] = \
#                    (+ UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/2.\
#                  - UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/2.\
#                  + VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/2.\
#                  - VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/2.)\
#                  / A[i  ,j  ]
#
#            # VERTICAL ADVECTION
#            if i_vert_adv:
#                if k == 0:
#                    vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
#                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
#                elif k == nz:
#                    vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
#                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ]) / dsigma[k]
#                else:
#                    vertAdv_POTT = COLP_NEW[i  ,j  ] * (\
#                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
#                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
#                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + vertAdv_POTT
#
#
#            # NUMERICAL DIFUSION 
#            if i_num_dif and (POTT_hor_dif_tau > 0.):
#                num_dif = POTT_hor_dif_tau *  exp(-(float(nz-k-1))) *\
#                            ( + COLP[i-1,j  ] * POTT[i-1,j  ,k  ] \
#                              + COLP[i+1,j  ] * POTT[i+1,j  ,k  ] \
#                              + COLP[i  ,j-1] * POTT[i  ,j-1,k  ] \
#                              + COLP[i  ,j+1] * POTT[i  ,j+1,k  ] \
#                         - 4. * COLP[i  ,j  ] * POTT[i  ,j  ,k  ] )
#                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + num_dif
#
#            # RADIATION 
#            if i_radiation:
#                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                                    dPOTTdt_RAD[i-1,j-1,k]*COLP[i,j] # TODO add boundaries
#            # MICROPHYSICS
#            if i_microphysics:
#                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                                    dPOTTdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries
#
#    cuda.syncthreads()



def run(comp_mode):

    def hor_adv(VAR, VAR_im1, VAR_ip1, VAR_jm1, VAR_jp1,
                UFLX, UFLX_ip1, VFLX, VFLX_jp1, A):
        out = (+ UFLX * (VAR_im1 + VAR)/2.
          - UFLX_ip1 * (VAR + VAR_ip1)/2.
          + VFLX * (VAR_jm1 + VAR)/2.\
          - VFLX_jp1 * (VAR + VAR_jp1)/2.)/A
        return(out)


    if comp_mode == 2:
        import cupy as cp
    else:
        import numpy as cp

    nx = 720
    ny = 360
    nb = 1
    nz = 32

    nxs = nx + 1
    nys = ny + 1

    tpb  = (1,       1,      nz)
    bpg  = (int(nx/tpb[0]),   int(ny/tpb[1]),  1)


    COLP        = cp.ones( ( nx +2*nb, ny +2*nb,   1      ), 
                                dtype=NL.wp)
    A = cp.ones_like(COLP)
    POTT        = cp.ones( ( nx +2*nb, ny +2*nb, nz  ), 
                        dtype=NL.wp)
    dPOTTdt = cp.ones_like(POTT)
    UFLX        = cp.ones( ( nxs +2*nb, ny +2*nb, nz  ), 
                        dtype=NL.wp)
    VFLX        = cp.ones( ( nx +2*nb, nys +2*nb, nz  ), 
                        dtype=NL.wp)


    if comp_mode == 2:
        print('gpu')
        hor_adv = njit(hor_adv, device=True)


        @cuda.jit()
        def calc_dPOTTdt_cuda(dPOTTdt, POTT, UFLX, VFLX, A):

            nx = dPOTTdt.shape[0] - 2
            ny = dPOTTdt.shape[1] - 2
            nz = dPOTTdt.shape[2]

            if i_temperature_tendency:

                i, j, k = cuda.grid(3)
                
                if i > 0 and i < nx+1 and j > 0 and j < ny+1:
                    # HORIZONTAL ADVECTION
                    if i_hor_adv:
                        dPOTTdt[i,j,k] = hor_adv(
                            POTT[i,j,k],
                            POTT[i-1,j,k], POTT[i+1,j,k],
                            POTT[i,j-1,k], POTT[i,j+1,k],
                            UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
                            VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
                            A[i,j,0])


        calc_dPOTTdt_cuda[bpg, tpb](dPOTTdt, POTT, UFLX, VFLX, A)
        cuda.synchronize()
        t0 = time.time()
        calc_dPOTTdt_cuda[bpg, tpb](dPOTTdt, POTT, UFLX, VFLX, A)
        cuda.synchronize()
        print(time.time() - t0)

        print(np.mean(cp.asnumpy(dPOTTdt)))
        print()

    elif comp_mode == 1:
        print('numba_par')
        hor_adv = njit(hor_adv)

        @njit(parallel=True)
        def calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, A, nx, ny, nz, nb):

            for i in prange(nb,nx+nb):
                for j in range(nb,ny+nb):
                    for k in range(0,nz):
                        if i_temperature_tendency:

                            # HORIZONTAL ADVECTION
                            if i_hor_adv:
                                dPOTTdt[i,j,k] = hor_adv(
                                    POTT[i,j,k],
                                    POTT[i-1,j,k], POTT[i+1,j,k],
                                    POTT[i,j-1,k], POTT[i,j+1,k],
                                    UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
                                    VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
                                    A[i,j,0])


        calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, A, nx, ny, nz, nb)
        t0 = time.time()
        calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, A, nx, ny, nz, nb)
        print(time.time() - t0)
        print(np.mean(dPOTTdt))
        print()



    elif comp_mode == 0:
        print('python')

        def calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, A, nx, ny, nz, nb):
            
            i = slice(nb,nx+nb)
            ip1 = slice(nb+1,nx+nb+1)
            im1 = slice(nb-1,nx+nb-1)

            j = slice(nb,ny+nb)
            jp1 = slice(nb+1,ny+nb+1)
            jm1 = slice(nb-1,ny+nb-1)

            k = slice(0,nz)
            #k0 = slice(0,1)

            if i_temperature_tendency:
                # HORIZONTAL ADVECTION
                if i_hor_adv:
                    dPOTTdt[i,j,k] = hor_adv(
                        POTT[i,j,k],
                        POTT[im1,j,k], POTT[ip1,j,k],
                        POTT[i,jm1,k], POTT[i,jp1,k],
                        UFLX[i  ,j  ,k], UFLX[ip1,j  ,k],
                        VFLX[i  ,j  ,k], VFLX[i  ,jp1,k],
                        A[i,j])


        calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, A, nx, ny, nz, nb)
        t0 = time.time()
        calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, A, nx, ny, nz, nb)
        print(time.time() - t0)
        print(np.mean(dPOTTdt))
        print()






if __name__ == '__main__':
    import debug_namelist as NL

    
    run(2)
    run(1)
    run(0)




