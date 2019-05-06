import numpy as np
import cupy as cp
import time
from namelist import POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics
from grid import nx,nxs,ny,nys,nz,nzs,nb
from numba import cuda, njit, prange, jit, vectorize

i_hor_adv   = 1
i_vert_adv  = 1
i_num_dif   = 1




import debug_namelist as NL
POTT_hor_dif_tau = 0.1

@jit([NL.wp_old+'[:,:,:], '+\
  NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:], '+\
  NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:], '+\
  #NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:], '+\
  NL.wp_old+'[:,:,:], '+NL.wp_old+'[:,:,:]  '
 ], target='gpu')
def calc_dPOTTdt_orig(dPOTTdt,
            POTT, POTTVB, COLP, COLP_NEW,
            UFLX, VFLX, WWIND,
            #dPOTTdt_RAD, dPOTTdt_MIC,
            A, dsigma):


    #nx = dPOTTdt.shape[0] - 2
    #ny = dPOTTdt.shape[1] - 2
    #nz = dPOTTdt.shape[2]

    if i_temperature_tendency:

        i, j, k = cuda.grid(3)
        if i > 0 and i < nx+1 and j > 0 and j < ny+1:

            # HORIZONTAL ADVECTION
            if i_hor_adv:
                dPOTTdt[i,j,k] = \
                    (+ UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/NL.wp(2.)\
                  - UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/NL.wp(2.)\
                  + VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/NL.wp(2.)\
                  - VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/NL.wp(2.))\
                  / A[i  ,j  ,0]

            # VERTICAL ADVECTION
            if i_vert_adv:
                if k == NL.wp_int(0):
                    vertAdv_POTT = COLP_NEW[i  ,j  ,0] * (\
                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[0, 0, k]
                elif k == nz:
                    vertAdv_POTT = COLP_NEW[i  ,j  ,0] * (\
                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ]) / dsigma[0, 0, k]
                else:
                    vertAdv_POTT = COLP_NEW[i  ,j  ,0] * (\
                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[0, 0, k]
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + vertAdv_POTT


            # NUMERICAL DIFUSION 
            #                #exp(-(float(nz-k-1))) *\
            if i_num_dif and (POTT_hor_dif_tau > 0.):
                #num_dif = POTT_hor_dif_tau*\
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
                            ( + COLP[i-1,j  ,0] * POTT[i-1,j  ,k  ] \
                              + COLP[i+1,j  ,0] * POTT[i+1,j  ,k  ] \
                              + COLP[i  ,j-1,0] * POTT[i  ,j-1,k  ] \
                              + COLP[i  ,j+1,0] * POTT[i  ,j+1,k  ] \
                         - NL.wp(4.) * COLP[i  ,j  ,0] * POTT[i  ,j  ,k  ] )
                #dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + num_dif

            ## RADIATION 
            #if i_radiation:
            #    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
            #                        dPOTTdt_RAD[i-1,j-1,k]*COLP[i,j] # TODO add boundaries
            ## MICROPHYSICS
            #if i_microphysics:
            #    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
            #                        dPOTTdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries

    cuda.syncthreads()



def run(comp_mode):

    from tendencies import  hor_adv, vert_adv, num_dif
    
    def POTT_calc(POTT, POTT_im1, POTT_ip1, POTT_jm1, POTT_jp1,
                UFLX, UFLX_ip1, VFLX, VFLX_jp1,
                COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1, A,
                POTTVB, POTTVB_kp1, WWIND, WWIND_kp1,
                COLP_NEW, dsigma, k):

        dPOTTdt = NL.wp(0.)

        if i_temperature_tendency:
            # HORIZONTAL ADVECTION
            if i_hor_adv:
                dPOTTdt = dPOTTdt + hor_adv(
                    POTT,
                    POTT_im1, POTT_ip1,
                    POTT_jm1, POTT_jp1,
                    UFLX, UFLX_ip1,
                    VFLX, VFLX_jp1,
                    A)
            # VERTICAL ADVECTION
            if i_vert_adv:
                dPOTTdt = dPOTTdt + vert_adv(
                    POTTVB, POTTVB_kp1,
                    WWIND, WWIND_kp1,
                    COLP_NEW, dsigma, k)
            # NUMERICAL HORIZONTAL DIFUSION
            if i_num_dif and (POTT_DIF_COEF > 0.):
                dPOTTdt = dPOTTdt + num_dif(
                    POTT, POTT_im1, POTT_ip1,
                    POTT_jm1, POTT_jp1,
                    COLP, COLP_im1, COLP_ip1,
                    COLP_jm1, COLP_jp1,
                    POTT_DIF_COEF)

        return(dPOTTdt)


    if comp_mode in [2,0]:
        import cupy as cp
    else:
        import numpy as cp


    i = slice(nb,nx+nb)
    j = slice(nb,ny+nb)

    tpb  = (1,       1,      nz)
    bpg  = (int(nx/tpb[0])+1,   int(ny/tpb[1])+1,  1)

    COLP        = cp.full( ( nx +2*nb, ny +2*nb,   1      ), 
                             np.nan,   dtype=NL.wp)
    A = cp.full_like(COLP, np.nan)
    COLP_NEW = cp.full_like(COLP, np.nan)
    POTT        = cp.full( ( nx +2*nb, ny +2*nb, nz  ), 
                        np.nan, dtype=NL.wp)
    dPOTTdt = cp.full_like(POTT, np.nan)
    UFLX        = cp.full( ( nxs +2*nb, ny +2*nb, nz  ), 
                        np.nan, dtype=NL.wp)
    VFLX        = cp.full( ( nx +2*nb, nys +2*nb, nz  ), 
                        np.nan, dtype=NL.wp)

    POTTVB        = cp.full( ( nx +2*nb, ny +2*nb, nzs  ), 
                        np.nan, dtype=NL.wp)
    WWIND = cp.full_like(POTTVB, np.nan)


    dsigma        = cp.full( ( 1, 1,   nz      ), 
                             np.nan,   dtype=NL.wp)


    A[:] = 1.
    COLP[:] = 1.
    COLP_NEW[:] = 2.
    UFLX[:] = 3.
    for k in range(0,nz):
        for i in range(1,100):
            UFLX[i,:,k] += i/(k+1)
    VFLX[:] = 3.
    POTT[:] = 2.
    for k in range(0,nz):
        for i in range(100,200):
            POTT[i,:,k] += i/(k+1)

    POTTVB[:] = 2.
    WWIND[:] = 1.
    for k in range(0,nz):
        for i in range(300,200):
            POTTVB[i,:,k] += i/(k+1)
            WWIND[i,:,k] += 20*i/(k+1) - k
    dsigma[:] = 7.


    POTT_DIF_COEF = NL.wp(1E-1)


    if comp_mode == 2:
        print('gpu')
        hor_adv = njit(hor_adv, device=True, inline=True)
        num_dif = njit(num_dif, device=True, inline=True)
        vert_adv = njit(vert_adv, device=True, inline=True)
        POTT_calc = njit(POTT_calc, device=True, inline=True)


        @cuda.jit('void('+NL.wp_str+','+NL.wp_str+','+NL.wp_str+
                        ','+NL.wp_str+','+NL.wp_str+','+NL.wp_str+
                        ','+NL.wp_str+','+NL.wp_str+','+NL.wp_str+
                        ','+NL.wp_str+')')
        def calc_dPOTTdt_cuda(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                             POTTVB, WWIND, COLP_NEW, dsigma):

            i, j, k = cuda.grid(3)
            
            if i >= nb and i < nx+nb and j >= nb and j < ny+nb:

                dPOTTdt[i  ,j  ,k] = \
                    POTT_calc(POTT[i  ,j  ,k],
                    POTT[i-1,j  ,k], POTT[i+1,j  ,k],
                    POTT[i  ,j-1,k], POTT[i  ,j+1,k],
                    UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
                    VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
                    COLP[i  ,j  ,0],
                    COLP[i-1,j  ,0], COLP[i+1,j  ,0],
                    COLP[i  ,j-1,0], COLP[i  ,j+1,0], A[i  ,j  ,0],
                    POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
                    WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
                    COLP_NEW[i  ,j  ,0], dsigma[0  ,0  ,k], k)

                

        calc_dPOTTdt_cuda[bpg, tpb](dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                                    POTTVB, WWIND, COLP_NEW, dsigma)
        cuda.synchronize()
        t0 = time.time()
        calc_dPOTTdt_cuda[bpg, tpb](dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                                    POTTVB, WWIND, COLP_NEW, dsigma)
        cuda.synchronize()
        print(time.time() - t0)

        print(np.mean(cp.asnumpy(dPOTTdt[i,j,:])))
        #print(cp.asnumpy(dPOTTdt))
        print()

    elif comp_mode == 1:
        print('numba_par')
        hor_adv = njit(hor_adv)
        vert_adv = njit(vert_adv)
        num_dif = njit(num_dif)
        POTT_calc = njit(POTT_calc)

        @njit(parallel=True)
        def calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                             POTTVB, WWIND, COLP_NEW, dsigma):

            for i in prange(nb,nx+nb):
                for j in range(nb,ny+nb):
                    for k in range(NL.wp_int(0),nz):

                        dPOTTdt[i  ,j  ,k] = \
                            POTT_calc(POTT[i  ,j  ,k],
                            POTT[i-1,j  ,k], POTT[i+1,j  ,k],
                            POTT[i  ,j-1,k], POTT[i  ,j+1,k],
                            UFLX[i  ,j  ,k], UFLX[i+1,j  ,k],
                            VFLX[i  ,j  ,k], VFLX[i  ,j+1,k],
                            COLP[i  ,j  ,0],
                            COLP[i-1,j  ,0], COLP[i+1,j  ,0],
                            COLP[i  ,j-1,0], COLP[i  ,j+1,0], A[i  ,j  ,0],
                            POTTVB[i  ,j  ,k], POTTVB[i  ,j  ,k+1],
                            WWIND[i  ,j  ,k], WWIND[i  ,j  ,k+1],
                            COLP_NEW[i  ,j  ,0], dsigma[0  ,0  ,k], k)


        calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                             POTTVB, WWIND, COLP_NEW, dsigma)
        t0 = time.time()
        calc_dPOTTdt(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                             POTTVB, WWIND, COLP_NEW, dsigma)
        print(time.time() - t0)
        print(np.mean(dPOTTdt[i,j,:]))
        print()




    elif comp_mode == 0:
        print('gpu_old')


        calc_dPOTTdt_orig[bpg, tpb](dPOTTdt,
            POTT, POTTVB, COLP, COLP_NEW,
            UFLX, VFLX, WWIND,
            A, dsigma)
        cuda.synchronize()
        t0 = time.time()
        calc_dPOTTdt_orig[bpg, tpb](dPOTTdt,
            POTT, POTTVB, COLP, COLP_NEW,
            UFLX, VFLX, WWIND,
            A, dsigma)
        cuda.synchronize()
        print(time.time() - t0)
        print(np.mean(cp.asnumpy(dPOTTdt[i,j,:])))
        print()



if __name__ == '__main__':
    import debug_namelist as NL

    
    run(2)
    run(1)
    run(0)




