####################################################################

####################################################################
import numpy as np
import cupy as cp
import time
from namelist import (POTT_dif_coef, \
                    i_POTT_main_switch,
                    i_POTT_radiation, i_POTT_microphys,
                    i_POTT_hor_adv, i_POTT_vert_adv, i_POTT_num_dif)
from namelist import (wp, wp_int, wp_old)
from grid import nx,nxs,ny,nys,nz,nzs,nb
from numba import cuda, njit, prange, vectorize
from GPU import cuda_kernel_decorator

from tendency_functions import  hor_adv_py, vert_adv_py, num_dif_py
####################################################################


####################################################################
### DEVICE UNSPECIFIC PYTHON FUNCTIONS
####################################################################
def add_up_tendencies_py(
            POTT, POTT_im1, POTT_ip1, POTT_jm1, POTT_jp1,
            UFLX, UFLX_ip1, VFLX, VFLX_jp1,
            COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1, A,
            POTTVB, POTTVB_kp1, WWIND, WWIND_kp1,
            COLP_NEW, dsigma, k):

    dPOTTdt = wp(0.)

    # HORIZONTAL ADVECTION
    if i_POTT_hor_adv:
        dPOTTdt = dPOTTdt + hor_adv(
            POTT,
            POTT_im1, POTT_ip1,
            POTT_jm1, POTT_jp1,
            UFLX, UFLX_ip1,
            VFLX, VFLX_jp1,
            A)
    # VERTICAL ADVECTION
    if i_POTT_vert_adv:
        dPOTTdt = dPOTTdt + vert_adv(
            POTTVB, POTTVB_kp1,
            WWIND, WWIND_kp1,
            COLP_NEW, dsigma, k)
    # NUMERICAL HORIZONTAL DIFUSION
    if i_POTT_num_dif and (POTT_dif_coef > wp(0.)):
        dPOTTdt = dPOTTdt + num_dif(
            POTT, POTT_im1, POTT_ip1,
            POTT_jm1, POTT_jp1,
            COLP, COLP_im1, COLP_ip1,
            COLP_jm1, COLP_jp1,
            POTT_dif_coef)

    return(dPOTTdt)






####################################################################
### SPECIALIZE FOR GPU
####################################################################
hor_adv = njit(hor_adv_py, device=True, inline=True)
num_dif = njit(num_dif_py, device=True, inline=True)
vert_adv = njit(vert_adv_py, device=True, inline=True)
add_up_tendencies = njit(add_up_tendencies_py, device=True, inline=True)

def launch_cuda_kernel(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                     POTTVB, WWIND, COLP_NEW, dsigma):

    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dPOTTdt[i  ,j  ,k] = \
            add_up_tendencies(POTT[i  ,j  ,k],
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


POTT_tendency_gpu = cuda.jit(cuda_kernel_decorator(launch_cuda_kernel))\
                            (launch_cuda_kernel)



####################################################################
### SPECIALIZE FOR CPU
####################################################################
hor_adv = njit(hor_adv_py)
vert_adv = njit(vert_adv_py)
num_dif = njit(num_dif_py)
add_up_tendencies = njit(add_up_tendencies_py)

def launch_numba_cpu(dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                 POTTVB, WWIND, COLP_NEW, dsigma):

    for i in prange(nb,nx+nb):
        for j in range(nb,ny+nb):
            for k in range(wp_int(0),nz):

                dPOTTdt[i  ,j  ,k] = \
                    add_up_tendencies(POTT[i  ,j  ,k],
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


POTT_tendency_cpu = njit(parallel=True)(launch_numba_cpu)




####################################################################
### OLD GPU FUNCTION
####################################################################
#@njit([wp_old+'[:,:,:], '+\
#  wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
#  wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
#  #wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
#  wp_old+'[:,:,:], '+wp_old+'[:,:,:]  '
# ], target='gpu')
#def calc_dPOTTdt_orig(dPOTTdt,
#            POTT, POTTVB, COLP, COLP_NEW,
#            UFLX, VFLX, WWIND,
#            #dPOTTdt_RAD, dPOTTdt_MIC,
#            A, dsigma):
#
#    if i_POTT_main_switch:
#
#        i, j, k = cuda.grid(3)
#        if i > 0 and i < nx+1 and j > 0 and j < ny+1:
#
#            # HORIZONTAL ADVECTION
#            if i_POTT_hor_adv:
#                dPOTTdt[i,j,k] = \
#                    (+ UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/wp(2.)\
#                  - UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/wp(2.)\
#                  + VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/wp(2.)\
#                  - VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/wp(2.))\
#                  / A[i  ,j  ,0]
#
#            # VERTICAL ADVECTION
#            if i_POTT_vert_adv:
#                if k == wp_int(0):
#                    vertAdv_POTT = COLP_NEW[i  ,j  ,0] * (\
#                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[0, 0, k]
#                elif k == nz:
#                    vertAdv_POTT = COLP_NEW[i  ,j  ,0] * (\
#                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ]) / dsigma[0, 0, k]
#                else:
#                    vertAdv_POTT = COLP_NEW[i  ,j  ,0] * (\
#                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
#                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[0, 0, k]
#                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + vertAdv_POTT
#
#
#            # NUMERICAL DIFUSION 
#            #                #exp(-(float(nz-k-1))) *\
#            if i_POTT_num_dif and (POTT_dif_coef > 0.):
#                num_dif = POTT_dif_coef*\
#                            ( + COLP[i-1,j  ,0] * POTT[i-1,j  ,k  ] \
#                              + COLP[i+1,j  ,0] * POTT[i+1,j  ,k  ] \
#                              + COLP[i  ,j-1,0] * POTT[i  ,j-1,k  ] \
#                              + COLP[i  ,j+1,0] * POTT[i  ,j+1,k  ] \
#                         - wp(4.) * COLP[i  ,j  ,0] * POTT[i  ,j  ,k  ] )
#                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + num_dif


@njit([wp_old+'[:,:,:], '+\
  wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:], '+wp_old+'[:,:], '+\
  wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
  #wp_old+'[:,:,:], '+wp_old+'[:,:,:], '+\
  wp_old+'[:,:], '+wp_old+'[:]  '
 ], target='gpu')
def calc_dPOTTdt_orig(dPOTTdt,
            POTT, POTTVB, COLP, COLP_NEW,
            UFLX, VFLX, WWIND,
            #dPOTTdt_RAD, dPOTTdt_MIC,
            A, dsigma):

    if i_POTT_main_switch:

        i, j, k = cuda.grid(3)
        if i > 0 and i < nx+1 and j > 0 and j < ny+1:

            # HORIZONTAL ADVECTION
            if i_POTT_hor_adv:
                dPOTTdt[i,j,k] = \
                    (+ UFLX[i  ,j  ,k] * (POTT[i-1,j  ,k] + POTT[i  ,j  ,k])/wp(2.)\
                  - UFLX[i+1,j  ,k] * (POTT[i  ,j  ,k] + POTT[i+1,j  ,k])/wp(2.)\
                  + VFLX[i  ,j  ,k] * (POTT[i  ,j-1,k] + POTT[i  ,j  ,k])/wp(2.)\
                  - VFLX[i  ,j+1,k] * (POTT[i  ,j  ,k] + POTT[i  ,j+1,k])/wp(2.))\
                  / A[i  ,j]

            # VERTICAL ADVECTION
            if i_POTT_vert_adv:
                if k == wp_int(0):
                    vertAdv_POTT = COLP_NEW[i  ,j] * (\
                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
                elif k == nz:
                    vertAdv_POTT = COLP_NEW[i  ,j] * (\
                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ]) / dsigma[k]
                else:
                    vertAdv_POTT = COLP_NEW[i  ,j] * (\
                            + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
                            - WWIND[i  ,j  ,k+1] * POTTVB[i  ,j  ,k+1]) / dsigma[k]
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + vertAdv_POTT


            # NUMERICAL DIFUSION 
            #                #exp(-(float(nz-k-1))) *\
            if i_POTT_num_dif and (POTT_dif_coef > 0.):
                num_dif = POTT_dif_coef*\
                            ( + COLP[i-1,j  ] * POTT[i-1,j  ,k  ] \
                              + COLP[i+1,j  ] * POTT[i+1,j  ,k  ] \
                              + COLP[i  ,j-1] * POTT[i  ,j-1,k  ] \
                              + COLP[i  ,j+1] * POTT[i  ,j+1,k  ] \
                         - wp(4.) * COLP[i  ,j  ] * POTT[i  ,j  ,k  ] )
                dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + num_dif

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

    from tendencies import TendencyFactory

    if comp_mode in [2,0]:
        import cupy as cp
    else:
        import numpy as cp


    i = slice(nb,nx+nb)
    j = slice(nb,ny+nb)


    COLP        = cp.full( ( nx +2*nb, ny +2*nb,   1      ), 
                             np.nan,   dtype=wp)
    A = cp.full_like(COLP, np.nan)
    COLP_NEW = cp.full_like(COLP, np.nan)
    POTT        = cp.full( ( nx +2*nb, ny +2*nb, nz  ), 
                        np.nan, dtype=wp)
    dPOTTdt = cp.full_like(POTT, np.nan)
    UFLX        = cp.full( ( nxs +2*nb, ny +2*nb, nz  ), 
                        np.nan, dtype=wp)
    VFLX        = cp.full( ( nx +2*nb, nys +2*nb, nz  ), 
                        np.nan, dtype=wp)

    POTTVB        = cp.full( ( nx +2*nb, ny +2*nb, nzs  ), 
                        np.nan, dtype=wp)
    WWIND = cp.full_like(POTTVB, np.nan)


    dsigma        = cp.full( ( 1, 1,   nz      ), 
                             np.nan,   dtype=wp)


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


    if comp_mode == 2:
        print('gpu')
        Tendencies = TendencyFactory(target='GPU')

        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        t0 = time.time()
        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        print(time.time() - t0)

        print(np.mean(cp.asnumpy(dPOTTdt[i,j,:])))
        print()

    elif comp_mode == 1:
        print('numba_par')
        Tendencies = TendencyFactory(target='CPU')

        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        t0 = time.time()
        dPOTTdt = Tendencies.POTT_tendency(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
        print(time.time() - t0)
        print(np.mean(dPOTTdt[i,j,:]))
        print()




    elif comp_mode == 0:
        print('gpu_old')

        from grid import tpb, bpg

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




