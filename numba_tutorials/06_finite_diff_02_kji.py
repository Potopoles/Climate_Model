import numpy as np
import numba, math
from numba import cuda, float32
import time
from inspect import signature


nz_tpb = 1
ny_tpb = 1
nx_tpb = 32*2
tpb_improve = (nz_tpb,ny_tpb,nx_tpb)
shared_memory_size = (nz_tpb,ny_tpb+2,nx_tpb+2)

def cuda_kernel_decorator(function):
    n_input_args = len(signature(function).parameters)
    decorator = 'void('
    for i in range(n_input_args):
        decorator += wp_3D+','
    decorator = decorator[:-1]
    decorator += ')'
    return(decorator)

def exchange_BC(VAR):
    VAR[:,jj,0] = VAR[:,jj,nx]
    VAR[:,jj,nx+nb] = VAR[:,jj,nb]
    VAR[:,0,ii] = VAR[:,ny,ii]
    VAR[:,ny+nb,ii] = VAR[:,nb,ii]

#def kernel_numpy(dVARdt, VAR, VAR1):
#    dVARdt[:,jj,ii] = (VAR[:,jj,ii+1] - VAR[:,jj,ii-1]) * VAR1[:,jj,ii]

def kernel_trivial(dVARdt, VAR, VAR1, VAR2):
    k, j, i = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVARdt[k,j,i] = (VAR[k,j,i+1] - VAR[k,j,i-1]) * VAR1[k,j,i]
        for c in range(10):
            dVARdt[k,j,i] = (dVARdt[k,j,i] + 
                            (VAR[k,j+1,i] - VAR[k,j-1,i]) * 
                            np.float32(0.5)*(VAR2[k,j,i+1] + VAR2[k,j,i-1]))

def kernel_improve(dVARdt, VAR, VAR1, VAR2):
    k, j, i = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVARdt[k,j,i] = (VAR[k,j,i+1] - VAR[k,j,i-1]) * VAR1[k,j,i]
        #dVARdt[k,j,i] = (VAR[k,j,i] - VAR[k,j,i-1]) * VAR1[k,j,i]
        for c in range(10):
            dVARdt[k,j,i] = (dVARdt[k,j,i] + 
                            (VAR[k,j+1,i] - VAR[k,j-1,i]) * 
                            np.float32(0.5)*(VAR2[k,j,i+1] + VAR2[k,j,i-1]))


def kernel_shared(dVARdt, VAR, VAR1, VAR2):
    sVAR = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    sVAR1 = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    sVAR2 = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    k, j, i = cuda.grid(3)
    tk = cuda.threadIdx.x
    tj = cuda.threadIdx.y
    ti = cuda.threadIdx.z + 1


    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        sVAR[tk,tj,ti]  = VAR[k,j,i]
        #if ti < 3:
        #    sVAR[tk,tj,ti-1]  = VAR[k,j,i-1]
        sVAR1[tk,tj,ti] = VAR1[k,j,i]
        sVAR2[tk,tj,ti] = VAR2[k,j,i]
        #if ti == 1:
        #    sVAR [tk,tj,0] = VAR [k,j,i-1]
        #    sVAR1[tk,tj,0] = VAR1[k,j,i-1]
        #    sVAR2[tk,tj,0] = VAR2[k,j,i-1]
        #if ti == cuda.blockDim.z:
        #    sVAR[tk,tj,-1]  = VAR[k,j,i+1]
        #    sVAR1[tk,tj,-1] = VAR1[k,j,i+1]
        #    sVAR2[tk,tj,-1] = VAR2[k,j,i+1]
        cuda.syncthreads()

        #dVARdt[k,j,i] = (sVAR[tk,tj,ti+1] - sVAR[tk,tj,ti-1]) * sVAR1[tk,tj,ti]
        dVARdt[k,j,i] = (sVAR[tk,tj,ti] - sVAR[tk,tj,ti-1]) * sVAR1[tk,tj,ti]
        #for c in range(10):
        #    dVARdt[k,j,i] = (dVARdt[k,j,i] + 
        #                    (sVAR[tk,tj+1,ti] - sVAR[tk,tj-1,ti]) * 
        #                np.float32(0.5)*(sVAR2[tk,tj,ti+1] + sVAR2[tk,tj,ti-1]))




if __name__ == '__main__':

    n_iter = 10

    wp = np.float32
    wp_3D = 'float32[:,:,:]'
    wp_numba = numba.float32

    zoom = 2
    nx = int(180*zoom)
    ny = int(90*zoom)
    nz = 32
    nb = 1

    #ii,jj = np.ix_(np.arange(0,nx)+nb,np.arange(0,ny)+nb)

    #VAR         = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)
    #dVARdt      = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)
    #VAR1        = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)
    #VAR2        = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)

    #VAR[ii,jj,:] = np.random.random((nx,ny,nz))
    #VAR1[ii,jj,:] = np.random.random((nx,ny,nz))
    #VAR2[ii,jj,:] = np.random.random((nx,ny,nz))
    #dVARdt[ii,jj,:] = 0.

    jj,ii = np.ix_(np.arange(0,ny)+nb, np.arange(0,nx)+nb)

    VAR         = np.full((nz, ny+2*nb,nx+2*nb), np.nan, dtype=wp)
    dVARdt      = np.full((nz, ny+2*nb,nx+2*nb), np.nan, dtype=wp)
    VAR1        = np.full((nz, ny+2*nb,nx+2*nb), np.nan, dtype=wp)
    VAR2        = np.full((nz, ny+2*nb,nx+2*nb), np.nan, dtype=wp)

    VAR   [:,jj,ii] = np.random.random((nz,ny,nx))
    VAR1  [:,jj,ii] = np.random.random((nz,ny,nx))
    VAR2  [:,jj,ii] = np.random.random((nz,ny,nx))
    dVARdt[:,jj,ii] = 0.

    exchange_BC(VAR)
    exchange_BC(VAR1)
    exchange_BC(VAR2)

    stream = cuda.stream()

    VARd    = cuda.to_device(VAR, stream)
    dVARdtd = cuda.to_device(dVARdt, stream)
    VAR1d   = cuda.to_device(VAR1, stream)
    VAR2d   = cuda.to_device(VAR2, stream)


    kernel_trivial = cuda.jit(cuda_kernel_decorator(kernel_trivial))\
                                (kernel_trivial)
    kernel_improve = cuda.jit(cuda_kernel_decorator(kernel_improve))\
                                (kernel_improve)
    kernel_shared = cuda.jit(cuda_kernel_decorator(kernel_shared))\
                                (kernel_shared)


    #print('python')
    #t0 = time.time()
    #for c in range(n_iter):
    #    kernel_numpy(dVARdt, VAR, VAR1)
    #print(time.time() - t0)
    ##print(np.mean(dVARdt[:,jj,ii]))
    #print(np.mean(dVARdt[:,jj,ii]))

    dVARdt[:,jj,ii] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)


    print('trivial')
    tpb = (nz, 1, 1)
    bpg = (math.ceil((nz)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
            math.ceil((nx+2*nb)/tpb[2]))
    print(tpb)
    print(bpg)
    kernel_trivial[bpg, tpb](dVARdtd, VARd, VAR1d, VAR2d)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_trivial[bpg, tpb](dVARdtd, VARd, VAR1d, VAR2d)
        cuda.synchronize()
    print(time.time() - t0)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[:,jj,ii]))


    dVARdt[:,jj,ii] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('improve')
    tpb = tpb_improve
    bpg = (math.ceil((nz)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
            math.ceil((nx+2*nb)/tpb[2]))
    print(tpb)
    print(bpg)
    kernel_improve[bpg, tpb](dVARdtd, VARd, VAR1d, VAR2d)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_improve[bpg, tpb](dVARdtd, VARd, VAR1d, VAR2d)
        cuda.synchronize()
    print(time.time() - t0)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[:,jj,ii]))



    dVARdt[:,jj,ii] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('shared')
    tpb = tpb_improve
    bpg = (math.ceil((nz)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
            math.ceil((nx+2*nb)/tpb[2]))
    print(tpb)
    print(bpg)
    kernel_shared[bpg, tpb](dVARdtd, VARd, VAR1d, VAR2d)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_shared[bpg, tpb](dVARdtd, VARd, VAR1d, VAR2d)
        cuda.synchronize()
    print(time.time() - t0)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[:,jj,ii]))
