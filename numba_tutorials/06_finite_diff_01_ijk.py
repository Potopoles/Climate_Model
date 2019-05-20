import numpy as np
import numba, math
from numba import cuda, float32
import time
from inspect import signature

nx_tpb = 1
ny_tpb = 2
nz_tpb = 32
tpb_improve = (nx_tpb,ny_tpb,nz_tpb)
shared_memory_size = (nx_tpb+2,ny_tpb+2,nz_tpb)

zoom = 2
nx = int(180*zoom)
ny = int(90*zoom)
nz = 32
nb = 1


def cuda_kernel_decorator(function):
    n_input_args = len(signature(function).parameters)
    decorator = 'void('
    for i in range(n_input_args):
        decorator += wp_3D+','
    decorator = decorator[:-1]
    decorator += ')'
    return(decorator)

def exchange_BC(VAR):
    VAR[0,jj,:] = VAR[nx,jj,:]
    VAR[nx+nb,jj,:] = VAR[nb,jj,:]
    VAR[ii,0,:] = VAR[ii,ny,:]
    VAR[ii,ny+nb,:] = VAR[ii,nb,:]

#def kernel_numpy(dVARdt, VAR, VAR1):
#    dVARdt[ii,jj,:] = (VAR[ii+1,jj,:] - VAR[ii-1,jj,:]) * VAR1[ii,jj,:]

def kernel_trivial(dVARdt, VAR, VAR1, VAR2):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVARdt[i,j,k] = (VAR[i+1,j,k] - VAR[i-1,j,k]) * VAR1[i,j,k]
        for c in range(10):
            dVARdt[i,j,k] = (dVARdt[i,j,k] + 
                            (VAR[i,j+1,k] - VAR[i,j-1,k]) * 
                        np.float32(0.5)*(VAR2[i+1,j,k] + VAR2[i-1,j,k]))

def kernel_improve(dVARdt, VAR, VAR1, VAR2):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVARdt[i,j,k] = (VAR[i+1,j,k] - VAR[i-1,j,k]) * VAR1[i,j,k]
        for c in range(10):
            dVARdt[i,j,k] = (dVARdt[i,j,k] + 
                            (VAR[i,j+1,k] - VAR[i,j-1,k]) * 
                            np.float32(0.5)*(VAR2[i+1,j,k] + VAR2[i-1,j,k]))


def kernel_shared(dVARdt, VAR, VAR1, VAR2):
    sVAR = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    sVAR1 = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    sVAR2 = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    i, j, k = cuda.grid(3)
    ti = cuda.threadIdx.x
    tj = cuda.threadIdx.y
    tk = cuda.threadIdx.z
    sVAR[ti,tj,tk] = VAR[i,j,k]
    sVAR1[ti,tj,tk] = VAR1[i,j,k]
    sVAR2[ti,tj,tk] = VAR2[i,j,k]
    cuda.syncthreads()
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVARdt[i,j,k] = (sVAR[ti+1,tj,tk] - sVAR[ti-1,tj,tk]) * sVAR1[ti,tj,tk]
        for c in range(10):
            dVARdt[i,j,k] = (dVARdt[i,j,k] + 
                            (sVAR[ti,tj+1,tk] - sVAR[ti,tj-1,tk]) * 
                        np.float32(0.5)*(sVAR2[ti+1,tj,tk] + sVAR2[ti-1,tj,tk]))


if __name__ == '__main__':

    n_iter = 50

    wp = np.float32
    wp_3D = 'float32[:,:,:]'
    wp_numba = numba.float32


    ii,jj = np.ix_(np.arange(0,nx)+nb,np.arange(0,ny)+nb)

    VAR         = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)
    dVARdt      = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)
    VAR1        = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)
    VAR2        = np.full((nx+2*nb,ny+2*nb,nz), np.nan, dtype=wp)

    VAR[ii,jj,:] = np.random.random((nx,ny,nz))
    VAR1[ii,jj,:] = np.random.random((nx,ny,nz))
    VAR2[ii,jj,:] = np.random.random((nx,ny,nz))

    dVARdt[ii,jj,:] = 0.

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

    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('trivial')
    tpb = (1, 1, nz)
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
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
    print(np.mean(dVARdt[ii,jj,:]))


    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('improve')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
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
    print(np.mean(dVARdt[ii,jj,:]))




    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('shared')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
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
    print(np.mean(dVARdt[ii,jj,:]))
