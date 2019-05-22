import numpy as np
import numba, math
from numba import cuda, float32
import time
from inspect import signature

zoom = 0.25
zoom = 2
nx = int(180*zoom)
ny = int(90*zoom)
nz = 32
nb = 1

nx_tpb = 2
ny_tpb = 2
nz_tpb = 32
tpb_improve = (nx_tpb,ny_tpb,nz_tpb)
shared_memory_size = (nx_tpb+2,ny_tpb+2,nz_tpb)



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


def kernel_trivial(dVARdt, VAR):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (VAR[i,j+1,k] - VAR[i,j-1,k])
        tmp += (VAR[i+1,j,k] - VAR[i-1,j,k])
        if k >= 1 and k < nz-1:
            tmp +=(VAR[i,j,k+1] - VAR[i,j,k-1]) 
        dVARdt[i,j,k] = tmp

def kernel_improve(dVARdt, VAR):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (VAR[i,j+1,k] - VAR[i,j-1,k])
        tmp += (VAR[i+1,j,k] - VAR[i-1,j,k])
        if k >= 1 and k < nz-1:
            tmp +=(VAR[i,j,k+1] - VAR[i,j,k-1]) 
        dVARdt[i,j,k] = tmp


def kernel_shared(dVARdt, VAR):
    sVAR = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    i, j, k = cuda.grid(3)
    si = cuda.threadIdx.x + 1
    sj = cuda.threadIdx.y + 1
    sk = cuda.threadIdx.z

    sVAR[si,sj,sk] = VAR[i,j,k]
    #cuda.syncthreads()

    if si == 1:
        sVAR[si-1,sj,sk] = VAR[i-1,j,k]
    if si == cuda.blockDim.x:
        sVAR[si+1,sj,sk] = VAR[i+1,j,k]
    #cuda.syncthreads()

    if sj == 1:
        sVAR[si,sj-1,sk] = VAR[i,j-1,k]
    if sj == cuda.blockDim.y:
        sVAR[si,sj+1,sk] = VAR[i,j+1,k]
    cuda.syncthreads()

    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (sVAR[si,sj+1,sk] - sVAR[si,sj-1,sk])
        tmp += (sVAR[si+1,sj,sk] - sVAR[si-1,sj,sk])
        if k >= 1 and k < nz-1:
            tmp += (sVAR[si,sj,sk+1] - sVAR[si,sj,sk-1]) 
        dVARdt[i,j,k] = tmp



if __name__ == '__main__':

    n_iter = 20

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


    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('trivial')
    tpb = (1, 1, nz)
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
    print(tpb)
    print(bpg)
    kernel_trivial[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_trivial[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
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
    kernel_improve[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_improve[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[ii,jj,:]))
    array1 = dVARdt.copy()




    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('shared')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
    print(tpb)
    print(bpg)
    kernel_shared[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_shared[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[ii,jj,:]))
    array2 = dVARdt

    #print(array1[:,5,5])
    #print()
    #print(array2[:,5,5])
    #print((np.abs(array2 - array1) > 0)[:,5,5])
    print(np.sum(np.abs(array2 - array1) > 0))
    quit()
