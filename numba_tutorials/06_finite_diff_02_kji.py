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

nz_tpb = 1
ny_tpb = 1
nx_tpb = 32*2
tpb_improve = (nz_tpb,ny_tpb,nx_tpb)
shared_memory_size = (nz_tpb+2,ny_tpb+2,nx_tpb+2)

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
        tmp = wp(0.)
        tmp += (VAR[k,j+1,i] - VAR[k,j-1,i])
        tmp += (VAR[k,j,i+1] - VAR[k,j,i-1])
        if k >= 1 and k < nz-1:
            tmp += (VAR[k+1,j,i] - VAR[k-1,j,i])
        dVARdt[k,j,i] = tmp


def kernel_improve(dVARdt, VAR, VAR1, VAR2):
    k, j, i = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (VAR[k,j+1,i] - VAR[k,j-1,i])
        tmp += (VAR[k,j,i+1] - VAR[k,j,i-1])
        if k >= 1 and k < nz-1:
            tmp += (VAR[k+1,j,i] - VAR[k-1,j,i])
        dVARdt[k,j,i] = tmp



def kernel_shared(dVARdt, VAR, VAR1, VAR2):
    sVAR = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    

    k, j, i = cuda.grid(3)
    si = cuda.threadIdx.z + 1
    sj = cuda.threadIdx.y + 1
    sk = cuda.threadIdx.x + 1

    sVAR[sk,sj,si] = VAR[k,j,i]
    #cuda.syncthreads()

    if si == 1:
        sVAR[sk,sj,si-1] = VAR[k,j,i-1]
    if si == cuda.blockDim.z:
        sVAR[sk,sj,si+1] = VAR[k,j,i+1]
    #cuda.syncthreads()

    if sj == 1:
        sVAR[sk,sj-1,si] = VAR[k,j-1,i]
    if sj == cuda.blockDim.y:
        sVAR[sk,sj+1,si] = VAR[k,j+1,i]
    #cuda.syncthreads()

    if sk == 1:
        sVAR[sk-1,sj,si] = VAR[k-1,j,i]
    if sk == cuda.blockDim.x:
        sVAR[sk+1,sj,si] = VAR[k+1,j,i]

    cuda.syncthreads()

    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (sVAR[sk,sj+1,si] - sVAR[sk,sj-1,si])
        tmp += (sVAR[sk,sj,si+1] - sVAR[sk,sj,si-1])
        if k >= 1 and k < nz-1:
            tmp += (sVAR[sk+1,sj,si] - sVAR[sk-1,sj,si])
        dVARdt[k,j,i] = tmp






if __name__ == '__main__':

    n_iter = 20

    wp = np.float32
    wp_3D = 'float32[:,:,:]'
    wp_numba = numba.float32

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
    print((time.time() - t0)/n_iter*1000)
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
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[:,jj,ii]))
    array1 = dVARdt.copy()



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
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[:,jj,ii]))
    array2 = dVARdt

    #k = 31
    #j = 1
    #print(array1[k,j,:])
    #print()
    #print(array2[k,j,:])
    #print((np.abs(array2 - array1) > 0)[k,j,:])
    print(np.sum(np.abs(array2 - array1) > 0))
    quit()

    #import matplotlib.pyplot as plt
    #plt.plot(line1)
    #plt.plot(line2)
    #plt.show()

