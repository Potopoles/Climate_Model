import numpy as np
import numba, math
from numba import cuda, float32
import time
from inspect import signature

zoom = 0.5
#zoom = 2
nx = int(180*zoom)
ny = int(90*zoom)
nz = 32
nb = 1

nz_tpb = 2
ny_tpb = 2
nx_tpb = 32*2
tpb_improve = (nz_tpb,ny_tpb,nx_tpb)
#shared_memory_size = (nz_tpb,ny_tpb+2,nx+2*nb)
shared_memory_size = (nz_tpb+2,ny_tpb+2,nx+2*nb)

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
        for c in range(10):
            dVARdt[k,j,i] = (VAR[k,j+1,i] - VAR[k,j-1,i])
            dVARdt[k,j,i] = (VAR[k,j+1,i] - VAR[k,j-1,i])
            if k >= 1 and k <= nz:
                dVARdt[k,j,i] = (VAR[k+1,j,i] - VAR[k-1,j,i])
        #dVARdt[k,j,i] = (VAR[k,j,i+1] - VAR[k,j,i-1]) * VAR1[k,j,i]
        #for c in range(10):
        #    dVARdt[k,j,i] = (dVARdt[k,j,i] + 
        #                    (VAR[k,j+1,i] - VAR[k,j-1,i]) * 
        #                    np.float32(0.5)*(VAR2[k,j,i+1] + VAR2[k,j,i-1]))

def kernel_improve(dVARdt, VAR, VAR1, VAR2):
    k, j, i = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        for c in range(10):
            dVARdt[k,j,i] = (VAR[k,j+1,i] - VAR[k,j-1,i])
            dVARdt[k,j,i] = (VAR[k,j,i+1] - VAR[k,j,i-1])
            if k >= 1 and k <= nz:
                dVARdt[k,j,i] = (VAR[k+1,j,i] - VAR[k-1,j,i])
        #dVARdt[k,j,i] = (VAR[k,j,i+1] - VAR[k,j,i-1]) * VAR1[k,j,i]
        #for c in range(10):
        #    dVARdt[k,j,i] = (dVARdt[k,j,i] + 
        #                    (VAR[k,j+1,i] - VAR[k,j-1,i]) * 
        #                    np.float32(0.5)*(VAR2[k,j,i+1] + VAR2[k,j,i-1]))


def kernel_shared(dVARdt, VAR, VAR1, VAR2):
    sVAR = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    #sVAR1 = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    #sVAR2 = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    

    #i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    #k = cuda.blockIdx.z
    k, j, i = cuda.grid(3)
    #if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
    #si = i + 1
    #si = i
    sj = cuda.threadIdx.y + 1
    sk = cuda.threadIdx.x + 1

    for i in range(cuda.threadIdx.z, nx+2*nb, cuda.blockDim.z):
        si = i
        #sVAR[sj,si] = VAR[k,j,i]
        sVAR[sk,sj,si] = VAR[k,j,i]

    cuda.syncthreads()

    #if si < 2:
    #    sVAR[sj,si-1] = sVAR[sj,si+nx-2]
    #    sVAR[sj,si+nx] = sVAR[sj,si+1]

    #if sj == 1:
    #    sVAR[sj-1,si] = VAR[k,j-1,i]
    #elif sj == cuda.blockDim.y:
    #    sVAR[sj+1,si] = VAR[k,j+1,i]
    if sj == 1:
        sVAR[sk,sj-1,si] = VAR[k,j-1,i]
    elif sj == cuda.blockDim.y:
        sVAR[sk,sj+1,si] = VAR[k,j+1,i]

    cuda.syncthreads()

    if sk == 1:
        sVAR[sk-1,sj,si] = VAR[k-1,j,i]
    elif sk == cuda.blockDim.x:
        sVAR[sk+1,sj,si] = VAR[k+1,j,i]

    cuda.syncthreads()

    #if si == 1:
    #    sVAR[sj,si-1] = VAR[k,j,i-1]
    #elif si == cuda.blockDim.x:
    #    sVAR[sj,si+1] = VAR[k,j,i+1]

    #cuda.syncthreads()
    
    for i in range(cuda.threadIdx.z, nx+2*nb, cuda.blockDim.z):
        si = i
        if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
            for c in range(10):
                dVARdt[k,j,i] = (sVAR[sk,sj+1,si] - sVAR[sk,sj-1,si])
                dVARdt[k,j,i] = (sVAR[sk,sj,si+1] - sVAR[sk,sj,si-1])
                if k >= 1 and k <= nz:
                    dVARdt[k,j,i] = (sVAR[sk+1,sj,si] - sVAR[sk-1,sj,si])
        



if __name__ == '__main__':

    n_iter = 10

    wp = np.float32
    wp_3D = 'float32[:,:,:]'
    wp_numba = numba.float32


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
    print(time.time() - t0)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[:,jj,ii]))
    array2 = dVARdt

    quit()
    
    print(array1[5,5,:])
    print()
    print(array2[5,5,:])
    print(((array2 - array1) > 0)[5,5,:])
    print(np.sum((array2 - array1) > 0))

    #import matplotlib.pyplot as plt
    #plt.plot(line1)
    #plt.plot(line2)
    #plt.show()

