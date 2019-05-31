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

nx_tpb = 1
ny_tpb = 4
nz_tpb = 32
tpb_improve = (nx_tpb,ny_tpb,nz_tpb)
shared_memory_size = (nx_tpb+2,ny_tpb+2,nz_tpb)
shared_memory_size_z = (nx_tpb,ny_tpb,nz_tpb)
shared_memory_size_x = (nx_tpb+2,ny_tpb,nz_tpb)
shared_memory_size_y = (ny_tpb+2,nz_tpb)

i_run_x = 0
i_run_y = 1
i_run_z = 0


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


def kernel_improve(dVARdt, VAR):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        if i_run_x:
            tmp += (VAR[i+1,j,k] - VAR[i-1,j,k] + VAR[i,j,k])
        if i_run_y:
            tmp += (VAR[i,j+1,k] - VAR[i,j-1,k] + VAR[i,j,k])
        if i_run_z:
            if k >= 1 and k < nz-1:
                tmp +=(VAR[i,j,k+1] - VAR[i,j,k-1] + VAR[i,j,k-1]) 
        dVARdt[i,j,k] = tmp


def kernel_x(dVARdt, VAR):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (VAR[i+1,j,k] - VAR[i-1,j,k] + VAR[i,j,k])
        dVARdt[i,j,k] = tmp
def kernel_y(dVARdt, VAR):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (VAR[i+1,j,k] - VAR[i-1,j,k] + VAR[i,j,k])
        dVARdt[i,j,k] = tmp
def kernel_z(dVARdt, VAR):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        if k >= 1 and k < nz-1:
            tmp +=(VAR[i,j,k+1] - VAR[i,j,k-1] + VAR[i,j,k-1]) 
        dVARdt[i,j,k] = tmp



def kernel_shared(dVARdt, VAR):
    sVAR = cuda.shared.array(shape=(shared_memory_size),dtype=float32)    
    i, j, k = cuda.grid(3)
    si = cuda.threadIdx.x + 1
    sj = cuda.threadIdx.y + 1
    sk = cuda.threadIdx.z

    sVAR[si,sj,sk] = VAR[i,j,k]
    #cuda.syncthreads()

    if i_run_x:
        if si == 1:
            sVAR[si-1,sj,sk] = VAR[i-1,j,k]
        if si == cuda.blockDim.x:
            sVAR[si+1,sj,sk] = VAR[i+1,j,k]
    #cuda.syncthreads()

    if i_run_y:
        if sj == 1:
            sVAR[si,sj-1,sk] = VAR[i,j-1,k]
        if sj == cuda.blockDim.y:
            sVAR[si,sj+1,sk] = VAR[i,j+1,k]

    cuda.syncthreads()

    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        if i_run_x:
            tmp += (sVAR[si+1,sj,sk] - sVAR[si-1,sj,sk] - sVAR[si,sj,sk])
        if i_run_y:
            tmp += (sVAR[si,sj+1,sk] - sVAR[si,sj-1,sk] - sVAR[si,sj,sk])
        if i_run_z:
            if k >= 1 and k < nz-1:
                tmp += (sVAR[si,sj,sk+1] - sVAR[si,sj,sk-1] + sVAR[si,sj,sk]) 
        dVARdt[i,j,k] = tmp




def kernel_shared_z(dVARdt, VAR):
    sVAR = cuda.shared.array(shape=(shared_memory_size_z),dtype=float32)    
    i, j, k = cuda.grid(3)
    si = cuda.threadIdx.x
    sj = cuda.threadIdx.y
    sk = cuda.threadIdx.z

    sVAR[si,sj,sk] = VAR[i,j,k]
    cuda.syncthreads()

    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        if k >= 1 and k < nz-1:
            tmp += (sVAR[si,sj,sk+1] - sVAR[si,sj,sk-1] + sVAR[si,sj,sk]) 
        dVARdt[i,j,k] = tmp

def kernel_shared_x(dVARdt, VAR):
    sVAR = cuda.shared.array(shape=(shared_memory_size_x),dtype=float32)    
    i, j, k = cuda.grid(3)
    si = cuda.threadIdx.x + 1
    sj = cuda.threadIdx.y
    sk = cuda.threadIdx.z

    sVAR[si,sj,sk] = VAR[i,j,k]
    if si == 1:
        sVAR[si-1,sj,sk] = VAR[i-1,j,k]
    if si == cuda.blockDim.x:
        sVAR[si+1,sj,sk] = VAR[i+1,j,k]
    cuda.syncthreads()

    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (sVAR[si+1,sj,sk] - sVAR[si-1,sj,sk] - sVAR[si,sj,sk])
        dVARdt[i,j,k] = tmp

def kernel_shared_y(dVARdt, VAR):
    sVAR = cuda.shared.array(shape=(shared_memory_size_y),dtype=float32)    
    i, j, k = cuda.grid(3)
    sj = cuda.threadIdx.y + 1
    sk = cuda.threadIdx.z

    sVAR[sj,sk] = VAR[i,j,k]
    if sj == 1:
        sVAR[sj-1,sk] = VAR[i,j-1,k]
    if sj == cuda.blockDim.y:
        sVAR[sj+1,sk] = VAR[i,j+1,k]
    cuda.syncthreads()

    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        tmp = wp(0.)
        tmp += (sVAR[sj+1,sk] - sVAR[sj-1,sk] - sVAR[sj,sk])
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


    kernel_x = cuda.jit(cuda_kernel_decorator(kernel_x))(kernel_x)
    kernel_y = cuda.jit(cuda_kernel_decorator(kernel_y))(kernel_y)
    kernel_z = cuda.jit(cuda_kernel_decorator(kernel_z))(kernel_z)
    kernel_improve = cuda.jit(cuda_kernel_decorator(kernel_improve))\
                                (kernel_improve)
    kernel_shared = cuda.jit(cuda_kernel_decorator(kernel_shared))\
                                (kernel_shared)
    kernel_shared_z = cuda.jit(cuda_kernel_decorator(kernel_shared_z))\
                                (kernel_shared_z)
    kernel_shared_x = cuda.jit(cuda_kernel_decorator(kernel_shared_x))\
                                (kernel_shared_x)
    kernel_shared_y = cuda.jit(cuda_kernel_decorator(kernel_shared_y))\
                                (kernel_shared_y)


    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('trivial single')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
    #print(tpb)
    #print(bpg)
    kernel_x[bpg, tpb](dVARdtd, VARd)
    #kernel_y[bpg, tpb](dVARdtd, VARd)
    kernel_z[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        if i_run_x:
            kernel_x[bpg, tpb](dVARdtd, VARd)
        if i_run_y:
            kernel_y[bpg, tpb](dVARdtd, VARd)
        if i_run_z:
            kernel_z[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    #print(np.mean(dVARdt[ii,jj,:]))
    array1 = dVARdt.copy()


    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('trivial together')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
    #print(tpb)
    #print(bpg)
    kernel_improve[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_improve[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    #print(np.mean(dVARdt[ii,jj,:]))
    array1 = dVARdt.copy()




    dVARdt[ii,jj,:] = 0.
    dVARdtd = cuda.to_device(dVARdt, stream)

    print('shared together')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
    #print(tpb)
    #print(bpg)
    kernel_shared[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        kernel_shared[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    #print(np.mean(dVARdt[ii,jj,:]))
    array2 = dVARdt




    print('shared singel')
    tpb = tpb_improve
    bpg = (math.ceil((nx+2*nb)/tpb[0]), math.ceil((ny+2*nb)/tpb[1]),
           math.ceil((nz)/tpb[2]))
    #print(tpb)
    #print(bpg)
    kernel_shared_z[bpg, tpb](dVARdtd, VARd)
    kernel_shared_x[bpg, tpb](dVARdtd, VARd)
    kernel_shared_y[bpg, tpb](dVARdtd, VARd)
    cuda.synchronize()
    t0 = time.time()
    for c in range(n_iter):
        if i_run_x:
            kernel_shared_x[bpg, tpb](dVARdtd, VARd)
        if i_run_y:
            kernel_shared_y[bpg, tpb](dVARdtd, VARd)
        if i_run_z:
            kernel_shared_z[bpg, tpb](dVARdtd, VARd)
        cuda.synchronize()
    print((time.time() - t0)/n_iter*1000)
    dVARdt = dVARdtd.copy_to_host()
    #print(np.mean(dVARdt[ii,jj,:]))
    array2 = dVARdt


    quit()


    #print(array1[:,5,5])
    #print()
    #print(array2[:,5,5])
    #print((np.abs(array2 - array1) > 0)[:,5,5])
    print(np.sum(np.abs(array2 - array1) > 0))
    quit()
