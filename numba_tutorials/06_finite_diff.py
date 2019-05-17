import numpy as np
import numba
from numba import cuda, float32
import time
from inspect import signature


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

def kernel_numpy(dVARdt, VAR, VAR1):
    dVARdt[ii,jj,:] = (VAR[ii+1,jj,:] - VAR[ii-1,jj,:]) * VAR1[ii,jj]

def kernel_trivial(dVARdt, VAR, VAR1):
    i, j, k = cuda.grid(3)
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        dVARdt[i,j,k] = 0.
        dVARdt[i,j,k] = (VAR[i+1,j,k] - VAR[i-1,j,k]) * VAR1[i,j,k]

def kernel_xline(dVARdt, VAR, VAR1):
    #sVAR = cuda.shared.array(shape=(66,1,1),dtype=float32)    
    #sVAR1 = cuda.shared.array(shape=(66,1,1),dtype=float32)    
    i, j, k = cuda.grid(3)
    #sVAR[i,0,0] = VAR[i,0,0]
    #sVAR1[i,0,0] = VAR1[i,0,0]
    if i >= nb and i < nx+nb and j >= nb and j < ny+nb:
        #dVARdt[i,j,k] = 0.
        #dVARdt[i,j,k] = (sVAR[i+1,0,0] - sVAR[i-1,0,0]) * VAR1[i,j,k]
        dVARdt[i,j,k] = (VAR[i+1,j,k] - VAR[i-1,j,k]) * VAR1[i,j,k]


if __name__ == '__main__':

    n_iter = 100

    wp = np.float32
    wp_3D = 'float32[:,:,:]'
    wp_numba = numba.float32

    nx = 720
    ny = 64
    nz = 32
    nb = 1

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

    stream = cuda.stream()

    VARd    = cuda.to_device(VAR, stream)
    dVARdtd = cuda.to_device(dVARdt, stream)
    VAR1d   = cuda.to_device(VAR1, stream)
    VAR2d   = cuda.to_device(VAR2, stream)


    kernel_trivial = cuda.jit(cuda_kernel_decorator(kernel_trivial))\
                                (kernel_trivial)
    kernel_xline = cuda.jit(cuda_kernel_decorator(kernel_xline))\
                                (kernel_xline)


    print('python')
    t0 = time.time()
    for c in range(n_iter):
        kernel_numpy(dVARdt, VAR, VAR1)
    print(time.time() - t0)
    print(np.mean(dVARdt[ii,jj,:]))


    print('trivial')
    tpb = (1, 1, nz)
    bpg = (int((nx+2*nb)/tpb[0]),   int((ny+2*nb)/tpb[1]),
            int((nz)/tpb[2]))
    bpg = (722, 66, 1)
    kernel_trivial[bpg, tpb](dVARdtd, VARd, VAR1d)
    t0 = time.time()
    for c in range(n_iter):
        kernel_trivial[bpg, tpb](dVARdtd, VARd, VAR1d)
    print(time.time() - t0)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[ii,jj,:]))

    print('xline')
    tpb = (722, 1, 1)
    bpg = (int((nx+2*nb)/tpb[0])+1,   int((ny+2*nb)/tpb[1]),
            int((nz)/tpb[2]))
    bpg = (1, 66, 32)
    kernel_xline[bpg, tpb](dVARdtd, VARd, VAR1d)
    t0 = time.time()
    for c in range(n_iter):
        kernel_xline[bpg, tpb](dVARdtd, VARd, VAR1d)
    print(time.time() - t0)
    dVARdt = dVARdtd.copy_to_host()
    print(np.mean(dVARdt[ii,jj,:]))
