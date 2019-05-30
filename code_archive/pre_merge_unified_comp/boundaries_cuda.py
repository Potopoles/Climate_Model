import numpy as np
from numba import cuda, jit
from org_namelist import wp_old


def exchange_BC_gpu(FIELD, zonal, merid, griddim, blockdim, stream,
                    stagx=False, stagy=False, array2D=False):

    if array2D:
        get_BCx2D[griddim, blockdim, stream](FIELD, zonal, stagx)
        stream.synchronize()
        set_BCx2D[griddim, blockdim, stream](FIELD, zonal, stagx)
        stream.synchronize()
        get_BCy2D[griddim, blockdim, stream](FIELD, merid, stagy)
        stream.synchronize()
        set_BCy2D[griddim, blockdim, stream](FIELD, merid, stagy)
        stream.synchronize()
    else:
        get_BCx[griddim, blockdim, stream](FIELD, zonal, stagx)
        stream.synchronize()
        set_BCx[griddim, blockdim, stream](FIELD, zonal, stagx)
        stream.synchronize()
        get_BCy[griddim, blockdim, stream](FIELD, merid, stagy)
        stream.synchronize()
        set_BCy[griddim, blockdim, stream](FIELD, merid, stagy)
        stream.synchronize()

    return(FIELD)



@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def get_BCx(FIELD, zonal, stagx):
    i, j, k = cuda.grid(3)
    if stagx: # staggered in x
        nxs = FIELD.shape[0] - 2
        if i == 1:
            zonal[0,j,k] = FIELD[i,j,k] 
        elif i == nxs-1:
            zonal[1,j,k] = FIELD[i,j,k] 
    else:     # unstaggered in x
        nx = FIELD.shape[0] - 2
        if i == 1:
            zonal[0,j,k] = FIELD[i,j,k] 
        elif i == nx:
            zonal[1,j,k] = FIELD[i,j,k] 
    cuda.syncthreads()

@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def get_BCy(FIELD, merid, stagy):
    i, j, k = cuda.grid(3)
    if stagy: # staggered in y
        pass
    else: # unstaggered in y
        ny = FIELD.shape[1] - 2
        if j == 1:
            merid[i,0,k] = FIELD[i,j,k] 
        elif j == ny:
            merid[i,1,k] = FIELD[i,j,k] 
    cuda.syncthreads()

@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def set_BCx(FIELD, zonal, stagx):
    i, j, k = cuda.grid(3)
    if stagx: # staggered in x
        nxs = FIELD.shape[0] - 2
        if i == 0:
            FIELD[i,j,k] = zonal[1,j,k] 
        elif i == nxs:
            FIELD[i,j,k] = zonal[0,j,k] 
    else:     # unstaggered in x
        nx = FIELD.shape[0] - 2
        if i == 0:
            FIELD[i,j,k] = zonal[1,j,k] 
        elif i == nx+1:
            FIELD[i,j,k] = zonal[0,j,k] 
    cuda.syncthreads()

@jit([wp_old+'[:,:,:], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def set_BCy(FIELD, merid, stagy):
    i, j, k = cuda.grid(3)
    if stagy: # staggered in y
        nys = FIELD.shape[1] - 2
        if j == 0 or j == 1 or j == nys or j == nys+1:
            FIELD[i,j,k] = 0.
    else: # unstaggered in y
        ny = FIELD.shape[1] - 2
        if j == 0:
            FIELD[i,j,k] = merid[i,0,k] 
        elif j == ny+1:
            FIELD[i,j,k] = merid[i,1,k] 
    cuda.syncthreads()






@jit([wp_old+'[:,:  ], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def get_BCx2D(FIELD, zonal, stagx):
    i, j = cuda.grid(2)
    if stagx: # staggered in x
        nxs = FIELD.shape[0] - 2
        if i == 1:
            zonal[0,j,0] = FIELD[i,j] 
        elif i == nxs-1:
            zonal[1,j,0] = FIELD[i,j] 
    else:     # unstaggered in x
        nx = FIELD.shape[0] - 2
        if i == 1:
            zonal[0,j,0] = FIELD[i,j] 
        elif i == nx:
            zonal[1,j,0] = FIELD[i,j] 
    cuda.syncthreads()

@jit([wp_old+'[:,:  ], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def get_BCy2D(FIELD, merid, stagy):
    i, j = cuda.grid(2)
    if stagy: # staggered in y
        pass
    else: # unstaggered in y
        ny = FIELD.shape[1] - 2
        if j == 1:
            merid[i,0,0] = FIELD[i,j] 
        elif j == ny:
            merid[i,1,0] = FIELD[i,j] 
    cuda.syncthreads()

@jit([wp_old+'[:,:  ], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def set_BCx2D(FIELD, zonal, stagx):
    i, j = cuda.grid(2)
    if stagx: # staggered in x
        nxs = FIELD.shape[0] - 2
        if i == 0:
            FIELD[i,j] = zonal[1,j,0] 
        elif i == nxs:
            FIELD[i,j] = zonal[0,j,0] 
    else:     # unstaggered in x
        nx = FIELD.shape[0] - 2
        if i == 0:
            FIELD[i,j] = zonal[1,j,0] 
        elif i == nx+1:
            FIELD[i,j] = zonal[0,j,0] 
    cuda.syncthreads()

@jit([wp_old+'[:,:  ], '+wp_old+'[:,:,:],  '+wp_old], target='gpu')
def set_BCy2D(FIELD, merid, stagy):
    i, j = cuda.grid(2)
    if stagy: # staggered in y
        nys = FIELD.shape[1] - 2
        if j == 0 or j == 1 or j == nys or j == nys+1:
            FIELD[i,j] = 0.
    else: # unstaggered in y
        ny = FIELD.shape[1] - 2
        if j == 0:
            FIELD[i,j] = merid[i,0,0] 
        elif j == ny+1:
            FIELD[i,j] = merid[i,1,0] 
    cuda.syncthreads()


