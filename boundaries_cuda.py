import numpy as np
from numba import cuda, jit
from namelist import wp
if wp == 'float64':
    from numba import float64



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



@jit([wp+'[:,:,:], '+wp+'[:,:,:],  '+wp], target='gpu')
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

@jit([wp+'[:,:,:], '+wp+'[:,:,:],  '+wp], target='gpu')
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

@jit([wp+'[:,:,:], '+wp+'[:,:,:],  '+wp], target='gpu')
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

@jit([wp+'[:,:,:], '+wp+'[:,:,:],  '+wp], target='gpu')
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






@jit([wp+'[:,:  ], '+wp+'[:,:,:],  '+wp], target='gpu')
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

@jit([wp+'[:,:  ], '+wp+'[:,:,:],  '+wp], target='gpu')
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

@jit([wp+'[:,:  ], '+wp+'[:,:,:],  '+wp], target='gpu')
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

@jit([wp+'[:,:  ], '+wp+'[:,:,:],  '+wp], target='gpu')
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



#def exchange_BC_gpu(FIELD, zonal, merid, griddim, blockdim, stream):
#    get_BC[griddim, blockdim, stream](FIELD, zonal, merid)
#    stream.synchronize()
#    set_BC[griddim, blockdim, stream](FIELD, zonal, merid)
#
#    stream.synchronize()
#    return(FIELD)


#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
#def get_BC(FIELD, zonal, merid):
#
#    nx = FIELD.shape[0] - 2
#    ny = FIELD.shape[1] - 2
#
#    i, j, k = cuda.grid(3)
#    if i == 1:
#        zonal[0,j,k] = FIELD[i,j,k] 
#    elif i == nx:
#        zonal[1,j,k] = FIELD[i,j,k] 
#
#    if j == 1:
#        merid[i,0,k] = FIELD[i,j,k] 
#    elif j == ny:
#        merid[i,1,k] = FIELD[i,j,k] 
#
#    cuda.syncthreads()
#
#
#@jit([wp+'[:,:,:], '+wp+'[:,:,:], '+wp+'[:,:,:]'], target='gpu')
#def set_BC(FIELD, zonal, merid):
#
#    nx = FIELD.shape[0] - 2
#    ny = FIELD.shape[1] - 2
#
#    i, j, k = cuda.grid(3)
#    if i == 0:
#        FIELD[i,j,k] = zonal[1,j,k] 
#    elif i == nx+1:
#        FIELD[i,j,k] = zonal[0,j,k] 
#
#    if j == 0:
#        FIELD[i,j,k] = merid[i,0,k] 
#    elif j == ny+1:
#        FIELD[i,j,k] = merid[i,1,k] 
#
#    cuda.syncthreads()









#def exchange_BC_periodic_x(GR, FIELD):
#
#    if np.ndim(FIELD) == 2:
#        dimx,dimy = FIELD.shape
#        binds = np.arange(0,GR.nb)
#
#        if dimx == GR.nx+2*GR.nb: # unstaggered in x
#            FIELD[binds,:] = FIELD[GR.nx+binds,:]
#            FIELD[GR.nx+GR.nb+binds,:] = FIELD[GR.nb+binds,:]
#        else: # staggered in x
#            FIELD[binds,:] = FIELD[GR.nxs+binds-1,:]
#            FIELD[GR.nxs+GR.nb+binds-1,:] = FIELD[GR.nb+binds,:]
#            #FIELD[GR.nxs+GR.nb+binds,:] = FIELD[GR.nb+binds+1,:]
#
#
#    elif np.ndim(FIELD) == 3:
#        dimx,dimy,dimz = FIELD.shape
#        binds = np.arange(0,GR.nb)
#
#        if dimx == GR.nx+2*GR.nb: # unstaggered in x
#            for k in range(0,dimz):
#                FIELD[binds,:,k] = FIELD[GR.nx+binds,:,k]
#                FIELD[GR.nx+GR.nb+binds,:,k] = FIELD[GR.nb+binds,:,k]
#        else: # staggered in x
#            for k in range(0,dimz):
#                FIELD[binds,:,k] = FIELD[GR.nxs+binds-1,:,k]
#                FIELD[GR.nxs+GR.nb+binds-1,:,k] = FIELD[GR.nb+binds,:,k]
#
#    return(FIELD)
#
#
#
#def exchange_BC_rigid_y(GR, FIELD):
#
#    if np.ndim(FIELD) == 2:
#        dimx,dimy = FIELD.shape
#        binds = np.arange(0,GR.nb)
#
#        if dimy == GR.ny+2*GR.nb: # unstaggered in y
#            for j in range(0,GR.nb):
#                FIELD[:,j] = FIELD[:,GR.nb]
#                FIELD[:,j+GR.ny+GR.nb] = FIELD[:,GR.ny+GR.nb-1]
#        else: # staggered in y
#            for j in range(0,GR.nb):
#                FIELD[:,j] = 0.
#                FIELD[:,j+1] = 0.
#                FIELD[:,j+GR.ny+GR.nb] = 0.
#                FIELD[:,j+GR.ny+GR.nb+1] = 0.
#
#    elif np.ndim(FIELD) == 3:
#        dimx,dimy,dimz = FIELD.shape
#        binds = np.arange(0,GR.nb)
#
#        if dimy == GR.ny+2*GR.nb: # unstaggered in y
#            for k in range(0,dimz):
#                for j in range(0,GR.nb):
#                    FIELD[:,j,k] = FIELD[:,GR.nb,k]
#                    FIELD[:,j+GR.ny+GR.nb,k] = FIELD[:,GR.ny+GR.nb-1,k]
#        else: # staggered in y
#            for k in range(0,dimz):
#                for j in range(0,GR.nb):
#                    FIELD[:,j,k] = 0.
#                    FIELD[:,j+1,k] = 0.
#                    FIELD[:,j+GR.ny+GR.nb,k] = 0.
#                    FIELD[:,j+GR.ny+GR.nb+1,k] = 0.
#
#    return(FIELD)
