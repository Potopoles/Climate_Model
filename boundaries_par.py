import numpy as np
import time
import random
from namelist import njobs

def exchange_BC_uvflx(GR, UFLX, VFLX, helix, helix_inds,
                        barrier, status, lock):

    ################# PUT INTO HELIX
    ################################
    # staggered
    #FIELD[GR.nb,:,k]
    #FIELD[GR.nxs-1,:,k]
    # unstaggered
    #FIELD[GR.nx,:]
    #FIELD[GR.nb,:]

    UFLX_left  = UFLX[GR.nb,1:-1].flatten()
    #UFLX_right = UFLX[GR.nxs+GR.nb-1,1:-1].flatten()
    UFLX_right = UFLX[GR.nxs-1,1:-1].flatten()
    VFLX_left  = VFLX[GR.nb,1:-1].flatten()
    #VFLX_right = VFLX[GR.nx +GR.nb-1,1:-1].flatten()
    VFLX_right = VFLX[GR.nx,1:-1].flatten()

    inds = helix_inds['give_UFLX']
    helix[inds[0][0]:inds[0][1]] = UFLX_left
    helix[inds[1][0]:inds[1][1]] = UFLX_right
    inds = helix_inds['give_VFLX']
    helix[inds[0][0]:inds[0][1]] = VFLX_left
    helix[inds[1][0]:inds[1][1]] = VFLX_right
    #print('nans in helix: ' + str(np.sum(np.isnan(np.asarray(helix[:])))))

    barrier.wait()
    ################ TAKE FROM HELIX
    ################################
    inds = helix_inds['take_UFLX']
    UFLX_left  = helix[inds[0][0]:inds[0][1]]
    UFLX_right = helix[inds[1][0]:inds[1][1]]
    inds = helix_inds['take_VFLX']
    VFLX_left  = helix[inds[0][0]:inds[0][1]]
    VFLX_right = helix[inds[1][0]:inds[1][1]]

    # staggered
    #FIELD[0,:,k] 
    #FIELD[GR.nxs+GR.nb-1,:,k]
    # unstaggered
    #FIELD[0,:]
    #FIELD[GR.nx+GR.nb,:]

    uflx_shape = UFLX[0,1:-1].shape
    UFLX[0,1:-1]              = np.asarray(UFLX_left ).reshape(uflx_shape)
    UFLX[GR.nxs+GR.nb-1,1:-1] = np.asarray(UFLX_right).reshape(uflx_shape) 
    vflx_shape = VFLX[0,1:-1].shape
    VFLX[0,1:-1]              = np.asarray(VFLX_left ).reshape(vflx_shape)
    VFLX[GR.nx +GR.nb  ,1:-1] = np.asarray(VFLX_right).reshape(vflx_shape) 

    #status.value += 1
    #if status.value == 1:
    #    print('first ' + str(status.value))
    #elif status.value == njobs:
    #    print('last ' + str(status.value))
    #    status.value = 0
    #else:
    #    print(status.value)
    #lock.release()

    UFLX = exchange_BC_rigid_y(GR, UFLX)
    VFLX = exchange_BC_rigid_y(GR, VFLX)

    barrier.wait()
    return(UFLX, VFLX)

def exchange_BC_par(GR, UFLX, VFLX, helix, helix_inds):
    pass

def exchange_BC_force_y_equal(GR, FIELD):
    FIELD = exchange_BC_periodic_x(GR, FIELD)
    FIELD = exchange_BC_rigid_y_force_y_equal(GR, FIELD)
    return(FIELD)

def exchange_BC_all(GR, COLP, UWIND, VWIND, POTT):

    COLP = exchange_BC_periodic_x(GR, COLP)
    UWIND = exchange_BC_periodic_x(GR, UWIND)
    VWIND = exchange_BC_periodic_x(GR, VWIND)
    POTT = exchange_BC_periodic_x(GR, POTT)

    COLP = exchange_BC_rigid_y(GR, COLP)
    UWIND = exchange_BC_rigid_y(GR, UWIND)
    VWIND = exchange_BC_rigid_y(GR, VWIND)
    POTT = exchange_BC_rigid_y(GR, POTT)

    return(COLP, UWIND, VWIND, POTT)







def exchange_BC_periodic_x(GR, FIELD):

    if np.ndim(FIELD) == 2:
        dimx,dimy = FIELD.shape
        binds = np.arange(0,GR.nb)

        if dimx == GR.nx+2*GR.nb: # unstaggered in x
            FIELD[binds,:] = FIELD[GR.nx+binds,:]
            FIELD[GR.nx+GR.nb+binds,:] = FIELD[GR.nb+binds,:]
        else: # staggered in x
            FIELD[binds,:] = FIELD[GR.nxs+binds-1,:]
            FIELD[GR.nxs+GR.nb+binds-1,:] = FIELD[GR.nb+binds,:]
            #FIELD[GR.nxs+GR.nb+binds,:] = FIELD[GR.nb+binds+1,:]


    elif np.ndim(FIELD) == 3:
        dimx,dimy,dimz = FIELD.shape
        binds = np.arange(0,GR.nb)

        if dimx == GR.nx+2*GR.nb: # unstaggered in x
            for k in range(0,dimz):
                FIELD[binds,:,k] = FIELD[GR.nx+binds,:,k]
                FIELD[GR.nx+GR.nb+binds,:,k] = FIELD[GR.nb+binds,:,k]
        else: # staggered in x
            for k in range(0,dimz):
                FIELD[binds,:,k] = FIELD[GR.nxs+binds-1,:,k]
                FIELD[GR.nxs+GR.nb+binds-1,:,k] = FIELD[GR.nb+binds,:,k]

    return(FIELD)



def exchange_BC_rigid_y(GR, FIELD):

    if np.ndim(FIELD) == 2:
        dimx,dimy = FIELD.shape
        binds = np.arange(0,GR.nb)

        if dimy == GR.ny+2*GR.nb: # unstaggered in y
            for j in range(0,GR.nb):
                FIELD[:,j] = FIELD[:,GR.nb]
                FIELD[:,j+GR.ny+GR.nb] = FIELD[:,GR.ny+GR.nb-1]
        else: # staggered in y
            for j in range(0,GR.nb):
                FIELD[:,j] = 0.
                FIELD[:,j+1] = 0.
                FIELD[:,j+GR.ny+GR.nb] = 0.
                FIELD[:,j+GR.ny+GR.nb+1] = 0.

    elif np.ndim(FIELD) == 3:
        dimx,dimy,dimz = FIELD.shape
        binds = np.arange(0,GR.nb)

        if dimy == GR.ny+2*GR.nb: # unstaggered in y
            for k in range(0,dimz):
                for j in range(0,GR.nb):
                    FIELD[:,j,k] = FIELD[:,GR.nb,k]
                    FIELD[:,j+GR.ny+GR.nb,k] = FIELD[:,GR.ny+GR.nb-1,k]
        else: # staggered in y
            for k in range(0,dimz):
                for j in range(0,GR.nb):
                    FIELD[:,j,k] = 0.
                    FIELD[:,j+1,k] = 0.
                    FIELD[:,j+GR.ny+GR.nb,k] = 0.
                    FIELD[:,j+GR.ny+GR.nb+1,k] = 0.

    return(FIELD)







