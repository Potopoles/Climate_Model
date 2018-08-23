import numpy as np
import random
from namelist import njobs

def give_to_helix(GR, VAR, var_name, helix_inds, helix, stagx=False):
    # staggered
    #FIELD[GR.nb,:,k]
    #FIELD[GR.nxs-1,:,k]
    # unstaggered
    #FIELD[GR.nb,:]
    #FIELD[GR.nx,:]

    VAR_left  = VAR[GR.nb,1:-1].flatten()
    if stagx:
        VAR_right = VAR[GR.nxs-1,1:-1].flatten()
    else:
        VAR_right = VAR[GR.nx   ,1:-1].flatten()

    inds = helix_inds['give_'+str(var_name)]
    #print(var_name+' ' + str(VAR_left.shape))
    #print(var_name+' ' + str(range(inds[0][0],inds[0][1])))
    helix[inds[0][0]:inds[0][1]] = VAR_left
    helix[inds[1][0]:inds[1][1]] = VAR_right

    return(helix)


def take_from_helix(GR, VAR, var_name, helix_inds, helix, stagx=False):
    ## staggered
    ##FIELD[0,:,k] 
    ##FIELD[GR.nxs+GR.nb,:,k]
    ####FIELD[GR.nxs+GR.nb-1,:,k] # why is this version wrong?
    ## unstaggered
    ##FIELD[0,:,k]
    ##FIELD[GR.nx+GR.nb,:]

    inds = helix_inds['take_'+str(var_name)]
    VAR_left  = helix[inds[0][0]:inds[0][1]]
    VAR_right = helix[inds[1][0]:inds[1][1]]

    shape = VAR[0,1:-1].shape
    VAR[0,1:-1]              = np.asarray(VAR_left ).reshape(shape)
    if stagx:
        VAR[GR.nxs+GR.nb,1:-1] = np.asarray(VAR_right).reshape(shape) 
    else:
        VAR[GR.nx +GR.nb,1:-1] = np.asarray(VAR_right).reshape(shape) 

    return(VAR)


def exchange_BC_prog(GR,
                    UWIND, VWIND, POTT, QV, QC,
                    helix, helix_inds,
                    barrier, status, lock):

    helix = give_to_helix(GR, UWIND, 'UWIND', helix_inds, helix, stagx=True)
    helix = give_to_helix(GR, VWIND, 'VWIND', helix_inds, helix, stagx=False)
    helix = give_to_helix(GR, POTT,  'POTT',  helix_inds, helix, stagx=False)
    helix = give_to_helix(GR, QV,    'QV',    helix_inds, helix, stagx=False)
    helix = give_to_helix(GR, QC,    'QC',    helix_inds, helix, stagx=False)
    barrier.wait()

    UWIND = take_from_helix(GR, UWIND, 'UWIND', helix_inds, helix, stagx=True)
    VWIND = take_from_helix(GR, VWIND, 'VWIND', helix_inds, helix, stagx=False)
    POTT  = take_from_helix(GR, POTT,  'POTT',  helix_inds, helix, stagx=False)
    QV    = take_from_helix(GR, QV,    'QV',    helix_inds, helix, stagx=False)
    QC    = take_from_helix(GR, QC,    'QC',    helix_inds, helix, stagx=False)

    UWIND = exchange_BC_rigid_y(GR, UWIND)
    VWIND = exchange_BC_rigid_y(GR, VWIND)
    POTT  = exchange_BC_rigid_y(GR, POTT )
    QV    = exchange_BC_rigid_y(GR, QV   )
    QC    = exchange_BC_rigid_y(GR, QC   )
    barrier.wait()

    return(UWIND, VWIND, POTT, QV, QC)


def exchange_BC_brflx(GR,
                        BFLX, CFLX, DFLX, EFLX,
                        RFLX, QFLX, SFLX, TFLX,
                        helix, helix_inds,
                        barrier, status, lock):

    #cdef double[:,:, ::1] BFLX = np.zeros( (nx +2*nb,ny +2*nb,nz) )
    helix = give_to_helix(GR, BFLX, 'BFLX', helix_inds, helix, stagx=False)
    #cdef double[:,:, ::1] CFLX = np.zeros( (nxs+2*nb,nys+2*nb,nz) )
    helix = give_to_helix(GR, CFLX, 'CFLX', helix_inds, helix, stagx=True)
    #cdef double[:,:, ::1] DFLX = np.zeros( (nx +2*nb,nys+2*nb,nz) )
    helix = give_to_helix(GR, DFLX, 'DFLX', helix_inds, helix, stagx=False)
    #cdef double[:,:, ::1] EFLX = np.zeros( (nx +2*nb,nys+2*nb,nz) )
    helix = give_to_helix(GR, EFLX, 'EFLX', helix_inds, helix, stagx=False)

    #cdef double[:,:, ::1] RFLX = np.zeros( (nx +2*nb,ny +2*nb,nz) )
    helix = give_to_helix(GR, RFLX, 'RFLX', helix_inds, helix, stagx=False)
    #cdef double[:,:, ::1] QFLX = np.zeros( (nxs+2*nb,nys+2*nb,nz) )
    helix = give_to_helix(GR, QFLX, 'QFLX', helix_inds, helix, stagx=True)
    #cdef double[:,:, ::1] SFLX = np.zeros( (nxs+2*nb,ny +2*nb,nz) )
    helix = give_to_helix(GR, SFLX, 'SFLX', helix_inds, helix, stagx=True)
    #cdef double[:,:, ::1] TFLX = np.zeros( (nxs+2*nb,ny +2*nb,nz) )
    helix = give_to_helix(GR, TFLX, 'TFLX', helix_inds, helix, stagx=True)

    barrier.wait()

    #cdef double[:,:, ::1] BFLX = np.zeros( (nx +2*nb,ny +2*nb,nz) )
    BFLX = take_from_helix(GR, BFLX, 'BFLX', helix_inds, helix, stagx=False)
    #cdef double[:,:, ::1] CFLX = np.zeros( (nxs+2*nb,nys+2*nb,nz) )
    CFLX = take_from_helix(GR, CFLX, 'CFLX', helix_inds, helix, stagx=True)
    #cdef double[:,:, ::1] DFLX = np.zeros( (nx +2*nb,nys+2*nb,nz) )
    DFLX = take_from_helix(GR, DFLX, 'DFLX', helix_inds, helix, stagx=False)
    #cdef double[:,:, ::1] EFLX = np.zeros( (nx +2*nb,nys+2*nb,nz) )
    EFLX = take_from_helix(GR, EFLX, 'EFLX', helix_inds, helix, stagx=False)

    #cdef double[:,:, ::1] RFLX = np.zeros( (nx +2*nb,ny +2*nb,nz) )
    RFLX = take_from_helix(GR, RFLX, 'RFLX', helix_inds, helix, stagx=False)
    #cdef double[:,:, ::1] QFLX = np.zeros( (nxs+2*nb,nys+2*nb,nz) )
    QFLX = take_from_helix(GR, QFLX, 'QFLX', helix_inds, helix, stagx=True)
    #cdef double[:,:, ::1] SFLX = np.zeros( (nxs+2*nb,ny +2*nb,nz) )
    SFLX = take_from_helix(GR, SFLX, 'SFLX', helix_inds, helix, stagx=True)
    #cdef double[:,:, ::1] TFLX = np.zeros( (nxs+2*nb,ny +2*nb,nz) )
    TFLX = take_from_helix(GR, TFLX, 'TFLX', helix_inds, helix, stagx=True)


    BFLX = exchange_BC_rigid_y(GR, BFLX)
    CFLX = exchange_BC_rigid_y(GR, CFLX)
    DFLX = exchange_BC_rigid_y(GR, DFLX)
    EFLX = exchange_BC_rigid_y(GR, EFLX)
    RFLX = exchange_BC_rigid_y(GR, RFLX)
    QFLX = exchange_BC_rigid_y(GR, QFLX)
    SFLX = exchange_BC_rigid_y(GR, SFLX)
    TFLX = exchange_BC_rigid_y(GR, TFLX)
    barrier.wait()


    return(BFLX, CFLX, DFLX, EFLX,
            RFLX, QFLX, SFLX, TFLX)


def exchange_BC_contin(GR, COLP, WWIND, helix, helix_inds,
                            barrier, status, lock):

    helix = give_to_helix(GR, COLP, 'COLP', helix_inds, helix, stagx=False)
    helix = give_to_helix(GR, WWIND, 'WWIND', helix_inds, helix, stagx=False)
    barrier.wait()

    COLP = take_from_helix(GR, COLP, 'COLP', helix_inds, helix, stagx=False)
    WWIND = take_from_helix(GR, WWIND, 'WWIND', helix_inds, helix, stagx=False)

    COLP = exchange_BC_rigid_y(GR, COLP)
    WWIND = exchange_BC_rigid_y(GR, WWIND)
    barrier.wait()

    return(COLP, WWIND)

def exchange_BC_uvflx(GR, UFLX, VFLX, helix, helix_inds,
                        barrier, status, lock):

    helix = give_to_helix(GR, UFLX, 'UFLX', helix_inds, helix, stagx=True)
    helix = give_to_helix(GR, VFLX, 'VFLX', helix_inds, helix, stagx=False)
    barrier.wait()

    UFLX = take_from_helix(GR, UFLX, 'UFLX', helix_inds, helix, stagx=True)
    VFLX = take_from_helix(GR, VFLX, 'VFLX', helix_inds, helix, stagx=False)

    UFLX = exchange_BC_rigid_y(GR, UFLX)
    VFLX = exchange_BC_rigid_y(GR, VFLX)
    barrier.wait()

    ################## PUT INTO HELIX
    #################################
    ## staggered
    ##FIELD[GR.nb,:,k]
    ##FIELD[GR.nxs-1,:,k]
    ## unstaggered
    ##FIELD[GR.nx,:]
    ##FIELD[GR.nb,:]

    #UFLX_left  = UFLX[GR.nb,1:-1].flatten()
    #UFLX_right = UFLX[GR.nxs-1,1:-1].flatten()
    #VFLX_left  = VFLX[GR.nb,1:-1].flatten()
    #VFLX_right = VFLX[GR.nx,1:-1].flatten()

    #inds = helix_inds['give_UFLX']
    #helix[inds[0][0]:inds[0][1]] = UFLX_left
    #helix[inds[1][0]:inds[1][1]] = UFLX_right
    #inds = helix_inds['give_VFLX']
    #helix[inds[0][0]:inds[0][1]] = VFLX_left
    #helix[inds[1][0]:inds[1][1]] = VFLX_right

    #barrier.wait()
    ################# TAKE FROM HELIX
    #################################
    #inds = helix_inds['take_UFLX']
    #UFLX_left  = helix[inds[0][0]:inds[0][1]]
    #UFLX_right = helix[inds[1][0]:inds[1][1]]
    #inds = helix_inds['take_VFLX']
    #VFLX_left  = helix[inds[0][0]:inds[0][1]]
    #VFLX_right = helix[inds[1][0]:inds[1][1]]

    ## staggered
    ##FIELD[0,:,k] 
    ##FIELD[GR.nxs+GR.nb-1,:,k]
    ## unstaggered
    ##FIELD[0,:]
    ##FIELD[GR.nx+GR.nb,:]

    #shape = UFLX[0,1:-1].shape
    #UFLX[0,1:-1]              = np.asarray(UFLX_left ).reshape(shape)
    ##UFLX[GR.nxs+GR.nb-1,1:-1] = np.asarray(UFLX_right).reshape(shape) 
    #UFLX[GR.nxs+GR.nb,1:-1] = np.asarray(UFLX_right).reshape(shape) 
    #shape = VFLX[0,1:-1].shape
    #VFLX[0,1:-1]              = np.asarray(VFLX_left ).reshape(shape)
    #VFLX[GR.nx +GR.nb,1:-1] = np.asarray(VFLX_right).reshape(shape) 

    return(UFLX, VFLX)





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







