import numpy as np

def exchange_BC(GR, FIELD):
    FIELD = exchange_BC_periodic_x(GR, FIELD)
    FIELD = exchange_BC_rigid_y(GR, FIELD)
    return(FIELD)

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







#def exchange_BC_rigid_y_horDifSpecial(GR, FIELD):
#
#    if np.ndim(FIELD) == 2:
#        dimx,dimy = FIELD.shape
#        binds = np.arange(0,GR.nb)
#
#        if dimy == GR.ny+2*GR.nb: # unstaggered in y
#            raise NotImplementedError()
#        else: # staggered in y
#            for j in range(0,GR.nb):
#                FIELD[:,j] = FIELD[:,j+2] 
#                FIELD[:,j+1] = FIELD[:,j+2]
#                FIELD[:,j+GR.ny+GR.nb] = FIELD[:,j+GR.ny+GR.nb-1]
#                FIELD[:,j+GR.ny+GR.nb+1] = FIELD[:,j+GR.ny+GR.nb-1] 
#
#    elif np.ndim(FIELD) == 3:
#        dimx,dimy,dimz = FIELD.shape
#        binds = np.arange(0,GR.nb)
#
#        if dimy == GR.ny+2*GR.nb: # unstaggered in y
#            raise NotImplementedError()
#        else: # staggered in y
#            for k in range(0,dimz):
#                for j in range(0,GR.nb):
#                    FIELD[:,j,k] = FIELD[:,j+2,k] 
#                    FIELD[:,j+1,k] = FIELD[:,j+2,k]
#                    FIELD[:,j+GR.ny+GR.nb,k] = FIELD[:,j+GR.ny+GR.nb-1,k]
#                    FIELD[:,j+GR.ny+GR.nb+1,k] = FIELD[:,j+GR.ny+GR.nb-1,k] 
#
#    return(FIELD)
