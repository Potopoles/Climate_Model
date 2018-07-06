import numpy as np

def exchange_BC(GR, FIELD):
    FIELD = exchange_BC_periodic_x(GR, FIELD)
    FIELD = exchange_BC_rigid_y(GR, FIELD)
    return(FIELD)

def exchange_BC_all(GR, COLP, UWIND, VWIND, TAIR):

    #VWIND[GR.ii,1] = 1
    #VWIND[GR.ii,2] = 2
    #VWIND[GR.ii,6] = 6
    
    COLP = exchange_BC_periodic_x(GR, COLP)
    UWIND = exchange_BC_periodic_x(GR, UWIND)
    VWIND = exchange_BC_periodic_x(GR, VWIND)
    TAIR = exchange_BC_periodic_x(GR, TAIR)

    COLP = exchange_BC_rigid_y(GR, COLP)
    UWIND = exchange_BC_rigid_y(GR, UWIND)
    VWIND = exchange_BC_rigid_y(GR, VWIND)
    TAIR = exchange_BC_rigid_y(GR, TAIR)

    return(COLP, UWIND, VWIND, TAIR)







def exchange_BC_periodic_x(GR, FIELD):

    dimx,dimy = FIELD.shape
    binds = np.arange(0,GR.nb)

    if dimx == GR.nx+2*GR.nb: # unstaggered in x
        FIELD[binds,:] = FIELD[GR.nx+binds,:]
        FIELD[GR.nx+GR.nb+binds,:] = FIELD[GR.nb+binds,:]
    else: # staggered in x
        FIELD[binds,:] = FIELD[GR.nxs+binds-1,:]
        FIELD[GR.nxs+GR.nb+binds-1,:] = FIELD[GR.nb+binds,:]
        #FIELD[GR.nxs+GR.nb+binds,:] = FIELD[GR.nb+binds+1,:]

    return(FIELD)



def exchange_BC_rigid_y(GR, FIELD):

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
            FIELD[:,j+GR.ny+GR.nb] = 0.
            FIELD[:,j+GR.ny+GR.nb+1] = 0.

    return(FIELD)
