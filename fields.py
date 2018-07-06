import numpy as np
from namelist import *
from boundaries import exchange_BC_all, exchange_BC_rigid_y, exchange_BC_periodic_x
from IO import load_topo
from geopotential import diag_geopotential

def initialize_fields(GR):
    # CREATE ARRAYS
    # pressure and geopotential
    COLP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    PHI = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    # wind velocities
    UWIND = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    VWIND = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)
    WIND = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    # mass fluxes at velocity points
    UFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    VFLX = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)
    # mass fluxes at mass points
    UFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    UFLXMP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    VFLXMP = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    # momentum fluxes at velocity points
    UUFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    VUFLX = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)
    UVFLX = np.full( (GR.nxs+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    VVFLX = np.full( (GR.nx+2*GR.nb,GR.nys+2*GR.nb), np.nan)
    # temperature
    TAIR = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    # surface height
    HSURF = load_topo(GR) 
    #HSURF[GR.iijj] = 0.
    #HSURF = exchange_BC_periodic_x(GR, HSURF)
    #HSURF = exchange_BC_rigid_y(GR, HSURF)

    # INITIAL CONDITIONS
    COLP[GR.iijj] = psurf - ptop 

    UWIND[GR.iisjj] = u0   
    VWIND[GR.iijjs] = 0.

    COLP = gaussian2D(GR, COLP, COLP_gauss_pert, np.pi*3/4, 0, np.pi/10, np.pi/10)

    TAIR[GR.iijj] = 760.
    TAIR = gaussian2D(GR, TAIR, TAIR_gauss_pert, np.pi*3/4, 0, np.pi/10, np.pi/10)
    TAIR = random2D(GR, TAIR, TAIR_rand_pert)

    PHI = diag_geopotential(GR, PHI, HSURF, TAIR, COLP)



    # BOUNDARY CONDITIONS
    COLP, UWIND, VWIND, TAIR = exchange_BC_all(GR, COLP, UWIND, VWIND, TAIR)

    return(COLP, PHI, UWIND, VWIND, WIND,
            UFLX, VFLX, UFLXMP, VFLXMP,
            UUFLX, VUFLX, UVFLX, VVFLX,
            HSURF, TAIR)



def random2D(GR, FIELD, pert):
    FIELD = FIELD + pert*np.random.rand(FIELD.shape[0], FIELD.shape[1])
    return(FIELD)

def gaussian2D(GR, FIELD, pert, lon0_rad, lat0_rad, lonSig_rad, latSig_rad):

    dimx,dimy = FIELD.shape

    if (dimy == GR.nys+2*GR.nb): # staggered in y 
        selinds = GR.iijjs
    elif (dimx == GR.nxs+2*GR.nb): # staggered in x 
        selinds = GR.iisjj
    else: # unstaggered in y and x 
        selinds = GR.iijj

    perturb = pert*np.exp( \
            - np.power(GR.lonjs_rad[selinds] - lon0_rad, 2)/(2*lonSig_rad**2) \
            - np.power(GR.latjs_rad[selinds] - lat0_rad, 2)/(2*latSig_rad**2) )
    FIELD[selinds] = FIELD[selinds] + perturb

    return(FIELD)


















#g%sigmaVB = REAL((kks - 1))/nz
#g%dsig = g%sigmaVB(kk+1) - g%sigmaVB(kk)
#g%sigma = g%sigmaVB(kk) + g%dsig/2 
#
#! READ IN TABLE WITH VALUES FOR INTERPOLATION
#nRows = 0
#open(unit=1, file='verticalProfileTable.dat', status='old')
#DO
#      read(1,*,iostat=stat) b
#      IF (stat < 0) exit
#      IF (stat /= 0) stop 'ERROR READING verticalProfileTable.dat!'
#      nRows = nRows + 1
#END DO
#rewind(1)
#allocate( tab_z(nRows), tab_g(nRows), tab_p(nRows), tab_T(nRows), &
#          tab_rho(nRows) )
#DO c = 1,nRows
#      read(1,*) tab_z(c), tab_g(c), tab_p(c), tab_T(c), tab_rho(c)
#END DO
#
#!! TODO
#!! VERTICAL INTERPOLATION IS DONE LINEARLY WHICH IS NOT GOOD.
#!! CHANGE ACCORDING TO BOOK FUNDAMENTALS OF ATMOSPHERIC MODELING
#DO i = (1+nb),(nx+nb)
#DO j = (1+nb),(ny+nb)
#    P_SURF(i,j) = interp1d(tab_z,tab_p,H_SURF(i,j))
#    T_SURF(i,j) = interp1d(tab_z,tab_T,H_SURF(i,j))
#    COL_P(i,j) = P_SURF(i,j) - g%pTop
#    P_AIR(i,j,kk) = COL_P(i,j)*g%sigma
#    DO k = 1,nz
#        T_AIR(i,j,k) = interp1d(tab_p,tab_T,P_AIR(i,j,k))
#
#        ! MERIDIONAL GRADIENT
#        !T_AIR(i,j,k) = T_AIR(i,j,k) + 150*(cos(abs(g%lat_rad(i,j))) - 0.5)
#    END DO
#END DO
#END DO
