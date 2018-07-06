import numpy as np
from constants import con_g, con_Rd
from namelist import ptop
from boundaries import exchange_BC


def diag_geopotential(GR, PHI, HSURF, TAIR, COLP):

    dsig = 1
    sigma = 0.5
    PHI[GR.iijj] = HSURF[GR.iijj]*con_g + con_Rd*TAIR[GR.iijj]*dsig*COLP[GR.iijj] /\
                    (COLP[GR.iijj]*sigma + ptop)

    PHI = exchange_BC(GR, PHI)

    return(PHI)
