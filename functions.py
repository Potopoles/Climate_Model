import numpy as np


def diagnostics(GR, WIND, UWIND, VWIND, COLP, POTT):
    vmax = 0
    mean_wind = 0
    mean_temp = 0
    for k in range(0,GR.nz):
        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )
        vmax = max(vmax, np.max(WIND[:,:,k][GR.iijj]))
        mean_wind += np.sum( WIND[:,:,k][GR.iijj]*COLP[GR.iijj]*GR.A[GR.iijj] ) / \
                        np.sum( COLP[GR.iijj]*GR.A[GR.iijj] )
        mean_temp += np.sum(POTT[:,:,k][GR.iijj]*GR.A[GR.iijj]*COLP[GR.iijj])/ \
                np.sum(GR.A[GR.iijj]*COLP[GR.iijj])
    mean_wind = mean_wind/GR.nz
    mean_temp = mean_temp/GR.nz

    mean_colp = np.sum(COLP[GR.iijj]*GR.A[GR.iijj])/np.sum(GR.A[GR.iijj])

    return(WIND, vmax, mean_wind, mean_temp, mean_colp)
