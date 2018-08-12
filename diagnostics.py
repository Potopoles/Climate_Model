import numpy as np
import time
from constants import con_kappa, con_g, con_Rd
from namelist import pTop


def diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB, TAIR, TAIRVB, RHO,\
                                PVTF, PVTFVB, UWIND, VWIND, WIND):

    t_start = time.time()

    TAIR[GR.iijj] = POTT[GR.iijj] * PVTF[GR.iijj]
    TAIRVB[GR.iijj] = POTTVB[GR.iijj] * PVTFVB[GR.iijj]
    PAIR[GR.iijj] = 100000*np.power(PVTF[GR.iijj], 1/con_kappa)
    RHO[GR.iijj] = PAIR[GR.iijj] / (con_Rd * TAIR[GR.iijj])

    for k in range(0,GR.nz):
        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + \
                                        UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )


    t_end = time.time()
    GR.diag_comp_time += t_end - t_start

    return(PAIR, TAIR, TAIRVB, RHO, WIND)
