import numpy as np
import time
from constants import con_kappa, con_g, con_Rd


def diagnose_secondary_fields(GR, PAIR, PHI, POTT, TAIR, RHO,\
                                PVTF, PVTFVB, UWIND, VWIND, WIND):

    t_start = time.time()

    for k in range(0,GR.nz):
        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + \
                                        UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )

        PAIR[:,:,k][GR.iijj] = 100000*np.power(PVTF[:,:,k][GR.iijj], 1/con_kappa)

        TAIR[:,:,k][GR.iijj] = POTT[:,:,k][GR.iijj] / \
                np.power(100000/PAIR[:,:,k][GR.iijj], con_kappa)

        RHO[:,:,k][GR.iijj] = PAIR[:,:,k][GR.iijj] / \
                (con_Rd * TAIR[:,:,k][GR.iijj])


    t_end = time.time()
    GR.diag_comp_time += t_end - t_start

    return(PAIR, TAIR, RHO, WIND)
