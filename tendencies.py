from numba import njit, cuda
from temperature_cuda import POTT_tendency_gpu, POTT_tendency_cpu
from namelist import (i_POTT_main_switch)
from grid import tpb, bpg



class TendencyFactory:
    
    def __init__(self, target):
        self.target = target


    def POTT_tendency(self, dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                            POTTVB, WWIND, COLP_NEW, dsigma):
        if i_POTT_main_switch:
            if self.target == 'GPU':
                POTT_tendency_gpu[bpg, tpb](
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
                cuda.synchronize()
            elif self.target == 'CPU':
                POTT_tendency_cpu(
                        dPOTTdt, POTT, UFLX, VFLX, COLP, A,
                        POTTVB, WWIND, COLP_NEW, dsigma)
                cuda.synchronize()
        return(dPOTTdt)
