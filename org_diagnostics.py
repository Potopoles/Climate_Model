import numpy as np
from diagnostics import diagnose_secondary_fields
from bin.diagnostics_cython import diagnose_secondary_fields_c
from diagnostics_cuda import diagnose_secondary_fields_gpu
from diagnostics_cuda import diagnose_secondary_fields_vb_gpu
from namelist import comp_mode

def secondary_diagnostics(GR, F):
    
    if comp_mode == 0:
        F.PAIR, F.TAIR, F.TAIRVB, F.RHO, F.WIND = \
                diagnose_secondary_fields(GR, F.COLP, F.PAIR, F.PHI, F.POTT, F.POTTVB,
                                        F.TAIR, F.TAIRVB, F.RHO,
                                        F.PVTF, F.PVTFVB, F.UWIND, F.VWIND, F.WIND)

    elif comp_mode == 1:
        PAIR, TAIR, TAIRVB, RHO, WIND = \
                diagnose_secondary_fields_c(GR, F.COLP, F.PAIR, F.PHI, F.POTT, F.POTTVB,
                                        F.TAIR, F.TAIRVB, F.RHO,
                                        F.PVTF, F.PVTFVB, F.UWIND, F.VWIND, F.WIND)
        F.PAIR = np.asarray(PAIR)
        F.TAIR = np.asarray(TAIR)
        F.TAIRVB = np.asarray(TAIRVB)
        F.RHO = np.asarray(RHO)
        F.WIND = np.asarray(WIND)

    elif comp_mode == 2:
        diagnose_secondary_fields_gpu[GR.griddim, GR.blockdim, GR.stream] \
                                (F.COLP, F.PAIR, F.PHI, F.POTT, F.POTTVB,
                                F.TAIR, F.RHO,
                                F.PVTF, F.PVTFVB, F.UWIND, F.VWIND, F.WIND)
        diagnose_secondary_fields_vb_gpu[GR.griddim_ks, GR.blockdim_ks, GR.stream] \
                                (F.POTTVB, F.TAIRVB, F.PVTFVB)

