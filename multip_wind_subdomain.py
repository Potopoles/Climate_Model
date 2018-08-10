import numpy as np
import time
import copy
import multiprocessing as mp
import matplotlib.pyplot as plt


from namelist import *
from grid import Grid
from fields import initialize_fields
from wind import wind_tendency_jacobson

from multiproc import create_subgrids 


if __name__ == '__main__':


    GR = Grid()
    subgrids = create_subgrids(GR, njobs)

    COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
    UFLX, VFLX, UFLXMP, VFLXMP, \
    HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
    RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids)

    COLP_NEW = copy.deepcopy(COLP)


    time0 = time.time()

    output = mp.Queue()
    processes = []
    for job_ind in range(0,njobs):

        SGR = subgrids[job_ind]

        #print(UFLX[SGR.map_iisjj])

        processes.append(
            mp.Process(\
                target=wind_tendency_jacobson,
                args = (job_ind, output, subgrids[job_ind],
                        UWIND[SGR.map_iisjj], VWIND[SGR.map_iijjs],
                        WWIND[SGR.map_iijj], 
                        UFLX[SGR.map_iisjj], VFLX[SGR.map_iijjs],
                        COLP[SGR.map_iijj], COLP_NEW[SGR.map_iijj],
                        HSURF[SGR.map_iijj], PHI[SGR.map_iijj],
                        POTT[SGR.map_iijj], PVTF[SGR.map_iijj],
                        PVTFVB[SGR.map_iijj])))

    for proc in processes:
        proc.start()

    results = [output.get() for p in processes]
    results.sort()
    for job_ind in range(0,njobs):
        duflxdt = results[job_ind][1]['dUFLXdt']
        dvflxdt = results[job_ind][1]['dVFLXdt']
        #print(duflxdt[5,5,5])

    for proc in processes:
        proc.join()


    time1 = time.time()
    print(time1 - time0)

