import numpy as np
#import time
import multiprocessing as mp

from grid import Grid

def set_map_indices(GR, SGR, x_shift_fact):
    SGR.map_ii  = np.arange(0,SGR.nx +2*SGR.nb).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.map_iis = np.arange(0,SGR.nxs+2*SGR.nb).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.map_jj  = np.arange(0,SGR.ny +2*SGR.nb).astype(np.int)
    SGR.map_jjs = np.arange(0,SGR.nys+2*SGR.nb).astype(np.int)

    SGR.map_iijj  = np.ix_(SGR.map_ii   ,SGR.map_jj  )
    SGR.map_iisjj = np.ix_(SGR.map_iis  ,SGR.map_jj  )
    SGR.map_iijjs = np.ix_(SGR.map_ii   ,SGR.map_jjs  )

    SGR.mapin_ii  = np.arange(0,SGR.nx ).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.mapin_iis = np.arange(0,SGR.nxs).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.mapin_jj  = np.arange(0,SGR.ny ).astype(np.int)
    SGR.mapin_jjs = np.arange(0,SGR.nys).astype(np.int)

    SGR.mapin_iijj  = np.ix_(SGR.mapin_ii   ,SGR.mapin_jj  )
    SGR.mapin_iisjj = np.ix_(SGR.mapin_iis  ,SGR.mapin_jj  )
    SGR.mapin_iijjs = np.ix_(SGR.mapin_ii   ,SGR.mapin_jjs  )
    return(SGR)

def create_subgrids(GR, njobs):
    subgrids = {}
    
    ########################################################################
    if njobs == 1:
        GR = set_map_indices(GR, GR, 0)
        subgrids[0] = GR
    ########################################################################
    if njobs == 2:
        # subgrid 0
        specs = {}
        specs['lon0_deg'] = GR.lon0_deg
        specs['lon1_deg'] = (GR.lon1_deg - GR.lon0_deg)/2
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 0)
        subgrids[0] = SGR

        # subgrid 1
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)/2)
        specs['lon1_deg'] = GR.lon1_deg
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 1/2)
        subgrids[1] = SGR 

        print(subgrids[0].ii)
        print(subgrids[1].ii)
        quit()
    ########################################################################
    if njobs == 3:
        # subgrid 0
        specs = {}
        specs['lon0_deg'] = GR.lon0_deg
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/3)
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 0)
        subgrids[0] = SGR

        # subgrid 1
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/3)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/3) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 1/3)
        subgrids[1] = SGR 

        # subgrid 2
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/3)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*3/3) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 2/3)
        subgrids[2] = SGR 
    ########################################################################
    if njobs == 4:
        # subgrid 0
        specs = {}
        specs['lon0_deg'] = GR.lon0_deg
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/4)
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 0)
        subgrids[0] = SGR

        # subgrid 1
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/4)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/4) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 1/4)
        subgrids[1] = SGR 

        # subgrid 2
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/4)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*3/4) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 2/4)
        subgrids[2] = SGR 

        # subgrid 3
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*3/4)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*4/4) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 3/4)
        subgrids[3] = SGR 
    ########################################################################

    return(subgrids)


if __name__ == '__main__':


    GR = Grid()

    from namelist import njobs
    subgrids = create_subgrids(GR, njobs)
