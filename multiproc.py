import numpy as np
import copy
#import time
import multiprocessing as mp
from namelist import njobs

from grid import Grid

def set_map_indices(GR, SGR, grid_no, ngrids):
    x_shift_fact = grid_no / ngrids
    # GRID MAP OUT
    SGR.GRmap_out_ii  = np.arange(0,SGR.nx +2*SGR.nb).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.GRmap_out_iis = np.arange(0,SGR.nxs+2*SGR.nb).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.GRmap_out_jj  = np.arange(0,SGR.ny +2*SGR.nb).astype(np.int)
    SGR.GRmap_out_jjs = np.arange(0,SGR.nys+2*SGR.nb).astype(np.int)
    SGR.GRmap_out_iijj      = np.ix_(SGR.GRmap_out_ii        ,SGR.GRmap_out_jj  )
    SGR.GRmap_out_iisjj     = np.ix_(SGR.GRmap_out_iis       ,SGR.GRmap_out_jj  )
    SGR.GRmap_out_iijjs     = np.ix_(SGR.GRmap_out_ii        ,SGR.GRmap_out_jjs  )

    # GRID MAP OUT INNER (WITHOUT BORDERS OF SGR ARRAY)
    SGR.GRimap_out_ii  = np.arange(0,SGR.nx ).astype(np.int) + int(GR.nx*x_shift_fact)
    SGR.GRimap_out_jj  = np.arange(0,SGR.ny ).astype(np.int)
    SGR.GRimap_out_iijj  = np.ix_(SGR.GRimap_out_ii, SGR.GRimap_out_jj )

    # GRID MAP IN 
    ind_range = np.arange(GR.nb,SGR.nx+GR.nb).astype(np.int)
    shift = int(GR.nx*x_shift_fact)
    SGR.GRmap_in_ii  = ind_range + shift
    ind_range = np.arange(GR.nb,SGR.nx+GR.nb).astype(np.int)
    shift = int(GR.nx*x_shift_fact)
    inds = ind_range + shift
    if ngrids - 1 == grid_no:
        inds = np.append(inds, GR.iis[-1])
    SGR.GRmap_in_iis = inds 
    SGR.GRmap_in_jj  = np.arange(GR.nb,SGR.ny +GR.nb).astype(np.int)
    SGR.GRmap_in_jjs = np.arange(GR.nb,SGR.nys+GR.nb).astype(np.int)
    SGR.GRmap_in_iijj  = np.ix_(SGR.GRmap_in_ii   ,SGR.GRmap_in_jj  )
    SGR.GRmap_in_iisjj = np.ix_(SGR.GRmap_in_iis  ,SGR.GRmap_in_jj  )
    SGR.GRmap_in_iijjs = np.ix_(SGR.GRmap_in_ii   ,SGR.GRmap_in_jjs  )

    # GRID MAP IN INNER (WITHOUT BORDERS OF SGR ARRAY)
    SGR.GRimap_in_ii  = SGR.GRmap_in_ii - 1
    SGR.GRimap_in_jj  = SGR.GRmap_in_jj - 1
    SGR.GRimap_in_iijj  = np.ix_(SGR.GRimap_in_ii, SGR.GRimap_in_jj )

    SGR.SGRmap_out_ii  = SGR.ii
    SGR.SGRmap_out_jj  = SGR.jj
    SGR.SGRmap_out_jjs = SGR.jjs
    if ngrids - 1 == grid_no:
        SGR.SGRmap_out_iis = SGR.iis
    else:
        SGR.SGRmap_out_iis = SGR.iis[:-1]
    SGR.SGRmap_out_iijj  = np.ix_(SGR.SGRmap_out_ii   ,SGR.SGRmap_out_jj  )
    SGR.SGRmap_out_iisjj = np.ix_(SGR.SGRmap_out_iis  ,SGR.SGRmap_out_jj  )
    SGR.SGRmap_out_iijjs = np.ix_(SGR.SGRmap_out_ii   ,SGR.SGRmap_out_jjs  )
    
    #print(SGR.GRmap_out_iijj)
    #print(SGR.GRimap_out_iijj)
    #print(SGR.GRmap_in_iijjs)
    #print(SGR.SGRmap_out_iijjs)
    #print()
    return(SGR)





def set_helix_give_inds(SGR, ind0):
    helix_inds = {}
    #######################################
    ########## uvflx_helix_inds ###########
    #######################################
    ######### UFLX #############
    helix_inds['give_UFLX'] = []
    length = SGR.ny*SGR.nz
    # left border
    helix_inds['give_UFLX'].append( ( ind0 + 0*length, \
                                 ind0 + 1*length) )
    ind0 += length
    # right border
    helix_inds['give_UFLX'].append( ( ind0 + 0*length, \
                                 ind0 + 1*length) )
    ind0 += length

    ######### VFLX #############
    helix_inds['give_VFLX'] = []
    length = SGR.nys*SGR.nz
    # left border
    helix_inds['give_VFLX'].append( ( ind0 + 0*length, \
                                 ind0 + 1*length) )
    ind0 += length
    # right border
    helix_inds['give_VFLX'].append( ( ind0 + 0*length, \
                                 ind0 + 1*length) )
    ind0 += length

    SGR.uvflx_helix_inds = helix_inds
    return(SGR, ind0)


def set_helix_take_inds(SGR, SGR_left, SGR_right):
    SGR.uvflx_helix_inds['take_UFLX'] = []
    SGR.uvflx_helix_inds['take_VFLX'] = []

    # left border
    SGR.uvflx_helix_inds['take_UFLX'].append(
                        SGR_left.uvflx_helix_inds['give_UFLX'][1])
    SGR.uvflx_helix_inds['take_VFLX'].append(
                        SGR_left.uvflx_helix_inds['give_VFLX'][1])
    # right border
    SGR.uvflx_helix_inds['take_UFLX'].append(
                        SGR_right.uvflx_helix_inds['give_UFLX'][0])
    SGR.uvflx_helix_inds['take_VFLX'].append(
                        SGR_right.uvflx_helix_inds['give_VFLX'][0])

    return(SGR)


def create_subgrids(GR, njobs):
    subgrids = {}

    if GR.nx % njobs > 0:
        raise ValueError('grid does not divide by number of jobs')
    
    ########################################################################
    if njobs == 1:
        GR = set_map_indices(GR, GR, 0, 1)
        GR, uvflx_max = set_helix_give_inds(GR, 0)
        subgrids[0] = GR

        GR.uvflx_helix_size = uvflx_max
        subgrids[0] = set_helix_take_inds(subgrids[0], subgrids[0], subgrids[0])
    ########################################################################
    if njobs == 2:
        # subgrid 0
        specs = {}
        specs['lon0_deg'] = GR.lon0_deg
        specs['lon1_deg'] = (GR.lon1_deg - GR.lon0_deg)/2
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 0, 2)
        SGR, uvflx_max = set_helix_give_inds(SGR, 0)
        subgrids[0] = SGR

        # subgrid 1
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)/2)
        specs['lon1_deg'] = GR.lon1_deg
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 1, 2)
        SGR, uvflx_max = set_helix_give_inds(SGR, uvflx_max)
        subgrids[1] = SGR 

        GR.uvflx_helix_size = uvflx_max
        # subgrid 0
        subgrids[0] = set_helix_take_inds(subgrids[0], subgrids[1], subgrids[1])
        # subgrid 1
        subgrids[1] = set_helix_take_inds(subgrids[1], subgrids[0], subgrids[0])
    ########################################################################
    if njobs == 3:
        # subgrid 0
        specs = {}
        specs['lon0_deg'] = GR.lon0_deg
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/3)
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 0, 3)
        SGR, uvflx_max = set_helix_give_inds(SGR, 0)
        subgrids[0] = SGR

        # subgrid 1
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/3)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/3) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 1, 3)
        SGR, uvflx_max = set_helix_give_inds(SGR, uvflx_max)
        subgrids[1] = SGR 

        # subgrid 2
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/3)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*3/3) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 2, 3)
        SGR, uvflx_max = set_helix_give_inds(SGR, uvflx_max)
        subgrids[2] = SGR 

        #print(GR.iisjj)
        ##print(GR.iijj)
        #quit()
        GR.uvflx_helix_size = uvflx_max
        # subgrid 0
        subgrids[0] = set_helix_take_inds(subgrids[0], subgrids[2], subgrids[1])
        # subgrid 1
        subgrids[1] = set_helix_take_inds(subgrids[1], subgrids[0], subgrids[2])
        # subgrid 2
        subgrids[2] = set_helix_take_inds(subgrids[2], subgrids[1], subgrids[0])
    ########################################################################
    if njobs == 4:
        # subgrid 0
        specs = {}
        specs['lon0_deg'] = GR.lon0_deg
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/4)
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 0, 4)
        SGR, uvflx_max = set_helix_give_inds(SGR, 0)
        subgrids[0] = SGR

        # subgrid 1
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*1/4)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/4) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 1, 4)
        SGR, uvflx_max = set_helix_give_inds(SGR, uvflx_max)
        subgrids[1] = SGR 

        # subgrid 2
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*2/4)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*3/4) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 2, 4)
        SGR, uvflx_max = set_helix_give_inds(SGR, uvflx_max)
        subgrids[2] = SGR 

        # subgrid 3
        specs = {}
        specs['lon0_deg'] = int((GR.lon1_deg - GR.lon0_deg)*3/4)
        specs['lon1_deg'] = int((GR.lon1_deg - GR.lon0_deg)*4/4) 
        specs['lat0_deg'] = GR.lat0_deg
        specs['lat1_deg'] = GR.lat1_deg
        SGR = Grid(1, specs)
        SGR = set_map_indices(GR, SGR, 3, 4)
        SGR, uvflx_max = set_helix_give_inds(SGR, uvflx_max)
        subgrids[3] = SGR 

        GR.uvflx_helix_size = uvflx_max
        # subgrid 0
        subgrids[0] = set_helix_take_inds(subgrids[0], subgrids[3], subgrids[1])
        # subgrid 1
        subgrids[1] = set_helix_take_inds(subgrids[1], subgrids[0], subgrids[2])
        # subgrid 2
        subgrids[2] = set_helix_take_inds(subgrids[2], subgrids[1], subgrids[3])
        # subgrid 3
        subgrids[3] = set_helix_take_inds(subgrids[3], subgrids[2], subgrids[0])
    ########################################################################

    return(GR, subgrids)


if __name__ == '__main__':


    GR = Grid()

    from namelist import njobs
    subgrids = create_subgrids(GR, njobs)
