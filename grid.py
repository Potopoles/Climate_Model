#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
####################################################################
File name:          grid.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190523
License:            MIT

Setup computational and geographical grid for simulation.
Alson includes coriolis parameter and grid cell area and similar
things.
####################################################################
"""
import math
from numba import cuda
import numpy as np
from namelist import *
from org_namelist import wp_int, wp
from constants import con_rE, con_omega
from boundaries import exchange_BC, exchange_BC_periodic_x
from IO import (load_restart_grid, set_up_sigma_levels)
from utilities import Timer

####################################################################

# FIXED GPU SETTINGS (have to here in code to be able to import them in another module)
tpbh  = 1    # tasks per block horizontal (CANNOT BE CHANGED!)
tpbv  = nz   # tasks per block vertical (CANNOT BE CHANGED!)
tpbvs = nz+1 # tasks per block vertical (CANNOT BE CHANGED!)


shared_nz = nz 
#shared_nzs = nz+1

nz = wp_int(nz)
nzs = wp_int(nz+1)
nx = wp_int((lon1_deg - lon0_deg)/dlon_deg)
nxs = wp_int(nx+1)
ny = wp_int((lat1_deg - lat0_deg)/dlat_deg)
nys = wp_int(ny+1)
nb = wp_int(nb)


# GPU computation
if nz > 32:
    raise NotImplementedError
tpb     = (2,       2,      nz )
tpb_ks  = (tpb[0],  tpb[1], nzs)
bpg = (math.ceil((nxs+2*nb)/tpb[0]), math.ceil((nys+2*nb)/tpb[1]), 1)
tpb_sc = (1,       1,      nz )
#tpb_sc_ks = (1,       1,      nzs)
bpg_sc = (math.ceil((nxs+2*nb)/tpb_sc[0]),
          math.ceil((nys+2*nb)/tpb_sc[1]), 1)

class Grid:

    def __init__(self, i_subgrid=0, specs=None, new=False):

        if i_load_from_restart:
            loadGR = load_restart_grid(dlat_deg, dlat_deg, nz)
            self.__dict__ = loadGR.__dict__
            self.i_sim_n_days = i_sim_n_days
            self.nts = i_sim_n_days*3600*24/self.dt
            self.i_out_nth_hour = i_out_nth_hour
            self.i_out_nth_ts = int(self.i_out_nth_hour*3600 / self.dt)
            self.i_restart_nth_day = i_restart_nth_day
            self.i_restart_nth_ts = self.i_restart_nth_day*24/ \
                                self.i_out_nth_hour*self.i_out_nth_ts
        else:
            self.new = new

            # TIMERS
            self.timer = Timer()
            #self.total_comp_time = 0
            #self.IO_time = 0
            #self.dyn_comp_time = 0
            #self.wind_comp_time = 0
            #self.temp_comp_time = 0
            #self.trac_comp_time = 0
            #self.cont_comp_time = 0
            #self.diag_comp_time = 0
            #self.step_comp_time = 0
            #self.special = 0
            #self.copy_time = 0

            self.rad_comp_time = 0
            self.mic_comp_time = 0
            self.soil_comp_time = 0

            # GRID DEFINITION IN DEGREES 
            if not i_subgrid:
                self.lon0_deg = lon0_deg
                self.lon1_deg = lon1_deg
                self.lat0_deg = lat0_deg
                self.lat1_deg = lat1_deg
            else:
                self.lon0_deg = specs['lon0_deg']
                self.lon1_deg = specs['lon1_deg']
                self.lat0_deg = specs['lat0_deg']
                self.lat1_deg = specs['lat1_deg']
            self.dlon_deg = dlon_deg
            self.dlat_deg = dlat_deg

            # GRID DEFINITION IN RADIANS
            self.lon0_rad = self.lon0_deg/180*np.pi
            self.lon1_rad = self.lon1_deg/180*np.pi
            self.lat0_rad = self.lat0_deg/180*np.pi
            self.lat1_rad = self.lat1_deg/180*np.pi
            self.dlon_rad = self.dlon_deg/180*np.pi
            self.dlat_rad = self.dlat_deg/180*np.pi

            # NUMBER OF GRID POINTS IN EACH DIMENSION
            possible_nz = [2,4,8,16,32,64]
            if nz not in possible_nz:
                raise NotImplementedError('Vertical reduction for GPU needs nz of 2**x')
            self.nz = nz
            self.nzs = nzs
            self.nx = nx
            self.nxs = nxs
            self.ny = ny
            self.nys = nys
            self.nb = nb

            # INDEX ARRAYS
            self.kk  = np.arange(0,self.nz)
            self.kks = np.arange(0,self.nzs)
            self.ii = np.arange((self.nb),(self.nx+self.nb)) 
            self.jj = np.arange((self.nb),(self.ny+self.nb)) 
            self.iis = np.arange((self.nb),(self.nxs+self.nb)) 
            self.jjs = np.arange((self.nb),(self.nys+self.nb)) 

            self.iijj           = np.ix_(self.ii   ,self.jj  )
            self.iijj_im1       = np.ix_(self.ii-1 ,self.jj  )
            self.iijj_im1_jp1   = np.ix_(self.ii-1 ,self.jj+1)
            self.iijj_ip1       = np.ix_(self.ii+1 ,self.jj  )
            self.iijj_ip1_jm1   = np.ix_(self.ii+1 ,self.jj-1)
            self.iijj_ip1_jp1   = np.ix_(self.ii+1 ,self.jj+1)
            self.iijj_jm1       = np.ix_(self.ii   ,self.jj-1)
            self.iijj_jp1       = np.ix_(self.ii   ,self.jj+1)

            self.iisjj          = np.ix_(self.iis  ,self.jj  )
            self.iisjj_jm1      = np.ix_(self.iis  ,self.jj-1)
            self.iisjj_jp1      = np.ix_(self.iis  ,self.jj+1)
            self.iisjj_im1      = np.ix_(self.iis-1,self.jj  )
            self.iisjj_im1_jm1  = np.ix_(self.iis-1,self.jj-1)
            self.iisjj_im1_jp1  = np.ix_(self.iis-1,self.jj+1)
            self.iisjj_ip1      = np.ix_(self.iis+1,self.jj  )
            self.iisjj_ip1_jm1  = np.ix_(self.iis+1,self.jj-1)
            self.iisjj_ip1_jp1  = np.ix_(self.iis+1,self.jj+1)

            self.iijjs          = np.ix_(self.ii  ,self.jjs  )
            self.iijjs_im1      = np.ix_(self.ii-1,self.jjs  )
            self.iijjs_ip1      = np.ix_(self.ii+1,self.jjs  )
            self.iijjs_jm1      = np.ix_(self.ii  ,self.jjs-1)
            self.iijjs_im1_jm1  = np.ix_(self.ii-1,self.jjs-1)
            self.iijjs_im1_jp1  = np.ix_(self.ii-1,self.jjs+1)
            self.iijjs_ip1_jm1  = np.ix_(self.ii+1,self.jjs-1)
            self.iijjs_ip1_jp1  = np.ix_(self.ii+1,self.jjs+1)
            self.iijjs_jp1      = np.ix_(self.ii  ,self.jjs+1)

            self.iisjjs         = np.ix_(self.iis  ,self.jjs  )
            self.iisjjs_im1     = np.ix_(self.iis-1,self.jjs  )
            self.iisjjs_im1_jm1 = np.ix_(self.iis-1,self.jjs-1)
            self.iisjjs_im1_jp1 = np.ix_(self.iis-1,self.jjs+1)
            self.iisjjs_ip1     = np.ix_(self.iis+1,self.jjs  )
            self.iisjjs_ip1_jm1 = np.ix_(self.iis+1,self.jjs-1)
            self.iisjjs_jm1     = np.ix_(self.iis  ,self.jjs-1)
            self.iisjjs_jp1     = np.ix_(self.iis  ,self.jjs+1)

            # 2D MATRIX OF LONGITUDES AND LATITUDES IN DEGREES
            self.lon_deg   = np.full( (self.nx +2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp)
            self.lat_deg   = np.full( (self.nx +2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp)
            self.lon_is_deg = np.full( (self.nxs+2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp)
            self.lat_is_deg = np.full( (self.nxs+2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp)
            self.lon_js_deg = np.full( (self.nx+2*self.nb,self.nys+2*self.nb), np.nan,
                                        dtype=wp)
            self.lat_js_deg = np.full( (self.nx+2*self.nb,self.nys+2*self.nb), np.nan,
                                        dtype=wp)

            self.dlon_rad_2D   = np.full( (self.nx +2*self.nb,self.nys+2*self.nb), self.dlon_rad,
                                        dtype=wp)
            self.dlat_rad_2D   = np.full( (self.nxs+2*self.nb,self.ny +2*self.nb), self.dlat_rad,
                                        dtype=wp)

            for j in range(self.nb, self.ny+self.nb):
                self.lon_deg[self.ii,j] = self.lon0_deg + \
                                        (self.ii-self.nb+0.5)*self.dlon_deg
                self.lon_is_deg[self.iis,j] = self.lon0_deg + \
                                        (self.iis-self.nb)*self.dlon_deg
            for j_s in range(self.nb, self.nys+self.nb):
                self.lon_js_deg[self.ii,j_s] = self.lon0_deg + \
                                        (self.ii-self.nb+0.5)*self.dlon_deg

            for i in range(self.nb, self.nx+self.nb):
                self.lat_deg[i,self.jj] = self.lat0_deg + \
                                        (self.jj-self.nb+0.5)*self.dlat_deg
                self.lat_js_deg[i,self.jjs] = self.lat0_deg + \
                                        (self.jjs-self.nb)*self.dlat_deg
            for i_s in range(self.nb, self.nxs+self.nb):
                self.lat_is_deg[i_s,self.jj] = self.lat0_deg + \
                                        (self.jj-self.nb+0.5)*self.dlat_deg


            # 2D MATRIX OF LONGITUDES AND LATITUDES IN RADIANS
            self.lon_rad = self.lon_deg/180*np.pi
            self.lat_rad = self.lat_deg/180*np.pi
            self.lon_is_rad = self.lon_is_deg/180*np.pi
            self.lat_is_rad = self.lat_is_deg/180*np.pi
            self.lon_js_rad = self.lon_js_deg/180*np.pi
            self.lat_js_rad = self.lat_js_deg/180*np.pi

            # 2D MATRIX OF GRID SPACING IN METERS
            self.dx   = np.full( (self.nx +2*self.nb,self.ny +2*self.nb),
                                np.nan, dtype=wp)
            self.dxjs = np.full( (self.nx +2*self.nb,self.nys+2*self.nb),
                                np.nan, dtype=wp)
            self.dyis = np.full( (self.nxs+2*self.nb,self.ny +2*self.nb),
                                np.nan, dtype=wp)

            self.dx[self.iijj] = np.cos( self.lat_rad[self.iijj] )*self.dlon_rad*con_rE 
            self.dxjs[self.iijjs] = np.cos( self.lat_js_rad[self.iijjs] )*self.dlon_rad*con_rE 
            self.dyis[self.iisjj] = self.dlat_rad*con_rE 
            self.dx = exchange_BC(self, self.dx)
            self.dxjs = exchange_BC(self, self.dxjs)
            self.dyis = exchange_BC(self, self.dyis)
            self.dy = self.dlat_rad*con_rE


            if not i_curved_earth:
                maxdx = np.max(self.dx[self.iijj])
                self.dx[self.iijj] = maxdx

            self.A = np.full( (self.nx+2*self.nb,self.ny+2*self.nb),
                                np.nan, dtype=wp)
            for i in self.ii:
                for j in self.jj:
                    self.A[i,j] = lat_lon_recangle_area(self.lat_rad[i,j],
                                        self.dlon_rad, self.dlat_rad, i_curved_earth)
            self.A = exchange_BC(self, self.A)

            if i_curved_earth:
                print('fraction of earth covered: ' + \
                        str(np.round(np.sum(self.A[self.iijj])/(4*np.pi*con_rE**2),2)))
            else:
                print('fraction of cylinder covered: ' + \
                        str(np.round(np.sum(self.A[self.iijj])/(2*np.pi**2*con_rE**2),2)))

            # CORIOLIS FORCE
            self.corf    = np.full( (self.nx +2*self.nb, self.ny +2*self.nb), 
                                    np.nan, dtype=wp)
            self.corf_is = np.full( (self.nxs+2*self.nb, self.ny +2*self.nb),
                                    np.nan, dtype=wp)
            self.corf[self.iijj] = 2*con_omega*np.sin(self.lat_rad[self.iijj])
            self.corf_is[self.iisjj] = 2*con_omega*np.sin(self.lat_is_rad[self.iisjj])

            # SIGMA LEVELS
            self.level  = np.arange(0,self.nz )
            self.levels = np.arange(0,self.nzs)
            # will be set in load_profile of IO
            self.sigma_vb = np.full( self.nzs, np.nan, dtype=wp)
            self.dsigma   = np.full( self.nz , np.nan, dtype=wp)


            # TIME STEP
            mindx = np.nanmin(self.dx)
            self.CFL = CFL
            self.i_out_nth_hour = i_out_nth_hour
            self.nc_output_count = 0
            self.i_sim_n_days = i_sim_n_days
            self.dt = int(self.CFL*mindx/340)
            while i_out_nth_hour*3600 % self.dt > 0:
                self.dt -= 1
            self.nts = i_sim_n_days*3600*24/self.dt
            self.ts = 0
            self.i_out_nth_ts = int(self.i_out_nth_hour*3600 / self.dt)
            self.i_restart_nth_day = i_restart_nth_day
            self.i_restart_nth_ts = int(self.i_restart_nth_day*24/ \
                    self.i_out_nth_hour*self.i_out_nth_ts)
            self.sim_time_sec = 0
            self.GMT = GMT_initialization

    
        # STUFF THAT SHOULD HAPPEN BOTH FOR NEW GRID OR LOADED GRID
        self.rad_1 = 0
        self.rad_2 = 0
        self.rad_sw = 0
        self.rad_lw = 0
        self.rad_lwsolv = 0
        self.rad_swsolv = 0



        # NEW GRID STYLE
        if self.new:
            self.corf         = np.expand_dims(self.corf,       axis=2)
            self.corf_is      = np.expand_dims(self.corf_is,    axis=2)
            self.lon_rad      = np.expand_dims(self.lat_rad,    axis=2)
            self.lat_rad      = np.expand_dims(self.lat_rad,    axis=2)
            self.lat_is_rad   = np.expand_dims(self.lat_is_rad, axis=2)
            self.dlon_rad     = np.expand_dims(self.dlon_rad_2D,axis=2)
            self.dlat_rad     = np.expand_dims(self.dlat_rad_2D,axis=2)
            self.A            = np.expand_dims(self.A,          axis=2)
            self.dyis         = np.expand_dims(self.dyis,       axis=2)
            self.dxjs         = np.expand_dims(self.dxjs,       axis=2)

            set_up_sigma_levels(self)

            self.dsigma       = np.expand_dims(
                                np.expand_dims(self.dsigma   , 0),0)
            self.sigma_vb     = np.expand_dims(
                                np.expand_dims(self.sigma_vb , 0),0)

            self.corfd        = cuda.to_device(self.corf       )
            self.corf_isd     = cuda.to_device(self.corf_is    )
            self.lat_radd     = cuda.to_device(self.lat_rad    )
            self.lat_is_radd  = cuda.to_device(self.lat_is_rad )
            self.dlon_radd    = cuda.to_device(self.dlon_rad   )
            self.dlat_radd    = cuda.to_device(self.dlat_rad   )
            self.Ad           = cuda.to_device(self.A          )
            self.dyisd        = cuda.to_device(self.dyis       )
            self.dxjsd        = cuda.to_device(self.dxjs       )
            self.dsigmad      = cuda.to_device(self.dsigma     )
            self.sigma_vbd    = cuda.to_device(self.sigma_vb   )


        
            self.i   = np.arange((self.nb),(self.nx +self.nb)) 
            self.i_s = np.arange((self.nb),(self.nxs+self.nb)) 
            self.j   = np.arange((self.nb),(self.ny +self.nb)) 
            self.js  = np.arange((self.nb),(self.nys+self.nb)) 
            self.ii,  self.jj  = np.ix_(self.i    ,self.j)
            self.iis, self.jjs = np.ix_(self.i_s  ,self.js)




def lat_lon_recangle_area(lat,dlon,dlat, i_curved_earth):
    if i_curved_earth:
        A = np.cos(lat) * \
                dlon * dlat * con_rE**2
    else:
        A = dlon * dlat * con_rE**2
    return(A)


