#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190701
License:            MIT

Set up computational and geographical grid for simulation.
Grid class:
    - Contains all grid parameters and is passed to almost all functions
      (except compiled njit/cuda.jit functions)
    - Also contains coriolis parameter
###############################################################################
"""
import math, pickle
import numpy as np
from numba import cuda

from namelist import (nz, nb,
                      lon0_deg, lon1_deg, dlon_deg,
                      lat0_deg, lat1_deg, dlat_deg,
                      CFL, i_load_from_IC, IC_file_name,
                      i_load_from_restart, i_restart_nth_day,
                      i_out_nth_hour, i_sim_n_days,
                      GMT_initialization)
from io_read_namelist import (wp_int, wp, gpu_enable, CPU, GPU,
                            POTT_dif_coef, UVFLX_dif_coef,
                            moist_dif_coef)
from io_constants import con_rE, con_omega
from io_restart import load_restart_grid
from io_initial_conditions import set_up_sigma_levels
from misc_utilities import Timer
###############################################################################

###############################################################################
# GLOBAL SETTINGS
###############################################################################
# vertical extent of cuda shared arrays
# (must be called before setting nz = wp_int(nz))
shared_nz = nz 

# domain extent to be imported in modules
nz = wp_int(nz)
nzs = wp_int(nz+1)
nx = wp_int((lon1_deg - lon0_deg)/dlon_deg)
nxs = wp_int(nx+1)
ny = wp_int((lat1_deg - lat0_deg)/dlat_deg)
nys = wp_int(ny+1)
nb = wp_int(nb)

###############################################################################
# GPU COMPUTATION SETTINGS
###############################################################################
if gpu_enable:
    if nz > 32:
        raise NotImplementedError('More than 32 vertical levels '+
                'not yet implemented for GPU.')

# THREADS PER BLOCK
# normal case
tpb     = (2,       2,      nz )
# case of vertically staggered variable
tpb_ks  = (tpb[0],  tpb[1], nzs)
# 2D (e.g. surface) field
tpb_2D  = (2,       2,      1 )
# BLOCKS PER GRID
# normal case
bpg = (math.ceil((nxs+2*nb)/tpb[0]), math.ceil((nys+2*nb)/tpb[1]), 1)
# case of single column (when shared arrays are used in vertical
# direction
tpb_sc = (1,       1,      nz )
bpg_sc = (math.ceil((nxs+2*nb)/tpb_sc[0]),
          math.ceil((nys+2*nb)/tpb_sc[1]), 1)


###############################################################################
# GRID CLASS
###############################################################################
class Grid:

    def __init__(self):

        #######################################################################
        # LOAD GRID FROM RESTART FILE
        #######################################################################
        if i_load_from_restart:
            loaded_GR = load_restart_grid(dlat_deg, dlat_deg, nz)
            self.__dict__ = loaded_GR.__dict__
            # update time settings
            self.i_sim_n_days = i_sim_n_days
            self.nts = i_sim_n_days*3600*24/self.dt
            self.i_out_nth_hour = i_out_nth_hour
            self.i_out_nth_ts = int(self.i_out_nth_hour*3600 / self.dt)
            self.i_restart_nth_day = i_restart_nth_day
            self.i_restart_nth_ts = self.i_restart_nth_day*24/ \
                                self.i_out_nth_hour*self.i_out_nth_ts
            self.copy_to_gpu()
        else:
            self.create_new_grid()

        # Get Grid from simulation providing initial conditions
        if i_load_from_IC:
            with open(IC_file_name, 'rb') as f:
                self.IC = pickle.load(f)['GR']
            self.GMT = self.IC.GMT

    #######################################################################
    # CREATE NEW GRID
    #######################################################################
    def create_new_grid(self):
        # GRID DEFINITION IN DEGREES 
        self.lon0_deg = lon0_deg
        self.lon1_deg = lon1_deg
        self.lat0_deg = lat0_deg
        self.lat1_deg = lat1_deg
        self.dlon_deg = dlon_deg
        self.dlat_deg = dlat_deg

        # GRID DEFINITION IN RADIANS
        self.lon0_rad    = self.lon0_deg/180*np.pi
        self.lon1_rad    = self.lon1_deg/180*np.pi
        self.lat0_rad    = self.lat0_deg/180*np.pi
        self.lat1_rad    = self.lat1_deg/180*np.pi
        self.dlon_rad_1D = self.dlon_deg/180*np.pi
        self.dlat_rad_1D = self.dlat_deg/180*np.pi

        # NUMBER OF GRID POINTS IN EACH DIMENSION
        self.nz = nz
        self.nzs = nzs
        self.nx = nx
        self.nxs = nxs
        self.ny = ny
        self.nys = nys
        self.nb = nb

        # INDEX ARRAYS
        self.i   = np.arange((self.nb),(self.nx +self.nb)) 
        self.i_s = np.arange((self.nb),(self.nxs+self.nb)) 
        self.j   = np.arange((self.nb),(self.ny +self.nb)) 
        self.js  = np.arange((self.nb),(self.nys+self.nb)) 
        self.k   = np.arange(self.nz) 
        self.ii,  self.jj  = np.ix_(self.i    ,self.j)
        self.iis, self.jjs = np.ix_(self.i_s  ,self.js)

        # 2D MATRIX OF LONGITUDES AND LATITUDES IN DEGREES
        self.lon_deg   = np.full(
                (self.nx +2*self.nb,self.ny+2*self.nb,1),
                                    np.nan, dtype=wp)
        self.lat_deg   = np.full(
                (self.nx +2*self.nb,self.ny+2*self.nb,1),
                                    np.nan, dtype=wp)
        self.lon_is_deg = np.full(
                (self.nxs+2*self.nb,self.ny+2*self.nb,1),
                                    np.nan, dtype=wp)
        self.lat_is_deg = np.full(
                (self.nxs+2*self.nb,self.ny+2*self.nb,1),
                                    np.nan, dtype=wp)
        self.lon_js_deg = np.full(
                (self.nx+2*self.nb,self.nys+2*self.nb,1),
                                    np.nan, dtype=wp)
        self.lat_js_deg = np.full(
                (self.nx+2*self.nb,self.nys+2*self.nb,1),
                                    np.nan, dtype=wp)
        self.dlon_rad = np.full(
                (self.nx +2*self.nb,self.nys+2*self.nb,1),
                                    self.dlon_rad_1D, dtype=wp)
        self.dlat_rad = np.full(
                (self.nxs+2*self.nb,self.ny +2*self.nb,1),
                                    self.dlat_rad_1D, dtype=wp)

        for j in range(self.nb, self.ny+self.nb):
            self.lon_deg[self.ii,j,0] = (self.lon0_deg +
                                (self.ii-self.nb+0.5)*self.dlon_deg)
            self.lon_is_deg[self.iis,j,0] = (self.lon0_deg +
                                (self.iis-self.nb)*self.dlon_deg)
        for j_s in range(self.nb, self.nys+self.nb):
            self.lon_js_deg[self.ii,j_s,0] = (self.lon0_deg +
                                (self.ii-self.nb+0.5)*self.dlon_deg)
        for i in range(self.nb, self.nx+self.nb):
            self.lat_deg[i,self.jj,0] = (self.lat0_deg +
                                (self.jj-self.nb+0.5)*self.dlat_deg)
            self.lat_js_deg[i,self.jjs,0] = (self.lat0_deg +
                                (self.jjs-self.nb)*self.dlat_deg)
        for i_s in range(self.nb, self.nxs+self.nb):
            self.lat_is_deg[i_s,self.jj,0] = (self.lat0_deg +
                                (self.jj-self.nb+0.5)*self.dlat_deg)

        # 2D MATRIX OF LONGITUDES AND LATITUDES IN RADIANS
        self.lon_rad = self.lon_deg/180*np.pi
        self.lat_rad = self.lat_deg/180*np.pi
        self.lon_is_rad = self.lon_is_deg/180*np.pi
        self.lat_is_rad = self.lat_is_deg/180*np.pi
        self.lon_js_rad = self.lon_js_deg/180*np.pi
        self.lat_js_rad = self.lat_js_deg/180*np.pi

        # 2D MATRIX OF GRID SPACING IN METERS
        self.dx   = np.full(
                    (self.nx +2*self.nb,self.ny +2*self.nb,1),
                            np.nan, dtype=wp)
        self.dxjs = np.full(
                    (self.nx +2*self.nb,self.nys+2*self.nb,1),
                            np.nan, dtype=wp)
        self.dyis = np.full(
                    (self.nxs+2*self.nb,self.ny +2*self.nb,1),
                            np.nan, dtype=wp)

        self.dx[self.ii,self.jj,0] = (
                    np.cos( self.lat_rad[self.ii,self.jj,0] ) *
                                self.dlon_rad_1D*con_rE )
        self.dxjs[self.ii,self.jjs,0] = ( 
                    np.cos( self.lat_js_rad[self.ii,self.jjs,0] ) *
                                self.dlon_rad_1D*con_rE )
        self.dyis[self.iis,self.jj,0] = self.dlat_rad_1D*con_rE 
        self.dx   = self.exchange_BC(self.dx)
        self.dxjs = self.exchange_BC(self.dxjs)
        self.dyis = self.exchange_BC(self.dyis)
        self.dy = self.dlat_rad*con_rE

        self.A = np.full( (self.nx+2*self.nb,self.ny+2*self.nb, 1),
                            np.nan, dtype=wp)
        for i in self.i:
            for j in self.j:
                self.A[i,j,0] = lat_lon_recangle_area(self.lat_rad[i,j,0],
                        self.dlon_rad_1D, self.dlat_rad_1D)
        self.A = self.exchange_BC(self.A)
        print('fraction of earth covered: ' +
                str(np.round(np.sum(
                self.A[self.ii,self.jj,0])/(4*np.pi*con_rE**2),2)))

        # CORIOLIS FORCE
        self.corf    = np.full(
                        (self.nx +2*self.nb, self.ny +2*self.nb,1), 
                                np.nan, dtype=wp)
        self.corf_is = np.full(
                        (self.nxs+2*self.nb, self.ny +2*self.nb,1),
                                np.nan, dtype=wp)
        self.corf[self.ii,self.jj,0] = 2*con_omega*np.sin(
                                    self.lat_rad[self.ii,self.jj,0])
        self.corf_is[self.iis,self.jj,0] = 2*con_omega*np.sin(
                                    self.lat_is_rad[self.iis,self.jj,0])

        # SIGMA LEVELS
        self.level  = np.arange(0,self.nz )
        self.levels = np.arange(0,self.nzs)
        # will be set in load_profile of IO
        self.sigma_vb = np.full( self.nzs, np.nan, dtype=wp)
        self.dsigma   = np.full( self.nz , np.nan, dtype=wp)

        set_up_sigma_levels(self)
        self.dsigma       = np.expand_dims(
                            np.expand_dims(self.dsigma   , 0),0)
        self.sigma_vb     = np.expand_dims(
                            np.expand_dims(self.sigma_vb , 0),0)

        # TIME STEP
        mindx = np.nanmin(self.dx)
        self.CFL = CFL
        self.i_out_nth_hour = i_out_nth_hour
        self.nc_output_count = 0
        self.i_sim_n_days = i_sim_n_days
        self.dt = int(self.CFL*mindx/400)
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

        # TIMER
        self.timer = Timer()

        # NUMERICAL DIFUSION
        self.UVFLX_dif_coef = np.zeros((1,1,nz), dtype=wp)
        self.POTT_dif_coef  = np.zeros((1,1,nz), dtype=wp)
        self.moist_dif_coef = np.zeros((1,1,nz), dtype=wp)
        # vert_reduce for 32 vertical levels:
        # 1.0 : uppermost level has 40% of lowest
        # 1.5 : uppermost level has 23% of lowest
        # 2.0 : uppermost level has 10% of lowest
        # 3.0 : uppermost level has 5% of lowest
        vert_reduce = .0
        self.UVFLX_dif_coef[0,0,self.k] = (UVFLX_dif_coef * 
                                np.exp(-vert_reduce*(nz-self.k-1)/nz))
        vert_reduce = 1.5
        self.POTT_dif_coef[0,0,self.k] = (POTT_dif_coef * 
                                np.exp(-vert_reduce*(nz-self.k-1)/nz))
        self.moist_dif_coef[0,0,self.k] = (moist_dif_coef * 
                                np.exp(-vert_reduce*(nz-self.k-1)/nz))

        self.copy_to_gpu()


    def copy_to_gpu(self):
        """
        Copy all necessary grid fields to gpu.
        """
        self.GRF = {CPU:{},GPU:{}}
        grid_field_names = ['corf', 'corf_is', 'A',
                            'sigma_vb', 'dsigma',
                            'dxjs', 'dyis',
                            'lat_rad', 'lat_is_rad', 'dlat_rad', 'dlon_rad',
                            'POTT_dif_coef', 'UVFLX_dif_coef',
                            'moist_dif_coef']
        for field_name in grid_field_names:
            exec('self.GRF[CPU][field_name] = self.'+field_name)
            if gpu_enable:
                self.GRF[GPU][field_name] = cuda.to_device(
                                    self.GRF[CPU][field_name])
                                    

        
    def exchange_BC(self, FIELD):
        """
        Python function (slow) to exchange boundaries.
        Should not be used within time step loop but just for initializiation.
        Advantage: Does not depend on import grid constants (e.g. nx)
                   Can therefore be used within grid.py and initial_conditiony.py
        """

        dim2 = False
        if len(FIELD.shape) == 2:
            dim2 = True
            fnx,fny = FIELD.shape
        elif len(FIELD.shape) == 3:
            fnx,fny,fnz = FIELD.shape

        # zonal boundaries
        if fnx == self.nxs+2*self.nb: # staggered in x
            FIELD[0,::] = FIELD[self.nxs-1,::] 
            FIELD[self.nxs,::] = FIELD[1,::] 
        else:     # unstaggered in x
            FIELD[0,::] = FIELD[self.nx,::] 
            FIELD[self.nx+1,::] = FIELD[1,::] 

        if dim2:
            # meridional boundaries
            if fny == self.nys+2*self.nb: # staggered in y
                for j in [0,1,self.nys,self.nys+1]:
                    FIELD[:,j] = wp(0.)
            else:     # unstaggered in y
                FIELD[:,0] = FIELD[:,1] 
                FIELD[:,self.ny+1] = FIELD[:,self.ny] 
        else:
            # meridional boundaries
            if fny == self.nys+2*self.nb: # staggered in y
                for j in [0,1,self.nys,self.nys+1]:
                    FIELD[:,j,:] = wp(0.)
            else:     # unstaggered in y
                FIELD[:,0,:] = FIELD[:,1,:] 
                FIELD[:,self.ny+1,:] = FIELD[:,self.ny,:] 

        return(FIELD)


def lat_lon_recangle_area(lat,dlon,dlat):
    A = np.cos(lat) * dlon * dlat * con_rE**2
    return(A)

