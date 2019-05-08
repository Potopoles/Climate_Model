import numpy as np
from namelist import *
from constants import con_rE, con_omega
from boundaries import exchange_BC, exchange_BC_periodic_x
from IO import load_restart_grid
from numba import cuda
if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np

from debug_namelist import wp, wp_int

# FIXED GPU SETTINGS (have to here in code to be able to import them in another module)
tpbh  = 1    # tasks per block horizontal (CANNOT BE CHANGED!)
tpbv  = nz   # tasks per block vertical (CANNOT BE CHANGED!)
tpbvs = nz+1 # tasks per block vertical (CANNOT BE CHANGED!)

nb = wp_int(1)
nx = wp_int(360)
ny = wp_int(180)
nz = wp_int(32)

nxs = nx + wp_int(1)
nys = ny + wp_int(1)
nzs = nz + wp_int(1)

# GPU computation
tpb  = (1,       1,      nz)
bpg  = (int(nx/tpb[0])+1,   int(ny/tpb[1])+1,  1)

class Grid:

    def __init__(self, i_subgrid=0, specs=None):

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

            # TIMERS
            self.total_comp_time = 0
            self.IO_time = 0
            self.dyn_comp_time = 0
            self.wind_comp_time = 0
            self.temp_comp_time = 0
            self.trac_comp_time = 0
            self.cont_comp_time = 0
            self.diag_comp_time = 0
            self.step_comp_time = 0
            self.copy_time = 0

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
            self.nzs = nz+1
            self.nx = int((self.lon1_deg - self.lon0_deg)/self.dlon_deg)
            self.nxs = self.nx + 1
            self.ny = int((self.lat1_deg - self.lat0_deg)/self.dlat_deg)
            self.nys = self.ny + 1
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
                                        dtype=wp_np)
            self.lat_deg   = np.full( (self.nx +2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp_np)
            self.lonis_deg = np.full( (self.nxs+2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp_np)
            self.latis_deg = np.full( (self.nxs+2*self.nb,self.ny+2*self.nb), np.nan,
                                        dtype=wp_np)
            self.lonjs_deg = np.full( (self.nx+2*self.nb,self.nys+2*self.nb), np.nan,
                                        dtype=wp_np)
            self.latjs_deg = np.full( (self.nx+2*self.nb,self.nys+2*self.nb), np.nan,
                                        dtype=wp_np)

            for j in range(self.nb, self.ny+self.nb):
                self.lon_deg[self.ii,j] = self.lon0_deg + \
                                        (self.ii-self.nb+0.5)*self.dlon_deg
                self.lonis_deg[self.iis,j] = self.lon0_deg + \
                                        (self.iis-self.nb)*self.dlon_deg
            for j_s in range(self.nb, self.nys+self.nb):
                self.lonjs_deg[self.ii,j_s] = self.lon0_deg + \
                                        (self.ii-self.nb+0.5)*self.dlon_deg

            for i in range(self.nb, self.nx+self.nb):
                self.lat_deg[i,self.jj] = self.lat0_deg + \
                                        (self.jj-self.nb+0.5)*self.dlat_deg
                self.latjs_deg[i,self.jjs] = self.lat0_deg + \
                                        (self.jjs-self.nb)*self.dlat_deg
            for i_s in range(self.nb, self.nxs+self.nb):
                self.latis_deg[i_s,self.jj] = self.lat0_deg + \
                                        (self.jj-self.nb+0.5)*self.dlat_deg


            # 2D MATRIX OF LONGITUDES AND LATITUDES IN RADIANS
            self.lon_rad = self.lon_deg/180*np.pi
            self.lat_rad = self.lat_deg/180*np.pi
            self.lonis_rad = self.lonis_deg/180*np.pi
            self.latis_rad = self.latis_deg/180*np.pi
            self.lonjs_rad = self.lonjs_deg/180*np.pi
            self.latjs_rad = self.latjs_deg/180*np.pi

            # 2D MATRIX OF GRID SPACING IN METERS
            self.dx   = np.full( (self.nx+2*self.nb,self.ny +2*self.nb), np.nan, dtype=wp_np)
            self.dxjs = np.full( (self.nx+2*self.nb,self.nys+2*self.nb), np.nan, dtype=wp_np)

            self.dx[self.iijj] = np.cos( self.lat_rad[self.iijj] )*self.dlon_rad*con_rE 
            self.dxjs[self.iijjs] = np.cos( self.latjs_rad[self.iijjs] )*self.dlon_rad*con_rE 
            self.dx = exchange_BC(self, self.dx)
            self.dxjs = exchange_BC(self, self.dxjs)
            self.dy = self.dlat_rad*con_rE


            if not i_curved_earth:
                maxdx = np.max(self.dx[self.iijj])
                self.dx[self.iijj] = maxdx

            self.A = np.full( (self.nx+2*self.nb,self.ny+2*self.nb), np.nan, dtype=wp_np)
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
                                    np.nan, dtype=wp_np)
            self.corf_is = np.full( (self.nxs+2*self.nb, self.ny +2*self.nb),
                                    np.nan, dtype=wp_np)
            self.corf[self.iijj] = 2*con_omega*np.sin(self.lat_rad[self.iijj])
            self.corf_is[self.iisjj] = 2*con_omega*np.sin(self.latis_rad[self.iisjj])

            # SIGMA LEVELS
            self.level  = np.arange(0,self.nz )
            self.levels = np.arange(0,self.nzs)
            # will be set in load_profile of IO
            self.sigma_vb = np.full( self.nzs, np.nan, dtype=wp_np)
            self.dsigma   = np.full( self.nz , np.nan, dtype=wp_np)


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

            ## GPU RELEVANT PART OF GRID
            ## TODO THIS IS VERY COMPLICATED. SIMPLIFY (e.g. fix blockdim)
            #if comp_mode == 2:
            #    self.stream = cuda.stream()

            #    if tpbh > 1:
            #        raise NotImplementedError('tpbh > 1 not yet possible see below')
            #    elif tpbv != self.nz:
            #        raise NotImplementedError('tpbv != nz not yet possible see below')
            #    self.blockdim      = (tpbh, tpbh, tpbv)
            #    self.blockdim_ks   = (tpbh, tpbh, tpbv+1)
            #    self.blockdim_xy   = (tpbh, tpbh, 1)
            #    self.griddim       = ((self.nx +2*self.nb)//self.blockdim[0], \
            #                          (self.ny +2*self.nb)//self.blockdim[1], \
            #                           self.nz //self.blockdim[2])
            #    self.griddim_is    = ((self.nxs+2*self.nb)//self.blockdim[0], \
            #                          (self.ny +2*self.nb)//self.blockdim[1], \
            #                           self.nz //self.blockdim[2])
            #    self.griddim_js    = ((self.nx +2*self.nb)//self.blockdim[0], \
            #                          (self.nys+2*self.nb)//self.blockdim[1], \
            #                           self.nz //self.blockdim[2])
            #    self.griddim_is_js = ((self.nxs+2*self.nb)//self.blockdim[0], \
            #                          (self.nys+2*self.nb)//self.blockdim[1], \
            #                           self.nz //self.blockdim[2])
            #    self.griddim_ks    = ((self.nx +2*self.nb)//self.blockdim_ks[0], \
            #                          (self.ny +2*self.nb)//self.blockdim_ks[1], \
            #                           self.nzs//self.blockdim_ks[2])
            #    self.griddim_is_ks = ((self.nxs+2*self.nb)//self.blockdim[0], \
            #                          (self.ny +2*self.nb)//self.blockdim[1], \
            #                           self.nzs//self.blockdim_ks[2])
            #    self.griddim_js_ks = ((self.nx +2*self.nb)//self.blockdim[0], \
            #                          (self.nys+2*self.nb)//self.blockdim[1], \
            #                           self.nzs//self.blockdim_ks[2])
            #    self.griddim_xy    = ((self.nx +2*self.nb)//self.blockdim_xy[0], \
            #                          (self.ny +2*self.nb)//self.blockdim_xy[1], \
            #                           1       //self.blockdim_xy[2])

            #    zonal   = np.zeros((2,self.ny +2*self.nb   ,self.nz  ), dtype=wp_np)
            #    zonals  = np.zeros((2,self.nys+2*self.nb   ,self.nz  ), dtype=wp_np)
            #    zonalvb = np.zeros((2,self.ny +2*self.nb   ,self.nz+1), dtype=wp_np)
            #    merid   = np.zeros((  self.nx +2*self.nb,2 ,self.nz  ), dtype=wp_np)
            #    merids  = np.zeros((  self.nxs+2*self.nb,2 ,self.nz  ), dtype=wp_np)
            #    meridvb = np.zeros((  self.nx +2*self.nb,2 ,self.nz+1), dtype=wp_np)

            #    self.zonal   = cuda.to_device(zonal,  self.stream)
            #    self.zonals  = cuda.to_device(zonals, self.stream)
            #    self.zonalvb = cuda.to_device(zonalvb, self.stream)
            #    self.merid   = cuda.to_device(merid,  self.stream)
            #    self.merids  = cuda.to_device(merids, self.stream)
            #    self.meridvb = cuda.to_device(meridvb, self.stream)

            #    self.Ad            = cuda.to_device(self.A, self.stream)
            #    self.dxjsd         = cuda.to_device(self.dxjs, self.stream)
            #    self.corfd         = cuda.to_device(self.corf, self.stream)
            #    self.corf_isd      = cuda.to_device(self.corf_is, self.stream)
            #    self.lat_radd      = cuda.to_device(self.lat_rad, self.stream)
            #    self.latis_radd    = cuda.to_device(self.latis_rad, self.stream)
            #    # dsigma and sigma_vb are copied in fields (not very nice... TODO)

    
        # STUFF THAT SHOULD HAPPEN BOTH FOR NEW GRID OR LOADED GRID
        self.rad_1 = 0
        self.rad_2 = 0
        self.rad_sw = 0
        self.rad_lw = 0
        self.rad_lwsolv = 0
        self.rad_swsolv = 0





def lat_lon_recangle_area(lat,dlon,dlat, i_curved_earth):
    if i_curved_earth:
        A = np.cos(lat) * \
                dlon * dlat * con_rE**2
    else:
        A = dlon * dlat * con_rE**2
    return(A)
