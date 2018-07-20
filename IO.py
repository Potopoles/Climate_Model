import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp2d
from boundaries import exchange_BC_rigid_y, exchange_BC_periodic_x
import pickle
import os
from namelist import pTop, n_topo_smooth, tau_topo_smooth
from geopotential import diag_pvt_factor
from constants import con_kappa
from scipy import interpolate

def load_profile(GR, COLP, HSURF, PSURF, PVTF, PVTFVB, POTT):
    filename = 'verticalProfileTable.dat'
    profile = np.loadtxt(filename)
    #print(profile)
    zsurf_test = np.mean(HSURF[GR.iijj])
    top_ind = np.argwhere(profile[:,2] >= pTop).squeeze()[-1]
    ztop_test = profile[top_ind,0] + (profile[top_ind,2] - pTop)/ \
                            (profile[top_ind,4]*profile[top_ind,1])


    ks = np.arange(0,GR.nzs)
    z_vb_test = np.zeros(GR.nzs)
    p_vb_test = np.zeros(GR.nzs)
    rho_vb_test = np.zeros(GR.nzs)
    g_vb_test = np.zeros(GR.nzs)

    z_vb_test[0] = ztop_test
    z_vb_test[ks] = zsurf_test + (ztop_test - zsurf_test)*(1 - ks/GR.nz)

    rho_vb_test = np.interp(z_vb_test, profile[:,0], profile[:,4]) 
    g_vb_test = np.interp(z_vb_test, profile[:,0], profile[:,1]) 
    p_vb_test[0] = pTop
    ks = 1
    for ks in range(1,GR.nzs):
        p_vb_test[ks] = p_vb_test[ks-1] + rho_vb_test[ks]*g_vb_test[ks]*(z_vb_test[ks-1] - z_vb_test[ks])
    
    GR.sigma_vb = (p_vb_test - pTop)/(p_vb_test[-1] - pTop)
    GR.dsigma = np.diff(GR.sigma_vb)

    for i in GR.ii:
        for j in GR.jj:
            PSURF[i,j] = np.interp(HSURF[i,j], profile[:,0], profile[:,2])

    COLP[GR.iijj] = PSURF[GR.iijj] - pTop
    PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)

    PAIR =  np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz ), np.nan)
    for k in range(0,GR.nz):
        PAIR[:,:,k][GR.iijj] = 100000*np.power(PVTF[:,:,k][GR.iijj], 1/con_kappa)

    TAIR =  np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz ), np.nan)
    interp = interpolate.interp1d(profile[:,2], profile[:,3])
    for i in GR.ii:
        for j in GR.jj:
            TAIR[i,j,:] = interp(PAIR[i,j,:])

    for k in range(0,GR.nz):
        POTT[:,:,k][GR.iijj] = TAIR[:,:,k][GR.iijj] * np.power(100000/PAIR[:,:,k][GR.iijj], con_kappa)

    return(GR, COLP, PSURF, POTT)


def write_restart(GR, COLP, PHI, UWIND, VWIND, WIND, WWIND,\
                        UFLX, VFLX, UFLXMP, VFLXMP, \
                        HSURF, POTT, POTTVB, PVTF, PVTFVB):
    filename = '../restart/'+str(GR.dlat_deg).zfill(2)+'.pkl'
    out = {}
    out['GR'] = GR
    out['COLP'] = COLP
    out['PHI'] = PHI
    out['UWIND'] = UWIND
    out['VWIND'] = VWIND
    out['WIND'] = WIND
    out['WWIND'] = WWIND
    out['UFLX'] = UFLX
    out['VFLX'] = VFLX
    out['UFLXMP'] = UFLXMP
    out['VFLXMP'] = VFLXMP
    out['HSURF'] = HSURF
    out['POTT'] = POTT
    out['POTTVB'] = POTTVB
    out['PVTF'] = PVTF
    out['PVTFVB'] = PVTFVB
    with open(filename, 'wb') as f:
        pickle.dump(out, f)

def load_restart_grid(dlat_deg):
    filename = '../restart/'+str(dlat_deg).zfill(2)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    else:
        raise ValueError('Restart File does not exist.')
    GR = inp['GR']
    return(GR)

def load_restart_fields(GR):
    filename = '../restart/'+str(GR.dlat_deg).zfill(2)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    COLP = inp['COLP']
    PHI = inp['PHI']
    UWIND = inp['UWIND']
    VWIND = inp['VWIND']
    WIND = inp['WIND']
    WWIND = inp['WWIND']
    UFLX = inp['UFLX']
    VFLX = inp['VFLX']
    UFLXMP = inp['UFLXMP']
    VFLXMP = inp['VFLXMP']
    HSURF = inp['HSURF']
    POTT = inp['POTT']
    POTTVB = inp['POTTVB']
    PVTF = inp['PVTF']
    PVTFVB = inp['PVTFVB']
    return(COLP, PHI, UWIND, VWIND, WIND, WWIND, \
                UFLX, VFLX, UFLXMP, VFLXMP, \
                HSURF, POTT, POTTVB, PVTF, PVTFVB)



def load_topo(GR):
    HSURF = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
    filename = '../elevation/elev.1-deg.nc'
    ncf = Dataset(filename, 'r', format='NETCDF4')
    lon_inp = ncf['lon'][:]
    lat_inp = ncf['lat'][:]
    hsurf_inp = ncf['data'][0,:,:]
    interp = interp2d(lon_inp, lat_inp, hsurf_inp)
    HSURF[GR.iijj] = interp(GR.lon_deg[GR.ii,GR.nb+1], GR.lat_deg[GR.nb+1,GR.jj]).T
    HSURF[HSURF < 0] = 0
    HSURF = exchange_BC_periodic_x(GR, HSURF)
    HSURF = exchange_BC_rigid_y(GR, HSURF)

    for i in range(0,n_topo_smooth):
        HSURF[GR.iijj] = HSURF[GR.iijj] + tau_topo_smooth*\
                                            (HSURF[GR.iijj_im1] + HSURF[GR.iijj_ip1] + \
                                            HSURF[GR.iijj_jm1] + HSURF[GR.iijj_jp1] - \
                                            4*HSURF[GR.iijj]) 
        HSURF = exchange_BC_periodic_x(GR, HSURF)
        HSURF = exchange_BC_rigid_y(GR, HSURF)

    return(HSURF)


def output_to_NC(GR, outCounter, COLP, PHI, UWIND, VWIND, WIND, WWIND,
                HSURF, POTT,
                mean_wind):
    print('###########################################')
    print('###########################################')
    print('write fields')
    print('###########################################')
    print('###########################################')

    filename = '../output/out'+str(outCounter).zfill(4)+'.nc'

    ncf = Dataset(filename, 'w', format='NETCDF4')
    ncf.close()

    ncf = Dataset(filename, 'a', format='NETCDF4')

    # DIMENSIONS
    time_dim = ncf.createDimension('time', None)
    bnds_dim = ncf.createDimension('bnds', 2)
    lon_dim = ncf.createDimension('lon', GR.nx)
    lons_dim = ncf.createDimension('lons', GR.nxs)
    lat_dim = ncf.createDimension('lat', GR.ny)
    lats_dim = ncf.createDimension('lats', GR.nys)
    level_dim = ncf.createDimension('level', GR.nz)
    levels_dim = ncf.createDimension('levels', GR.nzs)

    # DIMENSION VARIABLES
    time = ncf.createVariable('time', 'f8', ('time',) )
    bnds = ncf.createVariable('bnds', 'f8', ('bnds',) )
    lon = ncf.createVariable('lon', 'f4', ('lon',) )
    lons = ncf.createVariable('lons', 'f4', ('lons',) )
    lat = ncf.createVariable('lat', 'f4', ('lat',) )
    lats = ncf.createVariable('lats', 'f4', ('lats',) )
    level = ncf.createVariable('level', 'f4', ('level',) )
    levels = ncf.createVariable('levels', 'f4', ('levels',) )

    time[:] = outCounter
    bnds[:] = [0,1]
    lon[:] = GR.lon_rad[GR.ii,GR.nb+1]
    lons[:] = GR.lonis_rad[GR.iis,GR.nb+1]
    lat[:] = GR.lat_rad[GR.nb+1,GR.jj]
    lats[:] = GR.latjs_rad[GR.nb+1,GR.jjs]
    level[:] = GR.level
    levels[:] = GR.levels

    # VARIABLES
    COLP_out = ncf.createVariable('COLP', 'f4', ('time', 'lat', 'lon',) )
    PHI_out = ncf.createVariable('PHI', 'f4', ('time', 'level', 'lat', 'lon',) )
    UWIND_out = ncf.createVariable('UWIND', 'f4', ('time', 'level', 'lat', 'lons',) )
    VWIND_out = ncf.createVariable('VWIND', 'f4', ('time', 'level', 'lats', 'lon',) )
    WIND_out = ncf.createVariable('WIND', 'f4', ('time', 'level', 'lat', 'lon',) )
    WWIND_out = ncf.createVariable('WWIND', 'f4', ('time', 'levels', 'lat', 'lon',) )
    HSURF_out = ncf.createVariable('HSURF', 'f4', ('bnds', 'lat', 'lon',) )
    POTT_out = ncf.createVariable('POTT', 'f4', ('time', 'level', 'lat', 'lon',) )
    mean_wind_out = ncf.createVariable('mean_wind', 'f4', ('time', 'bnds',) )

    COLP_out[-1,:,:] = COLP[GR.iijj].T
    HSURF_out[0,:,:] = HSURF[GR.iijj].T
    for k in range(0,GR.nz):
        PHI_out[-1,k,:,:] = PHI[:,:,k][GR.iijj].T
        WIND_out[-1,k,:,:] = WIND[:,:,k][GR.iijj].T
        UWIND_out[-1,k,:,:] = UWIND[:,:,k][GR.iisjj].T
        VWIND_out[-1,k,:,:] = VWIND[:,:,k][GR.iijjs].T
        POTT_out[-1,k,:,:] = POTT[:,:,k][GR.iijj].T
    for ks in range(0,GR.nzs):
        WWIND_out[-1,ks,:,:] = WWIND[:,:,ks][GR.iijj].T*COLP[GR.iijj].T
    mean_wind_out[-1,:] = mean_wind

    ncf.close()
