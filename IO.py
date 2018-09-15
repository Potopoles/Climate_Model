import numpy as np
import os
import pickle
from netCDF4 import Dataset
from scipy.interpolate import interp2d
from boundaries import exchange_BC_rigid_y, exchange_BC_periodic_x
from namelist import pTop, n_topo_smooth, tau_topo_smooth
#from geopotential import diag_pvt_factor
from constants import con_kappa
from scipy import interpolate
from constants import con_g, con_Rd, con_kappa, con_cp

def diag_pvt_factor(GR, COLP, PVTF, PVTFVB):
    PAIRVB = np.full( (GR.nx+2*GR.nb, GR.ny+2*GR.nb, GR.nzs), np.nan )

    # TODO: WHY IS PAIRVB NOT FILLED AT UPPERMOST AND LOWER MOST HALFLEVEL??? 
    for ks in range(0,GR.nzs):
        PAIRVB[:,:,ks][GR.iijj] = pTop + GR.sigma_vb[ks] * COLP[GR.iijj]
    
    PVTFVB = np.power(PAIRVB/100000, con_kappa)

    for k in range(0,GR.nz):
        kp1 = k + 1
        PVTF[:,:,k][GR.iijj] = 1/(1+con_kappa) * \
                    ( PVTFVB[:,:,kp1][GR.iijj] * PAIRVB[:,:,kp1][GR.iijj] - \
                      PVTFVB[:,:,k  ][GR.iijj] * PAIRVB[:,:,k  ][GR.iijj] ) / \
                    ( PAIRVB[:,:,kp1][GR.iijj] - PAIRVB[:,:,k  ][GR.iijj] )

    return(PVTF, PVTFVB)



def load_profile(GR, subgrids, COLP, HSURF, PSURF, PVTF, PVTFVB, POTT, TAIR):
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
    z_vb_test[ks] = zsurf_test + (ztop_test - zsurf_test)*(1 - ks/GR.nz)**(2)
    #z_vb_test[ks] = zsurf_test + (ztop_test - zsurf_test)*(1 - ks/GR.nz)
    #print(z_vb_test)
    #print(np.diff(z_vb_test))
    #quit()

    rho_vb_test = np.interp(z_vb_test, profile[:,0], profile[:,4]) 
    g_vb_test = np.interp(z_vb_test, profile[:,0], profile[:,1]) 
    p_vb_test[0] = pTop
    ks = 1
    for ks in range(1,GR.nzs):
        p_vb_test[ks] = p_vb_test[ks-1] + \
                        rho_vb_test[ks]*g_vb_test[ks] * \
                        (z_vb_test[ks-1] - z_vb_test[ks])
    
    GR.sigma_vb = (p_vb_test - pTop)/(p_vb_test[-1] - pTop)
    GR.dsigma = np.diff(GR.sigma_vb)
    for key,subgrid in subgrids.items():
        subgrid.sigma_vb = GR.sigma_vb
        subgrid.dsigma = GR.dsigma
        subgrids[key] = subgrid

    for i in GR.ii:
        for j in GR.jj:
            PSURF[i,j] = np.interp(HSURF[i,j], profile[:,0], profile[:,2])

    COLP[GR.iijj] = PSURF[GR.iijj] - pTop
    PVTF, PVTFVB = diag_pvt_factor(GR, COLP, PVTF, PVTFVB)

    PAIR =  np.full( (GR.nx +2*GR.nb, GR.ny +2*GR.nb, GR.nz ), np.nan)
    for k in range(0,GR.nz):
        PAIR[:,:,k][GR.iijj] = 100000*np.power(PVTF[:,:,k][GR.iijj], 1/con_kappa)

    interp = interpolate.interp1d(profile[:,2], profile[:,3])
    for i in GR.ii:
        for j in GR.jj:
            TAIR[i,j,:] = interp(PAIR[i,j,:])

    for k in range(0,GR.nz):
        POTT[:,:,k][GR.iijj] = TAIR[:,:,k][GR.iijj] * \
                np.power(100000/PAIR[:,:,k][GR.iijj], con_kappa)

    return(GR, COLP, PSURF, POTT, TAIR)


def write_restart(GR, COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
                        UFLX, VFLX, \
                        HSURF, POTT, TAIR, TAIRVB, RHO,\
                        POTTVB, PVTF, PVTFVB,
                        RAD, SOIL, MIC, TURB):

    print('###########################################')
    print('###########################################')
    print('write RESTART')
    print('###########################################')
    print('###########################################')
 
    filename = '../restart/'+str(GR.dlat_deg).zfill(2) + '_' +\
                            str(GR.dlon_deg).zfill(2) + '_' +\
                            str(GR.nz).zfill(3)+'.pkl'
    out = {}
    #GR.stream  = None 
    #GR.zonal   = None 
    #GR.zonals  = None 
    #GR.zonalvb = None 
    #GR.merid   = None 
    #GR.merids  = None 
    #GR.meridvb = None 
    #GR.dsigma = None 
    #GR.sigma_vb = None 
    out['GR'] = GR
    out['COLP'] = COLP
    out['PAIR'] = PAIR
    out['PHI'] = PHI
    out['PHIVB'] = PHIVB
    out['UWIND'] = UWIND
    out['VWIND'] = VWIND
    out['WIND'] = WIND
    out['WWIND'] = WWIND
    out['UFLX'] = UFLX
    out['VFLX'] = VFLX
    out['HSURF'] = HSURF
    out['POTT'] = POTT
    out['TAIR'] = TAIR
    out['TAIRVB'] = TAIRVB
    out['RHO'] = RHO
    out['POTTVB'] = POTTVB
    out['PVTF'] = PVTF
    out['PVTFVB'] = PVTFVB
    out['RAD'] = RAD
    out['SOIL'] = SOIL
    out['MIC'] = MIC
    out['TURB'] = TURB
    with open(filename, 'wb') as f:
        pickle.dump(out, f)

def load_restart_grid(dlat_deg, dlon_deg, nz):
    filename = '../restart/'+str(dlat_deg).zfill(2) + '_' +\
                            str(dlon_deg).zfill(2) + '_' +\
                            str(nz).zfill(3)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    else:
        raise ValueError('Restart File does not exist.')
    GR = inp['GR']
    return(GR)

def load_restart_fields(GR):
    filename = '../restart/'+str(GR.dlat_deg).zfill(2) + '_' +\
                            str(GR.dlon_deg).zfill(2) + '_' +\
                            str(GR.nz).zfill(3)+'.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            inp = pickle.load(f)
    COLP = inp['COLP']
    PAIR = inp['PAIR']
    PHI = inp['PHI']
    PHIVB = inp['PHIVB']
    UWIND = inp['UWIND']
    VWIND = inp['VWIND']
    WIND = inp['WIND']
    WWIND = inp['WWIND']
    UFLX = inp['UFLX']
    VFLX = inp['VFLX']
    HSURF = inp['HSURF']
    POTT = inp['POTT']
    TAIR = inp['TAIR']
    TAIRVB = inp['TAIRVB']
    RHO = inp['RHO']
    POTTVB = inp['POTTVB']
    PVTF = inp['PVTF']
    PVTFVB = inp['PVTFVB']
    RAD = inp['RAD']
    SOIL = inp['SOIL'] 
    MIC = inp['MIC'] 
    TURB = inp['TURB'] 
    return(COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND, \
                UFLX, VFLX, \
                HSURF, POTT, TAIR, TAIRVB, RHO, \
                POTTVB, PVTF, PVTFVB,
                RAD, SOIL, MIC, TURB)

def load_topo(GR, HSURF):
    #HSURF = np.full( (GR.nx+2*GR.nb,GR.ny+2*GR.nb), np.nan)
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


