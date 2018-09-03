import numpy as np
import time
from netCDF4 import Dataset
from namelist import output_path, pTop
from IO_helper_functions import NC_output_diagnostics


def output_to_NC(GR, outCounter, COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,
                POTT, TAIR, RHO, PVTF, PVTFVB,
                RAD, SOIL, MIC):

    print('###########################################')
    print('###########################################')
    print('write fields')
    print('###########################################')
    print('###########################################')

    VORT, PAIR, TAIR, WWIND_ms,\
    WVP, CWP                 = NC_output_diagnostics(GR, UWIND, 
                        VWIND, WWIND, POTT, COLP, PVTF, PVTFVB,
                        PHI, PHIVB, RHO, MIC)

    filename = output_path+'/out'+str(outCounter).zfill(4)+'.nc'

    ncf = Dataset(filename, 'w', format='NETCDF4')
    ncf.close()

    ncf = Dataset(filename, 'a', format='NETCDF4')

    # DIMENSIONS
    time_dim = ncf.createDimension('time', None)
    #bnds_dim = ncf.createDimension('bnds', 2)
    bnds_dim = ncf.createDimension('bnds', 1)
    lon_dim = ncf.createDimension('lon', GR.nx)
    lons_dim = ncf.createDimension('lons', GR.nxs)
    lat_dim = ncf.createDimension('lat', GR.ny)
    lats_dim = ncf.createDimension('lats', GR.nys)
    level_dim = ncf.createDimension('level', GR.nz)
    levels_dim = ncf.createDimension('levels', GR.nzs)

    # DIMENSION VARIABLES
    dtime = ncf.createVariable('time', 'f8', ('time',) )
    bnds = ncf.createVariable('bnds', 'f8', ('bnds',) )
    lon = ncf.createVariable('lon', 'f4', ('lon',) )
    lons = ncf.createVariable('lons', 'f4', ('lons',) )
    lat = ncf.createVariable('lat', 'f4', ('lat',) )
    lats = ncf.createVariable('lats', 'f4', ('lats',) )
    level = ncf.createVariable('level', 'f4', ('level',) )
    levels = ncf.createVariable('levels', 'f4', ('levels',) )

    dtime[:] = GR.sim_time_sec/3600/24
    #bnds[:] = [0,1]
    bnds[:] = [0]
    lon[:] = GR.lon_rad[GR.ii,GR.nb+1]
    lons[:] = GR.lonis_rad[GR.iis,GR.nb+1]
    lat[:] = GR.lat_rad[GR.nb+1,GR.jj]
    lats[:] = GR.latjs_rad[GR.nb+1,GR.jjs]
    level[:] = GR.level
    levels[:] = GR.levels

    # VARIABLES
    ALBSFCSW_out = ncf.createVariable('ALBSFCSW', 'f4', ('time', 'lat', 'lon',) )
    PSURF_out = ncf.createVariable('PSURF', 'f4', ('time', 'lat', 'lon',) )
    #PAIR_out = ncf.createVariable('PAIR', 'f4', ('time', 'level', 'lat', 'lon',) )
    PHI_out = ncf.createVariable('PHI', 'f4', ('time', 'level', 'lat', 'lon',) )
    UWIND_out = ncf.createVariable('UWIND', 'f4', ('time', 'level', 'lat', 'lons',) )
    VWIND_out = ncf.createVariable('VWIND', 'f4', ('time', 'level', 'lats', 'lon',) )
    WIND_out = ncf.createVariable('WIND', 'f4', ('time', 'level', 'lat', 'lon',) )
    VORT_out = ncf.createVariable('VORT', 'f4', ('time', 'level', 'lat', 'lon',) )
    WWIND_out = ncf.createVariable('WWIND', 'f4', ('time', 'levels', 'lat', 'lon',) )
    POTT_out = ncf.createVariable('POTT', 'f4', ('time', 'level', 'lat', 'lon',) )
    TAIR_out = ncf.createVariable('TAIR', 'f4', ('time', 'level', 'lat', 'lon',) )
    #RHO_out = ncf.createVariable('RHO', 'f4', ('time', 'level', 'lat', 'lon',) )


    # RADIATION VARIABLES
    SWDIFFLXDO_out =ncf.createVariable('SWDIFFLXDO', 'f4', ('time', 'levels', 'lat', 'lon',) )
    SWDIRFLXDO_out =ncf.createVariable('SWDIRFLXDO', 'f4', ('time', 'levels', 'lat', 'lon',) )
    SWFLXUP_out = ncf.createVariable('SWFLXUP', 'f4', ('time', 'levels', 'lat', 'lon',) )
    SWFLXDO_out = ncf.createVariable('SWFLXDO', 'f4', ('time', 'levels', 'lat', 'lon',) )
    SWFLXNET_out = ncf.createVariable('SWFLXNET', 'f4', ('time', 'levels', 'lat', 'lon',) )
    LWFLXUP_out = ncf.createVariable('LWFLXUP', 'f4', ('time', 'levels', 'lat', 'lon',) )
    LWFLXDO_out = ncf.createVariable('LWFLXDO', 'f4', ('time', 'levels', 'lat', 'lon',) )
    LWFLXNET_out = ncf.createVariable('LWFLXNET', 'f4', ('time', 'levels', 'lat', 'lon',) )
    dPOTTdt_RAD_out=ncf.createVariable('dPOTTdt_RAD', 'f4', ('time', 'level', 'lat', 'lon',) )
    #SWFLXDIV_out = ncf.createVariable('SWFLXDIV', 'f4', ('time', 'level', 'lat', 'lon',) )
    #LWFLXDIV_out = ncf.createVariable('LWFLXDIV', 'f4', ('time', 'level', 'lat', 'lon',) )

    # MICROPHYSICS VARIABLES
    QV_out = ncf.createVariable('QV', 'f4', ('time', 'level', 'lat', 'lon',) )
    QC_out = ncf.createVariable('QC', 'f4', ('time', 'level', 'lat', 'lon',) )
    RH_out = ncf.createVariable('RH', 'f4', ('time', 'level', 'lat', 'lon',) )
    dQVdt_MIC_out = ncf.createVariable('dQVdt_MIC', 'f4', ('time', 'level', 'lat', 'lon',) )
    dQCdt_MIC_out = ncf.createVariable('dQCdt_MIC', 'f4', ('time', 'level', 'lat', 'lon',) )
    WVP_out = ncf.createVariable('WVP', 'f4', ('time', 'lat', 'lon',) )
    CWP_out = ncf.createVariable('CWP', 'f4', ('time', 'lat', 'lon',) )

    # SOIL VARIABLES
    TSURF_out = ncf.createVariable('TSURF', 'f4', ('time', 'lat', 'lon',) )
    SOILMOIST_out = ncf.createVariable('SOILMOIST', 'f4', ('time', 'lat', 'lon',) )
    RAINRATE_out = ncf.createVariable('RAINRATE', 'f4', ('time', 'lat', 'lon',) )
    ACCRAIN_out = ncf.createVariable('ACCRAIN', 'f4', ('time', 'lat', 'lon',) )
    EVAPITY_out = ncf.createVariable('EVAPITY', 'f4', ('time', 'lat', 'lon',) )
    dPOTTdt_MIC_out=ncf.createVariable('dPOTTdt_MIC', 'f4', ('time', 'level', 'lat', 'lon',) )

    # VERTICAL PROFILES
    POTTprof_out = ncf.createVariable('POTTprof', 'f4', ('time', 'level', 'lat',) )
    UWINDprof_out = ncf.createVariable('UWINDprof', 'f4', ('time', 'level', 'lat',) )
    VWINDprof_out = ncf.createVariable('VWINDprof', 'f4', ('time', 'level', 'lats',) )
    WWINDprof_out = ncf.createVariable('WWINDprof', 'f4', ('time', 'levels', 'lat',) )
    VORTprof_out = ncf.createVariable('VORTprof', 'f4', ('time', 'level', 'lat',) )
    QVprof_out = ncf.createVariable('QVprof', 'f4', ('time', 'level', 'lat',) )
    QCprof_out = ncf.createVariable('QCprof', 'f4', ('time', 'level', 'lat',) )

    ################################################################################
    ################################################################################
    ################################################################################

    ALBSFCSW_out[0,:,:] = SOIL.ALBEDOSW.T
    PSURF_out[-1,:,:] = COLP[GR.iijj].T + pTop
    TSURF_out[-1,:,:] = SOIL.TSOIL[:,:,0].T
    SOILMOIST_out[-1,:,:] = SOIL.MOIST.T
    RAINRATE_out[-1,:,:] = SOIL.RAINRATE.T*3600 # mm/h
    ACCRAIN_out[-1,:,:] = SOIL.ACCRAIN.T # mm
    EVAPITY_out[-1,:,:] = SOIL.EVAPITY.T

    WVP_out[-1,:,:] = WVP.T
    CWP_out[-1,:,:] = CWP.T

    for k in range(0,GR.nz):
        # DYNAMIC VARIABLES
        #PAIR_out[-1,k,:,:] = PAIR[:,:,k][GR.iijj].T
        PHI_out[-1,k,:,:] = PHI[:,:,k][GR.iijj].T
        UWIND_out[-1,k,:,:] = UWIND[:,:,k][GR.iisjj].T
        VWIND_out[-1,k,:,:] = VWIND[:,:,k][GR.iijjs].T
        WIND_out[-1,k,:,:] = WIND[:,:,k][GR.iijj].T
        VORT_out[-1,k,:,:] = VORT[:,:,k][GR.iijj].T
        POTT_out[-1,k,:,:] = POTT[:,:,k][GR.iijj].T
        TAIR_out[-1,k,:,:] = TAIR[:,:,k][GR.iijj].T
        #RHO_out[-1,k,:,:] = RHO[:,:,k][GR.iijj].T

        # RADIATION VARIABLES
        if RAD.i_radiation > 0:
            dPOTTdt_RAD_out[-1,k,:,:] = RAD.dPOTTdt_RAD[:,:,k].T * 3600
        #SWFLXDIV_out[-1,k,:,:] = RAD.SWFLXDIV[:,:,k].T 
        #LWFLXDIV_out[-1,k,:,:] = RAD.LWFLXDIV[:,:,k].T 

        # MICROPHYSICS VARIABLES
        QV_out[-1,k,:,:] = MIC.QV[:,:,k][GR.iijj].T
        QC_out[-1,k,:,:] = MIC.QC[:,:,k][GR.iijj].T
        RH_out[-1,k,:,:] = MIC.RH[:,:,k].T
        dQVdt_MIC_out[-1,k,:,:] = MIC.dQVdt_MIC[:,:,k].T * 3600
        dQCdt_MIC_out[-1,k,:,:] = MIC.dQCdt_MIC[:,:,k].T * 3600
        dPOTTdt_MIC_out[-1,k,:,:] = MIC.dPOTTdt_MIC[:,:,k].T * 3600

        # VERTICAL PROFILES
        POTTprof_out[-1,GR.nz-k-1,:] = np.mean(POTT[:,:,k][GR.iijj],axis=0)
        UWINDprof_out[-1,GR.nz-k-1,:] = np.mean(UWIND[:,:,k][GR.iijj],axis=0)
        VWINDprof_out[-1,GR.nz-k-1,:] = np.mean(VWIND[:,:,k][GR.iijjs],axis=0)
        VORTprof_out[-1,GR.nz-k-1,:] = np.mean(VORT[:,:,k][GR.iijj],axis=0)
        QVprof_out[-1,GR.nz-k-1,:] = np.mean(MIC.QV[:,:,k][GR.iijj],axis=0)
        QCprof_out[-1,GR.nz-k-1,:] = np.mean(MIC.QC[:,:,k][GR.iijj],axis=0)


    for ks in range(0,GR.nzs):
        # DYNAMIC VARIABLES
        #WWIND_out[-1,ks,:,:] = WWIND_ms[:,:,ks][GR.iijj].T
        WWIND_out[-1,ks,:,:] = (WWIND[:,:,ks][GR.iijj]*COLP[GR.iijj]).T


        # RADIATION VARIABLES
        if RAD.i_radiation > 0:
            SWDIFFLXDO_out[-1,ks,:,:] = RAD.SWDIFFLXDO[:,:,ks].T
            SWDIRFLXDO_out[-1,ks,:,:] = RAD.SWDIRFLXDO[:,:,ks].T
            SWFLXUP_out[-1,ks,:,:] = RAD.SWFLXUP[:,:,ks].T
            SWFLXDO_out[-1,ks,:,:] = RAD.SWFLXDO[:,:,ks].T
            SWFLXNET_out[-1,ks,:,:] = RAD.SWFLXNET[:,:,ks].T
            LWFLXUP_out[-1,ks,:,:] = RAD.LWFLXUP[:,:,ks].T
            LWFLXDO_out[-1,ks,:,:] = RAD.LWFLXDO[:,:,ks].T
            LWFLXNET_out[-1,ks,:,:] = RAD.LWFLXNET[:,:,ks].T

        # VERTICAL PROFILES
        WWINDprof_out[-1,GR.nzs-ks-1,:] = np.mean(WWIND_ms[:,:,ks][GR.iijj],axis=0)

    ncf.close()




def constant_fields_to_NC(GR, HSURF, RAD, SOIL):

    print('###########################################')
    print('###########################################')
    print('write constant fields')
    print('###########################################')
    print('###########################################')

    filename = output_path+'/constants.nc'

    ncf = Dataset(filename, 'w', format='NETCDF4')
    ncf.close()

    ncf = Dataset(filename, 'a', format='NETCDF4')

    # DIMENSIONS
    lon_dim = ncf.createDimension('lon', GR.nx)
    lons_dim = ncf.createDimension('lons', GR.nxs)
    lat_dim = ncf.createDimension('lat', GR.ny)
    lats_dim = ncf.createDimension('lats', GR.nys)
    level_dim = ncf.createDimension('level', GR.nz)
    levels_dim = ncf.createDimension('levels', GR.nzs)

    # DIMENSION VARIABLES
    lon = ncf.createVariable('lon', 'f4', ('lon',) )
    lons = ncf.createVariable('lons', 'f4', ('lons',) )
    lat = ncf.createVariable('lat', 'f4', ('lat',) )
    lats = ncf.createVariable('lats', 'f4', ('lats',) )
    level = ncf.createVariable('level', 'f4', ('level',) )
    levels = ncf.createVariable('levels', 'f4', ('levels',) )

    lon[:] = GR.lon_rad[GR.ii,GR.nb+1]
    lons[:] = GR.lonis_rad[GR.iis,GR.nb+1]
    lat[:] = GR.lat_rad[GR.nb+1,GR.jj]
    lats[:] = GR.latjs_rad[GR.nb+1,GR.jjs]
    level[:] = GR.level
    levels[:] = GR.levels

    # VARIABLES
    HSURF_out = ncf.createVariable('HSURF', 'f4', ('lat', 'lon',) )

    # RADIATION VARIABLES

    # SOIL VARIABLES
    OCEANMSK_out = ncf.createVariable('OCEANMSK', 'f4', ('lat', 'lon',) )

    HSURF_out[:,:] = SOIL.HSURF[GR.iijj].T
    OCEANMSK_out[:,:] = SOIL.OCEANMSK.T
    for k in range(0,GR.nz):
        pass

    ncf.close()


