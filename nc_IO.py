import numpy as np
import time
from netCDF4 import Dataset
from namelist import output_path, output_fields, pTop
from namelist import i_radiation, \
                     i_microphysics, i_surface_scheme
from IO_helper_functions import NC_output_diagnostics


def output_to_NC(GR, F, RAD, SURF, MIC):

    print('###########################################')
    print('###########################################')
    print('WRITE FIELDS '+str(GR.nc_output_count).zfill(4))
    print('###########################################')
    print('###########################################')

    # PREPARATIONS
    ####################################################################
    #VORT, F.PAIR, F.TAIR, WWIND_ms,\
    VORT, WWIND_ms, WVP, CWP = NC_output_diagnostics(GR, F, F.UWIND, 
                            F.VWIND, F.WWIND, F.POTT, F.COLP, F.PVTF, F.PVTFVB,
                            F.PHI, F.PHIVB, F.RHO, MIC)

    # CREATE AND OPEN FILE
    ####################################################################
    filename = output_path+'/out'+str(GR.nc_output_count).zfill(4)+'.nc'
    ncf = Dataset(filename, 'w', format='NETCDF4')
    ncf.close()
    ncf = Dataset(filename, 'a', format='NETCDF4')

    # DIMENSIONS
    ####################################################################
    time_dim = ncf.createDimension('time', None)
    bnds_dim = ncf.createDimension('bnds', 1)
    lon_dim = ncf.createDimension('lon', GR.nx)
    lons_dim = ncf.createDimension('lons', GR.nxs)
    lat_dim = ncf.createDimension('lat', GR.ny)
    lats_dim = ncf.createDimension('lats', GR.nys)
    level_dim = ncf.createDimension('level', GR.nz)
    levels_dim = ncf.createDimension('levels', GR.nzs)

    # DIMENSION VARIABLES
    ####################################################################
    dtime = ncf.createVariable('time', 'f8', ('time',) )
    bnds = ncf.createVariable('bnds', 'f8', ('bnds',) )
    lon = ncf.createVariable('lon', 'f4', ('lon',) )
    lons = ncf.createVariable('lons', 'f4', ('lons',) )
    lat = ncf.createVariable('lat', 'f4', ('lat',) )
    lats = ncf.createVariable('lats', 'f4', ('lats',) )
    level = ncf.createVariable('level', 'f4', ('level',) )
    levels = ncf.createVariable('levels', 'f4', ('levels',) )

    dtime[:] = GR.sim_time_sec/3600/24
    bnds[:] = [0]
    lon[:] = GR.lon_rad[GR.ii,GR.nb+1]
    lons[:] = GR.lon_is_rad[GR.iis,GR.nb+1]
    lat[:] = GR.lat_rad[GR.nb+1,GR.jj]
    lats[:] = GR.lat_js_rad[GR.nb+1,GR.jjs]
    level[:] = GR.level
    levels[:] = GR.levels

    ####################################################################
    ##############################################################################
    # 2D FIELDS
    ##############################################################################
    ####################################################################

    # pressure fields
    ##############################################################################
    if output_fields['PSURF']:
        PSURF_out = ncf.createVariable('PSURF', 'f4', ('time', 'lat', 'lon',) )
        PSURF_out[-1,:,:] = F.COLP[GR.iijj].T + pTop

    # flux fields
    ##############################################################################

    # velocity fields
    ##############################################################################

    # temperature fields
    ##############################################################################

    # primary diagnostic fields (relevant for dynamics)
    ##############################################################################

    # secondary diagnostic fields (not relevant for dynamics)
    ##############################################################################

    # constant fields
    ##############################################################################

    # radiation fields
    ##############################################################################

    # microphysics fields
    ##############################################################################



    ####################################################################
    ##############################################################################
    # 3D FIELDS
    ##############################################################################
    ####################################################################



    # pressure fields
    ##############################################################################

    # flux fields
    ##############################################################################
 
    # velocity fields
    ##############################################################################
    if output_fields['UWIND']:
        UWIND_out = ncf.createVariable('UWIND', 'f4', ('time', 'level', 'lat', 'lons',) )
        UWIND_out[-1,:,:,:] = F.UWIND[:,:,:][GR.iisjj].T
    if output_fields['VWIND']:
        VWIND_out = ncf.createVariable('VWIND', 'f4', ('time', 'level', 'lats', 'lon',) )
        VWIND_out[-1,:,:,:] = F.VWIND[:,:,:][GR.iijjs].T
    if output_fields['WIND']:
        WIND_out = ncf.createVariable('WIND', 'f4', ('time', 'level', 'lat', 'lon',) )
        WIND_out[-1,:,:,:] = F.WIND[:,:,:][GR.iijj].T
    if output_fields['WWIND']:
        WWIND_out = ncf.createVariable('WWIND', 'f4', ('time', 'levels', 'lat', 'lon',) )
        for ks in range(0,GR.nzs):
            WWIND_out[-1,ks,:,:] = (F.WWIND[:,:,ks][GR.iijj]*F.COLP[GR.iijj]).T
        #WWIND_out[-1,ks,:,:] = WWIND_ms[:,:,ks][GR.iijj].T
    if output_fields['VORT']:
        VORT_out = ncf.createVariable('VORT', 'f4', ('time', 'level', 'lat', 'lon',) )
        VORT_out[-1,:,:,:] = VORT[:,:,:][GR.iijj].T

    # temperature fields
    ##############################################################################
    if output_fields['POTT']:
        POTT_out = ncf.createVariable('POTT', 'f4', ('time', 'level', 'lat', 'lon',) )
        POTT_out[-1,:,:,:] = F.POTT[:,:,:][GR.iijj].T
    if output_fields['TAIR']:
        TAIR_out = ncf.createVariable('TAIR', 'f4', ('time', 'level', 'lat', 'lon',) )
        TAIR_out[-1,:,:,:] = F.TAIR[:,:,:][GR.iijj].T

    # primary diagnostic fields (relevant for dynamics)
    ##############################################################################
    if output_fields['PHI']:
        PHI_out = ncf.createVariable('PHI', 'f4', ('time', 'level', 'lat', 'lon',) )
        PHI_out[-1,:,:,:] = F.PHI[:,:,:][GR.iijj].T

    # secondary diagnostic fields (not relevant for dynamics)
    ##############################################################################
    if output_fields['PAIR']:
        PAIR_out = ncf.createVariable('PAIR', 'f4', ('time', 'level', 'lat', 'lon',) )
        PAIR_out[-1,:,:,:] = F.PAIR[:,:,:][GR.iijj].T
    if output_fields['RHO']:
        RHO_out = ncf.createVariable('RHO', 'f4', ('time', 'level', 'lat', 'lon',) )
        RHO_out[-1,:,:,:] = F.RHO[:,:,:][GR.iijj].T

    # constant fields
    ##############################################################################

    # radiation fields
    ##############################################################################

    # microphysics fields
    ##############################################################################
    if output_fields['QV']:
        QV_out = ncf.createVariable('QV', 'f4', ('time', 'level', 'lat', 'lon',) )
        QV_out[-1,:,:,:] = F.QV[:,:,:][GR.iijj].T
    if output_fields['QC']:
        QC_out = ncf.createVariable('QC', 'f4', ('time', 'level', 'lat', 'lon',) )
        QC_out[-1,:,:,:] = F.QC[:,:,:][GR.iijj].T
    if output_fields['WVP']:
        WVP_out = ncf.createVariable('WVP', 'f4', ('time', 'lat', 'lon',) )
        WVP_out[-1,:,:] = WVP.T
    if output_fields['CWP']:
        CWP_out = ncf.createVariable('CWP', 'f4', ('time', 'lat', 'lon',) )
        CWP_out[-1,:,:] = CWP.T



    ####################################################################
    ##############################################################################
    # PROFILES OF CERTAIN FIELDS
    ##############################################################################
    ####################################################################
    if output_fields['UWIND'] > 1:
        UWINDprof_out = ncf.createVariable('UWINDprof', 'f4', ('time', 'level', 'lat',) )
    if output_fields['VWIND'] > 1:
        VWINDprof_out = ncf.createVariable('VWINDprof', 'f4', ('time', 'level', 'lats',) )
    if output_fields['VORT'] > 1:
        VORTprof_out = ncf.createVariable('VORTprof', 'f4', ('time', 'level', 'lat',) )
    if output_fields['POTT'] > 1:
        POTTprof_out = ncf.createVariable('POTTprof', 'f4', ('time', 'level', 'lat',) )
    if output_fields['QV'] > 1:
        QVprof_out = ncf.createVariable('QVprof', 'f4', ('time', 'level', 'lat',) )
    if output_fields['QC'] > 1:
        QCprof_out = ncf.createVariable('QCprof', 'f4', ('time', 'level', 'lat',) )
    for k in range(0,GR.nz):
        if output_fields['UWIND'] > 1:
            UWINDprof_out[-1,GR.nz-k-1,:] = np.mean(F.UWIND[:,:,k][GR.iijj],axis=0)
        if output_fields['VWIND'] > 1:
            VWINDprof_out[-1,GR.nz-k-1,:] = np.mean(F.VWIND[:,:,k][GR.iijjs],axis=0)
        if output_fields['VORT'] > 1:
            VORTprof_out[-1,GR.nz-k-1,:] = np.mean(VORT[:,:,k][GR.iijj],axis=0)
        if output_fields['POTT'] > 1:
            POTTprof_out[-1,GR.nz-k-1,:] = np.mean(F.POTT[:,:,k][GR.iijj],axis=0)
        if output_fields['QV'] > 1:
            QVprof_out[-1,GR.nz-k-1,:] = np.mean(F.QV[:,:,k][GR.iijj],axis=0)
        if output_fields['QC'] > 1:
            QCprof_out[-1,GR.nz-k-1,:] = np.mean(F.QC[:,:,k][GR.iijj],axis=0)







    # RADIATION VARIABLES
    if i_radiation:
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

    # SURF VARIABLES
    if i_surface_scheme:
        if output_fields['SURFTEMP']:
            SURFTEMP_out = ncf.createVariable('SURFTEMP', 'f4', ('time', 'lat', 'lon',) )
            SURFTEMP_out[-1,:,:] = F.SOILTEMP[:,:,0].T
        #if i_microphysics:
        #    SOILMOIST_out = ncf.createVariable('SOILMOIST', 'f4', ('time', 'lat', 'lon',) )
        #    RAINRATE_out = ncf.createVariable('RAINRATE', 'f4', ('time', 'lat', 'lon',) )
        #    ACCRAIN_out = ncf.createVariable('ACCRAIN', 'f4', ('time', 'lat', 'lon',) )
        #    SOILEVAPITY_out = ncf.createVariable('SOILEVAPITY', 'f4', ('time', 'lat', 'lon',) )
        if i_radiation:
            if output_fields['SURFALBEDSW']:
                SURFALBEDSW_out = ncf.createVariable('SURFALBEDSW', 'f4',
                                                    ('time', 'lat', 'lon',) )
                SURFALBEDSW_out[0,:,:] = F.SURFALBEDSW.T
            if output_fields['SURFALBEDLW']:
                SURFALBEDLW_out = ncf.createVariable('SURFALBEDLW', 'f4',
                                                    ('time', 'lat', 'lon',) )
                SURFALBEDLW_out[0,:,:] = F.SURFALBEDLW.T


    # MICROPHYSICS VARIABLES
    if i_microphysics:
        RH_out         = ncf.createVariable('RH', 'f4', ('time', 'level', 'lat', 'lon',) )
        dQVdt_MIC_out  = ncf.createVariable('dQVdt_MIC', 'f4',
                                            ('time', 'level', 'lat', 'lon',) )
        dQCdt_MIC_out  = ncf.createVariable('dQCdt_MIC', 'f4',
                                            ('time', 'level', 'lat', 'lon',) )
        dPOTTdt_MIC_out=ncf.createVariable('dPOTTdt_MIC', 'f4',
                                            ('time', 'level', 'lat', 'lon',) )



    ################################################################################
    ################################################################################
    ################################################################################

    if i_surface_scheme:
        pass
        #if i_microphysics:
        #    SOILMOIST_out[-1,:,:] = SURF.MOIST.T
        #    RAINRATE_out[-1,:,:] = SURF.RAINRATE.T*3600 # mm/h
        #    ACCRAIN_out[-1,:,:] = SURF.ACCRAIN.T # mm
        #    SOILEVAPITY_out[-1,:,:] = SURF.SOILEVAPITY.T


    for k in range(0,GR.nz):

        # RADIATION VARIABLES
        if i_radiation > 0:
            dPOTTdt_RAD_out[-1,k,:,:] = F.dPOTTdt_RAD[:,:,k].T * 3600
            #SWFLXDIV_out[-1,k,:,:] = RAD.SWFLXDIV[:,:,k].T 
            #LWFLXDIV_out[-1,k,:,:] = RAD.LWFLXDIV[:,:,k].T 

        # MICROPHYSICS VARIABLES
        if i_microphysics:
            RH_out[-1,k,:,:] = MIC.RH[:,:,k].T
            dQVdt_MIC_out[-1,k,:,:] = F.dQVdt_MIC[:,:,k].T * 3600
            dQCdt_MIC_out[-1,k,:,:] = F.dQCdt_MIC[:,:,k].T * 3600
            dPOTTdt_MIC_out[-1,k,:,:] = F.dPOTTdt_MIC[:,:,k].T * 3600



    for ks in range(0,GR.nzs):


        # RADIATION VARIABLES
        if i_radiation > 0:
            SWDIFFLXDO_out[-1,ks,:,:] = RAD.SWDIFFLXDO[:,:,ks].T
            SWDIRFLXDO_out[-1,ks,:,:] = RAD.SWDIRFLXDO[:,:,ks].T
            SWFLXUP_out[-1,ks,:,:] = RAD.SWFLXUP[:,:,ks].T
            SWFLXDO_out[-1,ks,:,:] = RAD.SWFLXDO[:,:,ks].T
            SWFLXNET_out[-1,ks,:,:] = F.SWFLXNET[:,:,ks].T
            LWFLXUP_out[-1,ks,:,:] = RAD.LWFLXUP[:,:,ks].T
            LWFLXDO_out[-1,ks,:,:] = RAD.LWFLXDO[:,:,ks].T
            LWFLXNET_out[-1,ks,:,:] = F.LWFLXNET[:,:,ks].T


    ncf.close()




def constant_fields_to_NC(GR, F, RAD, SURF):

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
    lons[:] = GR.lon_is_rad[GR.iis,GR.nb+1]
    lat[:] = GR.lat_rad[GR.nb+1,GR.jj]
    lats[:] = GR.lat_js_rad[GR.nb+1,GR.jjs]
    level[:] = GR.level
    levels[:] = GR.levels


    # VARIABLES
    HSURF_out = ncf.createVariable('HSURF', 'f4', ('lat', 'lon',) )
    HSURF_out[:,:] = F.HSURF[GR.iijj].T

    # SURF VARIABLES
    if i_surface_scheme:
        OCEANMASK_out = ncf.createVariable('OCEANMASK', 'f4', ('lat', 'lon',) )
        OCEANMASK_out[:,:] = F.OCEANMASK.T

    # RADIATION VARIABLES
    ncf.close()


