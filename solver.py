import numpy as np

from grid import Grid
from fields import initialize_fields
from IO import output_to_NC, write_restart
from namelist import i_time_stepping, i_spatial_discretization, \
                    i_load_from_restart, i_save_to_restart
from functions import diagnostics
if i_time_stepping == 'EULER_FORWARD':
    from time_integration import euler_forward as time_stepper
elif i_time_stepping == 'MATSUNO':
    from time_integration import matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from time_integration import RK4 as time_stepper

GR = Grid()

COLP, PHI, UWIND, VWIND, WIND, WWIND,\
UFLX, VFLX, UFLXMP, VFLXMP, \
UUFLX, VUFLX, UVFLX, VVFLX, \
HSURF, POTT, POTTVB, PVTF, PVTFVB = initialize_fields(GR)


if not i_load_from_restart:
    outCounter = 0
    WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
                                    WIND, UWIND, VWIND, COLP, POTT)
    output_to_NC(GR, outCounter, COLP, PHI, UWIND, VWIND, WIND, WWIND,
                HSURF, POTT,
                mean_wind)
else:
    outCounter = GR.outCounter



while GR.ts < GR.nts:
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt

    WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
                                    WIND, UWIND, VWIND, COLP, POTT)
    print('#### ' + str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,2)) + \
            '  days  vmax: ' + str(np.round(vmax,1)) + '  m/s vmean: ' + \
            str(np.round(mean_wind,3)) + ' m/s Tmean: ' + \
            str(np.round(mean_temp,7)) + \
            '  K  COLP: ' + str(np.round(mean_colp,2)) + ' Pa')

    COLP, PHI, POTT, POTTVB, \
    UWIND, VWIND, WWIND,\
    UFLX, VFLX, UFLXMP, VFLXMP, \
    UUFLX, UVFLX, VUFLX, VVFLX, \
    HSURF = time_stepper(GR, COLP, PHI, POTT, POTTVB,
                    UWIND, VWIND, WIND, WWIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF, PVTF, PVTFVB, i_spatial_discretization)


    for k in range(0,GR.nz):
        if (np.sum(np.isnan(UWIND[:,:,k][GR.iisjj])) > 0) | \
                (np.max(UWIND[:,:,k][GR.iisjj]) > 500):
            quit()


    if GR.ts % GR.i_out_nth_ts == 0:
        outCounter += 1
        print('write fields')
        WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
                                        WIND, UWIND, VWIND, COLP, POTT)
        output_to_NC(GR, outCounter, COLP, PHI, UWIND, VWIND, WIND, WWIND,
                    HSURF, POTT,
                    mean_wind)

    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        GR.outCounter = outCounter
        write_restart(GR, COLP, PHI, UWIND, VWIND, WIND, WWIND,\
                                UFLX, VFLX, UFLXMP, VFLXMP, \
                                UUFLX, VUFLX, UVFLX, VVFLX, \
                                HSURF, POTT, POTTVB, PVTF, PVTFVB)
