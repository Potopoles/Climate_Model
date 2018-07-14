import numpy as np

from grid import Grid
from fields import initialize_fields
from IO import output_to_NC, write_restart
from namelist import i_time_stepping, i_spatial_discretization, \
                    i_load_from_restart, i_save_to_restart
if i_time_stepping == 'EULER_FORWARD':
    from time_integration import euler_forward as time_stepper
elif i_time_stepping == 'MATSUNO':
    from time_integration import matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from time_integration import RK4 as time_stepper

GR = Grid()

COLP, PHI, UWIND, VWIND, WIND, \
UFLX, VFLX, UFLXMP, VFLXMP, \
UUFLX, VUFLX, UVFLX, VVFLX, \
HSURF, POTT, PVTF, PVTFVB = initialize_fields(GR)


if not i_load_from_restart:
    outCounter = 0
    WIND[GR.iijj] = np.sqrt( ((UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2)**2 + \
                    ((VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2)**2 )
    mean_ekin = np.sum( 0.5*1*WIND[GR.iijj]**2*COLP[GR.iijj]*GR.A[GR.iijj] ) / \
                    np.sum( COLP[GR.iijj]*GR.A[GR.iijj] )
    output_to_NC(GR, outCounter, COLP, PHI, UWIND, VWIND, WIND,
                HSURF, POTT,
                mean_ekin)
else:
    outCounter = GR.outCounter



while GR.ts < GR.nts:
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt

    vmax = max(np.max(np.abs(UWIND[GR.iisjj])), np.max(np.abs(VWIND[GR.iijjs])))
    mean_hght = np.sum(COLP[GR.iijj]*GR.A[GR.iijj])/np.sum(GR.A[GR.iijj])
    mean_temp = np.sum(POTT[GR.iijj]*GR.A[GR.iijj]*COLP[GR.iijj])/ \
            np.sum(GR.A[GR.iijj]*COLP[GR.iijj])
    WIND[GR.iijj] = np.sqrt( ((UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2)**2 + \
                    ((VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2)**2 )
    #mean_ekin = np.sum( 0.5*1*WIND[GR.iijj]**2*COLP[GR.iijj]*GR.A[GR.iijj] ) / \
    #        np.sum( COLP[GR.iijj]*GR.A[GR.iijj] )
    mean_ekin = np.sum( WIND[GR.iijj]*COLP[GR.iijj]*GR.A[GR.iijj] ) / \
            np.sum( COLP[GR.iijj]*GR.A[GR.iijj] )
    print('#### ' + str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,2)) + \
            '  days  vmax: ' + str(np.round(vmax,1)) + '  m/s  hght: ' + \
            str(np.round(mean_hght,2)) + '  m ekin: ' + \
            str(np.round(mean_ekin,3)) + '  temp: ' + \
            str(np.round(mean_temp,7)))

    COLP, PHI, POTT, \
    UWIND, VWIND, \
    UFLX, VFLX, UFLXMP, VFLXMP, \
    UUFLX, UVFLX, VUFLX, VVFLX, \
    HSURF = time_stepper(GR, COLP, PHI, POTT,
                    UWIND, VWIND, WIND,
                    UFLX, VFLX, UFLXMP, VFLXMP,
                    UUFLX, UVFLX, VUFLX, VVFLX,
                    HSURF, PVTF, PVTFVB, i_spatial_discretization)

    if (np.sum(np.isnan(UWIND[GR.iisjj])) > 0) | (np.nanmax(UWIND) > 500):
        quit()


    if GR.ts % GR.i_out_nth_ts == 0:
        outCounter += 1
        print('write fields')
        WIND[GR.iijj] = np.sqrt( ((UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2)**2 + \
                        ((VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2)**2 )
        mean_ekin = np.sum( 0.5*1*WIND[GR.iijj]**2*COLP[GR.iijj]*GR.A[GR.iijj] ) / \
                        np.sum( COLP[GR.iijj]*GR.A[GR.iijj] )
        output_to_NC(GR, outCounter, COLP, PHI, UWIND, VWIND, WIND,
                    HSURF, POTT,
                    mean_ekin)

    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        GR.outCounter = outCounter
        write_restart(GR, COLP, PHI, UWIND, VWIND, WIND, \
                                UFLX, VFLX, UFLXMP, VFLXMP, \
                                UUFLX, VUFLX, UVFLX, VVFLX, \
                                HSURF, POTT, PVTF, PVTFVB)
