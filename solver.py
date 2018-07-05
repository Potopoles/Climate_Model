import numpy as np

from grid import Grid
from fields import initialize_fields
from IO import output_to_NC
from boundaries import exchange_BC

#from constants import con_g
from namelist import inpRate, outRate, i_pseudo_radiation
from height import height_tendency_upstream
from wind import masspoint_flux_tendency_upstream
from tracer import tracer_tendency_upstream


GR = Grid()

HGHT, HTOP, UWIND, VWIND, WIND, \
UFLX, VFLX, UFLXMP, VFLXMP, \
UUFLX, VUFLX, UVFLX, VVFLX, \
HSURF, TRACER = initialize_fields(GR)

outCounter = 0
WIND[GR.iijj] = np.sqrt( ((UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2)**2 + \
                ((VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2)**2 )
mean_ekin = np.sum( 0.5*1*WIND[GR.iijj]**2*HGHT[GR.iijj]*GR.A[GR.iijj] ) / np.sum( HGHT[GR.iijj]*GR.A[GR.iijj] )
output_to_NC(GR, outCounter, HGHT, HTOP, UWIND, VWIND, WIND,
            HSURF, TRACER,
            mean_ekin)

while GR.ts < GR.nts:
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt

    vmax = max(np.max(UWIND[GR.iisjj]), np.max(VWIND[GR.iijjs]))
    mean_hght = np.sum(HGHT[GR.iijj]*GR.A[GR.iijj])/np.sum(GR.A[GR.iijj])
    mean_tracer = np.sum(TRACER[GR.iijj]*GR.A[GR.iijj]*HGHT[GR.iijj])/np.sum(GR.A[GR.iijj]*HGHT[GR.iijj])
    WIND[GR.iijj] = np.sqrt( ((UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2)**2 + \
                    ((VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2)**2 )
    mean_ekin = np.sum( 0.5*1*WIND[GR.iijj]**2*HGHT[GR.iijj]*GR.A[GR.iijj] ) / np.sum( HGHT[GR.iijj]*GR.A[GR.iijj] )
    print('#### ' + str(GR.ts) + '  ' + str(np.round(GR.sim_time_sec/3600/24,2)) + \
            '  days  vmax: ' + str(np.round(vmax,1)) + '  m/s  hght: ' + \
            str(np.round(mean_hght,2)) + '  m ekin: ' + str(np.round(mean_ekin,3)) + '  tracer: ' + \
            str(np.round(mean_tracer,7)))

    # PROGNOSE HGHT
    dHGHTdt = height_tendency_upstream(GR, HGHT, UWIND, VWIND, UFLX, VFLX)

    # PROGNOSE WIND
    dUFLXMPdt, dVFLXMPdt = masspoint_flux_tendency_upstream(GR, UFLXMP, VFLXMP, HGHT,
                                                    UWIND, VWIND,
                                                    UUFLX, VUFLX, UVFLX, VVFLX,
                                                    HSURF)

    # PROGNOSE TRACER
    dTRACERdt = tracer_tendency_upstream(GR, TRACER, HGHT, UWIND, VWIND, UFLX, VFLX, WIND)

    # TIME STEPPING
    UFLXMP[GR.iijj] = UFLXMP[GR.iijj] + GR.dt*dUFLXMPdt
    VFLXMP[GR.iijj] = VFLXMP[GR.iijj] + GR.dt*dVFLXMPdt
    HGHT[GR.iijj] = HGHT[GR.iijj] + GR.dt*dHGHTdt
    TRACER[GR.iijj] = TRACER[GR.iijj] + GR.dt*dTRACERdt

    UFLXMP = exchange_BC(GR, UFLXMP)
    VFLXMP = exchange_BC(GR, VFLXMP)
    HGHT = exchange_BC(GR, HGHT)
    TRACER = exchange_BC(GR, TRACER)

    # DIAGNOSTICS 
    UWIND[GR.iisjj] = ( UFLXMP[GR.iisjj_im1] + UFLXMP[GR.iisjj] ) \
                    / ( HGHT[GR.iisjj_im1] + HGHT[GR.iisjj] )
    VWIND[GR.iijjs] = ( VFLXMP[GR.iijjs_jm1] + VFLXMP[GR.iijjs] ) \
                    / ( HGHT[GR.iijjs_jm1] + HGHT[GR.iijjs] )
    UWIND = exchange_BC(GR, UWIND)
    VWIND = exchange_BC(GR, VWIND)

    HTOP[GR.iijj] = HSURF[GR.iijj] + HGHT[GR.iijj] 

    if i_pseudo_radiation:
        HGHT[GR.iijj] = HGHT[GR.iijj] - GR.dt*outRate*HGHT[GR.iijj] + inpRate*np.cos(GR.lat_rad[GR.iijj])*GR.dt

    if GR.ts % GR.i_out_nth_ts == 0:
        outCounter = outCounter + 1
        print('write fields')
        WIND[GR.iijj] = np.sqrt( ((UWIND[GR.iijj] + UWIND[GR.iijj_ip1])/2)**2 + \
                        ((VWIND[GR.iijj] + VWIND[GR.iijj_jp1])/2)**2 )
        mean_ekin = np.sum( 0.5*1*WIND[GR.iijj]**2*HGHT[GR.iijj]*GR.A[GR.iijj] ) / np.sum( HGHT[GR.iijj]*GR.A[GR.iijj] )
        output_to_NC(GR, outCounter, HGHT, HTOP, UWIND, VWIND, WIND,
                    HSURF, TRACER,
                    mean_ekin)

