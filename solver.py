import numpy as np
import time
from datetime import timedelta
from grid import Grid
from fields import initialize_fields, CPU_Fields, GPU_Fields
from nc_IO import constant_fields_to_NC, output_to_NC
from org_model_physics import secondary_diagnostics
from IO import write_restart
from multiproc import create_subgrids
from namelist import i_time_stepping, \
                    i_load_from_restart, i_save_to_restart, \
                    i_radiation, njobs, comp_mode, \
                    i_microphysics, i_surface
from IO_helper_functions import print_ts_info, print_computation_time_info
if i_time_stepping == 'MATSUNO':
    from matsuno import step_matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from RK4 import step_RK4 as time_stepper

import _thread

####################################################################
# CREATE MODEL GRID
####################################################################
# main grid
GR = Grid()
# optional subgrids for domain decomposition (not completly implemented)
#GR, subgrids = create_subgrids(GR, njobs)
subgrids = {} # option

####################################################################
# CREATE MODEL FIELDS
####################################################################
CF = CPU_Fields(GR, subgrids)
CF, RAD, SURF, MIC, TURB = initialize_fields(GR, subgrids, CF)
if comp_mode == 2:
    GF = GPU_Fields(GR, subgrids, CF)
else:
    GF = None

####################################################################
# OUTPUT AT TIMESTEP 0 (before start of simulation)
####################################################################
constant_fields_to_NC(GR, CF, RAD, SURF)


####################################################################
####################################################################
####################################################################
# TIME LOOP START
while GR.ts < GR.nts:
    ####################################################################
    # SIMULATION STATUS
    ####################################################################
    real_time_ts_start = time.time()
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)
    t_start = time.time()
    print_ts_info(GR, CF, GF)
    t_end = time.time()

    ####################################################################
    # SECONDARY DIAGNOSTICS (related to physics)
    ####################################################################
    t_start = time.time()
    if comp_mode in [0,1]:
        secondary_diagnostics(GR, CF)
    elif comp_mode == 2:
        secondary_diagnostics(GR, GF)
    t_end = time.time()
    GR.diag_comp_time += t_end - t_start


    ####################################################################
    # RADIATION
    ####################################################################
    if i_radiation:
        #print('flag 1')
        t_start = time.time()
        # Asynchroneous Radiation
        if RAD.i_async_radiation:
            if RAD.done == 1:
                if comp_mode == 2:
                    GF.copy_radiation_fields_to_device(GR, CF)
                    GF.copy_radiation_fields_to_host(GR)
                RAD.done = 0
                _thread.start_new_thread(RAD.calc_radiation, (GR, CF))
        # Synchroneous Radiation
        else:
            if GR.ts % RAD.rad_nth_ts == 0:
                if comp_mode == 2:
                    GF.copy_radiation_fields_to_host(GR)
                RAD.calc_radiation(GR, CF)
                if comp_mode == 2:
                    GF.copy_radiation_fields_to_device(GR, CF)
        t_end = time.time()
        GR.rad_comp_time += t_end - t_start

    #print('RADIATION timerstarts:')
    #try:
    #    tot = GR.rad_1 + GR.rad_2 + GR.rad_lw + GR.rad_lwsolv + GR.rad_sw
    #    print(tot)
    #    print('rad_1     :  ' + str(int(100*GR.rad_1/tot)) + '\t\t' + str(GR.rad_1))
    #    print('rad_2     :  ' + str(int(100*GR.rad_2/tot)) + '\t\t' + str(GR.rad_2))
    #    print('rad_lw    :  ' + str(int(100*GR.rad_lw/tot)) + '\t\t' + str(GR.rad_lw))
    #    print('rad_lwsolv:  ' + str(int(100*GR.rad_lwsolv/tot)) + '\t\t' + str(GR.rad_lwsolv))
    #    print('rad_sw    :  ' + str(int(100*GR.rad_sw/tot)) + '\t\t' + str(GR.rad_sw))
    #except ZeroDivisionError:
    #    pass
    #print('RADIATION timersend:')
    

    ####################################################################
    # EARTH SURFACE
    ####################################################################
    if i_surface:
        t_start = time.time()
        SURF.advance_timestep(GR, CF, GF, RAD)
        t_end = time.time()
        GR.soil_comp_time += t_end - t_start


    ####################################################################
    # MICROPHYSICS
    ####################################################################
    #if i_microphysics:
    #    t_start = time.time()
    #    MIC.calc_microphysics(GR, WIND, SURF, TAIR, PAIR, RHO, PHIVB)
    #    t_end = time.time()
    #    GR.mic_comp_time += t_end - t_start


    ####################################################################
    # DYNAMICS
    ####################################################################
    t_dyn_start = time.time()
    if comp_mode in [0,1]:
        time_stepper(GR, subgrids, CF)
    elif comp_mode == 2:
        time_stepper(GR, subgrids, GF)
    t_dyn_end = time.time()
    GR.dyn_comp_time += t_dyn_end - t_dyn_start


    ####################################################################
    # WRITE NC OUTPUT
    ####################################################################
    if GR.ts % GR.i_out_nth_ts == 0:
        # copy GPU fields to CPU
        if comp_mode == 2:
            t_start = time.time()
            GF.copy_all_fields_to_host(GR)
            t_end = time.time()
            GR.copy_time += t_end - t_start

        # write file
        t_start = time.time()
        GR.nc_output_count += 1
        output_to_NC(GR, CF, RAD, SURF, MIC)
        t_end = time.time()
        GR.IO_time += t_end - t_start


    ####################################################################
    # WRITE RESTART FILE
    ####################################################################
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        # copy GPU fields to CPU
        if comp_mode == 2:
            t_start = time.time()
            GF.copy_all_fields_to_host(GR)
            t_end = time.time()
            GR.copy_time += t_end - t_start

        t_start = time.time()
        write_restart(GR, CF, RAD, SURF, MIC, TURB)
        t_end = time.time()
        GR.IO_time += t_end - t_start


    GR.total_comp_time += time.time() - real_time_ts_start
# TIME LOOP STOP
####################################################################
####################################################################
####################################################################



####################################################################
# FINALIZE SIMULATION
####################################################################
print_computation_time_info(GR)


