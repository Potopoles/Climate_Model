import numpy as np
import time
from datetime import timedelta
from grid import Grid
from fields import initialize_fields, CPU_Fields, GPU_Fields
from nc_IO import constant_fields_to_NC, output_to_NC
from IO import write_restart
from multiproc import create_subgrids
from namelist import i_time_stepping, \
                    i_load_from_restart, i_save_to_restart, \
                    i_radiation, njobs, comp_mode, \
                    i_microphysics, i_soil
from diagnostics import diagnose_secondary_fields
from bin.diagnostics_cython import diagnose_secondary_fields_c
from IO_helper_functions import print_ts_info, print_computation_time_info
if i_time_stepping == 'MATSUNO':
    from matsuno import step_matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from RK4 import step_RK4 as time_stepper

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
RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids, CF)
if comp_mode == 2:
    GF = GPU_Fields(GR, subgrids, CF)

####################################################################
# OUTPUT AT TIMESTEP 0 (before start of simulation)
####################################################################
constant_fields_to_NC(GR, CF, RAD, SOIL)


####################################################################
####################################################################
####################################################################
# TIME LOOP START
while GR.ts < GR.nts:
    ####################################################################
    # SIMULATION STATUS
    ####################################################################
    # test for crash
    #for k in range(0,GR.nz):
    #    if (np.sum(np.isnan(UWIND[:,:,k][GR.iisjj])) > 0) | \
    #            (np.max(UWIND[:,:,k][GR.iisjj]) > 500):
    #        quit()
    real_time_ts_start = time.time()
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)
    print_ts_info(GR, CF.WIND, CF.UWIND, CF.VWIND, CF.COLP, CF.POTT)

    ####################################################################
    # SECONDARY DIAGNOSTICS (not related to dynamics)
    ####################################################################
    #if i_radiation or i_microphysics or i_soil:
    #    t_start = time.time()
    #    PAIR, TAIR, TAIRVB, RHO, WIND = \
    #            diagnose_secondary_fields(GR, CF.COLP, PAIR, PHI, POTT, POTTVB,
    #                                    TAIR, TAIRVB, RHO,\
    #                                    PVTF, PVTFVB, UWIND, VWIND, WIND)
    #    #PAIR, TAIR, TAIRVB, RHO, WIND = \
    #    #        diagnose_secondary_fields_c(GR, CF.COLP, PAIR, PHI, POTT, POTTVB,
    #    #                                TAIR, TAIRVB, RHO,\
    #    #                                PVTF, PVTFVB, UWIND, VWIND, WIND)
    #    #PAIR = np.asarray(PAIR)
    #    #TAIR = np.asarray(TAIR)
    #    #TAIRVB = np.asarray(TAIRVB)
    #    #RHO = np.asarray(RHO)
    #    #WIND = np.asarray(WIND)
    #    t_end = time.time()
    #    GR.diag_comp_time += t_end - t_start

    ####################################################################
    # RADIATION
    ####################################################################
    #if i_radiation:
    #    t_start = time.time()
    #    RAD.calc_radiation(GR, TAIR, TAIRVB, RHO, PHIVB, SOIL, MIC)
    #    t_end = time.time()
    #    GR.rad_comp_time += t_end - t_start
    ##########

    ####################################################################
    # MICROPHYSICS
    ####################################################################
    #if i_microphysics:
    #    t_start = time.time()
    #    MIC.calc_microphysics(GR, WIND, SOIL, TAIR, PAIR, RHO, PHIVB)
    #    t_end = time.time()
    #    GR.mic_comp_time += t_end - t_start

    ####################################################################
    # SOIL
    ####################################################################
    #if i_soil:
    #    t_start = time.time()
    #    SOIL.advance_timestep(GR, RAD, MIC)
    #    t_end = time.time()
    #    GR.soil_comp_time += t_end - t_start

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
            GF.copy_fields_to_host(GR)
            t_end = time.time()
            GR.copy_time += t_end - t_start

        # write file
        t_start = time.time()
        GR.nc_output_count += 1
        #WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
        #                                WIND, UWIND, VWIND, COLP, POTT)
        output_to_NC(GR, CF, RAD, SOIL, MIC)
        t_end = time.time()
        GR.IO_time += t_end - t_start

    ####################################################################
    # WRITE RESTART FILE
    ####################################################################
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        write_restart(GR, CF.COLP, CF.PAIR, CF.PHI, CF.PHIVB, CF.UWIND,
                        CF.VWIND, CF.WIND, CF.WWIND,
                        CF.UFLX, CF.VFLX,
                        CF.HSURF, CF.POTT, CF.TAIR, CF.TAIRVB, CF.RHO,
                        CF.POTTVB, CF.PVTF, CF.PVTFVB,
                        RAD, SOIL, MIC, TURB)


    GR.total_comp_time += time.time() - real_time_ts_start
# TIME LOOP STOP
####################################################################
####################################################################
####################################################################



####################################################################
# FINALIZE SIMULATION
####################################################################
print_computation_time_info(GR)


