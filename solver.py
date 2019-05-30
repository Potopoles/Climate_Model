#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          solver.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190530
License:            MIT

Simple global climate model, hydrostatic and on a lat-lon grid.
Still in construction.

solver.py is the entry point to the program.

Implementation of dynamical core according to:
Jacobson 2005
Fundamentals of Atmospheric Modeling, Second Edition Chapter 7

last sucessful run:
Python: 3. 7.3
Numba:  0.43.1
Numpy:  1.16.3
"""
#if i == 0:
#    from pdb import set_trace
#    pdb.set_trace()
import time
from datetime import timedelta
import numpy as np

from grid import Grid
from fields import initialize_fields, CPU_Fields, GPU_Fields
from ModelFields import ModelFields
from nc_IO import constant_fields_to_NC, output_to_NC
from org_model_physics import secondary_diagnostics
from IO import write_restart
from multiproc import create_subgrids
from namelist import (i_time_stepping,
                    i_load_from_restart, i_save_to_restart,
                    i_radiation, njobs, comp_mode,
                    i_microphysics, i_surface_scheme)
from org_namelist import (gpu_enable, HOST, DEVICE)
from IO_helper_functions import (print_ts_info)
if i_time_stepping == 'MATSUNO':
    from matsuno import step_matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from RK4 import step_RK4 as time_stepper

import _thread
from namelist import i_run_new_style
import cupy as cp
from dyn_org_discretizations import DiagnosticsFactory
Diagnostics = DiagnosticsFactory()
####################################################################

####################################################################
# CREATE MODEL GRID
####################################################################
# main grid
GR = Grid()
GR_NEW = Grid(new=True)
# optional subgrids for domain decomposition (not completly implemented)
#GR, subgrids = create_subgrids(GR, njobs)
subgrids = {} # option

####################################################################
# CREATE MODEL FIELDS
####################################################################
CF = CPU_Fields(GR, subgrids)
CF, RAD, SURF, MIC, TURB = initialize_fields(GR, subgrids, CF)

F = ModelFields(GR_NEW, gpu_enable, CF)


if comp_mode == 2:
    GF = GPU_Fields(GR, subgrids, CF)
else:
    GF = None


#F.old_to_new(GF)

####################################################################
# OUTPUT AT TIMESTEP 0 (before start of simulation)
####################################################################
constant_fields_to_NC(GR, CF, RAD, SURF)


####################################################################
####################################################################
####################################################################
# TIME LOOP START
while GR.ts < GR.nts:
    GR.timer.start('total')
    ####################################################################
    # SIMULATION STATUS
    ####################################################################
    real_time_ts_start = time.time()
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)
    print_ts_info(GR, CF, GF)

    ####################################################################
    # SECONDARY DIAGNOSTICS (related to physics)
    ####################################################################
    GR.timer.start('diag2')
    # TODO
    if i_run_new_style == 1:
        if comp_mode == 1:
            CF.COLP          = np.expand_dims(CF.COLP, axis=2)
            CF.dCOLPdt       = np.expand_dims(CF.dCOLPdt, axis=2)
            CF.COLP_NEW      = np.expand_dims(CF.COLP_NEW, axis=2)
            CF.COLP_OLD      = np.expand_dims(CF.COLP_OLD, axis=2)
            CF.HSURF         = np.expand_dims(CF.HSURF, axis=2)
        elif comp_mode == 2:
            GF.COLP          = cp.expand_dims(GF.COLP, axis=2)
            GF.dCOLPdt       = cp.expand_dims(GF.dCOLPdt, axis=2)
            GF.COLP_NEW      = cp.expand_dims(GF.COLP_NEW, axis=2)
            GF.COLP_OLD      = cp.expand_dims(GF.COLP_OLD, axis=2)
            GF.HSURF         = cp.expand_dims(GF.HSURF, axis=2)
    if comp_mode == 1:
        if i_run_new_style:
            F.old_to_new(CF, host=True)
            Diagnostics.secondary_diag(HOST, GR_NEW,
                    **F.get(Diagnostics.fields_secondary_diag,
                            target=HOST))
            F.new_to_old(CF, host=True)
        else:
            secondary_diagnostics(GR, CF)
    elif comp_mode == 2:
        if i_run_new_style:
            F.old_to_new(GF, host=False)
            Diagnostics.secondary_diag(DEVICE, GR_NEW,
                    **F.get(Diagnostics.fields_secondary_diag,
                            target=DEVICE))
            F.new_to_old(GF, host=False)
        else:
            secondary_diagnostics(GR, GF)
    if i_run_new_style == 1:
        if comp_mode == 1:
            # TODO
            CF.COLP          = CF.COLP.squeeze()
            CF.dCOLPdt       = CF.dCOLPdt.squeeze()
            CF.COLP_NEW      = CF.COLP_NEW.squeeze()
            CF.COLP_OLD      = CF.COLP_OLD.squeeze()
            CF.HSURF         = CF.HSURF.squeeze()
        elif comp_mode == 2:
            # TODO
            GF.COLP          = GF.COLP.squeeze()
            GF.dCOLPdt       = GF.dCOLPdt.squeeze()
            GF.COLP_NEW      = GF.COLP_NEW.squeeze()
            GF.COLP_OLD      = GF.COLP_OLD.squeeze()
            GF.HSURF         = GF.HSURF.squeeze()
    GR.timer.stop('diag2')


    ####################################################################
    # RADIATION
    ####################################################################
    if i_radiation:
        #print('flag 1')
        GR.timer.start('rad')
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
        GR.timer.stop('rad')

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
    #quit()
    

    ####################################################################
    # EARTH SURFACE
    ####################################################################
    if i_surface_scheme:
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
    # TODO
    if i_run_new_style == 1:
        if comp_mode == 1:
            CF.COLP          = np.expand_dims(CF.COLP, axis=2)
            CF.dCOLPdt       = np.expand_dims(CF.dCOLPdt, axis=2)
            CF.COLP_NEW      = np.expand_dims(CF.COLP_NEW, axis=2)
            CF.COLP_OLD      = np.expand_dims(CF.COLP_OLD, axis=2)
            CF.HSURF         = np.expand_dims(CF.HSURF, axis=2)
        elif comp_mode == 2:
            GF.COLP          = cp.expand_dims(GF.COLP, axis=2)
            GF.dCOLPdt       = cp.expand_dims(GF.dCOLPdt, axis=2)
            GF.COLP_NEW      = cp.expand_dims(GF.COLP_NEW, axis=2)
            GF.COLP_OLD      = cp.expand_dims(GF.COLP_OLD, axis=2)
            GF.HSURF         = cp.expand_dims(GF.HSURF, axis=2)
    GR.timer.start('dyn')
    if comp_mode in [0,1]:
        time_stepper(GR, GR_NEW, subgrids, CF, F)
    elif comp_mode == 2:
        time_stepper(GR, GR_NEW, subgrids, GF, F)
    GR.timer.stop('dyn')
    if i_run_new_style == 1:
        if comp_mode == 1:
            # TODO
            CF.COLP          = CF.COLP.squeeze()
            CF.dCOLPdt       = CF.dCOLPdt.squeeze()
            CF.COLP_NEW      = CF.COLP_NEW.squeeze()
            CF.COLP_OLD      = CF.COLP_OLD.squeeze()
            CF.HSURF         = CF.HSURF.squeeze()
        elif comp_mode == 2:
            # TODO
            GF.COLP          = GF.COLP.squeeze()
            GF.dCOLPdt       = GF.dCOLPdt.squeeze()
            GF.COLP_NEW      = GF.COLP_NEW.squeeze()
            GF.COLP_OLD      = GF.COLP_OLD.squeeze()
            GF.HSURF         = GF.HSURF.squeeze()



    ####################################################################
    # WRITE NC OUTPUT
    ####################################################################
    if GR.ts % GR.i_out_nth_ts == 0:
        # copy GPU fields to CPU
        if comp_mode == 2:
            GF.copy_all_fields_to_host(GR)

        # write file
        GR.timer.start('IO')
        GR.nc_output_count += 1
        output_to_NC(GR, CF, RAD, SURF, MIC)
        GR.timer.stop('IO')


    ####################################################################
    # WRITE RESTART FILE
    ####################################################################
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        # copy GPU fields to CPU
        if comp_mode == 2:
            GR.timer.start('copy')
            GF.copy_all_fields_to_host(GR)
            GR.timer.stop('copy')

        GR.timer.start('IO')
        write_restart(GR, CF, RAD, SURF, MIC, TURB)
        GR.timer.stop('IO')

    GR.timer.stop('total')
# TIME LOOP STOP
####################################################################
####################################################################
####################################################################



####################################################################
# FINALIZE SIMULATION
####################################################################
GR.timer.print_report()

