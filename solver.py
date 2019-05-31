#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190531
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
###############################################################################
"""
#if i == 0:
#    from pdb import set_trace
#    pdb.set_trace()
import time, _thread
import numpy as np
from datetime import timedelta

from namelist import (i_time_stepping,
                    i_load_from_restart, i_save_to_restart,
                    i_radiation, njobs, comp_mode,
                    i_microphysics, i_surface_scheme)
from io_read_namelist import (gpu_enable, CPU, GPU)
from grid import Grid
from ModelFields import ModelFields
from io_nc_output import constant_fields_to_NC, output_to_NC
from io_restart import write_restart
from io_functions import (print_ts_info)
from dyn_matsuno import step_matsuno as time_stepper

from dyn_org_discretizations import DiagnosticsFactory
###############################################################################
if comp_mode == 1:
    Diagnostics = DiagnosticsFactory(target=CPU)
elif comp_mode == 2:
    Diagnostics = DiagnosticsFactory(target=GPU)

####################################################################
# CREATE MODEL GRID
####################################################################
GR = Grid()

####################################################################
# CREATE MODEL FIELDS
####################################################################
F = ModelFields(GR, gpu_enable)

# TODO: not yet moved to new model setup
RAD = None
SURF = None
MIC = None
TURB = None


####################################################################
# OUTPUT AT TIMESTEP 0 (before start of simulation)
####################################################################
constant_fields_to_NC(GR, F, RAD, SURF)


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
    print_ts_info(GR, F)

    ####################################################################
    # SECONDARY DIAGNOSTICS (related to physics)
    ####################################################################
    GR.timer.start('diag2')
    Diagnostics.secondary_diag(GR,
            **F.get(Diagnostics.fields_secondary_diag,
                    target=Diagnostics.target))
    GR.timer.stop('diag2')


    ####################################################################
    # RADIATION
    ####################################################################
    #if i_radiation:
    #    #print('flag 1')
    #    GR.timer.start('rad')
    #    # Asynchroneous Radiation
    #    if RAD.i_async_radiation:
    #        if RAD.done == 1:
    #            if comp_mode == 2:
    #                GF.copy_radiation_fields_to_device(GR, CF)
    #                GF.copy_radiation_fields_to_host(GR)
    #            RAD.done = 0
    #            _thread.start_new_thread(RAD.calc_radiation, (GR, CF))
    #    # Synchroneous Radiation
    #    else:
    #        if GR.ts % RAD.rad_nth_ts == 0:
    #            if comp_mode == 2:
    #                GF.copy_radiation_fields_to_host(GR)
    #            RAD.calc_radiation(GR, CF)
    #            if comp_mode == 2:
    #                GF.copy_radiation_fields_to_device(GR, CF)
    #    GR.timer.stop('rad')

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
    #if i_surface_scheme:
    #    t_start = time.time()
    #    SURF.advance_timestep(GR, CF, GF, RAD)
    #    t_end = time.time()
    #    GR.soil_comp_time += t_end - t_start


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
    GR.timer.start('dyn')
    time_stepper(GR, F)
    GR.timer.stop('dyn')


    ####################################################################
    # WRITE NC OUTPUT
    ####################################################################
    if GR.ts % GR.i_out_nth_ts == 0:
        # copy GPU fields to CPU
        if comp_mode == 2:
            F.copy_device_to_host(F.ALL_FIELDS)

        # write file
        GR.timer.start('IO')
        GR.nc_output_count += 1
        output_to_NC(GR, F, RAD, SURF, MIC)
        GR.timer.stop('IO')


    ####################################################################
    # WRITE RESTART FILE
    ####################################################################
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        pass
        # copy GPU fields to CPU
        #if comp_mode == 2:
        #    GF.copy_all_fields_to_host(GR)

        #GR.timer.start('IO')
        #write_restart(GR, CF, RAD, SURF, MIC, TURB)
        #GR.timer.stop('IO')

    GR.timer.stop('total')
# TIME LOOP STOP
####################################################################
####################################################################
####################################################################



####################################################################
# FINALIZE SIMULATION
####################################################################
GR.timer.print_report()





## DEBUG TEMPLATE
#F.host['POTT'][:] = np.nan
#F.host['UWIND'][:] = np.nan
#F.host['VWIND'][:] = np.nan
#n_iter = 10
#Prognostics.euler_forward(CPU, GR,
#        **NF.get(Prognostics.fields_prognostic, target=CPU))
#t0 = time.time()
#for i in range(n_iter):
#    Prognostics.euler_forward(CPU, GR,
#            **NF.get(Prognostics.fields_prognostic, target=CPU))
#print((time.time() - t0)/n_iter)

#F.COLP          = F.COLP.squeeze()
#F.dCOLPdt       = F.dCOLPdt.squeeze()
#F.COLP_NEW      = F.COLP_NEW.squeeze()
#F.COLP_OLD      = F.COLP_OLD.squeeze()
#F.HSURF         = F.HSURF.squeeze()

##TODO
#FIELD1 = np.asarray(F.VWIND)
#print(np.nanmean((FIELD1)))
#print()

#F.UWIND, F.VWIND, F.COLP, F.POTT, F.QV, F.QC \
#             = proceed_timestep_jacobson_c(GR,
#                    F.UWIND_OLD, F.UWIND, F.VWIND_OLD, F.VWIND,
#                    F.COLP_OLD, F.COLP, F.POTT_OLD, F.POTT,
#                    F.QV_OLD, F.QV, F.QC_OLD, F.QC,
#                    F.dUFLXdt, F.dVFLXdt, F.dPOTTdt, F.dQVdt, F.dQCdt)
#print((time.time() - t0)/n_iter)

##TODO
#FIELD2 = np.asarray(F.VWIND)
#print(np.nanmean((FIELD2)))
#
#print()
#print(np.sum(np.isnan(FIELD2[:,:,:])) -\
#             np.sum(np.isnan(FIELD1[:,:,:])))
#print(np.nanmean(FIELD2[:,:,:] - FIELD1[:,:,:]))
#quit()




## TODO
#if comp_mode == 1:
#    CF.COLP          = np.expand_dims(CF.COLP, axis=2)
#    CF.dCOLPdt       = np.expand_dims(CF.dCOLPdt, axis=2)
#    CF.COLP_NEW      = np.expand_dims(CF.COLP_NEW, axis=2)
#    CF.COLP_OLD      = np.expand_dims(CF.COLP_OLD, axis=2)
#    CF.HSURF         = np.expand_dims(CF.HSURF, axis=2)
#    F.old_to_new(CF, host=True)
#elif comp_mode == 2:
#    GF.COLP          = cp.expand_dims(GF.COLP, axis=2)
#    GF.dCOLPdt       = cp.expand_dims(GF.dCOLPdt, axis=2)
#    GF.COLP_NEW      = cp.expand_dims(GF.COLP_NEW, axis=2)
#    GF.COLP_OLD      = cp.expand_dims(GF.COLP_OLD, axis=2)
#    GF.HSURF         = cp.expand_dims(GF.HSURF, axis=2)
#    #F.old_to_new(GF, host=False)
