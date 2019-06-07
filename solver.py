#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190604
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
                    i_radiation, i_comp_mode,
                    i_microphysics, i_surface_scheme)
from io_read_namelist import (gpu_enable, CPU, GPU)
from io_nc_output import constant_fields_to_NC, output_to_NC
from io_restart import write_restart
from io_functions import (print_ts_info)
from main_grid import Grid
from main_fields import ModelFields
from dyn_matsuno import step_matsuno as time_stepper
from dyn_org_discretizations import DiagnosticsFactory
###############################################################################
if i_comp_mode == 1:
    Diagnostics = DiagnosticsFactory(target=CPU)
elif i_comp_mode == 2:
    Diagnostics = DiagnosticsFactory(target=GPU)

## show number of CPU threads for numba.
#import numba
#print(numba.config.NUMBA_DEFAULT_NUM_THREADS)
#quit()

####################################################################
# CREATE MODEL GRID AND FIELDS
####################################################################
GR = Grid()
F = ModelFields(GR, gpu_enable)


####################################################################
# OUTPUT AT TIMESTEP 0 (before start of simulation)
####################################################################
constant_fields_to_NC(GR, F)


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
    # SECONDARY DIAGNOSTICS (related to physics, but not affecting
    #                        dynamics)
    ####################################################################
    GR.timer.start('diag')
    Diagnostics.secondary_diag(
            **F.get(Diagnostics.fields_secondary_diag,
                    target=Diagnostics.target))
    GR.timer.stop('diag')


    ####################################################################
    # RADIATION
    ####################################################################
    if i_radiation:
        GR.timer.start('rad')
        F.RAD.launch_radiation_calc(GR, F)
        GR.timer.stop('rad')


    ####################################################################
    # EARTH SURFACE
    ####################################################################
    if i_surface_scheme:
        GR.timer.start('srfc')
        F.SURF.advance_timestep(GR, **F.get(F.SURF.fields_timestep,
                                target=F.SURF.target))
        GR.timer.stop('srfc')


    ####################################################################
    # MICROPHYSICS
    # TODO: implement
    ####################################################################
    #if i_microphysics:
    #    GR.timer.start('mic')
    #    F.MIC.calc_microphysics(GR, WIND, SURF, TAIR, PAIR, RHO, PHIVB)
    #    GR.timer.stop('mic')


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
        if i_comp_mode == 2:
            F.copy_device_to_host(GR, F.ALL_FIELDS)
        # write file
        GR.timer.start('IO')
        GR.nc_output_count += 1
        output_to_NC(GR, F)
        GR.timer.stop('IO')


    ####################################################################
    # WRITE RESTART FILE
    ####################################################################
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        # copy GPU fields to CPU
        if i_comp_mode == 2:
            F.copy_device_to_host(GR, F.ALL_FIELDS)
        # write file
        GR.timer.start('IO')
        write_restart(GR, F)
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
#if i_comp_mode == 1:
#    CF.COLP          = np.expand_dims(CF.COLP, axis=2)
#    CF.dCOLPdt       = np.expand_dims(CF.dCOLPdt, axis=2)
#    CF.COLP_NEW      = np.expand_dims(CF.COLP_NEW, axis=2)
#    CF.COLP_OLD      = np.expand_dims(CF.COLP_OLD, axis=2)
#    CF.HSURF         = np.expand_dims(CF.HSURF, axis=2)
#    F.old_to_new(CF, host=True)
#elif i_comp_mode == 2:
#    GF.COLP          = cp.expand_dims(GF.COLP, axis=2)
#    GF.dCOLPdt       = cp.expand_dims(GF.dCOLPdt, axis=2)
#    GF.COLP_NEW      = cp.expand_dims(GF.COLP_NEW, axis=2)
#    GF.COLP_OLD      = cp.expand_dims(GF.COLP_OLD, axis=2)
#    GF.HSURF         = cp.expand_dims(GF.HSURF, axis=2)
#    #F.old_to_new(GF, host=False)
