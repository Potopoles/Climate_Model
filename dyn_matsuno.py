#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190604
License:            MIT

Perform a matsuno time integration.
###############################################################################
"""
from namelist import i_comp_mode, i_moist_main_switch
from io_read_namelist import CPU, GPU, gpu_enable
from main_grid import tpb, bpg
from dyn_tendencies import compute_tendencies
from dyn_org_discretizations import (PrognosticsFactory, DiagnosticsFactory) 
if gpu_enable:
    from misc_gpu_functions import set_equal
###############################################################################
if i_comp_mode == 1:
    Prognostics = PrognosticsFactory(target=CPU)
    Diagnostics = DiagnosticsFactory(target=CPU)
elif i_comp_mode == 2:
    Prognostics = PrognosticsFactory(target=GPU)
    Diagnostics = DiagnosticsFactory(target=GPU)

def step_matsuno(GR, F):

    # UPDATE TIME LEVELS
    ##############################
    ##############################
    GR.timer.start('step')
    if i_comp_mode == 1:
        F.host['COLP_OLD'][:]  = F.host['COLP'][:]
        F.host['UWIND_OLD'][:] = F.host['UWIND'][:]
        F.host['VWIND_OLD'][:] = F.host['VWIND'][:]
        F.host['POTT_OLD'][:]  = F.host['POTT'][:]
        if i_moist_main_switch:
            F.host['QV_OLD'][:]    = F.host['QV'][:]
            F.host['QC_OLD'][:]    = F.host['QC'][:]
    elif i_comp_mode == 2:
        set_equal[bpg, tpb](F.device['COLP_OLD'],     F.device['COLP'])
        set_equal[bpg, tpb](F.device['UWIND_OLD'],    F.device['UWIND'])
        set_equal[bpg, tpb](F.device['VWIND_OLD'],    F.device['VWIND'])
        set_equal[bpg, tpb](F.device['POTT_OLD'],     F.device['POTT'])
        if i_moist_main_switch:
            set_equal[bpg, tpb](F.device['QV_OLD'],       F.device['QV'])
            set_equal[bpg, tpb](F.device['QC_OLD'],       F.device['QC'])
    GR.timer.stop('step')
    ##############################
    ##############################

    ############################################################
    ############################################################
    ##########     ESTIMATE
    ############################################################
    ############################################################

    # COMPUTE TENDENCIES
    ##############################
    ##############################
    compute_tendencies(GR, F)
    if i_comp_mode == 1:
        F.host['COLP'][:]  = F.host['COLP_NEW'][:]
    elif i_comp_mode == 2:
        set_equal[bpg, tpb](F.device['COLP'],     F.device['COLP_NEW'])
    ##############################
    ##############################

    # PROGNOSE NEXT TIME LEVEL
    ##############################
    ##############################
    GR.timer.start('step')
    Prognostics.euler_forward(GR, GR.GRF[Prognostics.target],
                        **F.get(Prognostics.fields_prognostic,
                            target=Prognostics.target))
    GR.timer.stop('step')
    ##############################
    ##############################

    # DIAGNOSE VARIABLES
    ##############################
    ##############################
    GR.timer.start('diag')
    Diagnostics.primary_diag(GR.GRF[Diagnostics.target],
                        **F.get(Diagnostics.fields_primary_diag,
                            target=Diagnostics.target))
    GR.timer.stop('diag')
    ##############################
    ##############################

    ############################################################
    ############################################################
    ##########     FINAL
    ############################################################
    ############################################################

    # COMPUTE TENDENCIES
    ##############################
    ##############################
    compute_tendencies(GR, F)
    if i_comp_mode == 1:
        F.host['COLP'][:]  = F.host['COLP_NEW'][:]
    elif i_comp_mode == 2:
        set_equal[bpg, tpb](F.device['COLP'],     F.device['COLP_NEW'])
    ##############################
    ##############################

    # PROGNOSE NEXT TIME LEVEL
    ##############################
    ##############################
    GR.timer.start('step')
    Prognostics.euler_forward(GR, GR.GRF[Prognostics.target],
                        **F.get(Prognostics.fields_prognostic,
                            target=Prognostics.target))
    GR.timer.stop('step')
    ##############################
    ##############################

    # DIAGNOSE VARIABLES
    ##############################
    ##############################
    GR.timer.start('diag')
    Diagnostics.primary_diag(GR.GRF[Diagnostics.target],
                        **F.get(Diagnostics.fields_primary_diag,
                            target=Diagnostics.target))
    GR.timer.stop('diag')
    ##############################
    ##############################

    


