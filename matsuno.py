#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          matsuno.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190530
License:            MIT

Perform a matsuno time step.
###############################################################################
"""
from namelist import comp_mode
from org_namelist import HOST, DEVICE
from grid import tpb, bpg
from GPU import set_equal
from jacobson import tendencies_jacobson, diagnose_fields_jacobson
from dyn_org_discretizations import (PrognosticsFactory) 
###############################################################################
Prognostics = PrognosticsFactory()

def step_matsuno(GR, NF):

    ##############################
    ##############################
    GR.timer.start('step')

    if comp_mode == 1:
        NF.host['COLP_OLD'][:]  = NF.host['COLP'][:]
        NF.host['UWIND_OLD'][:] = NF.host['UWIND'][:]
        NF.host['VWIND_OLD'][:] = NF.host['VWIND'][:]
        NF.host['POTT_OLD'][:]  = NF.host['POTT'][:]
        #NF.host['QV_OLD'][:]    = NF.host['QV'][:]
        #NF.host['QC_OLD'][:]    = NF.host['QC'][:]
    elif comp_mode == 2:
        set_equal[bpg, tpb](NF.device['COLP_OLD'],     NF.device['COLP'])
        set_equal[bpg, tpb](NF.device['UWIND_OLD'],    NF.device['UWIND'])
        set_equal[bpg, tpb](NF.device['VWIND_OLD'],    NF.device['VWIND'])
        set_equal[bpg, tpb](NF.device['POTT_OLD'],     NF.device['POTT'])
        #set_equal[bpg, tpb](NF.device['QV_OLD'],       NF.device['QV'])
        #set_equal[bpg, tpb](NF.device['QC_OLD'],       NF.device['QC'])

    GR.timer.stop('step')
    ##############################
    ##############################


    ############################################################
    ############################################################
    ##########     ESTIMATE
    ############################################################
    ############################################################

    ##############################
    ##############################
    tendencies_jacobson(GR, NF)
    if comp_mode == 1:
        NF.host['COLP'][:]  = NF.host['COLP_NEW'][:]
    elif comp_mode == 2:
        set_equal[bpg, tpb](NF.device['COLP'],     NF.device['COLP_NEW'])
    ##############################
    ##############################

    ##############################
    ##############################
    GR.timer.start('step')
    if comp_mode == 1:

        Prognostics.euler_forward(HOST, GR,
                **NF.get(Prognostics.fields_prognostic, target=HOST))

    elif comp_mode == 2:

        Prognostics.euler_forward(DEVICE, GR,
                **NF.get(Prognostics.fields_prognostic, target=DEVICE))

    GR.timer.stop('step')
    ##############################
    ##############################


    ##############################
    ##############################
    GR.timer.start('diag')
    diagnose_fields_jacobson(GR, NF)
    GR.timer.stop('diag')
    ##############################
    ##############################



    ############################################################
    ############################################################
    ##########     FINAL
    ############################################################
    ############################################################

    ##############################
    ##############################
    tendencies_jacobson(GR, NF)
    if comp_mode == 1:
        NF.host['COLP'][:]  = NF.host['COLP_NEW'][:]
    elif comp_mode == 2:
        set_equal[bpg, tpb](NF.device['COLP'],     NF.device['COLP_NEW'])
    ##############################
    ##############################


    ##############################
    ##############################
    GR.timer.start('step')
    if comp_mode == 1:

        Prognostics.euler_forward(HOST, GR,
                **NF.get(Prognostics.fields_prognostic, target=HOST))

    elif comp_mode == 2:

        Prognostics.euler_forward(DEVICE, GR,
                **NF.get(Prognostics.fields_prognostic, target=DEVICE))

    GR.timer.stop('step')
    ##############################
    ##############################


    ##############################
    ##############################
    GR.timer.start('diag')
    diagnose_fields_jacobson(GR, NF)
    GR.timer.stop('diag')
    ##############################
    ##############################

    


