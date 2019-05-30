#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          jacobson.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190530
License:            MIT

###############################################################################
"""
from namelist import comp_mode
from org_namelist import HOST, DEVICE
from dyn_org_discretizations import (TendencyFactory,
                                    DiagnosticsFactory) 
###############################################################################

Tendencies = TendencyFactory()
Diagnostics = DiagnosticsFactory()

def tendencies_jacobson(GR, GR_NEW, F, subgrids, NF):

    # PROGNOSE CONTINUITY
    ##############################
    ##############################
    GR.timer.start('cont')

    if comp_mode == 1:

        NF.old_to_new(F, host=True)
        Tendencies.continuity(HOST, GR_NEW,
                    **NF.get(Tendencies.fields_continuity, target=HOST))
        NF.new_to_old(F, host=True)

    elif comp_mode == 2:

        NF.old_to_new(F, host=False)
        Tendencies.continuity(DEVICE, GR_NEW,
                **NF.get(Tendencies.fields_continuity, target=DEVICE))
        NF.new_to_old(F, host=False)

    GR.timer.stop('cont')
    ##############################
    ##############################



    # PROGNOSE WIND
    ##############################
    ##############################
    GR.timer.start('wind')

    if comp_mode == 1:

        NF.old_to_new(F, host=True)
        Tendencies.momentum(HOST, GR_NEW,
                **NF.get(Tendencies.fields_momentum, target=HOST))
                        
        NF.new_to_old(F, host=True)
        
    elif comp_mode == 2:

        NF.old_to_new(F, host=False)
        Tendencies.momentum(DEVICE, GR_NEW,
                **NF.get(Tendencies.fields_momentum, target=DEVICE))
                        
        NF.new_to_old(F, host=False)

    GR.timer.stop('wind')
    ##############################
    ##############################



    ##############################
    ##############################
    GR.timer.start('temp')
    # PROGNOSE POTT
    if comp_mode == 1:

        NF.old_to_new(F, host=True)
        Tendencies.temperature(HOST, GR_NEW,
                    **NF.get(Tendencies.fields_temperature, target=HOST))
        NF.new_to_old(F, host=True)

    elif comp_mode == 2:

        NF.old_to_new(F, host=False)
        Tendencies.temperature(DEVICE, GR_NEW,
                    **NF.get(Tendencies.fields_temperature, target=DEVICE))
        NF.new_to_old(F, host=False)

    GR.timer.stop('temp')
    ##############################
    ##############################


    # MOIST VARIABLES
    ###############################
    ###############################
    #t_start = time.time()
    #if comp_mode == 0:
    #    F.dQVdt = water_vapor_tendency(GR, F.dQVdt, F.QV, F.COLP, F.COLP_NEW, \
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
    #    F.dQCdt = cloud_water_tendency(GR, F.dQCdt, F.QC, F.COLP, F.COLP_NEW, \
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)

    #elif comp_mode == 1:
    #    F.dQVdt = water_vapor_tendency_c(GR, njobs, F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
    #    F.dQVdt = np.asarray(F.dQVdt)
    #    F.dQCdt = cloud_water_tendency_c(GR, njobs, F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)
    #    F.dQCdt = np.asarray(F.dQCdt)

    #elif comp_mode == 2:
    #    water_vapor_tendency_gpu[GR.griddim, GR.blockdim, GR.stream] \
    #                                (F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
    #                                 F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC,
    #                                 GR.Ad, GR.dsigmad)
    #    GR.stream.synchronize()
    #    cloud_water_tendency_gpu[GR.griddim, GR.blockdim, GR.stream] \
    #                                (F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
    #                                 F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC,
    #                                 GR.Ad, GR.dsigmad)
    #    GR.stream.synchronize()

    #t_end = time.time()
    #GR.trac_comp_time += t_end - t_start
    ###############################
    ###############################




def diagnose_fields_jacobson(GR, GR_NEW, F, NF):

    ##############################
    ##############################
    if comp_mode == 1:

        NF.old_to_new(F, host=True)
        Diagnostics.primary_diag(HOST, GR_NEW,
                **NF.get(Diagnostics.fields_primary_diag, target=HOST))
        NF.new_to_old(F, host=True)

    elif comp_mode == 2:

        NF.old_to_new(F, host=False)
        Diagnostics.primary_diag(DEVICE, GR_NEW,
                **NF.get(Diagnostics.fields_primary_diag, target=DEVICE))
        NF.new_to_old(F, host=False)

    ##############################
    ##############################




