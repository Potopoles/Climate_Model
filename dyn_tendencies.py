#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          dyn_tendencies.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190531
License:            MIT

Compute tendencies during one time step.
###############################################################################
"""
from namelist import i_comp_mode
from io_read_namelist import CPU, GPU
from dyn_org_discretizations import TendencyFactory
###############################################################################
if i_comp_mode == 1:
    Tendencies = TendencyFactory(target=CPU)
elif i_comp_mode == 2:
    Tendencies = TendencyFactory(target=GPU)

def compute_tendencies(GR, F):

    # PROGNOSE CONTINUITY
    ##############################
    ##############################
    GR.timer.start('cont')
    Tendencies.continuity(GR, GR.GRF[Tendencies.target],
                    **F.get(Tendencies.fields_continuity,
                        target=Tendencies.target))
    GR.timer.stop('cont')
    ##############################
    ##############################


    # PROGNOSE WIND
    ##############################
    ##############################
    GR.timer.start('wind')
    Tendencies.momentum(GR.GRF[Tendencies.target],
                    **F.get(Tendencies.fields_momentum,
                        target=Tendencies.target))
    GR.timer.stop('wind')
    ##############################
    ##############################


    # PROGNOSE POTT
    ##############################
    ##############################
    GR.timer.start('temp')
    Tendencies.temperature(GR.GRF[Tendencies.target],
                    **F.get(Tendencies.fields_temperature,
                            target=Tendencies.target))
    GR.timer.stop('temp')
    ##############################
    ##############################


    # MOIST VARIABLES
    ###############################
    ###############################
    #t_start = time.time()
    #if i_comp_mode == 0:
    #    F.dQVdt = water_vapor_tendency(GR, F.dQVdt, F.QV, F.COLP, F.COLP_NEW, \
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
    #    F.dQCdt = cloud_water_tendency(GR, F.dQCdt, F.QC, F.COLP, F.COLP_NEW, \
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)

    #elif i_comp_mode == 1:
    #    F.dQVdt = water_vapor_tendency_c(GR, njobs, F.dQVdt, F.QV, F.COLP, F.COLP_NEW,
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQVdt_MIC)
    #    F.dQVdt = np.asarray(F.dQVdt)
    #    F.dQCdt = cloud_water_tendency_c(GR, njobs, F.dQCdt, F.QC, F.COLP, F.COLP_NEW,
    #                                    F.UFLX, F.VFLX, F.WWIND, F.dQCdt_MIC)
    #    F.dQCdt = np.asarray(F.dQCdt)

    #elif i_comp_mode == 2:
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

