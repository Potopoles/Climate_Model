#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
File name:          dyn_tendencies.py  
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190609
License:            MIT

Compute tendencies during one time step.
- 20190531: Created (CH)
- 20190609: Added moisture QV and QC (CH)
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


    # PROGNOSE MOISTURE VARIABLES
    ###############################
    ###############################
    GR.timer.start('moist')
    Tendencies.moisture(GR.GRF[Tendencies.target],
                    **F.get(Tendencies.fields_moisture,
                            target=Tendencies.target))
    GR.timer.stop('moist')
    ###############################
    ###############################

