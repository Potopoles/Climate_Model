#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190530
Last modified:      20190616
License:            MIT

Functions and classes used for support:
- Timer class to measure execution times.
- Function to create list with function input fields from function signature
###############################################################################
"""
import time
import numpy as np
from numba import cuda
from inspect import signature

from namelist import i_sync_context, i_comp_mode

class Timer:
    
    def __init__(self):
        self.i_sync_context = i_sync_context
        self.i_comp_mode    = i_comp_mode

        self.timings = {}
        self.flags = {}

    def start(self, timer_key):
        if timer_key not in self.timings.keys():
            self.timings[timer_key] = 0.
            self.flags [timer_key] = None
        self.flags[timer_key] = time.time()

    def stop(self, timer_key):
        if (timer_key not in self.flags.keys()
            or self.flags[timer_key] is None):
            raise ValueError('No time measurement in progress for timer ' +
                            str(timer_key) + '.')

        if self.i_comp_mode == 2 and self.i_sync_context:
            cuda.synchronize()
        self.timings[timer_key] += time.time() - self.flags[timer_key]
        self.flags[timer_key] = None

    def print_report(self):
        n_decimal_perc = 0
        n_decimal_sec = 1
        n_decimal_min = 2
        total = self.timings['total']
        print('took ' + str(np.round(total/60,n_decimal_min)) + ' min.')
        print('Detailed computing times:')
        print('#### gernal')
        for key,value in self.timings.items():
            print(key + '\t' + 
                str(np.round(100*value/total,n_decimal_perc)) +
                '\t%\t' + str(np.round(value,n_decimal_sec)) + ' \tsec')



def function_input_fields(function):
    """
    From function reads input arguments and extracts model fields.
    Returns list with string of model fields.
    """
    input_fields = list(signature(function).parameters)
    ignore = ['self', 'GR', 'GRF']
    for ign in ignore:
        if ign in input_fields:
            input_fields.remove(ign)
    return(input_fields)
