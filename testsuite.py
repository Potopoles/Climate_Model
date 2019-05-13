#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          testsuite.py  
Author:             Christoph Heim (CH)
Date created:       20190509
Last modified:      20190509
License:            MIT

Compare output fields of two simulations and check if they are
identical.
"""
import os
import numpy as np
import xarray as xr

# USER INPUT
####################################################################
tolerance = 1E-6

ref_path = '../output_ref'
test_path = '../output_test'

file = 'out0002.nc'

test_fields = ['UWIND','VWIND','WWIND','PSURF','POTT','PHI']
####################################################################


ds_ref = xr.open_dataset(os.path.join(ref_path,file))
ds_test = xr.open_dataset(os.path.join(test_path,file))


failed = False

deviation_sum = 0

for test_field in test_fields:
    print(test_field)
    diff_array = ds_test[test_field] - ds_ref[test_field]
    deviation = np.abs(diff_array).max().values
    deviation_sum += deviation
    print(deviation)
    if deviation > tolerance:
        print('Deviation!')
        failed = True

if failed:
    print('Test failed!')
else:
    print('Equal with tolerance ' + str(tolerance) + '    GREAT!')
    print('Deviation summed over variables is ' + str(deviation_sum))
    if deviation_sum == 0:
        print('Bitwise identical')

