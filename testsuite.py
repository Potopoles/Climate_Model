#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190601
License:            MIT

Compare output fields of two simulations and check if they are
identical.
###############################################################################
"""
import os
import numpy as np
import xarray as xr

# USER INPUT
####################################################################
tolerance = 1E-4

ref_path = '../output_ref'
test_path = '../output_test'

file = 'out0002.nc'

test_fields = ['UWIND','VWIND','WWIND','COLP','POTT','PHI']
test_fields.extend(['SURFTEMP'])
test_fields.extend(['SWFLXNET', 'LWFLXNET', 'dPOTTdt_RAD'])
####################################################################


ds_ref = xr.open_dataset(os.path.join(ref_path,file))
ds_test = xr.open_dataset(os.path.join(test_path,file))


failed = False

deviation_sum = 0

for test_field in test_fields:
    print(test_field)

    diff_array = ds_test[test_field] - ds_ref[test_field]
    deviation = np.abs(diff_array).max().values

    #vals_test = ds_test[test_field].values
    #vals_ref = ds_ref[test_field].values
    #vals_diff = vals_test - vals_ref
    #deviation = np.abs(np.max(vals_diff))
    
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

