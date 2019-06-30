#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20190509
Last modified:      20190630
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

test_fields = ['UWIND','VWIND','WWIND','COLP','PHI']
test_fields.extend(['SURFTEMP', 'SLHFLX', 'SMOMXFLX', 'SMOMYFLX',
                    'SURFALBEDSW', 'SSHFLX'])
test_fields.extend(['SWFLXNET', 'LWFLXNET'])
test_fields.extend(['QV', 'QC'])
test_fields.extend(['dVFLXdt_TURB'])
####################################################################


ds_ref = xr.open_dataset(os.path.join(ref_path,file))
ds_test = xr.open_dataset(os.path.join(test_path,file))


failed = False

deviation_sum = 0

for test_field in test_fields:
    print(test_field)

    try:
        diff_array = ds_test[test_field] - ds_ref[test_field]
        maxv_test = np.abs(ds_test[test_field]).max().values
        if maxv_test <= tolerance:
            print('all elements == 0')
            deviation = 0.
        else:
            deviation = np.abs(diff_array).max().values / maxv_test
         

        deviation_sum += deviation
        print(deviation)
        if deviation > tolerance:
            print('!!! Deviation !!!')
            print('max val is ' + str(maxv_test))
            failed = True
        print()
    except KeyError:
        print('Not in output file.')

if failed:
    print('Test failed!')
else:
    print('Equal with tolerance ' + str(tolerance) + '    GREAT!')
    print('Deviation summed over variables is ' + str(deviation_sum))
    if deviation_sum == 0:
        print('Bitwise identical')

