import os
import numpy as np
import xarray as xr

tolerance = 1E-7

ref_path = '../output_orig'
test_path = '../output'

file = 'out0002.nc'

test_fields = ['UWIND','VWIND','WWIND','PSURF','POTT','PHI']


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

