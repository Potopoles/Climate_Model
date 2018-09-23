from distutils.core import setup
import numpy
# BUGFIX for numpy stuff not found error. Run the following, and use the path for:
# sudo cp -r PATH/numpy /usr/local/include/
#print(numpy.get_include())
#quit()

import os
import glob
from Cython.Build import cythonize
from distutils.extension import Extension


remove_existing = 1
compile = 1
binary_dir = 'bin'

all = ['wind_cython',
        'geopotential_cython',
        'diagnostics_cython',
        'temperature_cython',
        'moisture_cython',
        'continuity_cython',
        'jacobson_cython']

all_par = ['wind_cython_par']

all_rad = ['longwave_cython']

strings = all_par
strings = all
strings = all_rad
#strings = ['wind_cython']
#strings = ['geopotential_cython']
#strings = ['diagnostics_cython']
strings = ['temperature_cython']
#strings = ['moisture_cython']
#strings = ['continuity_cython']
#strings = ['jacobson_cython']

folder = ''
#folder = 'radiation/'

for string in strings:
    if remove_existing:
        for filename in glob.glob(string+'.c*'):
            os.remove(filename)



    if compile:
        ext_modules = [
            Extension(
                    string,
                    [folder + string+'.pyx'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'],)]

        setup(name='Parallel',
                ext_modules=cythonize(ext_modules))

        for filename in glob.glob(string+'.c*'):
            os.rename(filename,binary_dir + '/' + filename)


## run:
# python compile_cython.py build_ext --inplace 
