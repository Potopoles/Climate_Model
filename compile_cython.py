from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

string = 'wind_cython'
string = 'geopotential_cython'
string = 'diagnostics_cython'
string = 'temperature_cython'
string = 'moisture_cython'
string = 'continuity_cython'

ext_modules = [
    Extension(
            string,
            [string+'.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],)]


setup(name='Parallel',
        ext_modules=cythonize(ext_modules))

#setup(name='Parallel',
#        ext_modules=cythonize('wind_cython.pyx'))


## run:
# python compile_cython.py build_ext --inplace 
