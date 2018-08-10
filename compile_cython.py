from distutils.core import setup
from Cython.Build import cythonize

#setup(name='Parallel',
#        ext_modules=cythonize('cython_loop.pyx'))

setup(name='Parallel',
        ext_modules=cythonize('wind_cython.pyx'))


## run:
# python compile_cython.py build_ext --inplace 
