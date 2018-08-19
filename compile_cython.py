from distutils.core import setup
from Cython.Build import cythonize

#setup(name='Parallel',
#        ext_modules=cythonize('cython_loop.pyx'))

# TODO
setup(name='Parallel',
        ext_modules=cythonize('temperature_cython.pyx'))


## run:
# python compile_cython.py build_ext --inplace 
