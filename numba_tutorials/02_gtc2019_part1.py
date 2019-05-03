import numpy as np
from numba import jit
import math, time

# PART I
x = np.zeros((2,3,4), dtype=np.float32)
print(repr(x))
print(x.dtype)
print(x.shape)
print(x.strides)


# PART II
orig = np.arange(20)
print(orig)
print(orig.strides)

# slice out every third element
view = orig[::3]
print(view)
print(view.strides)
view[1] = 99
print(view)
print(orig)

# PART III
b = np.arange(1,4)
print(b)
print(b[:,np.newaxis])


# PART IV
@jit
def hypot(x, y):
    x = abs(x); y = abs(y)
    t = min(x, y); x = max(x, y)
    t = t / x
    return(x * math.sqrt(1 + t*t))

# call it first for first time compilation
hypot(3.0,4.0)
t0 = time.time()
hypot(3.0,4.0)
print(time.time() - t0)
t0 = time.time()
hypot.py_func(3.0,4.0)
print(time.time() - t0)

print(hypot.inspect_types())



# PART V
#@jit(nopython=True)
@jit()
def cannot_compile(x):
    return(x['key'])

print(cannot_compile(dict(key='value')))


# PART VI
@jit(nopython=True)
def ex1(x, y, out):
    for i in range(x.shape[0]):
        #out[i] = x[i] + y[i]
        out[i] = hypot(x[i], y[i])

in1 = np.arange(10, dtype=np.float64)
in2 = 2 * in1+1
out = np.empty_like(in1)
print('in1: ' + str(in1))
print('in2: ' + str(in2))

ex1(in1, in2, out)
print('out: ' + str(out))

np.testing.assert_almost_equal(out, np.hypot(in1, in2))
