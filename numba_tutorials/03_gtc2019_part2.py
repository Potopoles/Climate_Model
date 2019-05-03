import numpy as np
import cupy as cp
from numba import vectorize
import time, math
import scipy.stats

ary = cp.arange(10).reshape((2,5))
print(repr(ary))
print(ary.dtype)
print(ary.shape)
print(ary.strides)

# move data from cpu to gpu
ary_cpu = np.arange(10)
ary_gpu = cp.asarray(ary_cpu)
print(type(ary_gpu))
print(ary_gpu.device)
ary_cpu_returned = cp.asnumpy(ary_gpu)
print(ary_cpu_returned)
print(type(ary_cpu_returned))


@vectorize(['int64(int64, int64)'], target='cuda')
def add_ufunc(x, y):
    return(x + y)
a = np.array([1,2,3,4])
b = np.array([10,20,30,40])
b_col = b[:, np.newaxis]
c = np.arange(4*4).reshape((4,4))

print(add_ufunc(a, b))
print(add_ufunc(b_col, c))



# fast example
SQRT_2PI = np.float32((2*math.pi)**0.5)

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian_pdf(x, xmean, sigma):
    return(math.exp(-0.5 * ((x - xmean) / sigma)**2) / (sigma * SQRT_2PI))

x = np.random.uniform(-3, 3, size=int(1E7)).astype(np.float32)
mean = np.float32(0.0)
sigma = np.float32(1.0)
print(gaussian_pdf(x[0], 0.0, 1.0))
norm_pdf = scipy.stats.norm
print('cpu')
t0 = time.time()
norm_pdf.pdf(x, loc=mean, scale=sigma)
print(time.time() - t0)
print('gpu')
t0 = time.time()
gaussian_pdf(x, mean, sigma)
print(time.time() - t0)
