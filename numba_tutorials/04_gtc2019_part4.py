from numba import cuda
import numpy as np
import cupy as cp
import time

@cuda.jit
def add_kernel(x, y, out):
    #tx = cuda.threadIdx.x
    #bx = cuda.blockIdx.x

    #block_size = cuda.blockDim.x
    #grid_size = cuda.gridDim.x

    #start = tx + bx*block_size
    #stride = block_size*grid_size
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(start, x.shape[0], stride):
        #for i in range(1000):
        #    out[i] = y[i] + x[i]
        if i > 0:
            out[i] = y[i] + x[i] + x[i-1]
        else:
            out[i] = y[i] + x[i]


n = int(2E7)
#x = np.arange(n).astype(np.float32)
x = cp.arange(n).astype(cp.float32)
y = 2*x
#out = np.empty_like(x)
out = cp.empty_like(x)

threads_per_block = 128
#threads_per_block = 512
blocks_per_grid = 30

add_kernel[blocks_per_grid, threads_per_block](x, y, out)
t0 = time.time()
add_kernel[blocks_per_grid, threads_per_block](x, y, out)
cuda.synchronize()
print(time.time() - t0)
print(out[:10])
