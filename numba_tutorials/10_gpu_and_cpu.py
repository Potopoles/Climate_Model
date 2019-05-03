import numba
from numba import cuda


@numba.jit
def clamp(x, xmin, xmax):
    if x < xmin:
        return(xmin)
    elif x > xmax:
        return(xmax)
    else:
        return(x)


@numba.cuda.jit
def clamp_array_gpu(x, xmim, xmax, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        out[i] = clamp(x[i], xmin, xmax)

def clamp_array_cpu(x, xmim, xmax, out):
