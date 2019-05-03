import numpy as np
import time
from numba import vectorize

target = 'cuda'
#target = 'cpu'

#@vectorize(['float64(float64, float64)'], target=target)
@vectorize(['float32(float32, float32)'], target=target)
def Add(a, b):
    c = a + b
    for i in range(100):
        c += a + b
        
    return(c)


N = int(1E8)
#dtype_float = np.float64
dtype_float = np.float32
A = np.ones(N, dtype=dtype_float)
B = np.ones(A.shape, dtype=dtype_float)
C = np.empty_like(A, dtype=dtype_float)

t0 = time.time()
C = Add(A, B)
print(time.time() - t0)

