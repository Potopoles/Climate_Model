from namelist import  wp
from numba import cuda, jit
if wp == 'float64':
    from numba import float64

@jit([wp+'[:,:  ], '+wp+'[:,:  ], '+wp+'[:,:  ], '+wp], target='gpu')
def time_step_2D(FIELD_NEW, FIELD_OLD, dFIELDdt, dt):
    nx = FIELD_NEW.shape[0] - 2
    ny = FIELD_NEW.shape[1] - 2
    i, j = cuda.grid(2)
    if i > 0 and i < nx+1 and j > 0 and j < ny+1:
        FIELD_NEW[i,j] = FIELD_OLD[i,j] + dt*dFIELDdt[i,j]
