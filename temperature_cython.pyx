import numpy as np
from namelist import wp, POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics

from libc.math cimport exp
cimport numpy as np
import cython
from cython.parallel import prange 

if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np
ctypedef fused wp_cy:
    double
    float

cdef int i_hor_adv = 1
cdef int i_vert_adv = 1
cdef int i_num_dif = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef temperature_tendency_jacobson_c( GR, njobs,\
        wp_cy[:,:, ::1] POTT,
        wp_cy[:,:, ::1] POTTVB,
        wp_cy[:,   ::1] COLP,
        wp_cy[:,   ::1] COLP_NEW,
        wp_cy[:,:, ::1] UFLX,
        wp_cy[:,:, ::1] VFLX,
        wp_cy[:,:, ::1] WWIND,
        wp_cy[:,:, ::1] dPOTTdt_RAD,
        wp_cy[:,:, ::1] dPOTTdt_MIC):

    cdef int c_njobs = njobs

    cdef int c_i_microphysics = i_microphysics
    cdef int c_i_radiation = i_radiation
   
    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz  = GR.nz
    cdef wp_cy[   ::1] dsigma    = GR.dsigma
    cdef wp_cy[:, ::1] A         = GR.A

    cdef int i, inb, im1, ip1, j, jnb, jm1, jp1, k, kp1
    cdef wp_cy hor_adv, vert_adv, num_diff

    cdef wp_cy c_POTT_hor_dif_tau = POTT_hor_dif_tau

    cdef wp_cy[:,:, ::1] dPOTTdt = np.zeros( (nx+2*nb,ny+2*nb,nz), dtype=wp_np)

    if i_temperature_tendency:
        for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
        #for i   in range(nb,nx +nb):
            im1 = i - 1
            ip1 = i + 1
            inb = i - nb
            for j   in range(nb,ny +nb):
                jm1 = j - 1
                jp1 = j + 1
                jnb = j - nb
                for k in range(0,nz):
                    kp1 = k + 1

                    # HORIZONTAL ADVECTION
                    if i_hor_adv:
                        hor_adv = (+ UFLX[i  ,j  ,k  ] *\
                                     (POTT[im1,j  ,k  ] +\
                                      POTT[i  ,j  ,k  ])/2. \
                                  - UFLX[ip1,j  ,k  ] *\
                                     (POTT[i  ,j  ,k  ] +\
                                      POTT[ip1,j  ,k  ])/2. \
                                  + VFLX[i  ,j  ,k  ] *\
                                     (POTT[i  ,jm1,k  ] +\
                                      POTT[i  ,j  ,k  ])/2. \
                                  - VFLX[i  ,jp1,k  ] *\
                                     (POTT[i  ,j  ,k  ] +\
                                      POTT[i  ,jp1,k  ])/2. \
                                 ) / A[i  ,j  ]

                        dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + hor_adv


                    # VERTICAL ADVECTION
                    if i_vert_adv:
                        if k == 0:
                            vert_adv = COLP_NEW[i  ,j  ] * (\
                                    - WWIND[i  ,j  ,kp1] * POTTVB[i  ,j  ,kp1] \
                                                           ) / dsigma[k]
                        elif k == nz:
                            vert_adv = COLP_NEW[i  ,j  ] * (\
                                    + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
                                                           ) / dsigma[k]
                        else:
                            vert_adv = COLP_NEW[i  ,j  ] * (\
                                    + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
                                    - WWIND[i  ,j  ,kp1] * POTTVB[i  ,j  ,kp1] \
                                                           ) / dsigma[k]

                        dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + vert_adv


                    # NUMERICAL DIFUSION 
                    if i_num_dif and (c_POTT_hor_dif_tau > 0):
                        num_diff = c_POTT_hor_dif_tau * exp(-(nz-k-1)) *\
                                    ( + COLP[im1,j  ] * POTT[im1,j  ,k  ] \
                                      + COLP[ip1,j  ] * POTT[ip1,j  ,k  ] \
                                      + COLP[i  ,jm1] * POTT[i  ,jm1,k  ]  \
                                      + COLP[i  ,jp1] * POTT[i  ,jp1,k  ] \
                                   - 4.*COLP[i  ,j  ] * POTT[i  ,j  ,k  ] )

                        dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + num_diff

                    # RADIATION 
                    if c_i_radiation:
                        dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + \
                                            dPOTTdt_RAD[inb,jnb,k  ]*COLP[i  ,j  ]
                    # MICROPHYSICS 
                    if c_i_microphysics:
                        dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + \
                                            dPOTTdt_MIC[inb,jnb,k  ]*COLP[i  ,j  ]


    return(dPOTTdt)


