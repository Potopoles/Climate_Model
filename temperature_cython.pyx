import numpy as np
import time
from namelist import POTT_hor_dif_tau, i_temperature_tendency, \
                    i_radiation, i_microphysics

cimport numpy as np
import cython
from cython.parallel import prange 

cdef int i_hor_adv = 1
cdef int i_vert_adv = 1
cdef int i_num_dif = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef temperature_tendency_jacobson_par( GR, njobs,\
        double[:,:, ::1] POTT,
        double[:,:, ::1] POTTVB,
        double[:,   ::1] COLP,
        double[:,   ::1] COLP_NEW,
        double[:,:, ::1] UFLX,
        double[:,:, ::1] VFLX,
        double[:,:, ::1] WWIND,
        double[:,:, ::1] dPOTTdt_RAD,
        double[:,:, ::1] dPOTTdt_MIC):

    cdef int c_njobs = njobs

    cdef int c_i_microphysics = i_microphysics
    cdef int c_i_radiation = i_radiation
   
    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz  = GR.nz
    cdef double[   ::1] dsigma    = GR.dsigma
    cdef double[:, ::1] A         = GR.A

    cdef int i, inb, im1, ip1, j, jnb, jm1, jp1, k, kp1
    cdef double hor_adv, vert_adv, num_diff

    cdef double c_POTT_hor_dif_tau = POTT_hor_dif_tau

    cdef double[:,:, ::1] dPOTTdt = np.zeros( (nx ,ny ,nz) )

    if i_temperature_tendency:
        for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs):
            im1 = i - 1
            ip1 = i + 1
            inb = i - nb
            for j   in prange(nb,ny +nb, nogil=False, num_threads=c_njobs):
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

                        dPOTTdt[inb,jnb,k] = dPOTTdt[inb,jnb,k] + hor_adv


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

                        dPOTTdt[inb,jnb,k] = dPOTTdt[inb,jnb,k] + vert_adv


                    # NUMERICAL DIFUSION 
                    if i_num_dif and (c_POTT_hor_dif_tau > 0):
                        num_diff = c_POTT_hor_dif_tau * \
                                     (+ COLP[im1,j  ] * POTT[im1,j  ,k  ] \
                                      + COLP[ip1,j  ] * POTT[ip1,j  ,k  ] \
                                      + COLP[i  ,jm1] * POTT[i  ,jm1,k  ]  \
                                      + COLP[i  ,jp1] * POTT[i  ,jp1,k  ] \
                                    - 4.*COLP[i  ,j  ] * POTT[i  ,j  ,k  ] )

                        dPOTTdt[inb,jnb,k] = dPOTTdt[inb,jnb,k] + num_diff

                    # RADIATION 
                    if c_i_radiation:
                        dPOTTdt[inb,jnb,k] = dPOTTdt[inb,jnb,k] + \
                                            dPOTTdt_RAD[inb,jnb,k  ]*COLP[i  ,j  ]
                    # MICROPHYSICS 
                    if c_i_microphysics:
                        dPOTTdt[inb,jnb,k] = dPOTTdt[inb,jnb,k] + \
                                            dPOTTdt_MIC[inb,jnb,k  ]*COLP[i  ,j  ]


    return(dPOTTdt)


