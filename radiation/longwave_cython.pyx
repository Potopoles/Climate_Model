import numpy as np
#cimport numpy as np
import cython
from namelist import wp
from radiation.namelist_radiation import njobs_rad, planck_n_lw_bins
from cython.parallel import prange 

if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np
ctypedef fused wp_cy:
    double
    float

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef calc_planck_intensity_c(wp_cy nu0, wp_cy nu1, wp_cy[::1]temp):

    cdef int nnu                = planck_n_lw_bins 
    cdef int c_njobs            = njobs_rad
    cdef int c_planck_n_lw_bins = planck_n_lw_bins

    #cdef wp_cy[   ::1] dsigma    = GR.dsigma
    cdef wp_cy[:, ::1] B         = np.zeros( (nnu, len(temp)), dtype=wp_np )
    cdef wp_cy[:, ::1] B_sum     = np.zeros( (len(temp)), dtype=wp_np )

    #cdef int i, inb, im1, ip1, j, jnb, jm1, jp1, k, kp1
    #cdef wp_cy hor_adv, vert_adv, num_diff

    #cdef wp_cy c_POTT_hor_dif_tau = POTT_hor_dif_tau

    #cdef wp_cy[:,:, ::1] dPOTTdt = np.zeros( (nx+2*nb,ny+2*nb,nz), dtype=wp_np)

    nu0 = nu0*100 # 1/m
    nu1 = nu1*100 # 1/m
    dnu = (nu1 - nu0)/(nnu - 1)
    #dnu = 5000 # 1/m
    nus = np.arange(nu0,nu1+1,dnu, dtype=wp_np)
    nus_center = nus[:-1]+dnu/2
    lambdas = 1./nus
    lambdas_center = 1./nus_center
    dlambdas = np.diff(lambdas)

    #B = 2*np.pi * (1 - omega_s) * 
    for c in range(0, nnu):
        B_sum[c] = B_sum[c] + 

    return(B)

    #if i_temperature_tendency:
    #    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #    #for i   in range(nb,nx +nb):
    #        im1 = i - 1
    #        ip1 = i + 1
    #        inb = i - nb
    #        for j   in range(nb,ny +nb):
    #            jm1 = j - 1
    #            jp1 = j + 1
    #            jnb = j - nb
    #            for k in range(0,nz):
    #                kp1 = k + 1

    #                # HORIZONTAL ADVECTION
    #                if i_hor_adv:
    #                    hor_adv = (+ UFLX[i  ,j  ,k  ] *\
    #                                 (POTT[im1,j  ,k  ] +\
    #                                  POTT[i  ,j  ,k  ])/2. \
    #                              - UFLX[ip1,j  ,k  ] *\
    #                                 (POTT[i  ,j  ,k  ] +\
    #                                  POTT[ip1,j  ,k  ])/2. \
    #                              + VFLX[i  ,j  ,k  ] *\
    #                                 (POTT[i  ,jm1,k  ] +\
    #                                  POTT[i  ,j  ,k  ])/2. \
    #                              - VFLX[i  ,jp1,k  ] *\
    #                                 (POTT[i  ,j  ,k  ] +\
    #                                  POTT[i  ,jp1,k  ])/2. \
    #                             ) / A[i  ,j  ]

    #                    dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + hor_adv


    #                # VERTICAL ADVECTION
    #                if i_vert_adv:
    #                    if k == 0:
    #                        vert_adv = COLP_NEW[i  ,j  ] * (\
    #                                - WWIND[i  ,j  ,kp1] * POTTVB[i  ,j  ,kp1] \
    #                                                       ) / dsigma[k]
    #                    elif k == nz:
    #                        vert_adv = COLP_NEW[i  ,j  ] * (\
    #                                + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
    #                                                       ) / dsigma[k]
    #                    else:
    #                        vert_adv = COLP_NEW[i  ,j  ] * (\
    #                                + WWIND[i  ,j  ,k  ] * POTTVB[i  ,j  ,k  ] \
    #                                - WWIND[i  ,j  ,kp1] * POTTVB[i  ,j  ,kp1] \
    #                                                       ) / dsigma[k]

    #                    dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + vert_adv


    #                # NUMERICAL DIFUSION 
    #                if i_num_dif and (c_POTT_hor_dif_tau > 0):
    #                    num_diff = c_POTT_hor_dif_tau * \
    #                                 (+ COLP[im1,j  ] * POTT[im1,j  ,k  ] \
    #                                  + COLP[ip1,j  ] * POTT[ip1,j  ,k  ] \
    #                                  + COLP[i  ,jm1] * POTT[i  ,jm1,k  ]  \
    #                                  + COLP[i  ,jp1] * POTT[i  ,jp1,k  ] \
    #                                - 4.*COLP[i  ,j  ] * POTT[i  ,j  ,k  ] )

    #                    dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + num_diff

    #                # RADIATION 
    #                if c_i_radiation:
    #                    dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + \
    #                                        dPOTTdt_RAD[inb,jnb,k  ]*COLP[i  ,j  ]
    #                # MICROPHYSICS 
    #                if c_i_microphysics:
    #                    dPOTTdt[i  ,j  ,k] = dPOTTdt[i  ,j  ,k] + \
    #                                        dPOTTdt_MIC[inb,jnb,k  ]*COLP[i  ,j  ]




