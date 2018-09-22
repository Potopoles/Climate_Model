import numpy as np
import cython
from namelist import wp
from radiation.namelist_radiation import njobs_rad, planck_n_lw_bins, emissivity_surface
from constants import con_c, con_h, con_kb
from libc.math cimport exp, pi

if wp == 'float64':
    from numpy import float64 as wp_np
elif wp == 'float32':
    from numpy import float32 as wp_np
ctypedef fused wp_cy:
    double
    float

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_planck_intensity_c(wp_cy[::1] temp,
                            wp_cy[::1] omega_s,
                            wp_cy[::1] planck_lambdas_center,
                            wp_cy[::1] planck_dlambdas):

    cdef int nnu                = planck_n_lw_bins 
    cdef int c_njobs            = njobs_rad
    cdef wp_cy c_con_c          = con_c
    cdef wp_cy c_con_h          = con_h
    cdef wp_cy c_con_kb         = con_kb

    cdef wp_cy[:, ::1] B              = np.zeros( (nnu, len(temp)) , dtype=wp_np )
    cdef wp_cy[   ::1] B_sum          = np.zeros( len(temp)        , dtype=wp_np )

    cdef int k, nu_ind
    cdef wp_cy dnu, spectral_radiance


    for k in range(0, len(temp)):
        for nu_ind in range(0,nnu):
            spectral_radiance = \
                2.*c_con_h*c_con_c**2. / planck_lambdas_center[nu_ind]**5. * \
                1. / ( exp( c_con_h*c_con_c / \
                (planck_lambdas_center[nu_ind]*c_con_kb*temp[k]) ) - 1. )
            B[nu_ind,k] = 2.*pi * (1. - omega_s[k]) * \
                          spectral_radiance * -planck_dlambdas[nu_ind] 

    for k in range(0, len(temp)):
        B_sum[k] = 0.
        for nu_ind in range(0, nnu):
            B_sum[k] = B_sum[k] + B[nu_ind,k]

    return(B_sum)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_surface_emission_c(wp_cy tsurf,
                            wp_cy[::1] planck_lambdas_center,
                            wp_cy[::1] planck_dlambdas):

    cdef int nnu                = planck_n_lw_bins 
    cdef int c_njobs            = njobs_rad
    cdef wp_cy c_con_c          = con_c
    cdef wp_cy c_con_h          = con_h
    cdef wp_cy c_con_kb         = con_kb
    cdef wp_cy c_emissivity_surface         = emissivity_surface

    cdef wp_cy[   ::1] B              = np.zeros( nnu , dtype=wp_np )
    cdef wp_cy         B_sum          = 0.

    cdef int nu_ind
    cdef wp_cy dnu, spectral_radiance


    for nu_ind in range(0,nnu):
        spectral_radiance = \
            2.*c_con_h*c_con_c**2. / planck_lambdas_center[nu_ind]**5. * \
            1. / ( exp( c_con_h*c_con_c / \
            (planck_lambdas_center[nu_ind]*c_con_kb*tsurf) ) - 1. )
        B[nu_ind] = emissivity_surface * pi *  \
                      spectral_radiance * - planck_dlambdas[nu_ind] 

    B_sum = 0.
    for nu_ind in range(0, nnu):
        B_sum = B_sum + B[nu_ind]

    return(B_sum)




#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef rad_calc_LW_RTE_matrix_c(int nz, int nzs,
                        wp_cy[::1] dtau, wp_cy[::1] gamma1, wp_cy[::1] gamma2,
                        wp_cy[::1] B_air, wp_cy B_surf, wp_cy albedo_surface):

    cdef wp_cy[   ::1] C1           = np.zeros(   nz        , dtype=wp_np )
    cdef wp_cy[   ::1] C2           = np.zeros(   nz        , dtype=wp_np )
    cdef wp_cy[   ::1] d0           = np.zeros( 2*nz        , dtype=wp_np )
    cdef wp_cy[   ::1] dp1          = np.zeros( 2*nz        , dtype=wp_np )
    cdef wp_cy[   ::1] dp2          = np.zeros( 2*nz        , dtype=wp_np )
    cdef wp_cy[   ::1] dm1          = np.zeros( 2*nz        , dtype=wp_np )
    cdef wp_cy[   ::1] dm2          = np.zeros( 2*nz        , dtype=wp_np )

    cdef wp_cy[   ::1] g_vec        = np.zeros( 2*nz+2      , dtype=wp_np )
    cdef wp_cy[:, ::1] A_mat        = np.zeros( (5, 2*nzs)  , dtype=wp_np )

    cdef int k, ind

    
    for k in range(0,2*nz+2):
        ind = k % nz
        if k == 2*nz+2:
            g_vec[k] = B_surf
        elif k > 0 and k % 2 == 0:
            g_vec[k] = - dtau[ind] * B_air[ind]
        else:# k > 0 and k % 2 == 1:
            g_vec[k] = dtau[ind] * B_air[ind]

    print(np.asarray(g_vec))
    quit()

    #C1 = 0.5 * dtau * gamma1
    #C2 = 0.5 * dtau * gamma2

    #d0_ = np.repeat(1+C1, 2)
    #d0_[range(1,len(d0_),2)] = - d0_[range(1,len(d0_),2)]
    #d0 = np.ones(2*nz+2); d0[1:-1] = d0_

    #dp1_ = np.repeat(-C2, 2)
    #dp1_[range(1,len(dp1_),2)] = - dp1_[range(1,len(dp1_),2)]
    #dp1 = np.zeros(2*nz+2); dp1[2:] = dp1_

    #dp2_ = np.repeat(-(1-C1), 2)
    #dp2_[range(0,len(dp2_),2)] = 0
    #dp2 = np.zeros(2*nz+2); dp2[2:] = dp2_

    #dm1_ = np.repeat(-C2, 2)
    #dm1_[range(1,len(dm1_),2)] = - dm1_[range(1,len(dm1_),2)]
    #dm1 = np.zeros(2*nz+2); dm1[:-2] = dm1_; dm1[-2] = -albedo_surface

    #dm2_ = np.repeat(1-C1, 2)
    #dm2_[range(1,len(dm2_),2)] = 0
    #dm2 = np.zeros(2*nz+2); dm2[:-2] = dm2_

    #A_mat = np.zeros( ( 5, 2*nzs ) )
    #A_mat[0,:] = dp2
    #A_mat[1,:] = dp1
    #A_mat[2,:] = d0
    #A_mat[3,:] = dm1
    #A_mat[4,:] = dm2

    return(A_mat, g_vec)
