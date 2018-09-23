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




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rad_calc_LW_RTE_matrix_c(int nz, int nzs,
                        wp_cy[::1] dtau, wp_cy[::1] gamma1, wp_cy[::1] gamma2,
                        wp_cy[::1] B_air, wp_cy B_surf, wp_cy albedo_surface):

    cdef wp_cy[   ::1] C1           = np.zeros(   nz        , dtype=wp_np )
    cdef wp_cy[   ::1] C2           = np.zeros(   nz        , dtype=wp_np )
    cdef wp_cy[   ::1] d0           = np.zeros( 2*nzs      , dtype=wp_np )
    cdef wp_cy[   ::1] dp1          = np.zeros( 2*nzs      , dtype=wp_np )
    cdef wp_cy[   ::1] dp2          = np.zeros( 2*nzs      , dtype=wp_np )
    cdef wp_cy[   ::1] dm1          = np.zeros( 2*nzs      , dtype=wp_np )
    cdef wp_cy[   ::1] dm2          = np.zeros( 2*nzs      , dtype=wp_np )

    cdef wp_cy[   ::1] g_vec        = np.zeros( 2*nzs      , dtype=wp_np )
    cdef wp_cy[:, ::1] A_mat        = np.zeros( (5,2*nzs)  , dtype=wp_np )

    cdef int k, ind


    
    for k in range(0,2*nz+2):
        ind = ((k+1) - (k+1)%2)/2 - 1
        if k == 2*nz+2-1:
            g_vec[k] = B_surf
        elif k > 0 and k % 2 == 0:
            g_vec[k] = - dtau[ind] * B_air[ind]
        elif k % 2 == 1:# k > 0 and k % 2 == 1:
            g_vec[k] = dtau[ind] * B_air[ind]

    for k in range(0,nz):
        C1[k] = 0.5 * dtau[k] * gamma1[k]
        C2[k] = 0.5 * dtau[k] * gamma2[k]

    for k in range(0,2*nz+2):
        ind = ((k+1) - (k+1)%2)/2 - 1
        #print(str(k) + '  ' + str(ind))
        if k == 0:
            d0[k] = 1.
        elif k == 2*nz+2-1:
            d0[k] = 1.
        elif k % 2 == 0:
            d0[k] = - (1. + C1[ind])
        elif k % 2 == 1:# k > 0 and k % 2 == 1:
            d0[k] = + (1. + C1[ind])

    for k in range(0,2*nz+2):
        ind = (k - k%2)/2 - 1
        #print(str(k) + '  ' + str(ind))
        if k == 0:
            dp1[k] = 0.
            dp2[k] = 0.
        elif k == 1:
            dp1[k] = 0.
            dp2[k] = 0.
        elif k % 2 == 0:
            dp1[k] = - C2[ind]
            dp2[k] = 0.
        elif k % 2 == 1:# k > 0 and k % 2 == 1:
            dp1[k] = + C2[ind]
            dp2[k] = - (1. - C1[ind])

    for k in range(0,2*nz+2):
        ind = (k - k%2)/2
        #print(str(k) + '  ' + str(ind))
        if k == 2*nz+2-1:
            dm1[k] = 0.
            dm2[k] = 0.
        elif k == 2*nz+2-2:
            dm1[k] = - albedo_surface
            dm2[k] = 0.
        elif k % 2 == 0:
            dm1[k] = - C2[ind]
            dm2[k] = + (1. - C1[ind])
        elif k % 2 == 1:# k > 0 and k % 2 == 1:
            dm1[k] = + C2[ind]
            dm2[k] = 0.

    for k in range(0,2*nz+2):
        A_mat[0,k] = dp2[k]
        A_mat[1,k] = dp1[k]
        A_mat[2,k] = d0 [k]
        A_mat[3,k] = dm1[k]
        A_mat[4,k] = dm2[k]



    return(A_mat, g_vec)
