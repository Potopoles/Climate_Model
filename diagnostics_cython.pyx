import numpy as np
from constants import con_kappa, con_g, con_Rd
from namelist import wp, pTop, njobs

cimport numpy as np
import cython
from cython.parallel import prange 

ctypedef fused wp_cy:
    double
    float




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef diagnose_secondary_fields_c(GR, \
                                wp_cy[:,   ::1] COLP,
                                wp_cy[:,:, ::1] PAIR,
                                wp_cy[:,:, ::1] PHI,
                                wp_cy[:,:, ::1] POTT,
                                wp_cy[:,:, ::1] POTTVB,
                                wp_cy[:,:, ::1] TAIR,
                                wp_cy[:,:, ::1] TAIRVB,
                                wp_cy[:,:, ::1] RHO,
                                wp_cy[:,:, ::1] PVTF,
                                wp_cy[:,:, ::1] PVTFVB,
                                wp_cy[:,:, ::1] UWIND,
                                wp_cy[:,:, ::1] VWIND,
                                wp_cy[:,:, ::1] WIND):

    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz = GR.nz
    cdef int nzs = GR.nzs
    cdef int i, j, k, ks
    cdef wp_cy c_con_kappa = con_kappa
    cdef wp_cy c_con_Rd = con_Rd


    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for j   in range(nb,ny +nb):
            for k in range(0,nz):
                TAIR[i,j,k] = POTT[i,j,k] * PVTF[i,j,k]
                PAIR[i,j,k] = 100000.*(PVTF[i,j,k])**(1./c_con_kappa)
                RHO[i,j,k] = PAIR[i,j,k] / (c_con_Rd * TAIR[i,j,k])
                WIND[i,j,k] = ( ((UWIND[i,j,k] + UWIND[i+1,j,k])/2.)**2. + \
                                ((VWIND[i,j,k] + VWIND[i,j+1,k])/2.)**2. )**(1./2.)

            for ks in range(0,nzs):
                TAIRVB[i,j,ks] = POTTVB[i,j,ks] * PVTFVB[i,j,ks]


    return(PAIR, TAIR, TAIRVB, RHO, WIND)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef diagnose_POTTVB_jacobson_c(GR, njobs,
        wp_cy[:,:, ::1] POTTVB,
        wp_cy[:,:, ::1] POTT,
        wp_cy[:,:, ::1] PVTF,
        wp_cy[:,:, ::1] PVTFVB):

    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nzs = GR.nzs

    cdef int i, j, ks, ksm1


    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for j   in range(nb,ny +nb):

            for ks in range(1,nzs-1):
                ksm1 = ks - 1

                POTTVB[i,j,ks  ] =   ( \
                        +   (PVTFVB[i,j,ks  ] - PVTF[i,j,ksm1]) * \
                            POTT[i,j,ksm1]
                        +   (PVTF[i,j,ks  ] - PVTFVB[i,j,ks  ]) * \
                            POTT[i,j,ks  ] ) / \
                            (PVTF[i,j,ks  ] - PVTF[i,j,ksm1])

            # extrapolate model bottom and model top POTTVB
            POTTVB[i,j, 0] = POTT[i,j, 0] - \
                    ( POTTVB[i,j, 1] - POTT[i,j, 0] )
            POTTVB[i,j,-1] = POTT[i,j,-1] - \
                    ( POTTVB[i,j,-2] - POTT[i,j,-1] )


    return(POTTVB)






@cython.boundscheck(False)
@cython.wraparound(False)
cpdef interp_COLPA_c(GR, njobs, 
    wp_cy[:, ::1] COLP):

    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys

    cdef wp_cy[:, ::1] A         = GR.A

    cdef int i, inb, im1, ip1, i_s, isnb, ism1, j, jnb, jm1, jp1, js, jsnb, jsm1

    cdef wp_cy[:, ::1] COLPA_is = np.zeros( (nxs,ny ), dtype=wp )
    cdef wp_cy[:, ::1] COLPA_js = np.zeros( (nx ,nys), dtype=wp )

    for i_s in prange(nb,nxs +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i_s in range(nb,nxs +nb):
        ism1 = i_s - 1
        isnb = i_s - nb
        for j   in range(nb,ny +nb):
            jp1 = j + 1
            jm1 = j - 1
            jnb = j - nb

            # ATTEMPT TO INTERPOLATE ONLY WITH TWO NEIGHBORING POINTS (JACOBSON)
            if j == nb:
                COLPA_is[isnb,jnb] = 1./4.*( \
                                    COLP[ism1,j  ] * A[ism1,j  ] + \
                                    COLP[i_s ,j  ] * A[i_s ,j  ] + \
                                    COLP[ism1,jp1] * A[ism1,jp1] + \
                                    COLP[i_s ,jp1] * A[i_s ,jp1]   )

            # ATTEMPT TO INTERPOLATE ONLY WITH TWO NEIGHBORING POINTS (JACOBSON)
            elif j == ny+nb-1:
                COLPA_is[isnb,jnb] = 1./4.*( \
                                    COLP[ism1,j  ] * A[ism1,j  ] + \
                                    COLP[i_s ,j  ] * A[i_s ,j  ] + \
                                    COLP[ism1,jm1] * A[ism1,jm1] + \
                                    COLP[i_s ,jm1] * A[i_s ,jm1]   )

            else:
                COLPA_is[isnb,jnb] = 1./8.*( \
                                    COLP[ism1,jp1] * A[ism1,jp1] + \
                                    COLP[i_s ,jp1] * A[i_s ,jp1] + \
                               2. * COLP[ism1,j  ] * A[ism1,j  ] + \
                               2. * COLP[i_s ,j  ] * A[i_s ,j  ] + \
                                    COLP[ism1,jm1] * A[ism1,jm1] + \
                                    COLP[i_s ,jm1] * A[i_s ,jm1]   )



    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        ip1 = i + 1
        im1 = i - 1
        inb = i - 1
        for js in range(nb,nys+nb):
            jsm1 = js - 1
            jsnb = js - 1

            COLPA_js[inb,jsnb] = 1./8.*( \
                                    COLP[ip1,jsm1] * A[ip1,jsm1] + \
                                    COLP[ip1,js  ] * A[ip1,js  ] + \
                               2. * COLP[i  ,jsm1] * A[i  ,jsm1] + \
                               2. * COLP[i  ,js  ] * A[i  ,js  ] + \
                                    COLP[im1,jsm1] * A[im1,jsm1] + \
                                    COLP[im1,js  ] * A[im1,js  ]   )


    return(COLPA_is, COLPA_js)




