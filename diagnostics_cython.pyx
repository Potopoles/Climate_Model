import numpy as np
import time
from constants import con_kappa, con_g, con_Rd
from namelist import pTop

cimport numpy as np
import cython
#from cython.parallel import prange 


#def diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB, TAIR, TAIRVB, RHO,\
#                                PVTF, PVTFVB, UWIND, VWIND, WIND):
#
#    t_start = time.time()
#
#    TAIR[GR.iijj] = POTT[GR.iijj] * PVTF[GR.iijj]
#    TAIRVB[GR.iijj] = POTTVB[GR.iijj] * PVTFVB[GR.iijj]
#    PAIR[GR.iijj] = 100000*np.power(PVTF[GR.iijj], 1/con_kappa)
#    RHO[GR.iijj] = PAIR[GR.iijj] / (con_Rd * TAIR[GR.iijj])
#
#    for k in range(0,GR.nz):
#        WIND[:,:,k][GR.iijj] = np.sqrt( ((UWIND[:,:,k][GR.iijj] + \
#                                        UWIND[:,:,k][GR.iijj_ip1])/2)**2 + \
#                        ((VWIND[:,:,k][GR.iijj] + VWIND[:,:,k][GR.iijj_jp1])/2)**2 )
#
#
#    t_end = time.time()
#    GR.diag_comp_time += t_end - t_start
#
#    return(PAIR, TAIR, TAIRVB, RHO, WIND)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef diagnose_POTTVB_jacobson_c(GR, njobs,
        double[:,:, ::1] POTTVB,
        double[:,:, ::1] POTT,
        double[:,:, ::1] PVTF,
        double[:,:, ::1] PVTFVB):

    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nzs = GR.nzs

    cdef int i, j, ks, ksm1

    t_start = time.time()

    #for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs):
    for i   in range(nb,nx +nb):
        #for j   in prange(nb,ny +nb, nogil=False, num_threads=c_njobs):
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

    t_end = time.time()
    GR.diag_comp_time += t_end - t_start

    return(POTTVB)






@cython.boundscheck(False)
@cython.wraparound(False)
cpdef interp_COLPA_c(GR, njobs, 
    double[:, ::1] COLP):


    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys

    cdef double[:, ::1] A         = GR.A

    cdef int i, inb, im1, ip1, i_s, isnb, ism1, j, jnb, jm1, jp1, js, jsnb, jsm1

    cdef double[:, ::1] COLPA_is = np.zeros( (nxs,ny ) )
    cdef double[:, ::1] COLPA_js = np.zeros( (nx ,nys) )

    t_start = time.time()

    #for i_s in prange(nb,nxs +nb, nogil=True, num_threads=c_njobs):
    for i_s in range(nb,nxs +nb):
        ism1 = i_s - 1
        isnb = i_s - nb
        #for j   in prange(nb,ny +nb, nogil=False, num_threads=c_njobs):
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



    #for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs):
    for i   in range(nb,nx +nb):
        ip1 = i + 1
        im1 = i - 1
        inb = i - 1
        #for js in prange(nb,nys+nb, nogil=False, num_threads=c_njobs):
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

    t_end = time.time()
    GR.diag_comp_time += t_end - t_start

    return(COLPA_is, COLPA_js)




