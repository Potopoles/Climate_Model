import numpy as np
from namelist import  i_colp_tendency, COLP_hor_dif_tau, njobs
from boundaries import exchange_BC

cimport numpy as np
import cython
from cython.parallel import prange 


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef colp_tendency_jacobson_c(GR, \
                                double[:,   ::1] COLP,
                                double[:,:, ::1] UWIND,
                                double[:,:, ::1] VWIND,
                                double[:,   ::1] dCOLPdt,
                                double[:,:, ::1] UFLX,
                                double[:,:, ::1] VFLX,
                                double[:,:, ::1] FLXDIV):

    cdef int c_njobs = njobs

    cdef int nb  = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys
    cdef int nz  = GR.nz
    cdef int i, im1, ip1, i_s, ism1, j, jm1, jp1, js, jsm1, k, km1

    cdef double dy = GR.dy
    cdef double num_dif
    cdef double c_COLP_hor_dif_tau = COLP_hor_dif_tau
    cdef double flux_div_sum

    cdef double[     ::1] dsigma  = GR.dsigma
    cdef double[:,   ::1] dxjs    = GR.dxjs
    cdef double[:,   ::1] A       = GR.A


    for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i_s in range(nb,nxs+nb):
        ism1 = i_s - 1
        for j   in range(nb,ny +nb):
            for k in range(0,nz):
                UFLX[i_s,j  ,k] = \
                        (COLP[ism1,j   ] + COLP[i_s,j   ])/2. *\
                         UWIND[i_s,j  ,k] * dy

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for js  in range(nb,nys+nb):
            jsm1 = js - 1
            for k in range(0,nz):
                VFLX[i  ,js ,k] = \
                        (COLP[i  ,jsm1] + COLP[i  ,js  ])/2. *\
                        VWIND[i  ,js  ,k] * dxjs[i  ,js   ]

    # TODO 1 NECESSARY
    UFLX = exchange_BC(GR, np.asarray(UFLX))
    VFLX = exchange_BC(GR, np.asarray(VFLX))

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        ip1 = i + 1
        for j   in range(nb,ny +nb):
            jp1 = j + 1

            flux_div_sum = 0.
            for k in range(0,nz):
                FLXDIV[i  ,j  ,k] = \
                        ( + UFLX[ip1,j  ,k  ] - UFLX[i  ,j  ,k  ] \
                          + VFLX[i  ,jp1,k  ] - VFLX[i  ,j  ,k  ] ) \
                          * dsigma[k] / A[i  ,j  ]

                flux_div_sum = flux_div_sum + FLXDIV[i   ,j   ,k]

            dCOLPdt[i  ,j  ] = - flux_div_sum



    ## NUMERICAL DIFUSION 
    if COLP_hor_dif_tau > 0:
        raise NotImplementedError('This part of the code is not yet tested!')
    #    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #    #for i   in range(nb,nx +nb):
    #        im1 = i - 1
    #        ip1 = i + 1
    #        for j   in range(nb,ny +nb):
    #            jm1 = j - 1
    #            jp1 = j + 1
    #            num_dif = c_COLP_hor_dif_tau * \
    #                         (+ COLP[im1,j  ] \
    #                          + COLP[ip1,j  ] \
    #                          + COLP[i  ,jm1] \
    #                          + COLP[i  ,jp1] \
    #                       - 4.*COLP[i  ,j  ] )

    #            dCOLPdt[i  ,j  ] = dCOLPdt[i  ,j  ] + num_dif

    return(dCOLPdt, UFLX, VFLX, FLXDIV)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef vertical_wind_jacobson_c(GR, \
                                double[:,   ::1] COLP_NEW,
                                double[:,   ::1] dCOLPdt,
                                double[:,:, ::1] FLXDIV,
                                double[:,:, ::1] WWIND):

    cdef int c_njobs = njobs

    cdef int nb  = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nzs = GR.nzs
    cdef int i, j, ks, kss

    cdef double flux_div_sum

    cdef double[     ::1] sigma_vb  = GR.sigma_vb


    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for j   in range(nb,ny +nb):

            flux_div_sum = 0.

            for ks in range(1,nzs-1):

                flux_div_sum = flux_div_sum + FLXDIV[i   ,j   ,ks-1]

                #WWIND[i  ,j  ,ks ] = - flux_div_sum / COLP_NEW[i  ,j  ] \
                #                     - sigma_vb[ks] * dCOLPdt[i-nb,j-nb] / COLP_NEW[i  ,j  ]
                WWIND[i  ,j  ,ks ] = - flux_div_sum / COLP_NEW[i  ,j  ] \
                                     - sigma_vb[ks] * dCOLPdt[i,j] / COLP_NEW[i  ,j  ]

    return(WWIND)


