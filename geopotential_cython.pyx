import numpy as np
from constants import con_g, con_kappa, con_cp
from namelist import pTop
from boundaries import exchange_BC

cimport numpy as np
import cython
from cython.parallel import prange 
from libc.math cimport pow

from geopotential import diag_pvt_factor


@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
cpdef diag_pvt_factor_c(GR, njobs,
        double[:,   ::1] COLP,
        double[:,:, ::1] PVTF,
        double[:,:, ::1] PVTFVB):

    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz  = GR.nz
    cdef int nzs = GR.nzs
    cdef double[   ::1] sigma_vb    = GR.sigma_vb

    cdef int i, j, k, ks, kp1

    cdef double c_con_kappa = con_kappa
    cdef double c_pTop = pTop

    cdef double[:,:, ::1] PAIRVB = np.full( (nx+2*nb ,ny+2*nb ,nzs), np.nan )

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for j   in range(nb,ny +nb):

            for ks in range(0,nzs):
                PAIRVB[i,j,ks  ] = c_pTop + sigma_vb[ks] * COLP[i,j]
                # TODO: IN THE FOLLOWING COMMAND THE RESULT GETS DIFFERENT COMPARED 
                # TO THE PYTHON VERSION!!!!
                #PVTFVB[i,j,ks  ] = (PAIRVB[i,j,ks ]/100000.) ** c_con_kappa

    # SOLUTION (inefficient): do it outside of loop
    PVTFVB = np.power(np.asarray(PAIRVB)/100000, con_kappa)

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for j   in range(nb,ny +nb):
            for k in range(0,nz):
                kp1 = k + 1
                PVTF[i,j,k  ] = 1./(1. + c_con_kappa) * \
                            ( PVTFVB[i,j,kp1] * PAIRVB[i,j,kp1] - \
                              PVTFVB[i,j,k  ] * PAIRVB[i,j,k  ] ) / \
                            ( PAIRVB[i,j,kp1] - PAIRVB[i,j,k  ] )


    return(PVTF, PVTFVB)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef diag_geopotential_jacobson_c(GR, njobs,
        double[:,:, ::1] PHI,
        double[:,:, ::1] PHIVB,
        double[:,   ::1] HSURF,
        double[:,:, ::1] POTT,
        double[:,   ::1] COLP,
        double[:,:, ::1] PVTF,
        double[:,:, ::1] PVTFVB):

    cdef int c_njobs = njobs

    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz  = GR.nz
    cdef int nzs = GR.nzs

    cdef int i, j, k, kp1
    cdef double dphi

    cdef double c_con_g = con_g
    cdef double c_con_cp = con_cp

    PVTF, PVTFVB = diag_pvt_factor(GR, np.asarray(COLP), np.asarray(PVTF), np.asarray(PVTFVB))

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        for j   in range(nb,ny +nb):

            PHIVB[i,j,nzs-1] = HSURF[i,j]*c_con_g
            PHI  [i,j,nz -1] = PHIVB[i,j,nzs-1] - c_con_cp*  \
                                        ( POTT[i,j,nz -1] * \
                                        (   PVTF[i,j,nz -1]    \
                                          - PVTFVB[i,j,nzs-1]  ) )
            for k in range(nz-2,-1,-1):
                kp1 = k + 1

                dphi = c_con_cp * POTT[i,j,kp1] * \
                                (PVTFVB[i,j,kp1] - PVTF[i,j,kp1])
                PHIVB[i,j,kp1] = PHI[i,j,kp1] - dphi

                # phi_k
                dphi = c_con_cp * POTT[i,j,k  ] * \
                                    (PVTF[i,j,k  ] - PVTFVB[i,j,kp1])
                PHI[i,j,k  ] = PHIVB[i,j,kp1] - dphi

            dphi = c_con_cp * POTT[i,j,0  ] * \
                            (PVTFVB[i,j,0  ] - PVTF[i,j,0  ])
            PHIVB[i,j,0  ] = PHI[i,j,0  ] - dphi


    # TODO 5 NECESSARY
    PVTF = exchange_BC(GR, np.asarray(PVTF))
    PVTFVB = exchange_BC(GR, np.asarray(PVTFVB))
    PHI = exchange_BC(GR, np.asarray(PHI))

    return(PHI, PHIVB, PVTF, PVTFVB)





