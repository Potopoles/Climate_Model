import numpy as np
from boundaries import exchange_BC
from namelist import njobs
from diagnostics_cython import interp_COLPA_c

cimport numpy as np
import cython
from cython.parallel import prange 







@cython.boundscheck(False)
@cython.wraparound(False)
cpdef proceed_timestep_jacobson_c(GR, \
                                double[:,:, ::1] UWIND,
                                double[:,:, ::1] VWIND,
                                double[:,   ::1] COLP_OLD,
                                double[:,   ::1] COLP,
                                double[:,:, ::1] POTT,
                                double[:,:, ::1] QV,
                                double[:,:, ::1] QC,
                                double[:,:, ::1] dUFLXdt,
                                double[:,:, ::1] dVFLXdt,
                                double[:,:, ::1] dPOTTdt,
                                double[:,:, ::1] dQVdt,
                                double[:,:, ::1] dQCdt):


    cdef int c_njobs = njobs

    cdef int nb  = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys
    cdef int nz  = GR.nz
    cdef int i, imb, i_s, ismb, j, jmb, js, jsmb, k

    cdef double dt = GR.dt

    cdef double[:, ::1] COLPA_is_OLD = np.zeros( (nxs,ny ) )
    cdef double[:, ::1] COLPA_is_NEW = np.zeros( (nxs,ny ) )
    cdef double[:, ::1] COLPA_js_OLD = np.zeros( (nx ,nys) )
    cdef double[:, ::1] COLPA_js_NEW = np.zeros( (nx ,nys) )


    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA_c(GR, c_njobs, COLP_OLD)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA_c(GR, c_njobs, COLP)

    for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i_s in range(nb,nxs+nb):
        ismb = i_s - nb
        for j   in range(nb,ny +nb):
            jmb = j - nb
            for k in range(0,nz):
                UWIND[i_s,j  ,k] = UWIND[i_s,j  ,k] * \
                                    COLPA_is_OLD[ismb,jmb]/COLPA_is_NEW[ismb,jmb] \
                                        + dt*dUFLXdt[ismb,jmb,k]/COLPA_is_NEW[ismb,jmb]

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        imb = i - nb
        for js  in range(nb,nys+nb):
            jsmb = js - nb
            for k in range(0,nz):
                VWIND[i  ,js ,k] = VWIND[i  ,js ,k] * \
                                    COLPA_js_OLD[imb,jsmb]/COLPA_js_NEW[imb,jsmb] \
                                        + dt*dVFLXdt[imb,jsmb,k]/COLPA_js_NEW[imb,jsmb]

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        imb = i - nb
        for j   in range(nb,ny +nb):
            jmb = j - nb
            for k in range(0,nz):
                POTT[i  ,j  ,k]  =  POTT[i  ,j  ,k] * \
                                    COLP_OLD    [i  ,j  ]/COLP        [i  ,j  ] \
                                        + dt*dPOTTdt[imb,jmb,k]/COLP        [i  ,j  ]
                QV[i  ,j  ,k]    =    QV[i  ,j  ,k] * \
                                    COLP_OLD    [i  ,j  ]/COLP        [i  ,j  ] \
                                        + dt*dQVdt  [imb,jmb,k]/COLP        [i  ,j  ]
                QC[i  ,j  ,k]    =    QC[i  ,j  ,k] * \
                                    COLP_OLD    [i  ,j  ]/COLP        [i  ,j  ] \
                                        + dt*dQCdt  [imb,jmb,k]/COLP        [i  ,j  ]

                if QV[i  ,j  ,k] < 0.:
                    QV[i  ,j  ,k] = 0.
                if QC[i  ,j  ,k] < 0.:
                    QC[i  ,j  ,k] = 0.

    # TODO 4 NECESSARY
    UWIND = exchange_BC(GR, np.asarray(UWIND))
    VWIND = exchange_BC(GR, np.asarray(VWIND))
    POTT = exchange_BC(GR, np.asarray(POTT))
    QV = exchange_BC(GR, np.asarray(QV))
    QC = exchange_BC(GR, np.asarray(QC))

    return(UWIND, VWIND, COLP, POTT, QV, QC)


