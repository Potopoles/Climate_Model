import numpy as np
from boundaries import exchange_BC
from namelist import wp, njobs
from diagnostics_cython import interp_COLPA_c

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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef proceed_timestep_jacobson_c(GR, \
                                wp_cy[:,:, ::1] UWIND_OLD,
                                wp_cy[:,:, ::1] UWIND,
                                wp_cy[:,:, ::1] VWIND_OLD,
                                wp_cy[:,:, ::1] VWIND,
                                wp_cy[:,   ::1] COLP_OLD,
                                wp_cy[:,   ::1] COLP,
                                wp_cy[:,:, ::1] POTT_OLD,
                                wp_cy[:,:, ::1] POTT,
                                wp_cy[:,:, ::1] QV_OLD,
                                wp_cy[:,:, ::1] QV,
                                wp_cy[:,:, ::1] QC_OLD,
                                wp_cy[:,:, ::1] QC,
                                wp_cy[:,:, ::1] dUFLXdt,
                                wp_cy[:,:, ::1] dVFLXdt,
                                wp_cy[:,:, ::1] dPOTTdt,
                                wp_cy[:,:, ::1] dQVdt,
                                wp_cy[:,:, ::1] dQCdt):


    cdef int c_njobs = njobs

    cdef int nb  = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys
    cdef int nz  = GR.nz
    cdef int i, imb, i_s, ismb, j, jmb, js, jsmb, k

    cdef wp_cy dt = GR.dt

    cdef wp_cy[:, ::1] COLPA_is_OLD = np.zeros( (nxs,ny ), dtype=wp_np )
    cdef wp_cy[:, ::1] COLPA_is_NEW = np.zeros( (nxs,ny ), dtype=wp_np )
    cdef wp_cy[:, ::1] COLPA_js_OLD = np.zeros( (nx ,nys), dtype=wp_np )
    cdef wp_cy[:, ::1] COLPA_js_NEW = np.zeros( (nx ,nys), dtype=wp_np )


    COLPA_is_OLD, COLPA_js_OLD = interp_COLPA_c(GR, c_njobs, COLP_OLD)
    COLPA_is_NEW, COLPA_js_NEW = interp_COLPA_c(GR, c_njobs, COLP)

    for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i_s in range(nb,nxs+nb):
        ismb = i_s - nb
        for j   in range(nb,ny +nb):
            jmb = j - nb
            for k in range(0,nz):
                UWIND[i_s,j  ,k] = UWIND_OLD[i_s,j  ,k] * \
                                    COLPA_is_OLD[ismb,jmb]/COLPA_is_NEW[ismb,jmb] \
                                        + dt*dUFLXdt[i_s,j  ,k]/COLPA_is_NEW[ismb,jmb]

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        imb = i - nb
        for js  in range(nb,nys+nb):
            jsmb = js - nb
            for k in range(0,nz):
                VWIND[i  ,js ,k] = VWIND_OLD[i  ,js ,k] * \
                                    COLPA_js_OLD[imb,jsmb]/COLPA_js_NEW[imb,jsmb] \
                                        + dt*dVFLXdt[i  ,js  ,k]/COLPA_js_NEW[imb,jsmb]

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        imb = i - nb
        for j   in range(nb,ny +nb):
            jmb = j - nb
            for k in range(0,nz):
                POTT[i  ,j  ,k]  =  POTT_OLD[i  ,j  ,k] * \
                                    COLP_OLD    [i  ,j  ]/COLP        [i  ,j  ] \
                                        + dt*dPOTTdt[i  ,j  ,k]/COLP        [i  ,j  ]
                QV[i  ,j  ,k]    =    QV_OLD[i  ,j  ,k] * \
                                    COLP_OLD    [i  ,j  ]/COLP        [i  ,j  ] \
                                        + dt*dQVdt  [i  ,j  ,k]/COLP        [i  ,j  ]
                QC[i  ,j  ,k]    =    QC_OLD[i  ,j  ,k] * \
                                    COLP_OLD    [i  ,j  ]/COLP        [i  ,j  ] \
                                        + dt*dQCdt  [i  ,j  ,k]/COLP        [i  ,j  ]

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


