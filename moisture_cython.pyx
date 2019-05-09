import numpy as np
from namelist import wp, QV_hor_dif_tau

cimport numpy as np
import cython
from cython.parallel import prange 

ctypedef fused wp_cy:
    double
    float

cdef int i_hor_adv      = 1
cdef int i_vert_adv     = 1
cdef int i_num_dif      = 1
cdef int i_microphysics = 1
#cdef int i_turb = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef water_vapor_tendency_c( GR, njobs,\
        wp_cy[:,:, ::1] dQVdt,
        wp_cy[:,:, ::1] QV,
        wp_cy[:,   ::1] COLP,
        wp_cy[:,   ::1] COLP_NEW,
        wp_cy[:,:, ::1] UFLX,
        wp_cy[:,:, ::1] VFLX,
        wp_cy[:,:, ::1] WWIND,
        wp_cy[:,:, ::1] dQVdt_MIC):

    cdef int c_njobs = njobs
   
    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz  = GR.nz
    cdef int nzs = GR.nzs
    cdef wp_cy[   ::1] dsigma    = GR.dsigma
    cdef wp_cy[:, ::1] A         = GR.A

    cdef int i, inb, im1, ip1, j, jnb, jm1, jp1, k, kp1, ks
    cdef wp_cy hor_adv, vert_adv, num_diff

    cdef wp_cy c_QV_hor_dif_tau = QV_hor_dif_tau

    cdef wp_cy[:,:, ::1] QVVB  = np.zeros( (nx ,ny ,nzs), dtype=wp)

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        im1 = i - 1
        ip1 = i + 1
        inb = i - nb
        for j   in range(nb,ny +nb):
            jm1 = j - 1
            jp1 = j + 1
            jnb = j - nb

            for ks in range(1,nzs-1):
                QVVB[inb,jnb,ks ] = (QV[i  ,j  ,ks-1] + QV[i  ,j  ,ks ]) / 2.

            for k in range(0,nz):
                kp1 = k + 1

                dQVdt[i,j,k] = 0.

                # HORIZONTAL ADVECTION
                if i_hor_adv:
                    hor_adv = (+ UFLX[i  ,j  ,k  ] *\
                                  (QV[im1,j  ,k  ] +\
                                   QV[i  ,j  ,k  ])/2. \
                               - UFLX[ip1,j  ,k  ] *\
                                  (QV[i  ,j  ,k  ] +\
                                   QV[ip1,j  ,k  ])/2. \
                               + VFLX[i  ,j  ,k  ] *\
                                  (QV[i  ,jm1,k  ] +\
                                   QV[i  ,j  ,k  ])/2. \
                               - VFLX[i  ,jp1,k  ] *\
                                  (QV[i  ,j  ,k  ] +\
                                   QV[i  ,jp1,k  ])/2. \
                              ) / A[i  ,j  ]

                    dQVdt[i,j,k] = dQVdt[i,j,k] + hor_adv


                # VERTICAL ADVECTION
                if i_vert_adv:
                    if k == 0:
                        vert_adv = COLP_NEW[i  ,j  ] * (\
                                - WWIND[i  ,j  ,kp1] * QVVB[inb,jnb,kp1] \
                                                       ) / dsigma[k]
                    elif k == nz:
                        vert_adv = COLP_NEW[i  ,j  ] * (\
                                + WWIND[i  ,j  ,k  ] * QVVB[inb,jnb,k  ] \
                                                       ) / dsigma[k]
                    else:
                        vert_adv = COLP_NEW[i  ,j  ] * (\
                                + WWIND[i  ,j  ,k  ] * QVVB[inb,jnb,k  ] \
                                - WWIND[i  ,j  ,kp1] * QVVB[inb,jnb,kp1] \
                                                       ) / dsigma[k]

                    dQVdt[i,j,k] = dQVdt[i,j,k] + vert_adv


                # NUMERICAL DIFUSION 
                if i_num_dif and (c_QV_hor_dif_tau > 0):
                    num_diff = c_QV_hor_dif_tau * \
                                 (+ COLP[im1,j  ] * QV[im1,j  ,k  ] \
                                  + COLP[ip1,j  ] * QV[ip1,j  ,k  ] \
                                  + COLP[i  ,jm1] * QV[i  ,jm1,k  ]  \
                                  + COLP[i  ,jp1] * QV[i  ,jp1,k  ] \
                                - 4.*COLP[i  ,j  ] * QV[i  ,j  ,k  ] )

                    dQVdt[i,j,k] = dQVdt[i,j,k] + num_diff

                # MICROPHYSICS 
                if i_microphysics:
                    dQVdt[i,j,k] = dQVdt[i,j,k] + \
                                        dQVdt_MIC[inb,jnb,k  ]*COLP[i  ,j  ]


    return(dQVdt)





@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cloud_water_tendency_c( GR, njobs,\
        wp_cy[:,:, ::1] dQCdt,
        wp_cy[:,:, ::1] QC,
        wp_cy[:,   ::1] COLP,
        wp_cy[:,   ::1] COLP_NEW,
        wp_cy[:,:, ::1] UFLX,
        wp_cy[:,:, ::1] VFLX,
        wp_cy[:,:, ::1] WWIND,
        wp_cy[:,:, ::1] dQCdt_MIC):

    cdef int c_njobs = njobs
   
    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int ny  = GR.ny
    cdef int nz  = GR.nz
    cdef int nzs = GR.nzs
    cdef wp_cy[   ::1] dsigma    = GR.dsigma
    cdef wp_cy[:, ::1] A         = GR.A

    cdef int i, inb, im1, ip1, j, jnb, jm1, jp1, k, kp1, ks
    cdef wp_cy hor_adv, vert_adv, num_diff

    cdef wp_cy c_QC_hor_dif_tau = QV_hor_dif_tau

    cdef wp_cy[:,:, ::1] QCVB  = np.zeros( (nx ,ny ,nzs), dtype=wp)

    for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='guided'):
    #for i   in range(nb,nx +nb):
        im1 = i - 1
        ip1 = i + 1
        inb = i - nb
        for j   in range(nb,ny +nb):
            jm1 = j - 1
            jp1 = j + 1
            jnb = j - nb

            for ks in range(1,nzs-1):
                QCVB[inb,jnb,ks ] = (QC[i  ,j  ,ks-1] + QC[i  ,j  ,ks ]) / 2.

            for k in range(0,nz):
                kp1 = k + 1

                dQCdt[i,j,k] = 0.

                # HORIZONTAL ADVECTION
                if i_hor_adv:
                    hor_adv = (+ UFLX[i  ,j  ,k  ] *\
                                 (QC[im1,j  ,k  ] +\
                                  QC[i  ,j  ,k  ])/2. \
                              - UFLX[ip1,j  ,k  ] *\
                                 (QC[i  ,j  ,k  ] +\
                                  QC[ip1,j  ,k  ])/2. \
                              + VFLX[i  ,j  ,k  ] *\
                                 (QC[i  ,jm1,k  ] +\
                                  QC[i  ,j  ,k  ])/2. \
                              - VFLX[i  ,jp1,k  ] *\
                                 (QC[i  ,j  ,k  ] +\
                                  QC[i  ,jp1,k  ])/2. \
                             ) / A[i  ,j  ]

                    dQCdt[i,j,k] = dQCdt[i,j,k] + hor_adv


                # VERTICAL ADVECTION
                if i_vert_adv:
                    if k == 0:
                        vert_adv = COLP_NEW[i  ,j  ] * (\
                                - WWIND[i  ,j  ,kp1] * QCVB[inb,jnb,kp1] \
                                                       ) / dsigma[k]
                    elif k == nz:
                        vert_adv = COLP_NEW[i  ,j  ] * (\
                                + WWIND[i  ,j  ,k  ] * QCVB[inb,jnb,k  ] \
                                                       ) / dsigma[k]
                    else:
                        vert_adv = COLP_NEW[i  ,j  ] * (\
                                + WWIND[i  ,j  ,k  ] * QCVB[inb,jnb,k  ] \
                                - WWIND[i  ,j  ,kp1] * QCVB[inb,jnb,kp1] \
                                                       ) / dsigma[k]

                    dQCdt[i,j,k] = dQCdt[i,j,k] + vert_adv


                # NUMERICAL DIFUSION 
                if i_num_dif and (c_QC_hor_dif_tau > 0):
                    num_diff = c_QC_hor_dif_tau * \
                                 (+ COLP[im1,j  ] * QC[im1,j  ,k  ] \
                                  + COLP[ip1,j  ] * QC[ip1,j  ,k  ] \
                                  + COLP[i  ,jm1] * QC[i  ,jm1,k  ]  \
                                  + COLP[i  ,jp1] * QC[i  ,jp1,k  ] \
                                - 4.*COLP[i  ,j  ] * QC[i  ,j  ,k  ] )

                    dQCdt[i,j,k] = dQCdt[i,j,k] + num_diff

                # MICROPHYSICS 
                if i_microphysics:
                    dQCdt[i,j,k] = dQCdt[i,j,k] + \
                                        dQCdt_MIC[inb,jnb,k  ] * COLP[i  ,j  ]


    return(dQCdt)
