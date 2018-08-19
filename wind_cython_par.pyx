import copy
import numpy as np
cimport numpy as np
from cython.parallel import prange 
#import time
from boundaries import exchange_BC
from constants import con_cp, con_rE, con_Rd
from namelist import WIND_hor_dif_tau, i_wind_tendency
from libc.stdio cimport printf
from libc.math cimport cos, sin
import cython

ctypedef np.double_t cDOUBLE

cdef int i_hor_adv = 1
cdef int i_vert_adv = 1
cdef int i_coriolis = 1
cdef int i_pre_grad = 1
cdef int i_num_dif = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef wind_tendency_jacobson_par( job_ind, output, GR, njobs,\
        double[:,:, ::1] UWIND,
        double[:,:, ::1] VWIND,
        double[:,:, ::1] WWIND,
        double[:,:, ::1] UFLX,
        double[:,:, ::1] VFLX,
        double[:,   ::1] COLP,
        double[:,   ::1] COLP_NEW,
        double[:,:, ::1] PHI,
        double[:,:, ::1] POTT,
        double[:,:, ::1] PVTF,
        double[:,:, ::1] PVTFVB):

    cdef int c_njobs = njobs
   
    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys
    cdef int nzs = GR.nzs
    cdef int nz  = GR.nz
    cdef double         dy        = GR.dy
    cdef double         dlon_rad  = GR.dlon_rad
    cdef double[   ::1] sigma_vb  = GR.sigma_vb
    cdef double[   ::1] dsigma    = GR.dsigma
    cdef double[:, ::1] dxjs      = GR.dxjs
    cdef double[:, ::1] corf_is   = GR.corf_is
    cdef double[:, ::1] corf      = GR.corf
    cdef double[:, ::1] latis_rad = GR.latis_rad
    cdef double[:, ::1] lat_rad   = GR.lat_rad
    cdef double[:, ::1] A         = GR.A

    cdef int k, kp1, ks, i, im1, ip1, i_s, ism1, isp1, j, jm1, jp1, js, jsm1, jsp1

    cdef double diff_UWIND, diff_VWIND, coriolis_UWIND, coriolis_VWIND, \
                preGrad_UWIND, preGrad_VWIND, horAdv_UWIND, horAdv_VWIND, \
                vertAdv_UWIND, vertAdv_VWIND

    cdef double c_WIND_hor_dif_tau = WIND_hor_dif_tau
    cdef double c_con_rE = con_rE
    cdef double c_con_cp = con_cp

    cdef double[:,:, ::1] dUFLXdt = np.zeros( (nxs,ny ,nz) )
    cdef double[:,:, ::1] dVFLXdt = np.zeros( (nx ,nys,nz) )

    cdef double[:,:, ::1] BFLX = np.zeros( (nx +2*nb,ny +2*nb,nz) )
    cdef double[:,:, ::1] CFLX = np.zeros( (nxs+2*nb,nys+2*nb,nz) )
    cdef double[:,:, ::1] DFLX = np.zeros( (nx +2*nb,nys+2*nb,nz) )
    cdef double[:,:, ::1] EFLX = np.zeros( (nx +2*nb,nys+2*nb,nz) )

    cdef double[:,:, ::1] RFLX = np.zeros( (nx +2*nb,ny +2*nb,nz) )
    cdef double[:,:, ::1] QFLX = np.zeros( (nxs+2*nb,nys+2*nb,nz) )
    cdef double[:,:, ::1] SFLX = np.zeros( (nxs+2*nb,ny +2*nb,nz) )
    cdef double[:,:, ::1] TFLX = np.zeros( (nxs+2*nb,ny +2*nb,nz) )

    cdef double[:,:, ::1] WWIND_UWIND_ks = np.zeros( (nxs+2*nb,ny +2*nb,nzs) )
    cdef double[:,:, ::1] WWIND_VWIND_ks = np.zeros( (nx +2*nb,nys+2*nb,nzs) )

    cdef double COLPAWWIND_is_ks, UWIND_ks
    cdef double COLPAWWIND_js_ks, VWIND_ks

    cdef int chunk = 1


    if i_wind_tendency:

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

        # HORIZONTAL ADVECTION
        if i_hor_adv:
            for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
                im1 = i - 1
                ip1 = i + 1
                for j   in range(nb,ny +nb):
                #for j   in prange(nb,ny +nb, nogil=False, num_threads=c_njobs):
                    jm1 = j - 1
                    jp1 = j + 1
                    for k in range(0,nz):

                        BFLX[i   ,j   ,k] = 1./12. * (  UFLX[i   ,jm1 ,k]   + \
                                                        UFLX[ip1 ,jm1 ,k]   + \
                                                   2.*( UFLX[i   ,j   ,k]   + \
                                                        UFLX[ip1 ,j   ,k] ) + \
                                                        UFLX[i   ,jp1 ,k]   + \
                                                        UFLX[ip1 ,jp1 ,k]   )

                        RFLX[i   ,j   ,k] = 1./12. * (  VFLX[im1 ,j   ,k]   + \
                                                        VFLX[im1 ,jp1 ,k]   + \
                                                   2.*( VFLX[i   ,j   ,k]   + \
                                                        VFLX[i   ,jp1 ,k] ) + \
                                                        VFLX[ip1 ,j   ,k]   + \
                                                        VFLX[ip1 ,jp1 ,k]   )

            #######################################################################

            for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
                ism1 = i_s - 1
                isp1 = i_s + 1
                for j   in range(nb,ny +nb):
                #for j   in prange(nb,ny +nb, nogil=False, num_threads=c_njobs):
                    jm1 = j - 1
                    jp1 = j + 1
                    for k in range(0,nz):

                        SFLX[i_s ,j   ,k]  = 1./24. * (  VFLX[ism1,j   ,k] + \
                                                         VFLX[ism1,jp1 ,k] + \
                                                         VFLX[i_s ,j   ,k] + \
                                                         VFLX[i_s ,jp1 ,k] + \
                                                         UFLX[ism1,j   ,k] + \
                                                      2.*UFLX[i_s ,j   ,k] + \
                                                         UFLX[isp1,j   ,k]   )

                        TFLX[i_s ,j   ,k]  = 1./24. * (  VFLX[ism1,j   ,k] + \
                                                         VFLX[ism1,jp1 ,k] + \
                                                         VFLX[i_s ,j   ,k] + \
                                                         VFLX[i_s ,jp1 ,k] + \
                                                       - UFLX[ism1,j   ,k] - \
                                                      2.*UFLX[i_s ,j   ,k] + \
                                                       - UFLX[isp1,j   ,k]   )

            #######################################################################

            for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
                im1 = i - 1
                ip1 = i + 1
                for js  in range(nb,nys+nb):
                #for js  in prange(nb,nys+nb, nogil=False, num_threads=c_njobs):
                    jsm1 = js - 1
                    jsp1 = js + 1
                    for k in range(0,nz):

                        DFLX[i   ,js  ,k]  = 1./24. * (  VFLX[i   ,jsm1,k]    + \
                                                      2.*VFLX[i   ,js  ,k]    +\
                                                         VFLX[i   ,jsp1,k]    + \
                                                         UFLX[i   ,jsm1,k]    +\
                                                         UFLX[i   ,js  ,k]    + \
                                                         UFLX[ip1 ,jsm1,k]    +\
                                                         UFLX[ip1 ,js  ,k]    )

                        EFLX[i   ,js  ,k]  = 1./24. * (  VFLX[i   ,jsm1,k]     + \
                                                      2.*VFLX[i   ,js  ,k]     +\
                                                         VFLX[i   ,jsp1,k]     - \
                                                         UFLX[i   ,jsm1,k]     +\
                                                       - UFLX[i   ,js  ,k]     - \
                                                         UFLX[ip1 ,jsm1,k]     +\
                                                       - UFLX[ip1 ,js  ,k]     )

            #######################################################################

            for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
                ism1 = i_s - 1
                isp1 = i_s + 1
                for js  in range(nb,nys+nb):
                #for js  in prange(nb,nys+nb, nogil=False, num_threads=c_njobs):
                    jsm1 = js - 1
                    jsp1 = js + 1
                    for k in range(0,nz):

                        CFLX[i_s ,js  ,k] = 1./12. * (  VFLX[ism1,jsm1,k]   + \
                                                        VFLX[i_s ,jsm1,k]   +\
                                                   2.*( VFLX[ism1,js  ,k]   + \
                                                        VFLX[i_s ,js  ,k] ) +\
                                                        VFLX[ism1,jsp1,k]   + \
                                                        VFLX[i_s ,jsp1,k]   )

                        QFLX[i_s ,js  ,k] = 1./12. * (  UFLX[ism1,jsm1,k]   + \
                                                        UFLX[ism1,js  ,k]   + \
                                                   2.*( UFLX[i_s ,jsm1,k]   + \
                                                        UFLX[i_s ,js  ,k] ) + \
                                                        UFLX[isp1,jsm1,k]   + \
                                                        UFLX[isp1,js  ,k]   )

            #######################################################################

            BFLX = exchange_BC(GR, np.asarray(BFLX))
            CFLX = exchange_BC(GR, np.asarray(CFLX))
            DFLX = exchange_BC(GR, np.asarray(DFLX))
            EFLX = exchange_BC(GR, np.asarray(EFLX))

            RFLX = exchange_BC(GR, np.asarray(RFLX))
            QFLX = exchange_BC(GR, np.asarray(QFLX))
            SFLX = exchange_BC(GR, np.asarray(SFLX))
            TFLX = exchange_BC(GR, np.asarray(TFLX))

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

        # VERTICAL ADVECTION
        if i_vert_adv:


            for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
                ism1 = i_s - 1
                isp1 = i_s + 1
                for j   in range(nb,ny +nb):
                #for j   in prange(nb,ny +nb, nogil=False, num_threads=c_njobs):
                    jm1 = j - 1
                    jp1 = j + 1

                    if j == nb:
                        for ks in range(1,nzs-1):
                            # INTERPOLATE DIFFERENTLY AT MERID. BOUNDARIES (JACOBSON)
                            COLPAWWIND_is_ks = 1./4.*( \
                                COLP_NEW[ism1,j  ] * A[ism1,j  ] * \
                                               WWIND[ism1,j  ,ks] + \
                                COLP_NEW[i_s ,j  ] * A[i_s ,j  ] * \
                                               WWIND[i_s ,j  ,ks] + \
                                COLP_NEW[ism1,jp1] * A[ism1,jp1] * \
                                               WWIND[ism1,jp1,ks] + \
                                COLP_NEW[i_s ,jp1] * A[i_s ,jp1] * \
                                               WWIND[i_s ,jp1,ks]   )

                            UWIND_ks = ( dsigma[ks  ] * UWIND[i_s ,j  ,ks-1] +   \
                                         dsigma[ks-1] * UWIND[i_s ,j  ,ks  ] ) / \
                                       ( dsigma[ks  ] + dsigma[ks-1] )

                            WWIND_UWIND_ks[i_s ,j  ,ks ] = COLPAWWIND_is_ks * UWIND_ks

                    elif j == ny+nb-1:
                        for ks in range(1,nzs-1):
                            # INTERPOLATE DIFFERENTLY AT MERID. BOUNDARIES (JACOBSON)
                            COLPAWWIND_is_ks = 1./4.*( \
                                COLP_NEW[ism1,j  ] * A[ism1,j  ] * \
                                                  WWIND[ism1,j  ,ks] + \
                                COLP_NEW[i_s ,j  ] * A[i_s ,j  ] * \
                                                  WWIND[i_s ,j  ,ks] + \
                                COLP_NEW[ism1,jm1] * A[ism1,jm1] * \
                                                  WWIND[ism1,jm1,ks] + \
                                COLP_NEW[i_s ,jm1] * A[i_s ,jm1] * \
                                                  WWIND[i_s ,jm1,ks]  )

                            UWIND_ks = ( dsigma[ks  ] * UWIND[i_s ,j  ,ks-1] +   \
                                         dsigma[ks-1] * UWIND[i_s ,j  ,ks  ] ) / \
                                       ( dsigma[ks  ] + dsigma[ks-1] )

                            WWIND_UWIND_ks[i_s ,j  ,ks ] = COLPAWWIND_is_ks * UWIND_ks

                    else:
                        for ks in range(1,nzs-1):

                            COLPAWWIND_is_ks = 1./8.*( \
                                COLP_NEW[ism1,jp1] * A[ism1,jp1] * \
                                                            WWIND[ism1,jp1,ks] + \
                                COLP_NEW[i_s ,jp1] * A[i_s ,jp1] * \
                                                            WWIND[i_s ,jp1,ks] + \
                           2. * COLP_NEW[ism1,j  ] * A[ism1,j  ] * \
                                                            WWIND[ism1,j  ,ks] + \
                           2. * COLP_NEW[i_s ,j  ] * A[i_s ,j  ] * \
                                                            WWIND[i_s ,j  ,ks] + \
                                COLP_NEW[ism1,jm1] * A[ism1,jm1] * \
                                                            WWIND[ism1,jm1,ks] + \
                                COLP_NEW[i_s ,jm1] * A[i_s ,jm1] * \
                                                            WWIND[i_s ,jm1,ks]   )

                            UWIND_ks = ( dsigma[ks  ] * UWIND[i_s ,j  ,ks-1] +   \
                                         dsigma[ks-1] * UWIND[i_s ,j  ,ks  ] ) / \
                                       ( dsigma[ks  ] + dsigma[ks-1] )

                            WWIND_UWIND_ks[i_s ,j  ,ks ] = COLPAWWIND_is_ks * UWIND_ks

            #######################################################################

            for i   in prange(nb,nx +nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
                im1 = i - 1
                ip1 = i + 1
                for js  in range(nb,nys+nb):
                #for js  in prange(nb,nys+nb, nogil=False, num_threads=c_njobs):
                    jsm1 = js - 1
                    jsp1 = js + 1
                    for ks in range(1,nzs-1):

                        COLPAWWIND_js_ks = 1./8.*( \
                             COLP_NEW[ip1,jsm1] * A[ip1,jsm1] * \
                                                 WWIND[ip1,jsm1,ks] + \
                             COLP_NEW[ip1,js  ] * A[ip1,js  ] * \
                                                 WWIND[ip1,js  ,ks] + \
                        2. * COLP_NEW[i  ,jsm1] * A[i  ,jsm1] * \
                                                 WWIND[i  ,jsm1,ks] + \
                        2. * COLP_NEW[i  ,js  ] * A[i  ,js  ] * \
                                                 WWIND[i  ,js  ,ks] + \
                             COLP_NEW[im1,jsm1] * A[im1,jsm1] * \
                                                 WWIND[im1,jsm1,ks] + \
                             COLP_NEW[im1,js  ] * A[im1,js  ] * \
                                                 WWIND[im1,js  ,ks]   )

                        VWIND_ks = ( dsigma[ks  ] * VWIND[i  ,js  ,ks-1] +   \
                                     dsigma[ks-1] * VWIND[i  ,js  ,ks  ] ) / \
                                   ( dsigma[ks  ] + dsigma[ks-1] )

                        WWIND_VWIND_ks[i  ,js ,ks] = COLPAWWIND_js_ks * VWIND_ks


        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #                         MAIN LOOP  UWIND                            #
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

        #for i_s in range(nb,nxs+nb):
        #for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            for i_s in prange(nb,nxs+nb, schedule='static', num_threads=4):

                ism1 = i_s - 1
                isp1 = i_s + 1

                for j in range(nb,ny+nb):
                #for j in prange(nb,ny+nb, nogil=False, num_threads=c_njobs):

                    jm1 = j - 1
                    jp1 = j + 1

                    for k in range(0,nz):
                    #for k in prange(0,nz, nogil=False, num_threads=c_njobs):

                        kp1 = k+1

                        # HORIZONTAL ADVECTION
                        if i_hor_adv:
                            horAdv_UWIND =  + BFLX [ism1,j  ,k] * \
                                            ( UWIND[ism1,j  ,k] + UWIND[i_s ,j  ,k] )/2. \
                                            - BFLX [i_s ,j  ,k] * \
                                            ( UWIND[i_s ,j  ,k] + UWIND[isp1,j  ,k] )/2. \
                                            \
                                            + CFLX [i_s ,j  ,k] * \
                                            ( UWIND[i_s ,jm1,k] + UWIND[i_s ,j  ,k] )/2. \
                                            - CFLX [i_s ,jp1,k] * \
                                            ( UWIND[i_s ,j  ,k] + UWIND[i_s ,jp1,k] )/2. \
                                            \
                                            + DFLX [ism1,j  ,k] * \
                                            ( UWIND[ism1,jm1,k] + UWIND[i_s ,j  ,k] )/2. \
                                            - DFLX [i_s ,jp1,k] * \
                                            ( UWIND[i_s ,j  ,k] + UWIND[isp1,jp1,k] )/2. \
                                            \
                                            + EFLX [i_s ,j  ,k] * \
                                            ( UWIND[isp1,jm1,k] + UWIND[i_s ,j  ,k] )/2. \
                                            - EFLX [ism1,jp1,k] * \
                                            ( UWIND[i_s ,j  ,k] + UWIND[ism1,jp1,k] )/2. 

                            dUFLXdt[i_s-nb,j-nb,k] = dUFLXdt[i_s-nb,j-nb,k] + horAdv_UWIND

                        #######################################################################

                        ## VERTICAL ADVECTION
                        if i_vert_adv == 1:
                            vertAdv_UWIND = (WWIND_UWIND_ks[i_s ,j  ,k  ] - \
                                             WWIND_UWIND_ks[i_s,j  ,k+1]  ) / dsigma[k]
                            dUFLXdt[i_s-nb,j-nb,k] = dUFLXdt[i_s-nb,j-nb,k] + vertAdv_UWIND

                        #######################################################################

                        ## HORIZONTAL DIFFUSION
                        if i_num_dif == 1:
                            diff_UWIND = c_WIND_hor_dif_tau * \
                                 (  UFLX[ism1,j  ,k] + UFLX[isp1,j  ,k] \
                                  + UFLX[i_s ,jm1,k] + UFLX[i_s ,jp1,k] - 4.*UFLX[i_s ,j  ,k])

                            dUFLXdt[i_s-nb,j-nb,k] = dUFLXdt[i_s-nb,j-nb,k] + diff_UWIND

                        #######################################################################

                        #### CORIOLIS
                        if i_coriolis == 1:
                            coriolis_UWIND = c_con_rE*dlon_rad*dlon_rad/2.*(\
                                    + COLP[ism1,j  ] * \
                                    ( VWIND[ism1,j  ,k] + VWIND[ism1,jp1,k] )/2. * \
                                    ( corf_is[i_s ,j  ] * c_con_rE *\
                                      cos(latis_rad[i_s ,j  ]) + \
                                      ( UWIND[ism1,j  ,k] + UWIND[i_s ,j  ,k] )/2. * \
                                      sin(latis_rad[i_s ,j  ]) ) \

                                    + COLP[i_s ,j  ] * \
                                    ( VWIND[i_s ,j  ,k] + VWIND[i_s ,jp1,k] )/2. * \
                                    ( corf_is[i_s ,j  ] * c_con_rE *\
                                      cos(latis_rad[i_s ,j  ]) + \
                                      ( UWIND[i_s ,j  ,k] + UWIND[isp1,j  ,k] )/2. * \
                                      sin(latis_rad[i_s ,j  ]) ) \
                                    )

                            dUFLXdt[i_s-nb,j-nb,k] = dUFLXdt[i_s-nb,j-nb,k] + coriolis_UWIND

                        #######################################################################

                        #### PRESSURE GRADIENT
                        if i_pre_grad == 1:
                            preGrad_UWIND = - dy * ( \
                                    ( PHI [i_s ,j  ,k]  - PHI [ism1,j  ,k] ) * \
                                    ( COLP[i_s ,j    ]  + COLP[ism1,j    ] ) /2. + \
                                    ( COLP[i_s ,j    ]  - COLP[ism1,j    ] ) * c_con_cp/2. * \
                                    (\
                                      + POTT[ism1,j  ,k] / dsigma[k] * \
                                        ( \
                                            sigma_vb[kp1] * \
                                            ( PVTFVB[ism1,j  ,kp1] - PVTF  [ism1,j  ,k  ] ) + \
                                            sigma_vb[k  ] * \
                                            ( PVTF  [ism1,j  ,k  ] - PVTFVB[ism1,j  ,k  ] )   \
                                        ) \
                                      + POTT[i_s ,j  ,k] / dsigma[k] * \
                                        ( \
                                            sigma_vb[kp1] * \
                                            ( PVTFVB[i_s ,j  ,kp1] - PVTF  [i_s ,j  ,k  ] ) + \
                                            sigma_vb[k  ] * \
                                            ( PVTF  [i_s ,j  ,k  ] - PVTFVB[i_s ,j  ,k  ] )   \
                                        ) \
                                    ) )

                            dUFLXdt[i_s-nb,j-nb,k] = dUFLXdt[i_s-nb,j-nb,k] + preGrad_UWIND

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #                         MAIN LOOP  VWIND                            #
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

        #for i in range(nb,nx+nb):
        #for i in prange(nb,nx+nb, nogil=True, num_threads=c_njobs, schedule='static', chunksize=chunk):
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            for i in prange(nb,nx +nb, schedule='static', num_threads=4):

                im1 = i - 1
                ip1 = i + 1

                for js in range(nb,nys+nb):
                #for js in prange(nb,nys+nb, nogil=False, num_threads=c_njobs):

                    jsm1 = js - 1
                    jsp1 = js + 1

                    for k in range(0,nz):
                    #for k in prange(0,nz, nogil=False, num_threads=c_njobs):

                        kp1 = k+1

                        # HORIZONTAL ADVECTION
                        if i_hor_adv:
                            horAdv_VWIND =  + RFLX [i  ,jsm1,k] * \
                                            ( VWIND[i  ,jsm1,k] + VWIND[i  ,js  ,k] )/2. \
                                            - RFLX [i  ,js  ,k] * \
                                            ( VWIND[i  ,js  ,k] + VWIND[i  ,jsp1,k] )/2. \
                                            \
                                            + QFLX [i  ,js  ,k] * \
                                            ( VWIND[im1,js  ,k] + VWIND[i  ,js  ,k] )/2. \
                                            - QFLX [ip1,js  ,k] * \
                                            ( VWIND[i  ,js  ,k] + VWIND[ip1,js  ,k] )/2. \
                                            \
                                            + SFLX [i  ,jsm1,k] * \
                                            ( VWIND[im1,jsm1,k] + VWIND[i  ,js  ,k] )/2. \
                                            - SFLX [ip1,js  ,k] * \
                                            ( VWIND[i  ,js  ,k] + VWIND[ip1,jsp1,k] )/2. \
                                            \
                                            + TFLX [ip1,jsm1,k] * \
                                            ( VWIND[ip1,jsm1,k] + VWIND[i  ,js  ,k] )/2. \
                                            - TFLX [i  ,js  ,k] * \
                                            ( VWIND[i  ,js  ,k] + VWIND[im1,jsp1,k] )/2. 

                            dVFLXdt[i-nb,js-nb,k] = dVFLXdt[i-nb,js-nb,k] + horAdv_VWIND

                        #######################################################################

                        ## VERTICAL ADVECTION
                        if i_vert_adv == 1:
                            vertAdv_VWIND = (WWIND_VWIND_ks[i  ,js  ,k  ] - \
                                             WWIND_VWIND_ks[i  ,js  ,k+1]  ) / dsigma[k]
                            dVFLXdt[i-nb,js-nb,k] = dVFLXdt[i-nb,js-nb,k] + vertAdv_VWIND

                        #######################################################################

                        ## HORIZONTAL DIFFUSION
                        if i_num_dif == 1:
                            diff_VWIND = c_WIND_hor_dif_tau * \
                                 (  VFLX[im1,js  ,k] + VFLX[ip1,js  ,k] \
                                  + VFLX[i  ,jsm1,k] + VFLX[i  ,jsp1,k] - 4.*VFLX[i ,js  ,k])


                            dVFLXdt[i-nb,js-nb,k] = dVFLXdt[i-nb,js-nb,k] + diff_VWIND

                        #######################################################################

                        #### CORIOLIS
                        if i_coriolis == 1:
                            coriolis_VWIND = - c_con_rE*dlon_rad*dlon_rad/2.*(\
                                    + COLP[i,jsm1] * \
                                    ( UWIND[i  ,jsm1,k] + UWIND[ip1,jsm1,k] )/2. * \
                                    ( corf[i  ,jsm1] * c_con_rE *\
                                      cos(lat_rad[i  ,jsm1]) + \
                                      ( UWIND[i  ,jsm1,k] + UWIND[ip1,jsm1,k] )/2. * \
                                      sin(lat_rad[i  ,jsm1]) ) \

                                    + COLP[i  ,js  ] * \
                                    ( UWIND[i  ,js  ,k] + UWIND[ip1,js ,k] )/2. * \
                                    ( corf[i  ,js  ] * c_con_rE *\
                                      cos(lat_rad[i  ,js  ]) + \
                                      ( UWIND[i  ,js ,k] + UWIND[ip1,js ,k] )/2. * \
                                      sin(lat_rad[i  ,js  ]) ) \
                                    )

                            dVFLXdt[i-nb,js-nb,k] = dVFLXdt[i-nb,js-nb,k] + coriolis_VWIND

                        #######################################################################

                        #### PRESSURE GRADIENT
                        if i_pre_grad == 1:
                            preGrad_VWIND = - dxjs[i  ,js  ] * ( \
                                    ( PHI [i ,js  ,k]  - PHI [i,jsm1 ,k] ) * \
                                    ( COLP[i ,js    ]  + COLP[i,jsm1   ] ) /2. + \
                                    ( COLP[i ,js    ]  - COLP[i,jsm1   ] ) * c_con_cp/2. * \
                                    (\
                                      + POTT[i  ,jsm1,k] / dsigma[k] * \
                                        ( \
                                            sigma_vb[kp1] * \
                                            ( PVTFVB[i ,jsm1,kp1] - PVTF  [i  ,jsm1,k  ] ) + \
                                            sigma_vb[k  ] * \
                                            ( PVTF  [i ,jsm1,k  ] - PVTFVB[i  ,jsm1,k  ] )   \
                                        ) \
                                      + POTT[i  ,js  ,k] / dsigma[k] * \
                                        ( \
                                            sigma_vb[kp1] * \
                                            ( PVTFVB[i ,js  ,kp1] - PVTF  [i  ,js ,k  ] ) + \
                                            sigma_vb[k  ] * \
                                            ( PVTF  [i ,js  ,k  ] - PVTFVB[i  ,js ,k  ] )   \
                                        ) \
                                    ) )

                            dVFLXdt[i-nb,js-nb,k] = dVFLXdt[i-nb,js-nb,k] + preGrad_VWIND

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

    out = {}
    out['dUFLXdt'] = np.asarray(dUFLXdt)
    out['dVFLXdt'] = np.asarray(dVFLXdt)
    output.put( (job_ind, out) )


    #result = (np.asarray(dUFLXdt),
    #          np.asarray(dVFLXdt)) 
    #return(result)




#cdef exchange_flux_BC_periodic_x(double[:,:, ::1] FIELD, nb, nx, nxs):
#
#    #dimx,dimy,dimz = FIELD.shape
#    #binds = np.arange(0,nb)
#    print(FIELD.shape)
#    quit()
#
#    cdef int bind
#    cdef int dimz = 3
#    cdef int dimx = 30
#
#
#    if dimx == nx+2*nb: # unstaggered in x
#        for k in range(0,dimz):
#            for bind in range(0,nb):
#                FIELD[bind,:,k] = FIELD[nx+bind,:,k]
#                FIELD[nx+nb+bind,:,k] = FIELD[nb+bind,:,k]
#    else: # staggered in x
#        for k in range(0,dimz):
#            for bind in range(0,nb):
#                FIELD[bind,:,k] = FIELD[nxs+bind-1,:,k]
#                FIELD[nxs+nb+bind-1,:,k] = FIELD[nb+bind,:,k]
#
#    return(FIELD)
