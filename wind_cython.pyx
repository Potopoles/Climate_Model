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

cdef int i_hor_adv = 0
i_vert_adv = 0
cdef int i_coriolis = 1
i_pre_grad = 0
cdef int i_num_dif = 1



#def wind_tendency_jacobson_par(GR, UWIND, VWIND, WWIND, UFLX, VFLX, 
#                                COLP, COLP_NEW, HSURF, PHI, POTT, PVTF, PVTFVB):
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef wind_tendency_jacobson_par( GR, njobs,\
        double[:,:, ::1] UWIND,
        double[:,:, ::1] VWIND,
        double[:,:, ::1] UFLX,
        double[:,:, ::1] VFLX,
        double[:, ::1] COLP):




    cdef int c_njobs = njobs
   
    cdef int nb = GR.nb
    cdef int nx  = GR.nx
    cdef int nxs = GR.nxs
    cdef int ny  = GR.ny
    cdef int nys = GR.nys
    cdef int nzs = GR.nzs
    cdef int nz  = GR.nz
    cdef double dlon_rad = GR.dlon_rad
    cdef double[:, ::1] corf_is = GR.corf_is
    cdef double[:, ::1] corf    = GR.corf
    cdef double[:, ::1] latis_rad = GR.latis_rad
    cdef double[:, ::1] lat_rad   = GR.lat_rad

    cdef int k, i, im1, ip1, i_s, ism1, isp1, j, jm1, jp1, js, jsm1, jsp1

    cdef double diff_UWIND, diff_VWIND, coriolis_UWIND, coriolis_VWIND

    cdef double c_WIND_hor_dif_tau = WIND_hor_dif_tau
    cdef double c_con_rE = con_rE

    cdef double[:,:, ::1] dUFLXdt = np.zeros( (nxs,ny ,nz) )
    cdef double[:,:, ::1] dVFLXdt = np.zeros( (nx ,nys,nz) )


    if i_wind_tendency:

        #for i_s in range(nb,nxs+nb):
        for i_s in prange(nb,nxs+nb, nogil=True, num_threads=c_njobs):

            ism1 = i_s - 1
            isp1 = i_s + 1

            #for j in range(nb,ny+nb):
            for j in prange(nb,ny+nb, nogil=False, num_threads=c_njobs):

                jm1 = j - 1
                jp1 = j + 1

                for k in range(0,nz):
                #for k in prange(0,nz, nogil=False, num_threads=c_njobs):

                    ## HORIZONTAL DIFFUSION
                    if i_num_dif == 1:
                        diff_UWIND = c_WIND_hor_dif_tau * \
                                     (  UFLX[ism1,j  ,k] + UFLX[isp1,j  ,k] \
                                      + UFLX[i_s ,jm1,k] + UFLX[i_s ,jp1,k] - 4.*UFLX[i_s ,j  ,k])

                        dUFLXdt[i_s-nb,j-nb,k] = dUFLXdt[i_s-nb,j-nb,k] + diff_UWIND

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



        #for i in range(nb,nx+nb):
        for i in prange(nb,nx+nb, nogil=True, num_threads=c_njobs):

            im1 = i - 1
            ip1 = i + 1

            #for js in range(nb,nys+nb):
            for js in prange(nb,nys+nb, nogil=False, num_threads=c_njobs):

                jsm1 = js - 1
                jsp1 = js + 1
                #printf('%i\n', ism1)
                #printf('%i\n', j)

                for k in range(0,nz):
                #for k in prange(0,nz, nogil=False, num_threads=c_njobs):

                    ## HORIZONTAL DIFFUSION
                    if i_num_dif == 1:
                        diff_VWIND = c_WIND_hor_dif_tau * \
                                     (  VFLX[im1,js  ,k] + VFLX[ip1,js  ,k] \
                                      + VFLX[i  ,jsm1,k] + VFLX[i  ,jsp1,k] - 4.*VFLX[i ,js  ,k])


                        dVFLXdt[i-nb,js-nb,k] = dVFLXdt[i-nb,js-nb,k] + diff_VWIND

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



    return(dUFLXdt, dVFLXdt)
