from debug_namelist import wp, wp_int
from grid import nx,nxs,ny,nys,nz,nzs,nb

from numba import cuda, njit

#class Tendencies:
#    
#    def __init__(self, target):
#        self.target = target
#
#        if target == 'CPU':
#            self.hor_adv = njit(hor_adv)
#            self.vert_adv = njit(vert_adv)
#            self.num_dif = njit(num_dif)




def hor_adv(VAR, VAR_im1, VAR_ip1, VAR_jm1, VAR_jp1,
            UFLX, UFLX_ip1, VFLX, VFLX_jp1, A):
    return(
        ( + UFLX * (VAR_im1 + VAR)/wp(2.)
          - UFLX_ip1 * (VAR + VAR_ip1)/wp(2.)
          + VFLX * (VAR_jm1 + VAR)/wp(2.)
          - VFLX_jp1 * (VAR + VAR_jp1)/wp(2.) )/A )



def vert_adv(VARVB, VARVB_kp1, WWIND, WWIND_kp1, COLP_NEW,
            dsigma, k):

    if k == wp_int(0):
        return(
            COLP_NEW * (
                - WWIND_kp1 * VARVB_kp1) / dsigma)
    elif k == nz:
        return(
            COLP_NEW * (
                + WWIND     * VARVB    ) / dsigma)
    else:
        return(
            COLP_NEW * (
                + WWIND     * VARVB 
                - WWIND_kp1 * VARVB_kp1) / dsigma)




def num_dif(VAR, VAR_im1, VAR_ip1, VAR_jm1, VAR_jp1,
            COLP, COLP_im1, COLP_ip1, COLP_jm1, COLP_jp1,
            VAR_DIF_COEF):
    return(
            VAR_DIF_COEF * (
                + COLP_im1 * VAR_im1
                + COLP_ip1 * VAR_ip1
                + COLP_jm1 * VAR_jm1
                + COLP_jp1 * VAR_jp1
                - wp(4.) * COLP * VAR )
            )



def radiation():
    raise NotImplementedError()
#    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                        dPOTTdt_RAD[i-1,j-1,k]*COLP[i,j] # TODO add boundaries


def microphysics():
    raise NotImplementedError()
#    dPOTTdt[i,j,k] = dPOTTdt[i,j,k] + \
#                        dPOTTdt_MIC[i-1,j-1,k]*COLP[i,j] # TODO add boundaries



