import numpy as np
import scipy
from radiation.namelist_radiation import con_h, con_c, con_kb, \
                            ext_coef_LW

def org_longwave(GR, dz, tair_col, rho_col, tsurf):

    # LONGWAVE
    nu0 = 50
    nu1 = 2500
    dtau = ext_coef_LW * dz * rho_col
    taus = np.zeros(GR.nzs)
    taus[1:] = np.cumsum(dtau)

    gamma1 = np.zeros(GR.nz)
    gamma1[:] = 1.73

    gamma2 = np.zeros(GR.nz)
    gamma2[:] = 0

    emissivity_surface = 0.9
    albedo_surface_LW = 0.126
    # single scattering albedo
    omega_s = 0

    B_air = 2*np.pi * (1 - omega_s) * \
            calc_planck_intensity(GR, nu0, nu1, tair_col)
    B_surf = emissivity_surface * np.pi * \
            calc_planck_intensity(GR, nu0, nu1, tsurf)

    A_mat, g_vec = rad_calc_LW_RTE_matrix(GR, dtau, gamma1, gamma2,
                        B_air, B_surf, albedo_surface_LW)
    fluxes = scipy.sparse.linalg.spsolve(A_mat, g_vec)

    lwflxup = fluxes[range(1,len(fluxes),2)]
    lwflxdo = fluxes[range(0,len(fluxes),2)]

    return(lwflxup, lwflxdo)


def rad_calc_LW_RTE_matrix(GR, dtau, gamma1, gamma2, \
                        B_air, B_surf, albedo_surface):

    g_vec = np.zeros(2*GR.nz+2)
    g_vec[1:-1] = np.repeat(dtau*B_air, 2)
    g_vec[range(2,len(g_vec),2)] = - g_vec[range(2,len(g_vec),2)]
    g_vec[-1] = B_surf

    C1 = 0.5 * dtau * gamma1
    C2 = 0.5 * dtau * gamma2

    d0_ = np.repeat(1+C1, 2)
    d0_[range(1,len(d0_),2)] = - d0_[range(1,len(d0_),2)]
    d0 = np.ones(2*GR.nz+2); d0[1:-1] = d0_
    #print(d0)

    dp1_ = np.repeat(-C2, 2)
    dp1_[range(1,len(dp1_),2)] = - dp1_[range(1,len(dp1_),2)]
    dp1 = np.zeros(2*GR.nz+2); dp1[2:] = dp1_
    #print(dp1)

    dp2_ = np.repeat(-(1-C1), 2)
    dp2_[range(0,len(dp2_),2)] = 0
    dp2 = np.zeros(2*GR.nz+2); dp2[2:] = dp2_
    #print(dp2)

    dm1_ = np.repeat(-C2, 2)
    dm1_[range(1,len(dm1_),2)] = - dm1_[range(1,len(dm1_),2)]
    dm1 = np.zeros(2*GR.nz+2); dm1[:-2] = dm1_; dm1[-2] = -albedo_surface
    #print(dm1)

    dm2_ = np.repeat(1-C1, 2)
    dm2_[range(1,len(dm2_),2)] = 0
    dm2 = np.zeros(2*GR.nz+2); dm2[:-2] = dm2_
    #print(dm2)

    A_mat = scipy.sparse.diags( (dm2, dm1, d0, dp1[1:], dp2[2:]),
                                ( -2,  -1,  0,       1,       2),
                                (GR.nzs*2, GR.nzs*2), format='csr' )

    return(A_mat, g_vec)




def calc_planck_intensity(GR, nu0, nu1, temp):

    nu0 = nu0*100 # 1/m
    nu1 = nu1*100 # 1/m
    dnu = 5000 # 1/m
    nus = np.arange(nu0,nu1,dnu)
    nus_center = nus[:-1]+dnu/2
    lambdas = 1./nus
    lambdas_center = 1./nus_center
    dlambdas = np.diff(lambdas)

    if type(temp) == np.float64:
        B = np.zeros( (len(nus_center), 1) )
    else:
        B = np.zeros( (len(nus_center), len(temp)) )

    for i in range(0,len(nus_center)):
        #print(temp)
        #print(
        #    2*con_h*con_c**2 / lambdas_center[i]**5 * \
        #    1 / ( np.exp( con_h*con_c / (lambdas_center[i]*con_kb*temp) ) - 1 )
        #    )
        #quit()
        spectral_radiance = \
            2*con_h*con_c**2 / lambdas_center[i]**5 * \
            1 / ( np.exp( con_h*con_c / (lambdas_center[i]*con_kb*temp) ) - 1 )
        #print(spectral_radiance)
        radiance = spectral_radiance * -dlambdas[i]
        #print(radiance)
        B[i,:] = radiance

    B = np.sum(B,0)

    return(B)
