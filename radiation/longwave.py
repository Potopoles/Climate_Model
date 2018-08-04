import numpy as np
import scipy
from radiation.namelist_radiation import con_h, con_c, con_kb, \
                            sigma_abs_gas_LW_in, sigma_sca_gas_LW_in, \
                            emissivity_surface


###################################################################################
###################################################################################
# METHOD: SELF CONSTRUCTED
###################################################################################
###################################################################################

def org_longwave(GR, dz, tair_col, rho_col, tsurf, albedo_surface_LW, qc_col):

    # LONGWAVE
    nu0 = 50
    nu1 = 2500
    g_a = 0.0
    #albedo_surface_LW = 1
    #qc_col = np.minimum(qc_col, 0.003)
    sigma_abs_gas_LW = np.repeat(sigma_abs_gas_LW_in, GR.nz)
    #sigma_abs_gas_LW = sigma_abs_gas_LW + qc_col*1E-5
    sigma_sca_gas_LW = np.repeat(sigma_sca_gas_LW_in, GR.nz)
    #sigma_sca_gas_LW = sigma_sca_gas_LW + qc_col*1E-5
    sigma_tot_LW = sigma_abs_gas_LW + sigma_sca_gas_LW

    # optical thickness
    dtau = sigma_tot_LW * dz * rho_col
    taus = np.zeros(GR.nzs)
    taus[1:] = np.cumsum(dtau)

    # single scattering albedo
    omega_s = sigma_sca_gas_LW/sigma_tot_LW
    omega_s[np.isnan(omega_s)] = 0

    # quadrature
    my1 = 1/np.sqrt(3)

    gamma1 = np.zeros(GR.nz)
    gamma1[:] = ( 1 - omega_s*(1+g_a)/2 ) / my1

    gamma2 = np.zeros(GR.nz)
    gamma2[:] = omega_s*(1-g_a) / (2*my1)

    # emission fields
    B_air = 2*np.pi * (1 - omega_s) * \
            calc_planck_intensity(GR, nu0, nu1, tair_col)
    B_surf = emissivity_surface * np.pi * \
            calc_planck_intensity(GR, nu0, nu1, tsurf)

    # calculate radiative fluxes
    A_mat, g_vec = rad_calc_LW_RTE_matrix(GR, dtau, gamma1, gamma2,
                        B_air, B_surf, albedo_surface_LW)
    fluxes = scipy.sparse.linalg.spsolve(A_mat, g_vec)

    up_diffuse = fluxes[range(1,len(fluxes),2)]
    down_diffuse = - fluxes[range(0,len(fluxes),2)]

    #net_flux = + down_diffuse + up_diffuse 
    #print(down_diffuse)
    #print(up_diffuse)
    #print(net_flux)
    #import matplotlib.pyplot as plt
    #zs = range(GR.nzs,0,-1)
    #line1, = plt.plot(down_diffuse, zs, label='down diff')
    #line3, = plt.plot(up_diffuse, zs, label='up diff', linestyle='--')
    #line4, = plt.plot(net_flux, zs, label='net', linewidth=2)
    #plt.axvline(x=0, color='black', lineWidth=1)
    #plt.legend([line1,line3,line4])
    #plt.show()
    #quit()

    return(down_diffuse, up_diffuse)


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

    dp1_ = np.repeat(-C2, 2)
    dp1_[range(1,len(dp1_),2)] = - dp1_[range(1,len(dp1_),2)]
    dp1 = np.zeros(2*GR.nz+2); dp1[2:] = dp1_

    dp2_ = np.repeat(-(1-C1), 2)
    dp2_[range(0,len(dp2_),2)] = 0
    dp2 = np.zeros(2*GR.nz+2); dp2[2:] = dp2_

    dm1_ = np.repeat(-C2, 2)
    dm1_[range(1,len(dm1_),2)] = - dm1_[range(1,len(dm1_),2)]
    dm1 = np.zeros(2*GR.nz+2); dm1[:-2] = dm1_; dm1[-2] = -albedo_surface

    dm2_ = np.repeat(1-C1, 2)
    dm2_[range(1,len(dm2_),2)] = 0
    dm2 = np.zeros(2*GR.nz+2); dm2[:-2] = dm2_

    A_mat = scipy.sparse.diags( (dm2, dm1, d0, dp1[1:], dp2[2:]),
                                ( -2,  -1,  0,       1,       2),
                                (GR.nzs*2, GR.nzs*2), format='csr' )

    return(A_mat, g_vec)

###################################################################################
###################################################################################
###################################################################################
###################################################################################


###################################################################################
###################################################################################
# METHOD: TOON ET AL 1989
###################################################################################
###################################################################################

#def org_longwave(GR, dz, tair_vb_col, rho_col, tsurf, albedo_surface_LW):
#
#    # TODO
#    nu0 = 50
#    nu1 = 2500
#    g_a = 0.0
#    #albedo_surface_LW = 0
#    #albedo_surface_LW = 0.5
#    #albedo_surface_LW = 1
#    sigma_abs_gas_LW = np.repeat(sigma_abs_gas_LW_in, GR.nz)
#    sigma_sca_gas_LW = np.repeat(sigma_sca_gas_LW_in, GR.nz)
#    #sigma_sca_gas_LW[1] = sigma_sca_gas_LW[1]*2
#    #sigma_abs_gas_LW[:] = 0
#    #sigma_sca_gas_LW[:] = 0 
#    #sigma_sca_gas_LW[:5] = 0
#    #sigma_sca_gas_LW[7:] = 0
#    sigma_tot_LW = sigma_abs_gas_LW + sigma_sca_gas_LW
#    #print(sigma_tot_LW)
#
#    dtau = sigma_tot_LW * dz * rho_col
#    taus = np.zeros(GR.nzs)
#    taus[1:] = np.cumsum(dtau)
#    tau = taus[:-1] + np.diff(taus)/2
#
#    #print(tau)
#    #print(taus)
#    #print()
#
#    # single scattering albedo
#    omega_s = sigma_sca_gas_LW/sigma_tot_LW
#    omega_s[np.isnan(omega_s)] = 0
#    #print('omega_s ' + str(omega_s))
#
#    # QUADRATURE
#    my1 = 1/np.sqrt(3)
#
#    gamma1 = np.zeros(GR.nz)
#    #gamma1[:] = 3**(1/2) * (2 - omega_s * (1 + g_a) ) / 2
#    gamma1[:] = ( 1 - omega_s*(1+g_a)/2 ) / my1
#
#    gamma2 = np.zeros(GR.nz)
#    #gamma2[:] = omega_s * 3**(1/2) * (1 - g_a) / 2
#    gamma2[:] = omega_s*(1-g_a) / (2*my1)
#
#    #print('gammas:')
#    #print(gamma1)
#    #print(gamma2)
#    #print()
#
#    lamb_2str = np.sqrt(gamma1**2 - gamma2**2)
#    tau_2str = gamma2 / (gamma1 + lamb_2str)
#        
#    #print(lamb_2str)
#    #print(tau_2str)
#    #print()
#
#    surf_emitted_LW = emissivity_surf * np.pi * \
#                    calc_planck_intensity(GR, nu0, nu1, tsurf) 
#    #print(surf_emitted_LW)
#    emission_vb  = calc_planck_intensity(GR, nu0, nu1, tair_vb_col)
#    B0n = emission_vb[:-1]
#    B1n = np.diff(emission_vb)/dtau
#    #B1n = np.diff(tair_vb_col)
#    #print(B0n)
#    #print(B1n)
#    #print()
#    
#    down_diffuse = np.zeros( GR.nzs )
#    up_diffuse = np.full( GR.nzs, np.nan)
#    down_diffuse[1:], up_diffuse[1:] = \
#                rad_calc_LW_fluxes_toon(GR, dtau, gamma1, gamma2, my1,
#                            lamb_2str, tau_2str, tau, taus, omega_s,
#                            surf_emitted_LW, B0n, B1n, albedo_surface_LW)
#
#
#    ## first order
#    #up_diffuse[0] = max(0, up_diffuse[1] - (up_diffuse[2] - up_diffuse[1]) * dz[0]/dz[1])
#    # second order
#    up_diffuse[0] = max(0, up_diffuse[1] - (- up_diffuse[3] + 4*up_diffuse[2] - 3*up_diffuse[1]) \
#                                            * dz[0]/(dz[1]+dz[2]))
#
#    return(down_diffuse, up_diffuse)
#
#def rad_calc_LW_fluxes_toon(GR, dtau, gamma1, gamma2, my1,
#                            lamb_2str, tau_2str, tau, taus, omega_s,
#                            surf_emitted_LW, B0n, B1n, albedo_surface_LW):
#
#    e1 = 1        + tau_2str * np.exp( - lamb_2str * dtau )
#    e2 = 1        - tau_2str * np.exp( - lamb_2str * dtau )
#    e3 = tau_2str +            np.exp( - lamb_2str * dtau )
#    e4 = tau_2str -            np.exp( - lamb_2str * dtau )
#    #print('e terms:')
#    #print(e1)
#    #print(e2)
#    #print(e3)
#    #print(e4)
#    #print()
#
#    lodd = np.arange(3,2*GR.nz,2) - 1
#    leven = np.arange(2,2*GR.nz,2) - 1
#    ninds = np.arange(0,GR.nz-1)
#    nindsp1 = np.arange(1,GR.nz)
#    #print('indices')
#    #print(lodd)
#    #print(leven)
#    #print(ninds)
#    #print(nindsp1)
#    #print()
#
#    dm1 = np.full(2*GR.nz, np.nan)
#    dm1[lodd]  = e2[ninds  ] * e3[ninds  ] - e4[ninds  ] * e1[ninds  ]
#    dm1[leven] = e2[nindsp1] * e1[ninds  ] - e3[ninds  ] * e4[nindsp1]
#    dm1[0] = 0
#    dm1[-1] = e1[-1] - albedo_surface_LW * e3[-1]
#    #print(dm1)
#
#    d0 = np.full(2*GR.nz, np.nan)
#    d0[lodd]  = e1[ninds  ] * e1[nindsp1] - e3[ninds  ] * e3[nindsp1]
#    d0[leven] = e2[ninds  ] * e2[nindsp1] - e4[ninds  ] * e4[nindsp1]
#    d0[0] = e1[0]
#    d0[-1] = e2[-1] - albedo_surface_LW * e4[-1]
#    #print(d0)
#
#    dp1 = np.full(2*GR.nz, np.nan)
#    dp1[lodd]  = e3[ninds  ] * e4[nindsp1] - e1[ninds  ] * e2[nindsp1]
#    dp1[leven] = e1[nindsp1] * e4[nindsp1] - e2[nindsp1] * e3[nindsp1]
#    dp1[0] = - e2[0]
#    dp1[-1] = 0
#    #print(dp1)
#    #print()
#
#    A_mat = scipy.sparse.diags( (dm1[1:], d0, dp1),
#                                ( -1,  0,   1),
#                                (GR.nz*2, GR.nz*2), format='csr' )
#    #print(A_mat.todense())
#    #print()
#
#    #print('C terms')
#    Cp_tau = 2 * np.pi * my1 * ( B0n + B1n * ( dtau + 1 / (gamma1 + gamma2) ) )
#    #print(Cp_tau)
#    Cp_0   = 2 * np.pi * my1 * ( B0n + B1n * ( 0    + 1 / (gamma1 + gamma2) ) )
#    #print(Cp_0)
#    Cm_tau = 2 * np.pi * my1 * ( B0n + B1n * ( dtau - 1 / (gamma1 + gamma2) ) )
#    #print(Cm_tau)
#    Cm_0   = 2 * np.pi * my1 * ( B0n + B1n * ( 0    - 1 / (gamma1 + gamma2) ) )
#    #print(Cm_0)
#     
#    src = np.full(2*GR.nz, np.nan)
#    src[lodd]  = e3[ninds ]  * (Cp_0  [nindsp1] - Cp_tau[ninds  ]) + \
#                 e1[ninds ]  * (Cm_tau[ninds  ] - Cm_0  [nindsp1])
#    src[leven] = e2[nindsp1] * (Cp_0  [nindsp1] - Cp_tau[ninds  ]) + \
#                 e4[nindsp1] * (Cm_0  [nindsp1] - Cm_tau[ninds  ])
#    src[0] = 0 - Cm_0[0]
#    src[-1] = surf_emitted_LW - Cp_tau[-1] + albedo_surface_LW * Cm_tau[-1]
#    #print('src')
#    #print(src)
#    #print()
#
#    fluxes = scipy.sparse.linalg.spsolve(A_mat, src)
#
#    Y1 = fluxes[range(0,len(fluxes),2)]
#    Y2 = fluxes[range(1,len(fluxes),2)]
#
#    down_diffuse = + ( - Y1*e3 - Y2*e4 - Cm_tau )
#    up_diffuse   = + ( + Y1*e1 + Y2*e2 + Cp_tau )
#
#    #net_flux = - down_diffuse + up_diffuse 
#    #print(down_diffuse)
#    #print(up_diffuse)
#    #print(net_flux)
#    #import matplotlib.pyplot as plt
#    #zs = range(GR.nz,0,-1)
#    #line1, = plt.plot(-down_diffuse, zs, label='down diff')
#    #line3, = plt.plot(up_diffuse, zs, label='up diff', linestyle='--')
#    #line4, = plt.plot(net_flux, zs, label='net', linewidth=2)
#    #plt.axvline(x=0, color='black', lineWidth=1)
#    #plt.legend([line1,line3,line4])
#    #plt.show()
#    #quit()
#
#    return(down_diffuse, up_diffuse)


###################################################################################
###################################################################################
###################################################################################
###################################################################################






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

    for c in range(0,len(nus_center)):
        #print(temp)
        #print(
        #    2*con_h*con_c**2 / lambdas_center[i]**5 * \
        #    1 / ( np.exp( con_h*con_c / (lambdas_center[i]*con_kb*temp) ) - 1 )
        #    )
        #quit()
        spectral_radiance = \
            2*con_h*con_c**2 / lambdas_center[c]**5 * \
            1 / ( np.exp( con_h*con_c / (lambdas_center[c]*con_kb*temp) ) - 1 )
        #print(spectral_radiance)
        radiance = spectral_radiance * -dlambdas[c]
        #print(radiance)
        B[c,:] = radiance

    B = np.sum(B,0)

    return(B)
