import numpy as np
import scipy
from datetime import timedelta
from radiation.namelist_radiation import solar_constant_0, \
                            sigma_abs_gas_SW_in, sigma_sca_gas_SW_in


def org_shortwave(GR, dz, solar_constant, rho_col, swintoa, mysun,
                    albedo_surface_SW):

    # TODO
    #mysun = 1.0
    g_a = 0.0
    #albedo_surface_SW = 0
    #albedo_surface_SW = 0.5
    #albedo_surface_SW = 1
    sigma_abs_gas_SW = np.repeat(sigma_abs_gas_SW_in, GR.nz)
    sigma_sca_gas_SW = np.repeat(sigma_sca_gas_SW_in, GR.nz)
    #sigma_sca_gas_SW[1] = sigma_sca_gas_SW[1]*2
    #sigma_abs_gas_SW[:] = 0
    #sigma_sca_gas_SW[:] = 0 
    #sigma_sca_gas_SW[:5] = 0
    #sigma_sca_gas_SW[7:] = 0
    sigma_tot_SW = sigma_abs_gas_SW + sigma_sca_gas_SW
    #print(sigma_tot_SW)

    dtau = sigma_tot_SW * dz * rho_col
    taus = np.zeros(GR.nzs)
    taus[1:] = np.cumsum(dtau)
    tau = taus[:-1] + np.diff(taus)/2

    #print(tau)
    #print(taus)
    #print()

    # single scattering albedo
    omega_s = sigma_sca_gas_SW/sigma_tot_SW
    omega_s[np.isnan(omega_s)] = 0
    #print('omega_s ' + str(omega_s))
    #quit()

    # QUADRATURE
    my1 = 1/np.sqrt(3)

    gamma1 = np.zeros(GR.nz)
    #gamma1[:] = 3**(1/2) * (2 - omega_s * (1 + g_a) ) / 2
    gamma1[:] = ( 1 - omega_s*(1+g_a)/2 ) / my1

    gamma2 = np.zeros(GR.nz)
    #gamma2[:] = omega_s * 3**(1/2) * (1 - g_a) / 2
    gamma2[:] = omega_s*(1-g_a) / (2*my1)

    gamma3 = np.zeros(GR.nz)
    #gamma3[:] = (1 - 3**(1/2) * g_a * mysun) / 2
    gamma3[:] = (1 - 3 * g_a * my1 * mysun) / 2

    gamma4 = np.zeros(GR.nz)
    gamma4[:] = 1 - gamma3

    #print('gammas:')
    #print(gamma1)
    #print(gamma2)
    #print(gamma3)
    #print(gamma4)
    #print()

    lamb_2str = np.sqrt(gamma1**2 - gamma2**2)
    tau_2str = gamma2 / (gamma1 + lamb_2str)
        
    #print(lamb_2str)
    #print(tau_2str)
    #print()

    surf_reflected_SW = albedo_surface_SW * mysun * solar_constant * \
                            np.exp(-taus[-1]/mysun)
    #print(surf_reflected_SW)
    #print()

    #sw_dir = omega_s * solar_constant * np.exp(-tau/mysun)
    #A_mat, g_vec = rad_calc_SW_RTE_matrix(GR, dtau, gamma1, gamma2, gamma3,
    #                        swintoa, surf_reflected_SW,
    #                        albedo_surface_SW, sw_dir)
    #fluxes = scipy.sparse.linalg.spsolve(A_mat, src)
    down_diffuse = np.zeros( GR.nzs )
    up_diffuse = np.full( GR.nzs, np.nan)
    down_direct = np.full( GR.nzs, np.nan)
    down_diffuse[1:], up_diffuse[1:], down_direct[1:] = \
                rad_calc_SW_fluxes_toon(GR, dtau, gamma1, gamma2,
                            gamma3, gamma4, my1, solar_constant,
                            lamb_2str, tau_2str, tau, taus, omega_s, mysun,
                            surf_reflected_SW,
                            albedo_surface_SW)

    # first order
    #up_diffuse[0] = max(0, up_diffuse[1] - (up_diffuse[2] - up_diffuse[1]) * dz[0]/dz[1])
    # second order
    up_diffuse[0] = max(0, up_diffuse[1] - (- up_diffuse[3] + 4*up_diffuse[2] - 3*up_diffuse[1]) \
                                            * dz[0]/(dz[1]+dz[2]))
    down_direct[0] = -swintoa

    return(down_diffuse, up_diffuse, down_direct)

def rad_calc_SW_fluxes_toon(GR, dtau, gamma1, gamma2,
        gamma3, gamma4, my1, solar_constant,
        lamb_2str, tau_2str, tau, taus, omega_s, mysun,
        surf_reflected_SW, albedo_surface_SW):

    e1 = 1        + tau_2str * np.exp( - lamb_2str * dtau )
    e2 = 1        - tau_2str * np.exp( - lamb_2str * dtau )
    e3 = tau_2str +            np.exp( - lamb_2str * dtau )
    e4 = tau_2str -            np.exp( - lamb_2str * dtau )
    #print('e terms:')
    #print(e1)
    #print(e2)
    #print(e3)
    #print(e4)
    #print()

    lodd = np.arange(3,2*GR.nz,2) - 1
    leven = np.arange(2,2*GR.nz,2) - 1
    ninds = np.arange(0,GR.nz-1)
    nindsp1 = np.arange(1,GR.nz)
    #print('indices')
    #print(lodd)
    #print(leven)
    #print(ninds)
    #print(nindsp1)
    #print()

    dm1 = np.full(2*GR.nz, np.nan)
    dm1[lodd]  = e2[ninds  ] * e3[ninds  ] - e4[ninds  ] * e1[ninds  ]
    dm1[leven] = e2[nindsp1] * e1[ninds  ] - e3[ninds  ] * e4[nindsp1]
    dm1[0] = 0
    dm1[-1] = e1[-1] - albedo_surface_SW * e3[-1]
    #print(dm1)

    d0 = np.full(2*GR.nz, np.nan)
    d0[lodd]  = e1[ninds  ] * e1[nindsp1] - e3[ninds  ] * e3[nindsp1]
    d0[leven] = e2[ninds  ] * e2[nindsp1] - e4[ninds  ] * e4[nindsp1]
    d0[0] = e1[0]
    d0[-1] = e2[-1] - albedo_surface_SW * e4[-1]
    #print(d0)

    dp1 = np.full(2*GR.nz, np.nan)
    dp1[lodd]  = e3[ninds  ] * e4[nindsp1] - e1[ninds  ] * e2[nindsp1]
    dp1[leven] = e1[nindsp1] * e4[nindsp1] - e2[nindsp1] * e3[nindsp1]
    dp1[0] = - e2[0]
    dp1[-1] = 0
    #print(dp1)
    #print()

    A_mat = scipy.sparse.diags( (dm1[1:], d0, dp1),
                                ( -1,  0,   1),
                                (GR.nz*2, GR.nz*2), format='csr' )
    #print(A_mat.todense())
    #print()
    #quit()

    #print('C terms')
    Cp_tau = omega_s * solar_constant * np.exp( - (taus[1:]) / mysun ) * \
            ( (gamma1 - 1 / mysun) * gamma3 + gamma4 * gamma2 ) / \
            ( lamb_2str**2 - 1 / mysun**2 )
    #print(Cp_tau)
    Cp_0 = omega_s * solar_constant * np.exp( - (taus[:-1]) / mysun ) * \
            ( (gamma1 - 1 / mysun) * gamma3 + gamma4 * gamma2 ) / \
            ( lamb_2str**2 - 1 / mysun**2 )
    #print(Cp_0)
    Cm_tau = omega_s * solar_constant * np.exp( - (taus[1:]) / mysun ) * \
            ( (gamma1 + 1 / mysun) * gamma4 + gamma2 * gamma3 ) / \
            ( lamb_2str**2 - 1 / mysun**2 )
    #print(Cm_tau)
    Cm_0 = omega_s * solar_constant * np.exp( - (taus[:-1]) / mysun ) * \
            ( (gamma1 + 1 / mysun) * gamma4 + gamma2 * gamma3 ) / \
            ( lamb_2str**2 - 1 / mysun**2 )
    #print(Cm_0)
     
    src = np.full(2*GR.nz, np.nan)
    src[lodd]  = e3[ninds ]  * (Cp_0  [nindsp1] - Cp_tau[ninds  ]) + \
                 e1[ninds ]  * (Cm_tau[ninds  ] - Cm_0  [nindsp1])
    src[leven] = e2[nindsp1] * (Cp_0  [nindsp1] - Cp_tau[ninds  ]) + \
                 e4[nindsp1] * (Cm_0  [nindsp1] - Cm_tau[ninds  ])
    #src[0] = mysun * solar_constant - Cm_0[0]
    src[0] = 0 - Cm_0[0]
    src[-1] = surf_reflected_SW - Cp_tau[-1] + albedo_surface_SW * Cm_tau[-1]
    #print('src')
    #print(src)
    #print()
    #quit()

    fluxes = scipy.sparse.linalg.spsolve(A_mat, src)

    Y1 = fluxes[range(0,len(fluxes),2)]
    Y2 = fluxes[range(1,len(fluxes),2)]

    down_diffuse = - Y1*e3 - Y2*e4 - Cm_tau
    down_direct = - mysun * solar_constant * np.exp( - taus[1:] / mysun)
    up_diffuse = Y1*e1 + Y2*e2 + Cp_tau

    #net_flux = down_diffuse + down_direct + up_diffuse 
    #print(net_flux)
    #import matplotlib.pyplot as plt
    #zs = range(GR.nz,0,-1)
    #line1, = plt.plot(down_diffuse, zs, label='down diff')
    #line2, = plt.plot(down_direct, zs, label='down_dir')
    #line3, = plt.plot(up_diffuse, zs, label='up diff', linestyle='--')
    #line4, = plt.plot(net_flux, zs, label='net down')
    #plt.axvline(x=0, color='black', lineWidth=1)
    #plt.legend([line1,line2,line3,line4])
    #plt.show()
    #quit()

    return(down_diffuse, up_diffuse, down_direct)


#def rad_calc_SW_RTE_matrix(GR, dtau, gamma1, gamma2, gamma3,
#        TOA_SW_in, surf_reflected_SW, albedo_surface, sw_dir):
#
#    g_vec_ = np.repeat(dtau*sw_dir, 2)
#    g_vec_[range(0,len(g_vec_),2)] = g_vec_[range(0,len(g_vec_),2)] *        gamma3
#    g_vec_[range(1,len(g_vec_),2)] = g_vec_[range(1,len(g_vec_),2)] * - (1 - gamma3)
#    g_vec = np.zeros(2*GR.nz+2); g_vec[1:-1] = g_vec_
#    g_vec[0] = TOA_SW_in; g_vec[-1] = surf_reflected_SW
#    #print(g_vec)
#    #quit()
#
#    C1 = 0.5 * dtau * gamma1
#    C2 = 0.5 * dtau * gamma2
#    #C2[:] = 0.1
#    #C1[:] = 0
#    #C2[:] = 0
#    #print(C1)
#    #print(C2)
#
#    d0_ = np.repeat(1+C1, 2)
#    d0_[range(1,len(d0_),2)] = - d0_[range(1,len(d0_),2)]
#    d0 = np.full(2*GR.nz+2,np.nan); d0[0] = 1; d0[-1] = 1; d0[1:-1] = d0_
#    #print(d0)
#
#    dp1_ = np.repeat(-C2, 2)
#    dp1_[range(1,len(dp1_),2)] = - dp1_[range(1,len(dp1_),2)]
#    #dp1 = np.zeros(2*GR.nz+2); dp1[2:] = dp1_
#    dp1 = np.full(2*GR.nz+2,np.nan); dp1[0] = 0; dp1[1:-1] = dp1_
#    #print(dp1)
#
#    dp2_ = np.repeat(-(1-C1), 2)
#    dp2_[range(0,len(dp2_),2)] = 0
#    #dp2 = np.zeros(2*GR.nz+2); dp2[2:] = dp2_
#    dp2 = np.full(2*GR.nz+2,np.nan); dp2[:-2] = dp2_
#    #print(dp2)
#
#    dm1_ = np.repeat(-C2, 2)
#    dm1_[range(1,len(dm1_),2)] = - dm1_[range(1,len(dm1_),2)]
#    #dm1 = np.zeros(2*GR.nz+2); dm1[:-2] = dm1_; dm1[-2] = -albedo_surface
#    dm1 = np.full(2*GR.nz+2,np.nan); dm1[:-2] = dm1_; dm1[-2] = -albedo_surface
#    #print(dm1)
#
#    dm2_ = np.repeat(1-C1, 2)
#    dm2_[range(1,len(dm2_),2)] = 0
#    #dm2 = np.zeros(2*GR.nz+2); dm2[:-2] = dm2_
#    dm2 = np.full(2*GR.nz+2,np.nan); dm2[:-2] = dm2_
#    #print(dm2)
#
#    #A_mat = scipy.sparse.diags( (dm2, dm1, d0, dp1[1:], dp2[2:]),
#    #                            ( -2,  -1,  0,       1,       2),
#    #                            (GR.nzs*2, GR.nzs*2), format='csr' )
#    A_mat = scipy.sparse.diags( (dm2, dm1, d0, dp1, dp2),
#                                ( -2,  -1,  0,       1,       2),
#                                (GR.nzs*2, GR.nzs*2), format='csr' )
#    #print()
#    #print(A_mat.todense())
#    #quit()
#    return(A_mat, g_vec)



def rad_solar_zenith_angle(GR, SOLZEN):

    #print(GR.GMT)
    #quit()

    # SOLAR DECLINATION ANGLE
    n_days_in_year = 365 # TODO: leap years
    D_J = GR.GMT.timetuple().tm_yday
    Y = GR.GMT.timetuple().tm_year

    if Y >= 2001:
        D_L = np.floor((Y - 2001)/4)
    else:
        D_L = np.floor((Y - 2000)/4) - 1
    N_JD = 364.5 + (Y - 2001)*365 + D_L + D_J

    eps_ob = 23.439 - 0.0000004 * N_JD

    L_M = 280.460 + 0.9856474 * N_JD;
    g_M = 357.528 + 0.9856003 * N_JD;

    lamb_ec = L_M + 1.915 * np.sin(g_M * np.pi/180) + \
                    0.020 * np.sin(2 * g_M * np.pi/180);

    sol_declin = np.arcsin( \
            np.sin(eps_ob/180*np.pi)*np.sin(lamb_ec/180*np.pi))            

    # HOUR ANGLE
    sec_of_local_noon = GR.lon_rad[GR.iijj]/(2*np.pi)*86400
    sec_of_day = timedelta(hours=GR.GMT.hour,
                              minutes=GR.GMT.minute,
                              seconds=GR.GMT.second).total_seconds()
    sec_past_local_noon = sec_of_day - sec_of_local_noon
    hour_angle = 2*np.pi * sec_past_local_noon / 86400;


    #SOLZEN[:,:] = np.cos(hour_angle)
    SOLZEN[:,:] = np.arccos( \
                    + np.sin(GR.lat_rad[GR.iijj]) * np.sin(sol_declin) \
                    + np.cos(GR.lat_rad[GR.iijj]) * \
                                np.cos(sol_declin) * np.cos(hour_angle) )

    return(SOLZEN)


def calc_current_solar_constant(GR):
    n_days_in_year = 365 # TODO: leap years
    D_J = GR.GMT.timetuple().tm_yday
    theta_J = 2 * np.pi * D_J / n_days_in_year
    earth_sun_dist = 1.00011 + 0.034221 * np.cos(theta_J) \
                    + 0.00128 * np.sin(theta_J) \
                    + 0.000719 * np.cos(2 * theta_J) \
                    + 0.000077 * np.sin(2 * theta_J)
    return(solar_constant_0 * earth_sun_dist)
