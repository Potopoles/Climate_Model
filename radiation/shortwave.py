import numpy as np
import scipy
from datetime import timedelta
from radiation.namelist_radiation import solar_constant_0, \
                            ext_coef_SW



def org_shortwave(GR, dz, solar_constant, rho_col, swintoa, mysun):

    # LONGWAVE
    nu0 = 50
    nu1 = 2500
    dtau = ext_coef_SW * dz * rho_col
    taus = np.zeros(GR.nzs)
    taus[1:] = np.cumsum(dtau)
    tau = taus[:-1] + np.diff(taus)/2

    gamma1 = np.zeros(GR.nz)
    gamma1[:] = 1.73

    gamma2 = np.zeros(GR.nz)
    gamma2[:] = 0

    gamma3 = np.zeros(GR.nz)
    gamma3[:] = 0.6

    albedo_surface_SW = 0.126
    # single scattering albedo
    omega_s = 0

    surf_reflected_SW = albedo_surface_SW * swintoa * \
                            np.exp(-taus[GR.nzs-1]/mysun)
    sw_dir = omega_s * solar_constant * np.exp(-tau/mysun)

    A_mat, g_vec = rad_calc_SW_RTE_matrix(GR, dtau, gamma1, gamma2, gamma3,
                            swintoa, surf_reflected_SW,
                            albedo_surface_SW, sw_dir)
    fluxes = scipy.sparse.linalg.spsolve(A_mat, g_vec)

    swflxup = fluxes[range(1,len(fluxes),2)]
    swflxdo = fluxes[range(0,len(fluxes),2)]

    return(swflxup, swflxdo)



def rad_calc_SW_RTE_matrix(GR, dtau, gamma1, gamma2, gamma3,
        TOA_SW_in, surf_reflected_SW, albedo_surface, sw_dir):

    g_vec_ = np.repeat(dtau*sw_dir, 2)
    g_vec_[range(0,len(g_vec_),2)] = g_vec_[range(0,len(g_vec_),2)] * gamma3
    g_vec_[range(1,len(g_vec_),2)] = g_vec_[range(1,len(g_vec_),2)] * -(1 - gamma3)
    g_vec = np.zeros(2*GR.nz+2); g_vec[1:-1] = g_vec_
    g_vec[0] = TOA_SW_in; g_vec[-1] = surf_reflected_SW

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

