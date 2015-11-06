"""Solve ODEs of various bubble models
using scipy.integrate.odeint()
or scipy.integrate.ode

H. Soehnholz
01.09.2014
"""

import numpy as np
import matplotlib.pyplot as plt
#import pylab as pl
from scipy.integrate import odeint, ode
from scipy.optimize import brentq

#
# initial conditions and parameters
#
#R0 = 1.036e-3 # Anfangsradius [m]
#v0 = 0. # Anfangsgeschw. [m/s]
#Requ = 100e-6 # Ruheradius [m]
pac = 0. # Schalldruck [Pa]
frequ = 0. # Frequenz [Hz]
T_l = 293. # Wassertemperatur [Kelvin] fuer Modell von Toegel et al.

pstat = 1e5 # statischer Druck [Pa]
rho0 = 998.21 # Dichte der Fluessigkeit [kg / m ^ 3]
mu = 1002e-6 # Viskositaet [Pa * s]
#mu = 653.2e-6 # bei T = 40 degC
#mu = 353.8e-6 # bei T = 80 degC
sigma = 0.07275 # Oberflaechenspannung [N / m]
#sigma = 0.06267 # bei T = 80 degC
Btait = 3213e5 # B aus Tait-Gleichung
ntait = 7. # n aus Tait-Gleichung
kappa = 4./3. # Polytropenexponent
t0 = 1e-6 # Skalierung der Zeit [s]
bvan = 0.0016 # van der Waals Konstante b
#bvan = 0.

# Dampf in der Blase
#pvapour = 7073. # Dampfdruck [Pa]
T0_Kelvin = 273.15 # Ice point [K]

# Gas in der Blase
mu_gas = 0.0000171 # Viskositaet von Luft [Pa * s]
c_p = 1005. # spez. Waerme bei konstantem Druck fuer Luft [J / (kg * K)]
lambda_g = 0.0262 # Waermeleitfaehigkeit von Luft [W / (m * K)]
#T_gas_0 = 273. + T_l # Gastemperatur am Anfang [K]
#print T_gas_0


# Skalierung
#scale_t = T0 # Zeit
#scale_R = Requ # Blasenradius
#scale_U = Requ / T0 # Geschwindigkeit
#scale_p = scale_U * scale_U * rho0 # Druck

# Parameter skalieren
# sc_pstat = pstat / scale_p
# sc_pac = pac / scale_p
# sc_pvapour = pvapour / scale_p
# sc_sigma = sigma / scale_R / scale_p
# sc_mu = mu / scale_R / scale_p * scale_U
# sc_Btait = Btait / scale_p
# sc_frequ = frequ * scale_t
# sc_omega = 2. * np.pi * sc_frequ
# sc_pequ = sc_pstat + 2. * sc_sigma - sc_pvapour
# sc_c0 = np.sqrt((sc_pstat + sc_Btait) * ntait)


# sc_mu_gas = mu_gas / scale_R / scale_p * scale_U
# sc_c_p = c_p / scale_U
# sc_lambda_g = lambda_g / rho0 \
#     / scale_U / scale_U / scale_U / scale_U / scale_t
# sc_Re = np.sqrt(sc_pstat - sc_pvapour) / sc_mu_gas
# sc_Pr = sc_mu_gas * sc_c_p / sc_lambda_g
# sc_Nu = 0.111 * np.sqrt(sc_Re) * sc_Pr ** (1. / 3.)


# Anfangswerte skalieren
# Anfangsgasdruck in der Blase [Pa]
#p0 = (pstat + 2. * sigma / Requ - pvapour) \
#    * ((1. - bvan) / ((R0 / Requ) ** 3. - bvan)) ** kappa

#R0 = R0 / scale_R; # Anfangsradius (skaliert)
#v0 = v0 / scale_U # Anfangsgeschwindigkeit (skaliert)
#p0 = p0 / scale_p # Anfangsgasdruck in der Blase (skaliert)
#p0 = sc_pequ

#print p0

# generate scaled time data
def create_tdata(t_start, t_end, t_step):
#    return np.linspace(t_start / scale_t, t_end / scale_t, \
#                           (t_end - t_start) / t_step)
    return np.arange(t_start / scale_t, t_end / scale_t, t_step / scale_t)

def set_scale(Requ):
    """Determine scaling factors."""
    global scale_t, scale_R, scale_U, scale_p

    scale_t = t0 # time
    scale_R = Requ # bubble radius
    scale_U = Requ / t0 # bubble wall velocity
    scale_p = scale_U * scale_U * rho0 # pressure

    return

def scale_parameters(pvapour_in):
    """Scale parameters according to scaling factors."""

    global sc_pstat, sc_pac, sc_pvapour, sc_Btait
    global sc_sigma, sc_mu
    global sc_pequ
    global sc_frequ, sc_omega
    global sc_c0
    global sc_mu_gas, sc_c_p, sc_lambda_g, sc_Re, sc_Pr, sc_Nu

    sc_pstat = pstat / scale_p
    sc_pac = pac / scale_p
    sc_pvapour = pvapour_in / scale_p
    sc_Btait = Btait / scale_p

    sc_sigma = sigma / scale_R / scale_p
    sc_mu = mu / scale_R / scale_p * scale_U

    sc_pequ = sc_pstat + 2. * sc_sigma - sc_pvapour

    sc_frequ = frequ * scale_t
    sc_omega = 2. * np.pi * sc_frequ

    sc_c0 = np.sqrt((sc_pstat + sc_Btait) * ntait)

    sc_mu_gas = mu_gas / scale_R / scale_p * scale_U
    sc_c_p = c_p / scale_U
    sc_lambda_g = lambda_g / rho0 \
        / scale_U / scale_U / scale_U / scale_U / scale_t
    sc_Re = np.sqrt(sc_pstat - sc_pvapour) / sc_mu_gas
    sc_Pr = sc_mu_gas * sc_c_p / sc_lambda_g
    sc_Nu = 0.111 * np.sqrt(sc_Re) * sc_Pr ** (1. / 3.)

    return

def scale_initconds(R0_in, v0_in, Requ, pvapour):
    """Scale initial conditions according to scaling factors."""

    global R0, v0, p0

    R0 = R0_in / scale_R
    v0 = v0_in / scale_U
    p0 = (pstat + 2. * sigma / Requ - pvapour) \
        * ((1. - bvan) / ((R0 / Requ) ** 3. - bvan)) ** kappa
    p0 = p0 / scale_p

    return

def get_vapour_pressure(T):
    """Vapour pressure of water as a function of the temperature

    Equation from
    W. Wagner und A. Prusz, J. Phys. Chem. Ref. Data 31, 387--535 (2002)
    Section 2.3.1

    Temperature scale: ITS-90
    """

    # Parameters
    pc = 22.064e6 # [Pa]
    Tc = 647.096 # [K]
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    # Conversion degree Celsius -> Kelvin
    #T0_Kelvin = 273.15 # [K]
    T = T + T0_Kelvin

    theta = 1 - T / Tc

    # Compute vapour pressure pv
    # as a function of the temperature T
    pv = pc * np.exp(Tc / T * (a1 * theta \
                                   + a2 * theta ** 1.5 \
                                   + a3 * theta ** 3 \
                                   + a4 * theta ** 3.5 \
                                   + a5 * theta ** 4 \
                                   + a6 * theta ** 7.5))

    return pv

def GilmoreEick_deriv(x, t):
    """Compute one integration step
    using the extended Gilmore equations
    with additional equation for the gas pressure inside the bubble.
    """

    global T

    R = x[0]
    R_dot = x[1]
    pg = x[2]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t);
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t);

    T_gas = T_gas_0 * pg * R ** 3 / sc_pequ
    # if (t < 1.):
    #     print pg
    #     print T_gas
    T = np.append(T, [t, T_gas])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot  = - 3. * kappa * pg * R * R * R_dot \
        / (R ** 3 - bvan) \
        + 1.5 * (kappa - 1.) * sc_lambda_g * sc_Nu \
        * (T_gas_0 - T_gas) / R / R

    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))
    dpg = pg_dot
    return (dR, dR_dot, dpg)

def Gilmore_deriv(x, t):
    """Compute one integration step
    using the Gilmore equations.
    """

    global p_gas

    R = x[0]
    R_dot = x[1]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t);
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t);

    pg = (sc_pstat + 2. * sc_sigma - sc_pvapour) \
    * ((1. - bvan) / (R ** 3. - bvan)) ** kappa
#    print pg
    p_gas = np.append(p_gas, [t, pg])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot  = - 3. * kappa * pg * R * R * R_dot / (R ** 3 - bvan)
    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))

    return (dR, dR_dot)

def GilmoreEick(R0_in, v0_in, Requ, \
                    t_start, t_end, t_step, \
                    T_l = 20.):
    """Run the calculation (Gilmore + Eick)
    with the given initial conditions and parameters.
    returns: t, R, R_dot, pg, T, i
    """

    global T
    global T_gas_0, sc_pvapour

    T_gas_0 = T0_Kelvin + T_l # initial gas temperature inside bubble [K]

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)
    #print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
    #print scale_R, R0

    # solve system of ODEs
    T = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

    xsol, i = odeint(GilmoreEick_deriv, (R0, v0, p0), t_data, \
                     full_output = True)
    
    R = xsol[:, 0] * scale_R
    R_dot = xsol[:, 1] * scale_U
    pg = xsol[:, 2] * scale_p
    t = t_data * scale_t
    T = np.reshape(T, (-1, 2))

#    np.savetxt('GilmoreEick_result.dat', (t / 1e-6, R / 1e-6, R_dot, pg), \
#                   delimiter = '\t')
#    np.savetxt('GilmoreEick_Temp.dat', (T[:, 0], T[:, 1]))

    return (t, R, R_dot, pg, T, i)

#
# nochmal fuer das normale Gilmore-Modell (ohne Erweiterung)
#
def Gilmore(R0_in, v0_in, Requ, \
                t_start, t_end, t_step, \
                T_l = 20.):
    """Run the calculation (Gilmore)
    with the given initial conditions and parameters.
    returns: t, R, R_dot, pg, i
    """

    global p_gas

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)
#    print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
#    print scale_R, R0

    # solve system of ODEs
    p_gas = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

#    print (R0, v0)

    xsol, i = odeint(Gilmore_deriv, (R0, v0), t_data, full_output = True)

    R = xsol[:, 0] * scale_R
    R_dot = xsol[:, 1] * scale_U
    p_gas = np.reshape(p_gas, (-1, 2))
    t = t_data * scale_t

#    np.savetxt('Gilmore_result.dat', (t / 1e-6, R / 1e-6, R_dot))
#    np.savetxt('Gilmore_pg.dat', (p_gas[:, 0], p_gas[:, 1]))

    return (t, R, R_dot, p_gas, i)


def Gilmore_equation(t, x):
    """Compute one integration step
    using the Gilmore equations.
    """

    global p_gas

    R = x[0]
    R_dot = x[1]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t);
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t);

    pg = (sc_pstat + 2. * sc_sigma - sc_pvapour) \
    * ((1. - bvan) / (R ** 3. - bvan)) ** kappa
#    print pg
    p_gas = np.append(p_gas, [t, pg])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot  = - 3. * kappa * pg * R * R * R_dot / (R ** 3 - bvan)
    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))

    return [dR, dR_dot]

def Gilmore_ode(R0_in, v0_in, Requ, \
                t_start, t_end, t_step, \
                T_l=20.):
    """Solve Gilmore ODE in single steps using scipy.integrate.ode
    """
    
    global p_gas

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)
#    print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
#    print scale_R, R0

    # solve system of ODEs
    p_gas = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

#    print (R0, v0)

    #xsol, i = odeint(Gilmore_deriv, (R0, v0), t_data, full_output = True)
    o = ode(Gilmore_equation).set_integrator('dopri5',
#                                             atol=[1e-6, 1e0],
#                                             rtol=[1e-3, 1e-3],
#                                             first_step=1e-9,
#                                             verbosity=1,
                                             )
    o.set_initial_value([R0, v0], t_start)

    nsteps = (t_end - t_start) / t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    i = 0
    R_prev = R0
    growing = False
    while o.successful() and o.t < t_end:
        o.integrate(o.t + t_step)
        print("%g\t%g\t%g" % (o.t, o.y[0], o.y[1]))
        t[i] = o.t * scale_t
        R[i] = o.y[0] * scale_R
        R_dot[i] = o.y[1] * scale_U
        i += 1
        
        if o.y[0] >= R_prev:
            growing = True
#            print('Bubble is growing...')
        elif o.y[0] < R_prev and growing:
            # max. reached
            print('max!')
            
            # decrease Requ (condensation, diffusion)
            R0_in = o.y[0] * scale_R
            v0_in = o.y[1] * scale_U
            Requ = 0.6 * Requ
            set_scale(Requ)
            scale_parameters(pvapour_in)
            scale_initconds(R0_in, v0_in, Requ, pvapour_in)
            o.set_initial_value([R0, v0], o.t)
            
            growing = False
        R_prev = o.y[0]

    plt.figure()
#    plt.axis([0, 100, 0, 600])
    plt.plot(t / 1e-6, R / 1e-6, '.')
    plt.show()
    
#    R = xsol[:, 0] * scale_R
#    R_dot = xsol[:, 1] * scale_U
#    p_gas = np.reshape(p_gas, (-1, 2))
#    t = t_data * scale_t

    return t, R, R_dot


def GilmoreEick_equation(t, x):
    """Compute one integration step
    using the Gilmore--Eick equations.
    """

    global T

    R = x[0]
    R_dot = x[1]
    pg = x[2]

    pinf = sc_pstat - sc_pac * np.sin(sc_omega * t);
    pinf_dot = -sc_pac * sc_omega * np.cos(sc_omega * t);

    T_gas = T_gas_0 * pg * R ** 3 / sc_pequ
    # if (t < 1.):
    #     print pg
    #     print T_gas
    T = np.append(T, [t, T_gas])
    pb = pg + sc_pvapour # Druck in der Blase
    pg_dot  = - 3. * kappa * pg * R * R * R_dot \
        / (R ** 3 - bvan) \
        + 1.5 * (kappa - 1.) * sc_lambda_g * sc_Nu \
        * (T_gas_0 - T_gas) / R / R

    p = pb - (2.* sc_sigma + 4. * sc_mu * R_dot) / R

    p_over_pinf = (p + sc_Btait) / (pinf + sc_Btait)

    H = ntait / (ntait - 1.) * (pinf + sc_Btait) \
        * (p_over_pinf ** (1. - 1. / ntait) - 1.)
    H1 = p_over_pinf ** (- 1. / ntait)
    H2 = p_over_pinf ** (1. - 1. / ntait) / (ntait - 1.) \
        - ntait / (ntait - 1.)
    C = np.sqrt(sc_c0 * sc_c0 + (ntait - 1.) * H)

    dR = R_dot
    dR_dot = (- 0.5 * (3. - R_dot / C) * R_dot * R_dot \
                    + (1. + R_dot / C) * H \
                    + (1. - R_dot / C) * R \
                    * (H1 * (pg_dot \
                                 + (2. * sc_sigma + 4. * sc_mu * R_dot) \
                                 * R_dot / R / R) \
                           + H2 * pinf_dot) / C) \
                           / ((1. - R_dot / C) \
                                  * (R + 4. * sc_mu \
                                         * p_over_pinf ** (-1. / ntait) / C))
    dpg = pg_dot
    return [dR, dR_dot, dpg]

def GilmoreEick_ode(R0_in, v0_in, Requ, \
                    t_start, t_end, t_step, \
                    T_l=20.):
    """Solve Gilmore--Eick ODE in single steps using scipy.integrate.ode.
    Decrease Requ by a factor during each rebound.
    """

    global T
    global T_gas_0, sc_pvapour

    # initial gas temperature inside bubble [K]
    T_gas_0 = T0_Kelvin + T_l

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)

    # solve system of ODEs
    T = np.zeros(0)
#    t_data = create_tdata(t_start, t_end, t_step)

    o = ode(GilmoreEick_equation).set_integrator('dopri5',
#                                             atol=[1e-6, 1e0],
#                                             rtol=[1e-3, 1e-3],
#                                             first_step=1e-9,
#                                             verbosity=1,
                                             )
    o.set_initial_value([R0, v0, p0], t_start)

    nsteps = (t_end - t_start) / t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    pg = np.zeros(nsteps)
    i = 0
    R_prev = R0
    growing = False
    while o.successful() and o.t < t_end:
        o.integrate(o.t + t_step)
#        print("%g\t%g\t%g\t%g" % (o.t, o.y[0], o.y[1], o.y[2]))
        t[i] = o.t * scale_t
        R[i] = o.y[0] * scale_R
        R_dot[i] = o.y[1] * scale_U
        pg[i] = o.y[2] * scale_p
        i += 1
        
        if o.y[0] >= R_prev:
            growing = True
#            print('Bubble is growing...')
        elif o.y[0] < R_prev and growing:
            # max. reached
            print('Max. radius in rebound reached!')
            
            # decrease Requ (condensation, diffusion)
            R0_in = o.y[0] * scale_R
            v0_in = o.y[1] * scale_U
            Requ = 0.60 * Requ
            set_scale(Requ)
            scale_parameters(pvapour_in)
            scale_initconds(R0_in, v0_in, Requ, pvapour_in)
            o.set_initial_value([R0, v0, p0], o.t)
            
            growing = False
        R_prev = o.y[0]

#    plt.figure()
#    plt.axis([0, 100, 0, 600])
#    plt.plot(t / 1e-6, R / 1e-6, '.')
#    plt.show()

    T = np.reshape(T, (-1, 2))

    return t, R, R_dot, pg, T


def Toegel_equation(t, x):
    """Compute one integration step
    using the equations from Toegel et al., Phys. Rev. Lett. 85, 3165 (2000).
    """

    #
    # noch nicht fertig!
    #

    global p_g_prev # letzter Wert fuer Druck in der Blase
    global T_l # Wassertemperatur [Kelvin]
    
    R = x[0]
    R_dot = x[1]
    N = x[2]
    T = x[3]

    # Konstanten
    n_R = 1. # Teilchenzahldichte im Gleichgewicht
    D = 1. # Diffusionskonstante
    chi = 1. # Temperaturleitfaehigkeit (thermal diffusivity)
    k_B = 1. # Boltzmann-Konstante
    c = 1. # Schallgeschwindigkeit

    # Zusammenhang zwischen Ruheradius R0 und Teilchenzahl N
    def f(R_equ):
        return pstat * (1 - 1 / 8.86 ** 3) * R_equ ** 3 \
          + 2 * sigma * (1 - 1 / 8.86 ** 3) * R_equ ** 2 \
          - 3 * N * k_B * T_l / (4 * np.pi)

    # Eine Nullstelle von f(R_equ) finden
    # (Intervall muss angegeben werden!)
    R_equ = brentq(f, 10e-6, 100e-6)
    R_equ_dot = 1. # Wie berechnet man das?
    
    # Teilchenzahl
    l_diff = np.min([np.sqrt(D * R / R_dot), R / np.pi])
    dN = 4 * np.pi * R ** 2 * D \
      * (n_R - N / (4 * np.pi * R ** 3 / 3)) / l_diff

    # Temperatur
    l_th = np.min([np.sqrt(chi * R / R_dot), R / np.pi])
    Q_dot = 4 * np.pi * R ** 2 * lambda_mix * (T_l - T) / l_th
    V_dot = 4 * np.pi * R ** 2 * R_dot
    C_v = 3. * N * k_B
    dT = Q_dot / C_v - p_b * V_dot / C_v \
      + (4. * T_l - 3. * T) * dN * k_B / C_v

    # Druck in der Blase
    p_g = N * k_B * T / ((R ** 3 - (R_equ / 8.86) ** 3) * 4 * np.pi / 3)
    p_g_dot = (p_g - p_g_prev) / dt
    #    p_g_dot = k_B * (dN * T + N * dT) \
#      / (4 * np.pi / 3. * (R ** 3 - R_equ ** 3 / 8.86 ** 3)) \
#      - N * k_B * T / (4 * np.pi / 3.) \
#      * 3. * (R * R * R_dot - R_equ * R_equ * R_equ_dot / 8.86 ** 3) \
#      / (R ** 3 - R_equ ** 3 / 8.86 ** 3) ** 2
          
    p_inf = pstat - pac * np.sin(omega * t);

    dR = R_dot
    dR_dot = (-0.5 * 3. * R_dot * R_dot * (1. - R_dot / (3. * c)) / R \
              + (1. + R_dot / c) * (p_g - p_inf - p_stat) / (rho * R) \
              + p_g_dot / (rho * c) \
              - 4. * mu * R_dot / R / R \
              - 2. * sigma / (rho * R * R)) \
              / (1. - R_dot / c)
# oben schon berechnet:
#    dN = 4 * np.pi * R ** 2 * D \
#      * (n_R - N / (4 * np.pi * R ** 3 / 3)) / l_diff
#    dT = Q_dot / C_v - p_b * V_dot / C_v \
#      + (4. * T0 - 3. * T) * dN * k_B / C_v
    
    return [dR, dR_dot, dN, dT]

def Toegel_ode(R0, v0, N0, T0):
    global p_g_prev
    global T_l
    
    o = ode(GilmoreEick_equation).set_integrator('dopri5',
#                                             atol=[1e-6, 1e0],
#                                             rtol=[1e-3, 1e-3],
#                                             first_step=1e-9,
#                                             verbosity=1,
                                             )
    o.set_initial_value([R0, v0, p0], t_start)

    nsteps = (t_end - t_start) / t_step + 1
    t = np.zeros(nsteps)
    R = np.zeros(nsteps)
    R_dot = np.zeros(nsteps)
    pg = np.zeros(nsteps)
    i = 0
    R_prev = R0
    growing = False
    while o.successful() and o.t < t_end:
        o.integrate(o.t + t_step)
#        print("%g\t%g\t%g\t%g" % (o.t, o.y[0], o.y[1], o.y[2]))
        t[i] = o.t * scale_t
        R[i] = o.y[0] * scale_R
        R_dot[i] = o.y[1] * scale_U
        pg[i] = o.y[2] * scale_p
        i += 1
