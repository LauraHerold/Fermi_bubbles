# calculation of gamma-ray spectra for different processes
import numpy as np
from matplotlib import pyplot
from matplotlib import rc
import scipy
from scipy import special
from scipy import interpolate
import os
import numeric as num
#import sca

#################################################################### constants

home_dir = os.path.expanduser('~')
epsilon = 1.e-15
micro = 1.e-6
Giga = 1.e9
n_H_ref = 0.01
GeV2eV = 1.e9
erg2eV = 6.24e11
erg2GeV = 6.24e2
eV2K = 1.16e4
yr2s = 3.15e7 # 365.*24.*3600.       # seconds in a year
mb2cm2 = 1.e-27 # millibarn to cm^2 conversion

ee = 4.8e-10 # electron charge in Gaussian units
me = 5.11e-4 # (GeV) electron mass
mp = 0.938 # (GeV) proton mass
mpGeV = mp
mpi = 0.135 # (GeV) pi0 mass
sigma_T = 6.65e-25 # (cm^2) Thomson cross section (low-energy photon scattering on single electron)
r02 = 3 * sigma_T / (8. * np.pi)
c_light = 3.e10 # (cm/s) speed of light
hPl_eV = 4.14e-15  # eV s
hPl_GeV = hPl_eV / GeV2eV  # GeV s
#hPl = 6.63e-27 # erg s
hPl_erg = 6.63e-27 # erg s
kB = 1.38e-16 # erg / K
#emuG2GeV2 = 5.92e-27 # eB to GeV^2 conversion
emuG2GeV2 = hPl_erg * c_light * ee * micro * erg2GeV**2

#print emuG2GeV2
#print hPl_erg * c_light * ee * micro * erg2GeV**2
#exit()

#sigmaB = 5.67e-5 # erg / (cm^3 s K^4)                     #?
alpha_fine = ee**2 / (hPl_erg * c_light) * 2 * np.pi

# Stefan-Boltzmann constant for black body radiation density
aB = 4. * np.pi * 12. * special.zeta(4. ,1.) * kB**4 / (c_light * hPl_erg)**3
aB_eV = 4. * np.pi * 12. * special.zeta(4. ,1.) / (c_light * hPl_eV)**3
#aB = 7.566e-15 # erg/cm^3/K^4
#aB_eV = aB * erg2eV * eV2K**4 # ev/cm^3/ev^4

################################################################# define functions

step = lambda x: (1. + np.sign(x))/2. # returns 1 if x>0, 0.5 if x==0, 0 if x<0
# def step(x): return (1. + np.sign(x))/2.
positive = lambda x: x * step(x) # returns x if x>0 else 0

def plaw_cut(pars): # power law with cutoff a * x^b * exp(-x/c) with parameters a,b,c
    return lambda x: pars[0] * x**pars[1] * np.exp(-x/max(pars[2],1))

#def Es2dEs(Es):                                    #?
#    f = np.sqrt(Es[1] / Es[0])
#    return Es * (f - 1/f)


#def Es2dlogEs(Es):                                 #?
#    f = np.sqrt(Es[1] / Es[0])
#    return (f - 1/f) * np.ones_like(Es)



#############################################################################
#                                                                           #
#          Inverse Compton Scattering                                       #
#                                                                           #
#############################################################################


def sigmaIC(E_g, E_irf, E_e): ################## Extreme Klein-Nishina cross section (large fraction of energy emitted in one photon)
    '''
    IC cross section
    Blumenthal and Gould 1970 equation (2.48)
    INPUT:
        E_g - number: final photon energy (GeV) 
        E_irf - array_like, shape (n,) or a number: radiation field energies (eV)
        E_e - array_like, shape (k,) or a number: electron distribution energies (GeV)
    OUTPUT:
        sigma - array_like, shape (n, k): scattering cross section
    
    '''
    # define useful parameters
    E_irf_GeV = E_irf / GeV2eV
    if not isinstance(E_irf_GeV, np.ndarray):
        b = 4 * E_irf_GeV * E_e / me**2
    else:
        b = 4 * np.outer(E_irf_GeV, E_e) / me**2
    z = E_g / E_e
    z = np.minimum(z, 1-epsilon)
    x = z / b
    q = x / (1 - z)


    # define the mask to satisfy the conditions for E_g > E_irf and E_g < E_g_max = E_e b / (1 + b)
    if isinstance(E_irf_GeV, np.ndarray):
        if E_irf_GeV[-1] < E_g:
            irf_mask = 1.
        else:
            irf_mask = np.outer(step(E_g - E_irf_GeV), np.ones_like(E_e))
    else:
        if E_irf_GeV < E_g:
            irf_mask = 1.
        else:
            return 0.

    
    E_g_max = E_e * b / (1. + b)
    e_mask = step(E_g_max - E_g)
    
    tot_mask = irf_mask * e_mask


    # calculate cross section (Blumenthal and Gould 1970)
    brackets = 2. * q * np.log(q) + (1 + 2 * q) * (1 - q) \
            + 0.5 * z**2 / (1 - z) * (1 - q)


    return 3 * sigma_T * x * brackets * tot_mask

def thomson_b(U): # energy loss rate dE/dt of particles, U: energy density of incoming radiation
    return 4./3. * sigma_T * c_light * U * 1.e-9 / me**2 # Longair 9.41: b(U) =  4/3 sigma_T * c * U * (v^2/c^2) * gamma_Lorentz

def cooling_time0(U, E): # E(t) = t * b(U) --> E(T_cool) = 1\E --> T_cool = 1/E/b(U)
    return 1. / thomson_b(U) / E

def cooling_time(E_e, EdNdE_irf, E_irf): # cooling time if energy loss of particles depends on U_irf
    return E_e / ICS_Edot(E_e, EdNdE_irf, E_irf)


########################################################################## Use this for lifetime calculations:

#def ICS_Edot(E_e, EdNdE_irf, E_irf, gamma_spectrum=False):
def ICS_Edot(E_e, EdNdE_irf, E_irf): # = dE/dt (Longair 9.33)
    f = np.sqrt(E_irf[1] / E_irf[0]) # (f-1/f) \approx log(f)
    dE_irf = E_irf * (f - 1/f) # size of one energy bins

    E_gb = np.logspace(np.log10(E_e) - 10., np.log10(E_e), 101) # logspace of final photon energies (E_e /10^10, E_e, in 101 steps)
    E_g = np.sqrt(E_gb[1:] * E_gb[:-1])
    dE_g = E_gb[1:] - E_gb[:-1]

    sgm = np.zeros_like(E_irf)

    for i in range(len(E_g)):
        sgm += sigmaIC(E_g[i], E_irf, E_e)[:,0] * dE_g[i]
    return c_light * np.sum(sgm * EdNdE_irf/E_irf * dE_irf)


def IC_gamma_spectrum(EdNdE_irf, E_irf, E_e): # spectrum of one electron, not used
    dLogE_irf = np.log(E_irf[1] / E_irf[0]) * np.ones_like(E_irf)
    dN_irf = EdNdE_irf * dLogE_irf

    def EdNdE_gamma(E_g): # returns function of spectrum EdN/dE_gamma(E_gamma)
        sigma = sigmaIC(E_g, E_irf, E_e)
        return c_light * np.dot(dN_irf, sigma)
    return EdNdE_gamma #  = sigma * c * density of IRF 


def IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e): # spectrum of distribution of electrons
    '''
    calculate the IC emissivity given the spectrum of IRF and electrons
    INPUT:
    OUTPUT:
        EdN/dE function in units of [ph / cm^3 s]
    '''
    dlogE_e = np.log(E_e[1] / E_e[0]) * np.ones_like(E_e)
    dN_e = EdNdE_e * dlogE_e # dN = E dN/dE * dlogE = dN/dlogE * dlogE

    dLogE_irf = np.log(E_irf[1] / E_irf[0]) * np.ones_like(E_irf)
    dN_irf = EdNdE_irf * dLogE_irf

    def EdNdE_gamma(E_g):
        sigma = sigmaIC(E_g, E_irf, E_e)
        return c_light * np.dot(dN_irf, np.dot(sigma, dN_e))
    
    return EdNdE_gamma # = sigma * c * density of IRF * density of electrons (per energy)


################################################################################# Thermal radiation


def U_black_body(T): # energy density radiated from black body
    '''
    INPUT:
        T - temperature (eV)
    OUTPUT:
        U - energy density of black body radiation (eV/cm^3)
    '''
    return aB_eV * T**4 # Sfefan-Boltzmann law / T^4-Gesetz


def thermal_spectrum(T, U=None): # thermal photon spectrum of black body with temp T
    '''
    calculate thermal spectrum of photons
    INPUT:
        T - temperature (eV)
        U - energy density of radiation (eV / cm^3) (nu dU / d nu)
    OUTPUT:
        function(Eg) - spectrum of photons as a function of Eg in eV

    '''
    W0 = U_black_body(T) # energy density in the black body with temperature T
    #print 'energy density of CMB ev / cm^3'
    #print W0
    #exit()
    if U is None:
        U = W0
    def func(Eg): # Plancksches Strahlungsgesetz N(nu == v) 8 * pi * v^2 / c^3 * hv/(exp(hv/k_b/T)-1)
        k = Eg / hPl_eV / c_light

        expf = 1 / (np.exp(Eg/T) - 1) # T[ev] == k_B * T[K]
        return 8. * np.pi * Eg * k**3 * expf * U / W0
    return func # --> U(v) = N(v) * v in eV/cm^3


#############################################################################
#                                                                           #
#           synchrotron spectrum                                            #
#                                                                           #
#############################################################################

def nu_crit_norm(B): # critical frequency for electron velocity perpendicular to B-field (alpha = pi) 
    norm = 3 * emuG2GeV2 * B / (4*np.pi * me**3 * hPl_GeV) # Longair 8.54
    #norm = 3 * ee * micro * B / (4*np.pi * me**3) * c_light * erg2GeV
    return norm

def nu_critical(B, sin_al, E): # critical frequency, Longair 8.54
    '''
    critical frequency
    INPUT:
        B - magnetic field (micro Gauss)
        sin_al = sin(alpha),
            where alpha - angle between magnetic field and electron velocity
        E - electron energy (GeV)
    OUTPUT:
        critical frequency (Hz)

    '''
    norm = nu_crit_norm(B)
    
    if type(E) is np.ndarray and type(sin_al) is np.ndarray:
        return norm * np.outer(sin_al, E**2)
    else:
        return  norm * sin_al * E**2 # E/m == gamma_Lorentz

def E_critical(B, sin_al, nu): # critical energy
    '''
    energy corresponding to a critical frequency
    INPUT:
        B - magnetic field (micro Gauss)
        sin_al = sin(alpha),
            where alpha - angle between magnetic field and electron velocity
        nu - critical frequency (Hz)
    OUTPUT:
        E - electron energy (GeV)

    '''
    norm = nu_crit_norm(B)
    return np.sqrt(nu / (norm * sin_al))



def bessel_int_values():
    #dr = 0.00005
    #rs = np.arange(0. + dr/2., 5., dr)
    rsb = np.logspace(-7., 1., 10001)
    rs = np.sqrt(rsb[1:] * rsb[:-1])
    dr = (rsb[1:] - rsb[:-1])
    

    nn = 5./3.
    ks = scipy.special.kv(nn, rs)
    k_int = np.zeros_like(ks)

    k_int[0] = ks[0] * dr[0]
    for i in range(1, len(ks)):
        k_int[i] = k_int[i-1] + ks[i] * dr[i]
    k_tot = np.sum(ks * dr)
    k_int = k_tot - k_int

    return rs, k_int

rs, k_int = bessel_int_values()

func0 = scipy.interpolate.interp1d(rs, k_int, fill_value=0.)
def func(r):
    if r > 2.e-7 and r < 5.:
        return func0(r)
    else:
        return 0.

bessel_int = np.frompyfunc(func, 1, 1)

def synch_norm(B):
    '''
    normalization factor in synchrotron energy loss
    INPUT:
        B - magn field (micro Gauss)
    OUTPUT:
        normalization factor (GeV / s)

    '''
    return np.sqrt(3.) * ee**3 * micro * B / me * erg2GeV**2


def synch_power(nu, B, EdNdE_e, E_e, dlogE_e=None):
    '''
    synchrotron power for isotropic distribution of electrons
    INPUT:
        nu - synchrotron frequency (Hz)
        B - magnetic field (micro Gauss)
        EdNdE_e - electron spectrum dN / dV dlogE (1/cm^3)
        E_e - electron energies (GeV)
    OUTPUT:
        synchrotron spectrum - dW/dnu dt (GeV / Hz s cm^3)
        
    '''
    
    # vector of sin(alpha) dOm = 2pi sin(alpha) d cos(alpha)
    dcos_al = 0.02
    cos_al = np.arange(-1.+ dcos_al/2, 1., dcos_al)
    sin_al = np.sqrt(1. - cos_al**2)
    sin_al_dOm = 2. * np.pi * sin_al * dcos_al

    # electron density vector
    if len(E_e) > 1:
        dlogE_e = np.log(E_e[1] / E_e[0]) * np.ones_like(E_e)
    elif len(E_e) == 1 and dlogE_e == None:
        print 'in synch_power either length of E_e should be larger than 1 or dlogE_e should be defined'
        exit()
        
    dN_e = EdNdE_e * dlogE_e

    # find the matrix of critical frequencies
    nuc = nu_critical(B, sin_al, E_e)
    
    S_mat = nu / nuc * bessel_int(nu / nuc)

    norm = synch_norm(B) / (4. * np.pi)

    return norm * np.dot(sin_al_dOm, np.dot(S_mat, dN_e))

def synch_Edot(B, sin_al, E_e):
    '''
    calculate the synchrotron energy loss for an electron with energy E_e
    INPUT:
        B - magn field (micro Gauss)
        sin_al - sin of alpha (pitch angle)
        E_e - electron energy (GeV)
    OUTPUT:
        Edot - energy loss

    '''
    nuc = nu_critical(B, sin_al, E_e)
    print 'nuc = %.2e GHz' % (nuc / Giga)
    nus = np.logspace(np.log10(nuc)-7., np.log10(nuc)+1., 200)
    f = np.sqrt(nus[1] / nus[0]) # size of energy bin
    d_nus = nus * (f - 1./f)
    SS = nus / nuc * bessel_int(nus / nuc)

    return synch_norm(B) * np.sum(SS * d_nus) # dE/dt = normalization_factor * 

def B_field_Edensity(B): # energy density of B-field U_B = B^2/mu_0 (in SI units, Longair p. 195)
    '''
    magnetic field energy density
    INPUT:
        B - magn field (micro Gauss)
    OUTPUT:
        dE / dV - magn field energy density (ev / cm^3)
        
    '''
    return (micro * B)**2 / (8. * np.pi) * erg2eV

def synch_Edot_Thomson(B, sin_al, E_e): # dE/dt in ultra relativistic limit (Thomson approx.)
    '''
    calculate the synchrotron energy loss for an electron with energy E_e in Thomson approximation
    INPUT:
        B - magn field (micro Gauss)
        sin_al - sin of alpha (pitch angle)
        E_e - electron energy (GeV)
    OUTPUT:
        Edot - energy loss (GeV / s)

    '''
    U_B = (micro * B)**2 / (8. * np.pi) # energy density of B-field U_B = B^2/mu_0 (in SI units, Longair p. 195)
    print U_B * erg2eV
    return 2. * sigma_T * c_light * U_B * sin_al**2 * (E_e / me)**2 * erg2GeV # Longair 8.8 (E_e/me == gamma_Lorentz)



#############################################################################
#                                                                           #
#           Bremsstrahlung                                                  #
#                                                                           #
#############################################################################


def brems_spectrum(EdNdE_e, E_e, N_atom=1.):
    '''
    calculate the brems emissivity given the spectrum electrons and density of atoms
    INPUT:
        EdNdE_e - electron spectrum dN / dV dlogE (1/cm^3)
        E_e - electron energies (GeV)
        N_atom - density of atoms (1/cm^3)

    OUTPUT:
        EdN/dE function in units of [ph / cm^3 s]
    '''
    dlogE_e = np.log(E_e[1] / E_e[0]) * np.ones_like(E_e)
    dN_e = EdNdE_e * dlogE_e

    def EdNdE_gamma(E_g):
        sigma = sigmaBrems(E_g, E_e)
        return c_light * N_atom * np.dot(sigma, dN_e)
    
    return EdNdE_gamma


def sigmaBrems(E_g, E_e):
    E_f = np.abs(E_e - E_g) + epsilon
    x = E_f / E_e
    factor = np.log(2. * E_e * E_f / (me * E_g)) - 0.5
    factor *= step(factor) * step(E_e - E_g)
    return 4. * r02 * alpha_fine * (1. + x*x - 2. * x / 3.) * factor
    
    


#############################################################################
#                                                                           #
#           hadronic spectrum                                               #
#                                                                           #
#############################################################################


def sigma_pp(E_kin): # proton-proten cross-section
    '''
    compute pp cross section (from Aharonian book)
    INPUT:
        Ekin - kinetic energy of the proton Ekin = E - m (GeV)
    OUTPUT:
        scattering cross section (cm^2)
    '''
    E_cut = 1.
    sigma = 30. * (0.95 + 0.06 * np.log(E_kin)) * step(E_kin - E_cut)
    return sigma * mb2cm2
    

def pi0_spectrum(dNdp_p, p_p, n_H=1.):
    kappa_pi = 0.17
    dp_p = p_p[1:] - p_p[:-1]
    dNdp_p = np.sqrt(dNdp_p[1:] * dNdp_p[:-1])
    
    E_p = np.sqrt(p_p**2 + mp**2)
    E_p = np.sqrt(E_p[1:] * E_p[:-1])
    E_kin = E_p - mp
    kin_mask = step(E_kin**2 / kappa_pi**2 - mpi**2)

    p_pi = np.sqrt(positive(E_kin**2 / kappa_pi**2 - mpi**2)) + epsilon

    def EdNdE_gamma(E_g):
        E_pi_min = E_g + mpi**2 / (4 * E_g)
        E_p_min = mp + E_pi_min / kappa_pi
        dNpi = c_light * n_H * sigma_pp(E_kin) * dNdp_p * dp_p
        mask = kin_mask * step(E_p - E_p_min)
        return E_g * 2 * np.sum(dNpi / p_pi * mask)

    return EdNdE_gamma

# calculation of the pp to gamma, elec, positron etc. using cparamlib
data_dir = 'pp_data/'
Tp = np.load(data_dir + 'Tp.npy')

IDs = range(7)
ID_dict = {0:'gamma',
         1:'elec',
         2:'posi',
         3:'nue',
         4:'numu',
         5:'antinue',
         6:'antinumu'
         }

pp_csec = {}
for ID in IDs:
    xx = np.load(data_dir + 'x_%s.npy' % ID_dict[ID])
    EsigmaE = np.load(data_dir + 'EsigmaE_%s.npy' % ID_dict[ID])
    pp_csec[ID] = num.interpolate_linear2d(Tp, xx, EsigmaE)

def Tp2pp(Tp): # Kinetic energy to momentum ?
    return np.sqrt(Tp*(Tp+2*mpGeV))

def EdQdE_pp(dNdp_p, p_p, n_H=1., ID_PARTICLE=0):
    '''
    calculate pp to PARTICLE source function
    INPUT:
        dNdp_p - array_like, shape (n,):
            proton density dN / dp, where p is the momentum (1/GeV/cm^3)
        p_p - array_like, shape (n,):
            proton momenta (GeV)
        n_H - float:
            target gas density (1/cm^3)
        ID_PARTICLE - int:
            particle ID: gamma = 0, electron = 1, positron = 2 etc.
    OUTPUT:
        E dQ/ dE - function of energy:
            spectrum of produced particles (1/cm^3/s)
    '''
    dNdp_p = np.sqrt(dNdp_p[1:] * dNdp_p[:-1])
    E_p0 = np.sqrt(p_p**2 + mp**2)
    T_p0 = E_p0 - mp
    
    
    E_p = np.sqrt(E_p0[1:] * E_p0[:-1])
    
    # proton CR kinetic energy
    T_p = E_p - mp
    dT_p = T_p * np.log(T_p0[1:]/T_p0[:-1])
    

    def EdNdE(EE):
        csec = lambda TT: pp_csec[ID_PARTICLE](TT, EE/TT) * step(TT - EE)
        csec_vec = np.frompyfunc(csec, 1, 1)
        
        res = c_light * n_H * np.sum(csec_vec(T_p) * dNdp_p * dT_p)
        return res

    func_vec = np.frompyfunc(EdNdE, 1, 1)

    return func_vec


def pi0_sp_tune(index, cutoff=np.inf):

    Tp = np.logspace(0., 5., 100)
    p_p = Tp2pp(Tp)
    if cutoff == np.inf:
        dNdp_p = p_p**index
    else:
        dNdp_p = p_p**index * np.exp(-p_p/cutoff)
    func = EdQdE_pp(dNdp_p, p_p, n_H=n_H_ref, ID_PARTICLE=0)

    return np.frompyfunc(func, 1, 1)



################################################################################## test

if __name__ == '__main__':

    #Set up figure
    #Plotting parameters
    fig_width = 8   # width in inches
    fig_height = 7  # height in inches
    fig_size =  [fig_width, fig_height]
    params = {'axes.labelsize': 18,
              'font.size': 18,
              'legend.fontsize': 12,
              'xtick.labelsize':13,
              'ytick.labelsize':13,
              #'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.minor.size' : 3,
              'ytick.minor.size' : 3,
              'figure.subplot.left' : 0.14,
              'figure.subplot.right' : 0.92,
              'figure.subplot.bottom' : 0.12
                }
    pyplot.rcParams.update(params)
    rc('text.latex', preamble=r'\usepackage{amsmath}')

    E_e = 30.
    U_CMB = 0.26

    test_e = 0 # plot spectra of gamma, e+, e- created in hadronic collisions
    test_pi0 = 0 # plot pi0 gamma-ray spectrum from a power-law with a cutoff proton CR spectrum
    test_Wtot = 0 # compare Thomson E loss with full IC E loss on CMB
    test_synch = 0 # calculate synchrotron spectrum
    test_sigmaIC = 0 # test IC scattering cross section
    test_spectra = 1 # calculate IC spectra for three thermal fields: 1.e-4, 1.e-2, and 1. eV
    
    
    if test_e:
        pyplot.figure()
        
        Tps = np.logspace(0., 3., 3)
        xs = np.logspace(-5., 0., 70)
        ID_check = range(0, 3)
        lss = {0:'-', 1:'--', 2:':', 3:'-.'}
        for Tp0 in Tps:
            for ID in ID_check:
                csec_fn = lambda xx: pp_csec[ID](Tp0, xx)
                csec_vec = np.frompyfunc(csec_fn, 1, 1)

                label = "%s, Tp = %.1f GeV" % (ID_dict[ID], Tp0)
                pyplot.loglog(xs * Tp0, csec_vec(xs) / mb2cm2, ls=lss[ID%len(lss)], label=label)

                
        lg = pyplot.legend(loc = 'upper left', ncol=2)
        lg.get_frame().set_linewidth(0)  #To get rid of the box

        pyplot.ylim(1.e-3, 1.e4)

        pyplot.show()

        


    if test_pi0:
        E_g = 10.**np.arange(-2., 4., 0.1)
        
        p_p = 10**np.arange(-0.5, 5, 0.02)
        pars_p = [1., -2., 1000.]
        dNdp_p = plaw_cut(pars_p)(p_p)
        
        #EdNdE_gamma = pi0_spectrum(dNdp_p, p_p)
        EdNdE_gamma = EdQdE_pp(dNdp_p, p_p)
        
        EdNdE_gamma_vec = np.frompyfunc(EdNdE_gamma, 1, 1)
        gamma_spec = E_g*EdNdE_gamma_vec(E_g)

        pyplot.figure()

        pyplot.loglog(E_g, gamma_spec, label='')

        pyplot.xlabel(r"$\rm E_\gamma\; (GeV)$")
        ylabel = r'${E_\gamma}^2\frac{d N}{d E_\gamma}$'

        
        ymax = np.max(gamma_spec) * 10
        ymin = np.max(gamma_spec) / 100
        #pyplot.ylim(ymin, ymax)
        pyplot.title('pi0 gamma-ray spectrum')


    if test_Wtot:
        T = 2.35e-4 # eV
        U = U_CMB
        E_e = 1.e3
        E_irf = np.logspace(-6, -2, 101)

        Wtot_check = thomson_b(U) * E_e**2 

        EdNdE_irf = thermal_spectrum(T)(E_irf)/E_irf
        Wtot = ICS_Edot(E_e, EdNdE_irf, E_irf)
        
        print 'Edot = %.2e ' % Wtot
        print 'Thomson Edot = %.2e ' % Wtot_check
        print 't cool= %.2e yr' % (cooling_time(E_e, EdNdE_irf, E_irf) / yr2s)
        print 'Thomson t cool= %.2e yr' % (cooling_time0(U, E_e) / yr2s)
        
        

    if test_synch:
        dr = 0.001
        rs = np.arange(0. + dr/2., 4.5, dr)


        # synchrotron spectrum as a function of nu / nu_c
        if 0:
            pyplot.figure()
            ys = 9. * np.sqrt(3.) / (8. * np.pi) * rs * bessel_int(rs)
            pyplot.loglog(rs, ys)
            pyplot.loglog(rs, 1.333 * rs**(0.333), ls='--')
            pyplot.loglog(rs, 0.777 * np.sqrt(rs) / np.exp(rs), ls=':')

        #pyplot.show()


        # synchrotron spectrum for a population of electrons
        Es_e = 10.**np.arange(-1., 4.001, 0.1) # 100 MeV to 10 TeV
        pars_e = [1., -.8, 1000.]
        EdNdE_e = plaw_cut(pars_e)(E_e)
        #B = np.sqrt(U_CMB * 8. * np.pi / erg2eV) / micro
        B = 3 # muG

        sin_al = 1.
        Edot = synch_Edot(B, sin_al, E_e)
        Edot_Th = synch_Edot_Thomson(B, sin_al, E_e)
        print 'Synch E loss at %i GeV = %.3e GeV / s' % (E_e, Edot)
        print 'Synch E loss at %i GeV = %.3e GeV / s (Thomson)' % (E_e, Edot_Th)

        #pyplot.show()
        #exit()

        
        nus = np.logspace(0., 9., 50) * 1.e9 # Hz
        
        synch = lambda nu: synch_power(nu, B, EdNdE_e, Es_e)
        synch_vec = np.frompyfunc(synch, 1, 1)
        
        W_synch = synch_vec(nus)

        pyplot.figure()
        pyplot.loglog(hPl_eV * nus, GeV2eV * nus * W_synch)
        pyplot.xlabel(r'$\rm E\; (eV)$')
        ylabel = r'$\nu F_\nu\; \left({\rm \frac{eV}{cm^3\, s}} \right)$'
        pyplot.ylabel(ylabel)

        #pyplot.show()
        #exit()


        
        
    if test_sigmaIC:

        #E_g = 0.1
        #E_irf = 0.01
        #E_e = 100.
        #print sigmaIC(E_g, E_irf, E_e)

        
        E_g = .1 # GeV final gamma-ray energy
        E_irf = 10**np.arange(-4., 1.001, 1.) # IRF energies: 0.0001 eV to 10 eV
        E_e = 10.**np.arange(0., 4.001, 0.1) # electron energies: 100 MeV to 10 TeV
        sigmas = sigmaIC(E_g, E_irf, E_e) / sigma_T # cross sections
        
        pyplot.figure()

        for i in range(len(E_irf)):
            pyplot.loglog(E_e, sigmas[i], label='E\_irf = %.1e' % E_irf[i])
        
        lg = pyplot.legend(loc = 'upper right', ncol=2)
        lg.get_frame().set_linewidth(0)  #To get rid of the box

        pyplot.xlabel(r'$\rm E_e\; (GeV)$')
        ylabel = r'$\frac{E_\gamma}{\sigma_T} \frac{d \sigma_{\rm IC}}{d E_\gamma}$'
        pyplot.ylabel(ylabel)

        ymax = 10.
        ymin = 0.01
        pyplot.ylim(ymin, ymax)
        pyplot.xlim(E_e[0], E_e[-1])

        

    if test_spectra:

        #Ts = np.array([1.e-4, 1.e-2, 1.]) # starlight spectrum with 1eV temperature
        Ts = 10.**np.arange(-4., 1., 2.)
        nTs = len(Ts)

        Us = np.ones(nTs)

        E_irf = 10**np.arange(-6., 4.001, 0.1) # eV
        k = np.sqrt(E_irf[1] / E_irf[0])
        dE = (k - 1/k) * E_irf


        E_e = 10.**np.arange(-1., 4.001, 0.1) # 100 MeV to 10 TeV
        pars_e = [1., -1., 500.]
        EdNdE_e = plaw_cut(pars_e)(E_e)
        
        E_g = 10.**np.arange(-2., 4., 0.1)


        pyplot.figure()

        for i in range(nTs):

            irf_sp = thermal_spectrum(Ts[i], Us[i])
            EdNdE_irf = irf_sp(E_irf) / E_irf

            print 'EdNdE_irf:' + str(EdNdE_irf.shape)
            print 'E_irf:' + str(E_irf.shape)
            print 'EdNdE_e:' + str(EdNdE_e.shape)
            print 'E_e:' + str(E_e.shape)
            
            EdNdE_gamma = IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)

            EdNdE_gamma_vec = np.frompyfunc(EdNdE_gamma, 1, 1)

            gamma_spec = E_g * EdNdE_gamma_vec(E_g)


            pyplot.loglog(E_g, gamma_spec, label='T = %.1e eV' % Ts[i])

        lg = pyplot.legend(loc = 'upper right', ncol=2)
        lg.get_frame().set_linewidth(0)  #To get rid of the box


        pyplot.ylim(1.e-20, 1.e-11)
        #pyplot.xlim(E_irf[0], E_irf[-1])

    
    
    pyplot.show()
    exit()
        



