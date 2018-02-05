""" Plots the SED of all latitude stripes necessary to observe the Fermi bubbles. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

import dio
from yaml import load
import gamma_spectra

########################################################################################################################## Parameters

low_energy_range = 3                                           # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)
lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]

input_data = 'lowE'                                            # data, lowE, boxes, GALPROP
data_class = 'source'

fit_plaw = True
fit_IC  = False
fit_pi0 = False

cutoff = False

bin_start_fit = 6                                              # Energy bin where fit starts
fitmin = 3
fitmax = 17                                                    # bins: 0-15 for low-energy range 3, 0-17 else

fn_ending = '_nocutoff.pdf'
colours = ['blue', 'red']


########################################################################################################################## Constants

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575                                                              # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/Low_energy_range' + str(low_energy_range) +'/'

c_light = 2.9979e8                                                                      # m/s speed of light
h_Planck = 4.1357e-15                                                                   # eV * s Planck constant
kB = 8.6173303e-5                                                                       # eV/K
T_CMB = 2.73 * kB                                                                       # CMB temperature

ISFR_heights = [10, 10, 5, 5, 2, 1, 0.5, 0, 0.5, 1, 2, 5, 5, 10, 10]
E_e = 10.**np.arange(-1., 8.001, 0.1)                                                   # Electron-energies array (0.1 - 10^8 GeV)
p_p = 10**np.arange(-0.5, 6., 0.1)                                                      # Proton-momenta array

########################################################################################################################## Load dictionaries

dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) +'/dct_' + input_data + '_counts_' + data_class + '.yaml')

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']

nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = len(diff_profiles[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)
fitmax = min(nE-1, fitmax)
print 'fitmax: ' + str(fitmax)
#E_g = Es                                                                                # Final photon energies array in GeV
 

expo_dct = dio.loaddict('dct/Low_energy_range0/dct_expo_' + data_class + '.yaml')
expo_profiles = expo_dct['6) Exposure_profiles'][0:nE] # shape: (nB, nL, nE)
print "expo_profiles shape: " + str(len(expo_profiles)) + ", " + str(len(expo_profiles[0])) + ", " + str(len(expo_profiles[0][0])) + ", "
deltaE = expo_dct['8) deltaE']
dOmega = expo_dct['7) dOmega_profiles']


########################################################################################################################## Define likelihood class and powerlaw fct

class likelihood:                                                         # fct = sum over pixel in one lat stripe: k * low + c - high * log(k * low + c)
    def __init__(self, model_fct, data):
        self.model_fct = model_fct
        self.data = data
        print data 
    def __call__(self, N_0, gamma):                                             # c: isotropic background, k: fit low energies to high energies
        model_fct = self.model_fct
        data = self.data
        L = sum(model_fct(N_0, gamma)(E) - data[E] * np.log(model_fct(N_0, gamma)(E)) for E in range(binmin,binmax))
        return L
    def __call__(self, N_0, gamma, Ecut_inv):                                             # c: isotropic background, k: fit low energies to high energies 
        model_fct = self.model_fct
        data = self.data
        L = sum(model_fct(N_0, gamma, Ecut_inv)(E) - data[E] * np.log(model_fct(N_0, gamma, Ecut_inv)(E)) for E in range(fitmin,fitmax-5))
        return L

def plaw(N_0, gamma, Ecut_inv = 0.):  # powerlaw
    return lambda E: N_0 * Es[E]**(-gamma) * np.exp(-Es[E] * Ecut_inv)

########################################################################################################################## Define particle-spectra functions

E_zero = Es[bin_start_fit]

for b in xrange(nB):
    print Bc[b]
    pyplot.figure()
    colour_index = 0

    
    IRFmap_fn = '../../data/ISRF_flux/Standard_0_0_' + str(ISFR_heights[b]) + '_Flux.fits.gz'   # Model for the ISRF
    hdu = pyfits.open(IRFmap_fn)                                                                # Physical unit of field: 'micron'
    wavelengths = hdu[1].data.field('Wavelength') * 1.e-6                                       # in m
    E_irf_galaxy = c_light * h_Planck / wavelengths[::-1]                                       # Convert wavelength in eV, invert order
    EdNdE_irf_galaxy = hdu[1].data.field('Total')[::-1] / E_irf_galaxy                          # in 1/cm^3. Since unit of 'Total': eV/cm^3
    dlogE_irf = 0.0230258509299398                                                              # Wavelength bin size

    E_irf = np.e**np.arange(np.log(E_irf_galaxy[len(E_irf_galaxy)-1]), -6.* np.log(10.), -dlogE_irf)[:0:-1] # CMB-energies array with same log bin size as IRF_galaxy in eV
    irf_CMB = gamma_spectra.thermal_spectrum(T_CMB)                                             # Use thermal_spectrum from gamma_spectra.py, returns IRF in eV/cm^3
    EdNdE_CMB = irf_CMB(E_irf) / E_irf                                                          # in 1/cm^3

    EdNdE_irf = EdNdE_CMB + np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), EdNdE_irf_galaxy) # Differential flux in 1/cm^3 

    
    for l in xrange(nL):

        def IC_model(N_0, gamma, E_cut_inv = 0.):
            EdNdE_e = plaw([N_0, gamma, E_cut_inv])(E_e) # E_cut/c_light???
            EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
            EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
            return lambda E: EdNdE_gamma_IC_vec[E] * exposure_profiles[(b,l,E)] * dOmega[(b,l)] * deltaE[E] / Es[E]

        def pi0_model(N_0, gamma, E_cut_inv):
            dNdp_p = plaw([N_0, gamma, E_cut_inv])(p_p)
            EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
            EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
            return lambda E: EdNdE_gamma_pi0_vec[E] * exposure_profiles[(b,l,E)] * dOmega[(b,l)] * deltaE[E] / Es[E]
        

########################################################################################################################## Plot SED
        
        map  = np.asarray(diff_profiles[b][l])
        std_map = np.asarray(std_profiles[b][l])
        std_first_fit =  np.asarray(std_profiles[b][l])

        label = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2)
        pyplot.errorbar(Es, map, std_map, color=colours[colour_index], marker='s', markersize=4, markeredgewidth=0.4, linestyle = '', linewidth=0.1, label=label)



########################################################################################################################## Fit spectra



        if fit_plaw:
            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -      Powerlaw      -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "
            
            N_0, gamma = 100., 2.
            fit = likelihood(model_fct = plaw, data = map[binmin:binmax])                                                          # Fit model = (lowE * k + c) to highE
            print "Test"
            m = Minuit(fit, N_0 = N_0, gamma = gamma)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]
            label = r'$\mathrm{PL}:\ \gamma = %.2f, $' %gamma
            
            if cutoff:
                N_0, gamma, Ecut_inv = 100., 2., 0.
                fit = likelihood(plaw, map[binmin:binmax])                                                          # Fit model = (lowE * k + c) to highE
                m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv)
                m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                label = r'$\mathrm{PL}:\ n = %.2f,$ ' %(-2-gamma) + r'$E_{\mathrm{cut}} = %.1e\ \mathrm{GeV},$ ' %E_cut
                
            pyplot.errorbar(Es[binmin:binmax], [plaw(N_0, gamma)(E) for E in xrange(binmin,binmax)], label = label, color = colours[colour_index])

                

                 


        if fit_IC:
            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -  -  -          IC         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "
            

            N_0, gamma = 100., 2.
            fit = likelihood(IC_model, map[binmin:binmax])                                                          # Fit model = (lowE * k + c) to highE
            m = Minuit(fit, N_0, gamma)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]
            label  = r'$\mathrm{IC}:\ n = %.2f,$ ' %gamma
            
            if cutoff:
                N_0, gamma, E_cut_inv = 100., 2., 0.
                fit = likelihood(plaw, map[binmin:binmax])                                                          # Fit model = (lowE * k + c) to highE
                m = Minuit(fit,N_0, gamma, E_cut_inv)
                m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                label = r'$\mathrm{IC}:\ n = %.2f,$ ' %n + r'$E_\mathrm{cut} = %.1e\ \mathrm{GeV},$ ' %E_cut
                
            pyplot.errorbar(Es[binmin:binmax], [IC_model([fit_pars])(E) for E in Es[binmin:binmax]], label = label, color = colours[colour_index])




            

        if fit_pi0:

            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -         pi0         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "




            N_0, gamma = 100., 2.
            fit = likelihood(pi0_model, map[binmin:binmax])                                                          # Fit model = (lowE * k + c) to highE
            m = Minuit(fit,N_0, gamma)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]
            label  = r'$\pi^0:\ n = %.2f,$ ' %gamma
            
            if cutoff:
                N_0, gamma, E_cut_inv = 100., 2., 0.
                fit = likelihood(plaw, map[binmin:binmax])                                                          # Fit model = (lowE * k + c) to highE
                m = Minuit(fit,N_0, gamma, E_cut_inv)
                m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                label = r'$\pi^0:\ n = %.2f,$ ' %n + r'$p_\mathrm{cut} = %.1e\ \mathrm{GeV},$ ' % E_cut
                
            pyplot.errorbar(Es[binmin:binmax], [pi0_model([fit_pars])(E) for E in Es[binmin:binmax]], label = label, color = colours[colour_index])


        colour_index += 1
        

                    
########################################################################################################################## Cosmetics, safe plot

    lg = pyplot.legend(loc='upper left', ncol=2, fontsize = 'x-small')
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$E$ [GeV]')
    pyplot.ylabel(r'$ E^2\frac{dN}{dE}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    pyplot.title(r'SED in latitude stripes, $b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2))

    name = 'SED_'+ input_data +'_' + data_class + '_' + str(int(Bc[b]))
    fn = plot_dir + name + fn_ending
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.ylim((1.e-8,4.e-4))
    pyplot.savefig(fn, format = 'pdf')

