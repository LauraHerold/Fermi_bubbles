""" Plots the SED of all latitude stripes necessary to observe the Fermi bubbles. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit
from optparse import OptionParser
import dio
from yaml import load
import gamma_spectra
import scipy.integrate as integrate
from math import factorial
import auxil
from scipy import special

########################################################################################################################## Parameters

print_total_energy_output = True
plot_contour = False
print_upper_limits_Ecut = False


latitude = 7 # 7 is GP

fit_plaw = False
fit_logpar = False
fit_IC  = True
fit_pi0 = True

parser = OptionParser()
parser.add_option("-c", "--data_class", dest = "data_class", default = "source", help="data class (source or ultraclean)")
parser.add_option("-E", "--lowE_range", dest="lowE_range", default='0', help="There are 3 low-energy ranges: (3,5), (3,3), (4,5), (6,7)")
parser.add_option("-i", "--input_data", dest="input_data", default="boxes", help="Input data can be: data, lowE, boxes, GALPROP")
parser.add_option("-o", "--cutoff", dest="cutoff", default="True", help="Write true if you want cutoff")
(options, args) = parser.parse_args()

data_class = str(options.data_class)
low_energy_range = int(options.lowE_range) # 0: baseline, 4: test
input_data = str(options.input_data) # data, lowE, boxes, GALPROP

fn_ending = ".pdf"
cutoff = False
if str(options.cutoff) == "True":
    cutoff = True
    fn_ending = "cutoff_PL.pdf"
    #print_total_energy_output = False
    print_contour = False
    print_upper_limits_Ecut = False

########################################################################################################################## Constants


lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]

Save_as_dct = True
Save_plot = True

bin_start_fit = 5                                              # Energy bin where fit starts
fitmin = 3
fitmax = 18                                                    # bins: 0-15 for low-energy range 3, 0-17 else


colours = ['blue', 'red']

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
p_p = 10.**np.arange(-0.5, 6., 0.1)                                                      # Proton-momenta array

if cutoff:
    dof = fitmax - fitmin - 3
else:
    dof = fitmax - fitmin - 2

binmin = 0    
if low_energy_range == 3:
    binmin = 2

erg2GeV = 624.151
kpc2cm = 3.086e21
E_SN = 1.e49 * erg2GeV

R_GC = 8. * kpc2cm # cm




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
fitmax = min(nE, fitmax)
print 'fitmax: ' + str(fitmax)
#E_g = Es                                                                                # Final photon energies array in GeV

total_data_profiles = dio.loaddict('dct/Low_energy_range0/dct_data_counts_' + data_class + '.yaml')['6) Differential_flux_profiles']
print "total_data_profiles: ", len(total_data_profiles)

expo_dct = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) +'/dct_expo_' + data_class + '.yaml')
exposure_profiles = expo_dct['6) Exposure_profiles'] # shape: (nB, nL, nE)
print "expo_profiles shape: " + str(len(exposure_profiles)) + ", " + str(len(exposure_profiles[0])) + ", " + str(len(exposure_profiles[0][0]))
deltaE = expo_dct['8) deltaE'][binmin :]
dOmega = expo_dct['7) dOmega_profiles'][binmin :]


########################################################################################################################## Define likelihood class and powerlaw fct




class likelihood1:                                                        # First fit: over 4 energy bins only
    def __init__(self, model_fct, background_map, total_data_map):
        self.model_fct = model_fct
        self.background_map = background_map
        self.total_data_map = total_data_map
    def __call__(self, N_0, gamma):
        background_map  = self.background_map
        model_fct = self.model_fct
        total_data_map = self.total_data_map
        L =  sum(background_map[E] + model_fct(N_0, gamma)(E) - total_data_map[E] * np.log(background_map[E] + model_fct(N_0, gamma)(E)) for E in range(fitmin + 1,fitmin + 4))
        #print "N_0, gamma: " + str(N_0) + ", " + str(gamma) + " --> " + str(L)
        return L


class likelihood2:                                                       # Second fit: without cutoff  
    def __init__(self, model_fct, background_map, total_data_map):
        self.model_fct = model_fct
        self.background_map = background_map
        self.total_data_map = total_data_map
    def __call__(self, N_0, gamma):
        background_map  = self.background_map
        model_fct = self.model_fct
        total_data_map = self.total_data_map
        L = sum(background_map[E] + model_fct(N_0, gamma)(E) - total_data_map[E] * np.log(background_map[E] + model_fct(N_0, gamma)(E)) for E in range(fitmin,fitmax))
        #print "N_0, gamma: " + str(N_0) + ", " + str(gamma) + " --> " + str(L)
        return L
    
class likelihood_cutoff:                                                 # Optional third fit: with cutoff       
    def __init__(self, model_fct, background_map, total_data_map):
        self.model_fct = model_fct
        self.background_map = background_map
        self.total_data_map = total_data_map
    def __call__(self, Ecut_inv, N_0, gamma):
        background_map  = self.background_map
        model_fct = self.model_fct
        total_data_map = self.total_data_map
        L = sum(background_map[E] + model_fct(N_0, gamma, Ecut_inv)(E) - total_data_map[E] * np.log(background_map[E] + model_fct(N_0, gamma, Ecut_inv)(E)) for E in range(fitmin,fitmax))
        #print "N_0, alpha, beta: " + str(N_0) + ", " + str(gamma) + ", " + str(Ecut_inv) + " --> " + str(L)
        return L
    

def plaw(N_0, gamma, Ecut_inv = 0.):  # powerlaw
    return lambda E: N_0 * (Es[E]/Es[bin_start_fit])**(-gamma) * np.exp(-Es[E] * Ecut_inv)

def logpar(N_0, alpha, beta):
    return lambda E: N_0 * Es[E]**(-alpha - beta * np.log(Es[E]))



########################################################################################################################## Define particle-spectra functions

E_zero = Es[bin_start_fit]

print Bc

b = latitude

#l_ROI = R_GC * np.tan(dL * np.pi /180.) # cm
#h_ROI = R_GC * np.tan(dB[b] * np.pi / 180.) # cm
#V_ROI = l_ROI**2 * h_ROI  #cm^3
#print "V_ROI: ", V_ROI                                                                    #This is actually not completely right
#h_alt = R_GC * 2 * np.tan(dB[b]/2. * np.pi/180.)
#V_alt = l_ROI**2 * h_alt
#print "V_alt: ", V_alt
V_ROI = 1.2e65/3 #cm^3

print "Bc[b]: ",  Bc[b]
auxil.setup_figure_pars(plot_type = 'spectrum')
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

    def IC_model(N_0, gamma, Ecut_inv = 0.):
        EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)
        EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
        
        EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
        return lambda E: EdNdE_gamma_IC_vec(Es[E])  * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]

    def pi0_model(N_0, gamma, Ecut_inv = 0.):
        dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
        EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
        EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
        return lambda E: EdNdE_gamma_pi0_vec(Es[E])  * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]
        

########################################################################################################################## Plot SED
    
    map  = np.asarray(diff_profiles[b][l])
    expo_map = np.asarray(exposure_profiles[b][l])[binmin:]
    std_map = np.asarray(std_profiles[b][l])
    total_data_map = np.asarray(total_data_profiles[b][l])
    total_data_map = total_data_map[len(total_data_map)-nE:]
    print "len(total_data_profiles)-nE: ", (len(total_data_profiles)-nE)
    background_map = total_data_map - map

    for E in range(nE):
        if np.abs(std_map[E]) < 1.:
            std_map[E] = 1.
                
    label = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2)

    print map.shape, Es.shape, len(deltaE), expo_map.shape
    flux_map = map * Es**2 / dOmega[b][l] / deltaE / expo_map
    flux_std_map = std_map * Es**2 / dOmega[b][l] / deltaE / expo_map

    pyplot.errorbar(Es, flux_map, flux_std_map, color=colours[colour_index], marker='s', markersize=4, linestyle = '', label=label)



########################################################################################################################## Fit spectra

    if fit_plaw:
        def flux_plaw_in_counts(F_0, gamma, Ecut_inv = 0.):  # powerlaw
            return lambda E: (plaw(F_0, gamma, Ecut_inv)(E) * dOmega[b][l] * deltaE[E] * expo_map[E] / Es[E]**2)
        print " "
        print " "
        print "-  -  -  -  -  -  -  -  -  -      Powerlaw      -  -  -  -  -  -  -  -  -  -  -  -  -  -"
        print " "

        dct = {"x" : Es[fitmin:fitmax]}
        N_0, gamma, Ecut_inv = 1.e-6, 0.3, 0.

        fit = likelihood1(flux_plaw_in_counts, background_map, total_data_map)        # First fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, limit_N_0 = (0., 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]
            
        fit = likelihood2(flux_plaw_in_counts, background_map, total_data_map)        # Second fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, limit_N_0 = (0., 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]
        #cov = m.matrix()
        

            
        TS = 2 * sum(flux_plaw_in_counts(N_0, gamma)(E) - map[E] * np.log(flux_plaw_in_counts(N_0, gamma)(E)) for E in range(fitmin,fitmax))
        # gamma is spectral index of E^2dN/dE
        label = r'$\mathrm{PL}:\ \gamma = %.2f$' %(gamma+2) #+ r', $-\log L = %.2f$' %TS
        dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_Plaw_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) + ".yaml"
            
        if cutoff:
            fit = likelihood_cutoff(flux_plaw_in_counts, background_map, total_data_map)     # Third fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_gamma = 1., error_Ecut_inv = 1., limit_Ecut_inv = (0., 1.), limit_N_0 = (0., 1.), errordef = 0.5)
            m.migrad()
            N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
            m.hesse()
            sgm_Ecut = m.errors["Ecut_inv"]
            
            print "%"
            print r"Best-fit value $1/E_\cut$: ", Ecut_inv, r" 1/GeV\\"
            print r"Parameter error printed by MIGRAD: ", sgm_Ecut, r" 1/GeV\\"
            print r"Best-fit value $E_\cut$: ", (1/Ecut_inv), r" GeV\\"
            upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
            print r"Upper limit for $1/E_\cut$: ", upper_bound, r" 1/GeV\\"
            print r"Lower limit for $E_\cut$:  ", (1/upper_bound), r" GeV\\"
            print "%"


            
            #cov = m.matrix()
            #m.hesse()
            #m.minos()
            #print "m.get_merrors(): ", m.get_merrors()

            TS =  2 * sum(flux_plaw_in_counts(N_0, gamma, Ecut_inv)(E) - map[E] * np.log(flux_plaw_in_counts(N_0, gamma, Ecut_inv)(E)) for E in range(fitmin,fitmax))
            if Ecut_inv == 0:
                label = r'$\mathrm{PL}:\ \gamma = %.2f,$ ' %(gamma+2.) + r'$E_{\mathrm{cut}} = \infty$ ' + ',\n' + r'$-\log L = %.2f$' %TS
            else:
                label = r'$\mathrm{PL}:\ \gamma = %.2f,$ ' %(gamma+2.) + r'$E_{\mathrm{cut}} = %.1e\ \mathrm{GeV}$ ' %(1./Ecut_inv) + ',\n' + r'$-\log L = %.2f$' %TS
            dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_Plaw_cutoff_l=" + str(Lc[l]) + "_b=" + str(Bc[b])  + ".yaml"
            dct["E_cut"] = 1./Ecut_inv

        flux_plaw = [plaw(N_0, gamma, Ecut_inv)(E) for E in range(fitmin,fitmax)]
            
        chi2 = sum((flux_plaw - flux_map[fitmin:fitmax])**2 / flux_std_map[fitmin:fitmax]**2)
        print "chi2 = ", chi2
        #label += r', $\chi^2 = %.2f$' %chi2
            
        pyplot.errorbar(Es[fitmin:fitmax], flux_plaw, label = label, color = colours[colour_index])
            
        if Save_as_dct:
                
            dct["y"] = np.array(flux_plaw)
            dct["chi^2/d.o.f."] = chi2/dof
            dct["-logL"] = TS
            dct["gamma"] = (gamma + 2)                       # The fit returns gammas close to 0
            dct["N_0"] = N_0
            dio.saveyaml(dct, dct_fn, expand = True)

        if plot_contour:
            contour = pyplot.figure()
            m.draw_mncontour('gamma', 'Ecut_inv')
            contour.savefig(plot_dir + "Contour_powerlaw_l=" + str(Lc[l]) + ".pdf")

            

                

                 


    if fit_IC:
        print " "
        print " "
        print "-  -  -  -  -  -  -  -  -  -  -  -          IC         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
        print " "
            
        dct = {"x" : Es[fitmin:fitmax]}
        N_0, gamma, Ecut_inv = 1.e-14, 2.2, 0.
            
        fit = likelihood1(IC_model, background_map, total_data_map)                        # First fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (1.5,2.5), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        fit = likelihood2(IC_model, background_map, total_data_map)                        # Second fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = .5, limit_N_0 = (1.e-16, 1.e-7), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]
            
        TS =  2 * sum(IC_model(N_0, gamma)(E) - map[E] * np.log(IC_model(N_0, gamma)(E)) for E in range(fitmin,fitmax))
        cov = m.matrix()
        # gamma is spectral index of EdN/dE
        label  = r'$\mathrm{IC}:\ \gamma = %.2f$' %(gamma+1.) #+ r', $-\log L = %.2f$' %TS
        dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_IC_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) + ".yaml"
            
        if cutoff:
            fit = likelihood_cutoff(IC_model, background_map, total_data_map)            # Third fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_Ecut_inv = 1., error_gamma = 1., errordef = 0.5, limit_N_0 = (1.e-16, 1.e-7), limit_Ecut_inv = (0.,1.))
            m.migrad()
            #m.hesse()
            #m.minos()
            N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
            TS =  2 * sum(IC_model(N_0, gamma, Ecut_inv)(E) - map[E] * np.log(IC_model(N_0, gamma, Ecut_inv)(E)) for E in range(fitmin,fitmax))
            if Ecut_inv == 0:
                label = r'$\mathrm{IC}:\ \gamma = %.2f,$ ' %(gamma+1) + r'$E_\mathrm{cut} = \infty'+ ',\n' + r'$-\log L = %.2f$' %TS
            else:
                label = r'$\mathrm{IC}:\ \gamma = %.2f,$ ' %(gamma+1) + r'$E_\mathrm{cut} = %.1e\ \mathrm{GeV}$ ' %(1./Ecut_inv) + ',\n' + r'$-\log L = %.2f$' %TS
            dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_IC_cutoff_l=" + str(Lc[l]) + "_b=" + str(Bc[b])  + ".yaml"
            dct["E_cut"] = 1./Ecut_inv
            #cov = m.matrix
            N_0_e, gamma_e, Ecut_inv_e  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
            sgm_Ecut = m.errors["Ecut_inv"]
            
            print "%"
            print r"Best-fit value $1/E_\cut$: ", Ecut_inv, r" 1/GeV\\"
            print r"Parameter error printed by MIGRAD: ", sgm_Ecut, r" 1/GeV\\"
            print r"Best-fit value $E_\cut$: ", (1/Ecut_inv), r" GeV\\"
            upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
            print r"Upper limit for $1/E_\cut$: ", upper_bound, r" 1/GeV\\"
            print r"Lower limit for $E_\cut$:  ", (1/upper_bound), r" GeV\\"
            print "%"
            
            print "bf_EdNdE(E) = ", N_0_e, " * E**(-", gamma_e, ") * exp(E * ", Ecut_inv_e, ")" 
            def bf_EdNdE_e(E):
                return N_0_e * E**(-gamma_e) * np.exp(-E * Ecut_inv_e)



        flux_IC = [(IC_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 / dOmega[b][l] / deltaE[E] / expo_map[E]) for E in range(fitmin,fitmax)]

        chi2 = sum((flux_IC - flux_map[fitmin:fitmax])**2/flux_std_map[fitmin:fitmax]**2)
        print "chi2 = ", chi2
        #label += r', $\chi^2 = %.2f$' %chi2
            
        pyplot.errorbar(Es[fitmin:fitmax], flux_IC, label = label, color = colours[colour_index], ls = ':')
        #EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)

        
        
        if Save_as_dct:
                
            dct["y"] = np.array(flux_IC)
            dct["chi^2/d.o.f."] = chi2/dof
            dct["-logL"] = TS
            dct["gamma"] = gamma
            dct["N_0"] = N_0
            dio.saveyaml(dct, dct_fn, expand = True)

        if plot_contour:
            contour = pyplot.figure()
            m.draw_mncontour('gamma', 'Ecut_inv')
            contour.savefig(plot_dir + "Contour_IC_l=" + str(Lc[l]) + ".pdf")

            


            

    if fit_pi0:

        print " "
        print " "
        print "-  -  -  -  -  -  -  -  -  -         pi0         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
        print " "



        dct = {"x" : Es[fitmin:fitmax]}
        N_0, gamma, Ecut_inv = 1.e-10, 2.4, 0.
            
        fit = likelihood1(pi0_model, background_map, total_data_map)              # First fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        fit = likelihood2(pi0_model, background_map, total_data_map)              # Second fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        fit = likelihood2(pi0_model, background_map, total_data_map)              # Second fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]
            
        TS =  2 * sum(pi0_model(N_0, gamma)(E) - map[E] * np.log(pi0_model(N_0, gamma)(E)) for E in range(fitmin,fitmax))
        # gamma is spectral index of dN/dp
        label  = r'$\pi^0:\ \gamma = %.2f$' %gamma #+ r', $-\log L = %.2f$' %TS
        dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_pi0_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) + ".yaml"
        cov = m.matrix()

        
            
        if cutoff:
            Ecut_inv = 0.000001
            fit = likelihood_cutoff(pi0_model, background_map, total_data_map)   # Third fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_Ecut_inv = .1, error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), limit_Ecut_inv = (1.e-20,1.), errordef = 0.5)
            m.migrad()
            N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
            TS =  2 * sum(pi0_model(N_0, gamma, Ecut_inv)(E) - map[E] * np.log(pi0_model(N_0, gamma, Ecut_inv)(E)) for E in range(fitmin,fitmax))
            if Ecut_inv == 0:
                label = r'$\pi^0:\ \gamma = %.2f,$ ' %gamma + r'$p_\mathrm{cut} = \infty,$' + '\n' + r'$-\log L = %.2f$' %TS
            else:
                label = r'$\pi^0:\ \gamma = %.2f,$ ' %gamma + r'$p_\mathrm{cut} = %.1e\ \mathrm{GeV},$ ' % (1./Ecut_inv) + '\n' +  r'$-\log L = %.2f$' %TS
            dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_pi0_cutoff_l=" + str(Lc[l]) + "_b=" + str(Bc[b])  + ".yaml"
            dct["E_cut"] = 1./Ecut_inv
            #cov = m.matrix()
            #m.hesse()
            #m.minos()

            N_0_p, gamma_p, Ecut_inv_p = N_0, gamma, Ecut_inv
            sgm_Ecut = m.errors["Ecut_inv"]
            
            print "%"
            print r"Best-fit value $1/E_\cut$: ", Ecut_inv, r" 1/GeV\\"
            print r"Parameter error printed by MIGRAD: ", sgm_Ecut, r" 1/GeV\\"
            print r"Best-fit value $E_\cut$: ", (1/Ecut_inv), r" GeV\\"
            upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
            print r"Upper limit for $1/E_\cut$: ", upper_bound, r" 1/GeV\\"
            print r"Lower limit for $E_\cut$:  ", (1/upper_bound), r" GeV\\"
            print "%"
            
            #dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
            def bf_pdNdp_p(p):
                return p * N_0_p * p**(-gamma_p) * np.exp(-p * Ecut_inv_p)
            print "bf_dNdp_p() = ", N_0_p, " * E**(-", gamma_p, ") * exp(p * ", Ecut_inv_p, ")"
            
        flux_pi0 = [(pi0_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 / dOmega[b][l] / deltaE[E] / expo_map[E]) for E in range(fitmin,fitmax)]
        
        chi2 = sum((flux_pi0 - flux_map[fitmin:fitmax])**2/flux_std_map[fitmin:fitmax]**2)
        print "chi2 = ", chi2
        #label += r', $\chi^2 = %.2f$' %chi2
        
        pyplot.errorbar(Es[fitmin:fitmax], flux_pi0, label = label, color = colours[colour_index], ls = '-.')

        

        if Save_as_dct:
            dct["y"] = np.array(flux_pi0)
            dct["chi^2/d.o.f."] = chi2/dof
            dct["-logL"] = TS
            dct["gamma"] = gamma
            dct["N_0"] = N_0
            dio.saveyaml(dct, dct_fn, expand = True)
            
        if plot_contour:
            contour = pyplot.figure()
            m.draw_mncontour('gamma', 'Ecut_inv')
            contour.savefig(plot_dir + "Contour_pi0_l=" + str(Lc[l]) + ".pdf")



    if fit_logpar:
        def flux_logpar_in_counts(F_0, alpha, beta):  # Log parabola in counts
            return lambda E: (logpar(F_0, alpha, beta)(E) * dOmega[b][l] * deltaE[E] * expo_map[E] / Es[E]**2)
        print " "
        print " "
        print "-  -  -  -  -  -  -  -  -  -     Log parabola      -  -  -  -  -  -  -  -  -  -  -  -  -  -"
        print " "

        dct = {"x" : Es[fitmin:fitmax]}
        N_0, alpha, beta = 1.e-6, -0.26, 0.09

        fit = likelihood_cutoff(flux_logpar_in_counts, background_map, total_data_map)        # First fit
        m = Minuit(fit, N_0 = N_0, gamma = alpha, Ecut_inv = beta, limit_N_0 = (0., 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
        m.migrad()
        N_0, alpha, beta  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]

            
        TS = 2 * sum(flux_logpar_in_counts(N_0, alpha, beta)(E) - map[E] * np.log(flux_logpar_in_counts(N_0, alpha, beta)(E)) for E in range(fitmin,fitmax))
        label = r'$\mathrm{LogPar}:\ \alpha = %.2f, $' %alpha + r'$\beta = %.2f, $' %beta + '\n' + r'$-\log L = %.2f$' %TS
        dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_LogPar_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) + ".yaml"
            

        flux_logpar = [logpar(N_0, alpha, beta)(E) for E in range(fitmin,fitmax)]
        
        chi2 = sum((flux_logpar - flux_map[fitmin:fitmax])**2 / flux_std_map[fitmin:fitmax]**2)
        label += r', $\chi^2 = %.2f$' %chi2
            
        pyplot.errorbar(Es[fitmin:fitmax], flux_logpar, label = label, color = colours[colour_index], ls = '--')

        logpar_n = - alpha - 2. * beta * np.log(500.)
        print "logpar_n: ", logpar_n
            
        if Save_as_dct:
                
            dct["y"] = np.array(flux_logpar)
            dct["chi^2/d.o.f."] = chi2/dof
            dct["-logL"] = TS
            dct["alpha"] = alpha
            dct["beta"] = beta
            dct["N_0"] = N_0
            dio.saveyaml(dct, dct_fn, expand = True)

###################################################################################################################### Print total energy output

    if print_total_energy_output:
        for lower_bound_particle_energy in [1.]: #GeV
    
            print lower_bound_particle_energy

            #total_area = 4. * (8. * 3.086e21)**2 * np.pi # cm 
            
            E_tot_e = integrate.quad(bf_EdNdE_e, lower_bound_particle_energy, 1000.)[0] #GeV/cm^3
            
            E_tot_p = integrate.quad(bf_pdNdp_p, lower_bound_particle_energy, 1000.)[0]
            
            print "E_tot_e = ", (E_tot_e * 1.e12), " meV/cm^3 = ", (E_tot_e/erg2GeV), " erg/cm^3"
            print "E_tot_p = ", (E_tot_p * 1.e12), " meV/cm^3 = ", (E_tot_p/erg2GeV), " erg/cm^3"

    colour_index += 1
        

                    
########################################################################################################################## Cosmetics, safe plot

    
if Save_plot:
    lg = pyplot.legend(loc='upper left', ncol=2)
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$E\ \mathrm{[GeV]}$')
    #pyplot.ylabel('Counts')
    pyplot.ylabel(r'$ E^2\frac{\mathrm{d}N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    pyplot.title(r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2))

    name = 'SED_'+ input_data +'_' + data_class + '_' + str(int(Bc[b]))
    fn = plot_dir + name + fn_ending
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.ylim((5.e-8,4.e-4))
    pyplot.savefig(fn, format = 'pdf')



'''
right

bf_EdNdE(E) =  3.55753268274e-12  * E**(- 1.72868812878 ) * exp(E *  5.41278133426e-12)
bf_dNdp_p() =  9.98590822311e-11  * E**(- 2.18441413698 ) * exp(p *  8.42235703403e-09 )

1.0
E_tot_e =  4.87373582087  meV/cm^3 =  7.80858449457e-15  erg/cm^3
E_tot_p =  84.314091718  meV/cm^3 =  1.35086047636e-13  erg/cm^3
10.0
E_tot_e =  0.90343512013  meV/cm^3 =  1.44746242517e-15  erg/cm^3
E_tot_p =  5.5172140637  meV/cm^3 =  8.83955014685e-15  erg/cm^3
100.0
E_tot_e =  0.161491868184  meV/cm^3 =  2.58738459418e-16  erg/cm^3
E_tot_p =  0.364736406547  meV/cm^3 =  5.84372061484e-16  erg/cm^3


left

bf_EdNdE(E) =  9.99999234588e-11  * E**(- 2.98267837491 ) * exp(E *  0.0237152141157 )
bf_dNdp_p(p) =  9.99999983897e-11  * E**(- 2.32611658858 ) * exp(p *  0.02407928633 )

1.0
E_tot_e =  48.1612350707  meV/cm^3 =  7.71627940525e-14  erg/cm^3
E_tot_p =  70.2013693745  meV/cm^3 =  1.12474977008e-13  erg/cm^3
10.0
E_tot_e =  0.347238732073  meV/cm^3 =  5.5633770045e-16  erg/cm^3
E_tot_p =  2.09599720455  meV/cm^3 =  3.35815724808e-15  erg/cm^3
100.0
E_tot_e =  0.000208125231032  meV/cm^3 =  3.33453332658e-19  erg/cm^3
E_tot_p =  0.00466415754417  meV/cm^3 =  7.47280312644e-18  erg/cm^3
'''
