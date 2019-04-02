""" Plots the SED of all latitude stripes necessary to observe the Fermi bubbles. """
# isrf=Popescu
# v54, R12, F98
# cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/9-years
# python Plot_SED_IC_ISRF-components_dima.py -o0 -r $isrf
# cp ../../plots/Plots_9-year/Low_energy_range0/SED_ISRF_componentsboxes_source_0_"$isrf".pdf ../../paper/plots

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit
from optparse import OptionParser
from yaml import load
import gamma_spectra
import scipy.integrate as integrate
from math import factorial
from scipy import special

import auxil
import dio

########################################################################################################################## Parameters

print_total_energy_output = False
plot_contour = False
print_upper_limits_Ecut = False

no_accurate_matrix = False
acc_matrix = True

fit_IC  = True
fit_pi0 = False

Ecut_inv_min0 = 1.e-6 # 1 PeV cutoff for electron and proton CR spectra
Ecut_inv_min = 2.e-6 # effective 500 TeV cutoff for the proton spectra (there are some problems in the parametrization above 500 TeV)

parser = OptionParser()
parser.add_option("-c", "--data_class", dest = "data_class", default = "source", help="data class (source or ultraclean)")
parser.add_option("-E", "--lowE_range", dest="lowE_range", default='0', help="There are 3 low-energy ranges: (3,5), (3,3), (4,5), (6,7)")
parser.add_option("-i", "--input_data", dest="input_data", default="boxes", help="Input data can be: data, lowE, boxes, GALPROP")
parser.add_option("-o", "--cutoff", dest="cutoff", default="True", help="Write true if you want cutoff")
parser.add_option("-l", "--latitude", dest="latitude", default='7', help="Number of latitude stripe")
parser.add_option("-r", "--isrf", dest="isrf", default='v54', help="ISRF model for the GC: v54 (default), R12, F98, Popescu")
(options, args) = parser.parse_args()

data_class = str(options.data_class)
low_energy_range = int(options.lowE_range) # 0: baseline, 4: test
input_data = str(options.input_data) # data, lowE, boxes, GALPROP
latitude = int(options.latitude)
ISRF_model = options.isrf # 'v54', 'Popescu', 'R12', 'F98'

cre_sp_fn = '../dima/data/ISRF_CRe.yaml'
cre_sp_dict = dio.loaddict(cre_sp_fn)


fn_ending = ".pdf"
dct_fn_ending = ".yaml"
cutoff = False
if str(options.cutoff) == "True":
    cutoff = True
    fn_ending = "_cutoff.pdf"
    #print_total_energy_output = False
    print_contour = False
    print_upper_limits_Ecut = False


Save_as_dct = True 
Save_plot = True
without_last_data_point = False

plot_till_100TeV = True


########################################################################################################################## Constants
if plot_till_100TeV:
    E_gamma = 10**np.arange(0., 6., 0.1)

kpc2cm = 3.086e21
R_GC = 8.5 * kpc2cm # cm
V_ROI_tot = 4.e64 #cm^3
line_of_sight = 1.5 * kpc2cm
dOmega_tot = 0.012   # dOmega of unmasked ROI
mk2m = 1.e-6

lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]


    
fitmin = 3
fitmax = 18                                                    # bins: 0-15 for low-energy range 3, 0-19 else


if without_last_data_point:
    fitmax = 17
    fn_ending = "_without_last_data_point.pdf"
    

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
E_e = 10.**np.arange(-1., 10.001, 0.1)                                                   # Electron-energies array (0.1 - 10^8 GeV)
p_p = 10.**np.arange(-0.5, 7., 0.1)                                                      # Proton-momenta array

if cutoff:
    dof = fitmax - fitmin - 3
else:
    dof = fitmax - fitmin - 2

binmin = 0    
if low_energy_range == 3:
    binmin = 2

erg2GeV = 624.151

E_SN = 1.e49 * erg2GeV

plot_dict = {'Popescu': r'Popescu\ et\ al.\ (2017)',
             'R12': r'Porter\ et\ al.\ (2017)\ R12',
             'F98': r'Porter\ et\ al.\ (2017)\ F98',
             'v54': r'Porter\ et\ al.\ (2008)'
}






########################################################################################################################## Load dictionaries

dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) +'/dct_' + input_data + '_counts_' + data_class + '.yaml')

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']


print "diff_profiles: ", diff_profiles[7][0]

nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = len(diff_profiles[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)
fitmax = min(nE, fitmax)
print 'fitmax: ' + str(fitmax)
#E_g = Es                                                                                # Final photon energies array in GeV

total_data_profiles = dio.loaddict('dct/Low_energy_range0/dct_data_counts_' + data_class + '.yaml')['6) Differential_flux_profiles']
print "total_data_profiles: ", len(total_data_profiles)

std_total_data_profiles = dio.loaddict('dct/Low_energy_range0/dct_data_counts_' + data_class + '.yaml')['7) Standard_deviation_profiles']

expo_dct = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) +'/dct_expo_' + data_class + '.yaml')
exposure_profiles = expo_dct['6) Exposure_profiles'] # shape: (nB, nL, nE)
print "expo_profiles shape: " + str(len(exposure_profiles)) + ", " + str(len(exposure_profiles[0])) + ", " + str(len(exposure_profiles[0][0]))
deltaE = expo_dct['8) deltaE'][binmin :]
dOmega = expo_dct['7) dOmega_profiles']


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

class likelihood_cutoff2:                                                 # Optional third fit: with cutoff       
    def __init__(self, model_fct, background_map, total_data_map):
        self.model_fct = model_fct
        self.background_map = background_map
        self.total_data_map = total_data_map
    def __call__(self, Ecut_inv, N_0, gamma):
        background_map  = self.background_map
        model_fct = self.model_fct
        total_data_map = self.total_data_map
        L = sum(background_map[E] + model_fct(N_0, gamma, Ecut_inv)(E) - total_data_map[E] * np.log(background_map[E] + model_fct(N_0, gamma, Ecut_inv)(E)) for E in range(3,7))
        #print "N_0, alpha, beta: " + str(N_0) + ", " + str(gamma) + ", " + str(Ecut_inv) + " --> " + str(L)
        return L
    

def plaw(N_0, gamma, Ecut_inv = Ecut_inv_min0):  # powerlaw
    return lambda E: N_0 * (Es[E])**(-gamma) * np.exp(-Es[E] * Ecut_inv)

def logpar(N_0, alpha, beta):
    return lambda E: N_0 * Es[E]**(-alpha - beta * np.log(Es[E]))

def lambda2eV(ld):
    """
        transform lambda (m) to energy (eV)
    """
    return c_light * h_Planck / ld

########################################################################################################################## Define particle-spectra functions

TS_values = np.zeros((2,nL))
Ecut_values =  np.zeros((2,nL))

print Bc

b = latitude

#l_ROI = R_GC * np.tan(dL * np.pi /180.) # cm
#h_ROI = R_GC * np.tan(dB[b] * np.pi / 180.) # cm
#V_ROI = l_ROI**2 * h_ROI  #cm^3
#print "V_ROI: ", V_ROI                                                                    #This is actually not completely right
#h_alt = R_GC * 2 * np.tan(dB[b]/2. * np.pi/180.)
#V_alt = l_ROI**2 * h_alt
#print "V_alt: ", V_alt


print "Bc[b]: ",  Bc[b]
auxil.setup_figure_pars(plot_type = 'spectrum')
colour_index = 0

print 'Take %s ISRF model' % ISRF_model
IRFmap_fn = '../../data/ISRF_average/ldUld_GC_average_%s.csv' % ISRF_model


data = np.loadtxt(IRFmap_fn, delimiter=',', skiprows=1).T
wavelengths = data[0] * mk2m
ld_dUdld = data[1]
#else:
#    hdu = pyfits.open(IRFmap_fn)                                                                # Physical unit of field: 'micron'
#    ld_dUdld = hdu[1].data.field('Total')
#    wavelengths = hdu[1].data.field('Wavelength') * mk2m                                       # in m



E_irf_galaxy = lambda2eV(wavelengths)[::-1]                                       # Convert wavelength to eV, invert order

E_SL2IR_transition = 0.1 # transition energy between IR and starlight - 0.1 eV
transition_bin_IR_starlight = np.argmin(np.abs(E_irf_galaxy - E_SL2IR_transition))
print "transition energy IR - starlght: ", E_irf_galaxy[transition_bin_IR_starlight]
#transition_bin_IR_starlight0 = 73 * 4 - 1 # Energy = 0.1 eV
#print "transition energy IR - starlght old: ", E_irf_galaxy[transition_bin_IR_starlight0]


EdNdE_irf_galaxy = ld_dUdld[::-1] / E_irf_galaxy                          # in 1/cm^3. Since unit of 'Total': eV/cm^3
dlogE_irf = np.log(E_irf_galaxy[1] / E_irf_galaxy[0])                                                              # Wavelength bin size
# 0.0230258509299398

E_irf = np.e**np.arange(-6.* np.log(10.), np.log(E_irf_galaxy[-1]) + dlogE_irf, dlogE_irf) # CMB-energies array with same log bin size as IRF_galaxy in eV
irf_CMB = gamma_spectra.thermal_spectrum(T_CMB)                                             # Use thermal_spectrum from gamma_spectra.py, returns IRF in eV/cm^3
EdNdE_CMB = irf_CMB(E_irf) / E_irf                                                          # in 1/cm^3

EdNdE_irf_IR = np.zeros_like(EdNdE_irf_galaxy)
EdNdE_irf_IR[:transition_bin_IR_starlight] = EdNdE_irf_galaxy[:transition_bin_IR_starlight]
EdNdE_irf_IR = np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), EdNdE_irf_IR)
#EdNdE_irf_IR =  np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), np.append(EdNdE_irf_galaxy[0:transition_bin_IR_starlight], np.zeros(len(E_irf_galaxy)-transition_bin_IR_starlight)))

EdNdE_irf_starlight = np.zeros_like(EdNdE_irf_galaxy)
EdNdE_irf_starlight[transition_bin_IR_starlight:] = EdNdE_irf_galaxy[transition_bin_IR_starlight:]
EdNdE_irf_starlight = np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), EdNdE_irf_starlight)

EdNdE_irf_galaxy = np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), EdNdE_irf_galaxy)


EdNdE_irf = EdNdE_CMB + EdNdE_irf_galaxy # Differential flux in 1/cm^3 


fig = pyplot.figure()
for l in [0]:
    V_ROI = dOmega[b][l] * R_GC**2 * line_of_sight

    def IC_model(N_0, gamma, Ecut_inv = Ecut_inv_min0):
        EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)
        EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
        
        EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
        return lambda E: EdNdE_gamma_IC_vec(Es[E]) * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]
    def pi0_model(N_0, gamma, Ecut_inv = Ecut_inv_min0):
        dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
        EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
        EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
        return lambda E: EdNdE_gamma_pi0_vec(Es[E]) * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]
        

########################################################################################################################## Plot SED
    
    map  = np.asarray(diff_profiles[b][l])
    expo_map = np.asarray(exposure_profiles[b][l])[binmin:]
    std_map = np.asarray(std_profiles[b][l])
    total_data_map = np.asarray(total_data_profiles[b][l])
    total_data_map = total_data_map[len(total_data_map)-nE:]
    std_total_map = np.asarray(std_total_data_profiles[b][l])
    std_total_map = std_total_map[len(std_total_map)-nE:]
    print "len(total_data_profiles)-nE: ", (len(total_data_profiles)-nE)
    background_map = total_data_map - map
    print std_map

    print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    print "map: ", map
        
    #for E in range(nE):
    #    if np.abs(std_map[E]) < 1.:
    #        std_map[E] = 1.
                
    label = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2)

    print map.shape, Es.shape, len(deltaE), expo_map.shape
    flux_map = map * Es**2 / dOmega[b][l] / deltaE / expo_map
    flux_std_map = std_map * Es**2 / dOmega[b][l] / deltaE / expo_map

    pyplot.errorbar(Es, flux_map, flux_std_map, color=colours[colour_index], marker='s', markersize=4, linestyle = '', label=label)
    gamma_ray_luminosity = np.sum(flux_map * deltaE / Es) * dOmega_tot * 4. * np.pi * R_GC**2 / erg2GeV
    print "Gamma-ray luminosity: ", gamma_ray_luminosity # From 1 GeV to 1 TeV



########################################################################################################################## Fit spectra

    if fit_IC:
        print " "
        print " "
        print "-  -  -  -  -  -  -  -  -  -  -  -          IC         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
        print " "
            
        dct = {"x" : Es[fitmin:fitmax]}
        N_0, gamma, Ecut_inv = 7.e-12, 1.7, Ecut_inv_min0
            
        fit = likelihood1(IC_model, background_map, total_data_map)                        # First fit
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (1.5,2.5), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        if 1:
            fit = likelihood2(IC_model, background_map, total_data_map)                        # Second fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = .5, limit_N_0 = (1.e-16, 1.e-7), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]


        TS_nocut = -2. * np.sum((total_data_map[E] * np.log(background_map[E] + IC_model(N_0, gamma)(E)) - (background_map[E] + IC_model(N_0, gamma)(E))) for E in range(fitmin,fitmax))
        TS = TS_nocut
        #TS =  2 * sum(IC_model(N_0, gamma)(E) - map[E] * np.log(IC_model(N_0, gamma)(E)) for E in range(fitmin,fitmax))
        chi2_nocut = np.sum((total_data_map[E] - background_map[E] - IC_model(N_0, gamma)(E))**2/std_total_map[E]**2 for E in range(fitmin,fitmax))
        
        cov = m.matrix()
        # gamma is spectral index of EdN/dE
        label  = r'$\mathrm{IC}:\ \gamma = %.2f$' %(gamma+1.) #+ r', $-\log L = %.2f$' %TS
        dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_IC_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) + dct_fn_ending
            
        if cutoff:
            while True:
                Ecut_inv = np.random.random(1) * 1.e-5
                fit = likelihood_cutoff(IC_model, background_map, total_data_map)            # Third fit
                m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1.e-13, error_Ecut_inv = 1.e-5, error_gamma = 1., errordef = 0.5, limit_N_0 = (1.e-11, 1.e-7), limit_Ecut_inv = (0.,1.))
                m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                #fit = likelihood_cutoff(IC_model, background_map, total_data_map)            # Third fit
                #m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_Ecut_inv = 1.e-4, error_gamma = 1., errordef = 0.5, limit_N_0 = (1.e-16, 1.e-7), limit_Ecut_inv = (0.,1.))
                #m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                if m.matrix_accurate() or no_accurate_matrix:
                    break
            
            dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_IC_cutoff_l=" + str(Lc[l]) + "_b=" + str(Bc[b])  + dct_fn_ending
            dct["3) E_cut"] = 1./Ecut_inv
            #cov = m.matrix
            N_0_e, gamma_e, Ecut_inv_e  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
            sgm_Ecut = m.errors["Ecut_inv"]

            TS_cut = -2. * sum((total_data_map[E] * np.log(background_map[E] + IC_model(N_0, gamma, Ecut_inv)(E)) - (background_map[E] + IC_model(N_0, gamma, Ecut_inv)(E))) for E in range(fitmin,fitmax))
            
            TS = TS_cut
            print r"$-2\Delta\log(L) = $", TS_nocut, " - ", TS_cut, " = ", (TS_nocut - TS_cut)
            TS_values[0][l] = (TS_nocut - TS_cut)



        # replace the cutoff in the spectrum with min value
        print 'parameters for the final CRe spectrum:'
        print N_0, gamma, Ecut_inv
        print 'save to the dict:'
        print cre_sp_fn
        sub_dict = {}
        sub_dict['N_0'] = N_0
        sub_dict['gamma'] = gamma
        cre_sp_dict[ISRF_model] = sub_dict
        dio.savedict(cre_sp_dict, cre_sp_fn, expand=True)
        print
        
        EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)
        #EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv_min0)
        irf_components = [EdNdE_irf, EdNdE_irf_starlight, EdNdE_irf_IR, EdNdE_CMB]
        lss = [ "-", "--", "-.", ":",]
        labels = ["Total IC",  "SL", "IR", "CMB",]
        colors = ["blue", "darkorange", "red", "cadetblue"]
        lws = [1.6, 1.6, 1.6, 1.6]
        
        for component in range(len(labels)):
            EdNdE_irf_comp = irf_components[component]
            EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf_comp, E_irf, EdNdE_e, E_e)
            EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)

            if plot_till_100TeV:
                IC_model = EdNdE_gamma_IC_vec(E_gamma) * V_ROI/ (4. * R_GC**2 * np.pi) / E_gamma
            
                flux_IC = IC_model * E_gamma**2 / dOmega[b][l]
                pyplot.errorbar(E_gamma, flux_IC, label = labels[component], color = colors[component], ls = lss[component], linewidth = lws[component])
            else:
                IC_model = EdNdE_gamma_IC_vec(Es[fitmin:fitmax]) * V_ROI * exposure_profiles[b][l][fitmin:fitmax] / (4. * R_GC**2 * np.pi) * deltaE[fitmin:fitmax] / Es[fitmin:fitmax]
            
                flux_IC = IC_model * Es[fitmin:fitmax]**2 / dOmega[b][l] / deltaE[fitmin:fitmax] / expo_map[fitmin:fitmax]
                pyplot.errorbar(Es[fitmin:fitmax], flux_IC, label = labels[component], color = colors[component], ls = lss[component], linewidth = lws[component])
        #EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)

        colour_index += 1
        
"""        
        if Save_as_dct:
                
            dct["y"] = np.array(flux_IC)
            dct["chi^2/d.o.f."] = chi2/dof
            dct["-logL"] = TS
            dct["N_0"] = N_0
            
            dct["1) N_0_los"] = N_0 * line_of_sight
            dct["2) gamma"] = (gamma + 1)
            dct["4) -2 Delta logL"] = TS_values[0][l]
            dct["5) lower bound E_cut"] = Ecut_values[0][l]
            
            dio.saveyaml(dct, dct_fn, expand = True)

        if plot_contour:
            contour = pyplot.figure()
            m.draw_mncontour('gamma', 'Ecut_inv')
            contour.savefig(plot_dir + "Contour_IC_l=" + str(Lc[l]) + ".pdf")
"""
            
if fit_pi0:

        print " "
        print " "
        print "-  -  -  -  -  -  -  -  -  -         pi0         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
        print " "



        N_0, gamma, Ecut_inv = 9.e-10, 1.8, Ecut_inv_min
            
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
            
        
        
        # gamma is spectral index of dN/dp
        label  = r'$\pi^0:\ \gamma = %.2f$' %gamma #+ r', $-\log L = %.2f$' %TS
        dct_fn = "plot_dct/Low_energy_range" + str(low_energy_range) + "/" + input_data + "_" + data_class + "_pi0_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) +dct_fn_ending
        cov = m.matrix()

        
            
        if cutoff:
            while True:
                Ecut_inv = np.random.random(1) * 1.e-3
                fit = likelihood_cutoff(pi0_model, background_map, total_data_map)   # Third fit
                m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1.e-13, error_Ecut_inv = .1, error_gamma = 1., limit_N_0 = (8.e-11, 1.e-7), limit_Ecut_inv = (0.,1.), errordef = 0.5)
                m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                fit = likelihood_cutoff(pi0_model, background_map, total_data_map)   # Third fit
                m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_Ecut_inv = .05, error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), limit_Ecut_inv = (0.,1.), errordef = 0.5)
                m.migrad()
                N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                if m.matrix_accurate() or no_accurate_matrix:
                    break

            N_0_p, gamma_p, Ecut_inv_p = N_0, gamma, Ecut_inv
            sgm_Ecut = m.errors["Ecut_inv"]
            
            def bf_pdNdp_p(p):
                return p * N_0_p * p**(-gamma_p) * np.exp(-p * Ecut_inv_p)
            print "bf_dNdp_p() = ", N_0_p, " * E**(-", gamma_p, ") * exp(p * ", Ecut_inv_p, ")"

        if plot_till_100TeV:
            # MC
            Ecut_inv = Ecut_inv_min
            dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
            EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
            EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
            pi0_model = EdNdE_gamma_pi0_vec(E_gamma) * V_ROI  / (4. * R_GC**2 * np.pi)  / E_gamma
            flux_pi0 = pi0_model * E_gamma**2 / dOmega[b][l]
            pyplot.errorbar(E_gamma, flux_pi0, label=r"$\pi^0$", color="black", ls='-', lw=1.6)

            # Aharonian parametrization
            Ecut_inv = Ecut_inv_min0
            dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
            EdNdE_gamma_pi0_v0 = gamma_spectra.pi0_spectrum(dNdp_p, p_p)
            EdNdE_gamma_pi0_vec0 = np.frompyfunc(EdNdE_gamma_pi0_v0, 1, 1)
            pi0_model0 = EdNdE_gamma_pi0_vec0(E_gamma) * V_ROI  / (4. * R_GC**2 * np.pi)  / E_gamma

            # relative norm
            E0 = 1.e3
            norm = EdNdE_gamma_pi0_vec(E0) / EdNdE_gamma_pi0_vec0(E0)

            # plot Aharonian param-n
            flux_pi0_0 = norm * pi0_model0 * E_gamma**2 / dOmega[b][l]
            #pyplot.errorbar(E_gamma, flux_pi0_0, label = r"$\pi^0$", c='cyan', ls = '-')

        else:
            flux_pi0 = [(pi0_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 / dOmega[b][l] / deltaE[E] / expo_map[E]) for E in range(fitmin,fitmax)]
            pyplot.errorbar(Es[fitmin:fitmax], flux_pi0, label = r"$\pi^0$", color = "red", ls = '-')


                  
########################################################################################################################## Cosmetics, safe plot

    
if Save_plot:
    #pyplot.title(plot_dict[ISRF_model])
    lg = pyplot.legend(loc='upper left', ncol=2)
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$E\ \mathrm{[GeV]}$')
    #pyplot.ylabel('Counts')
    pyplot.ylabel(r'$ E^2\frac{\mathrm{d}N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    #pyplot.title(r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2))

    ax = fig.add_subplot(111)
    props = dict(facecolor='white', alpha=1, edgecolor = "white")
    st = r'$ISRF\ from\ %s$' % plot_dict[ISRF_model]
    ax.text(0.03, 0.72, st, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

    name = 'SED_ISRF_components'+ input_data +'_' + data_class + '_' + str(int(Bc[b]))
    name += '_%s' % ISRF_model
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.ylim((5.e-8,4.e-4))
    
    fn = plot_dir + name + fn_ending
    print 'save figure to file'
    print fn
    pyplot.savefig(fn, format = 'pdf')

#pyplot.show()
