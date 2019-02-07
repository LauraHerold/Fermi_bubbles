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


latitude = 7   # 7 is GP

fit_plaw = True
fit_IC  = False
fit_pi0 = False


fn_ending = ".pdf"
dct_fn_ending = ".yaml"
cutoff = True
no_accurate_matrix = False

Save_dct = True



########################################################################################################################## Constants


lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]
colours = ['blue', 'red']
titles = ["West", "East"]

fitmin = 3 
fitmax = 18                                                    #  3-18 


plot_dir = '../../plots/Plots_9-year/'

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575                                                              # logarithmic distance between two energy bins

kpc2cm = 3.086e21
R_GC = 8.5 * kpc2cm # cm
V_ROI_tot = 4.e64 #cm^3
line_of_sight = 1.5 * kpc2cm
dOmega_tot = 0.012   # dOmega of unmasked ROI

c_light = 2.9979e8                                                                      # m/s speed of light
h_Planck = 4.1357e-15                                                                   # eV * s Planck constant
kB = 8.6173303e-5                                                                       # eV/K
T_CMB = 2.73 * kB                                                                       # CMB temperature

ISFR_heights = [10, 10, 5, 5, 2, 1, 0.5, 0, 0.5, 1, 2, 5, 5, 10, 10]
E_e = 10.**np.arange(-1., 8.001, 0.1)                                                   # Electron-energies array (0.1 - 10^8 GeV)
p_p = 10.**np.arange(-0.5, 6., 0.1)                                                      # Proton-momenta array

erg2GeV = 624.151

E_SN = 1.e49 * erg2GeV





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
    

def plaw(N_0, gamma, Ecut_inv = 0.):  # powerlaw
    return lambda E: N_0 * (Es[E])**(-gamma) * np.exp(-Es[E] * Ecut_inv)



########################################################################################################################## Load dictionaries


dct_boxes = dio.loaddict('dct/Low_energy_range0/dct_boxes_source.yaml')
dct_lowE = dio.loaddict('dct/Low_energy_range0/dct_lowE_source.yaml')
dct_GALPROP = dio.loaddict('dct/Low_energy_range0/dct_GALPROP_source.yaml')
dct_lowE1 = dio.loaddict('dct/Low_energy_range1/dct_lowE_source.yaml')
dct_lowE2 = dio.loaddict('dct/Low_energy_range2/dct_lowE_source.yaml')
dct_lowE3 = dio.loaddict('dct/Low_energy_range3/dct_lowE_source.yaml')
dct_boxes1 = dio.loaddict('dct/Low_energy_range1/dct_boxes_source.yaml')
dct_boxes2 = dio.loaddict('dct/Low_energy_range2/dct_boxes_source.yaml')
dct_boxes3 = dio.loaddict('dct/Low_energy_range3/dct_boxes_source.yaml')

SED_boxes = dct_boxes['6) Differential_flux_profiles']
SED_lowE = dct_lowE['6) Differential_flux_profiles']
SED_GALPROP = dct_GALPROP['6) Differential_flux_profiles']
SED_lowE1 = dct_lowE1['6) Differential_flux_profiles']
SED_lowE2 = dct_lowE2['6) Differential_flux_profiles']
SED_lowE3 = [0,0] + dct_lowE3['6) Differential_flux_profiles']
SED_boxes1 = dct_boxes1['6) Differential_flux_profiles']
SED_boxes2 = dct_boxes2['6) Differential_flux_profiles']
SED_boxes3 = dct_boxes3['6) Differential_flux_profiles']

dct_boxes_counts = dio.loaddict('dct/Low_energy_range0/dct_boxes_counts_source.yaml')
dct_lowE_counts = dio.loaddict('dct/Low_energy_range0/dct_lowE_counts_source.yaml')
dct_GALPROP_counts = dio.loaddict('dct/Low_energy_range0/dct_GALPROP_counts_source.yaml')
dct_lowE1_counts = dio.loaddict('dct/Low_energy_range1/dct_lowE_counts_source.yaml')
dct_lowE2_counts = dio.loaddict('dct/Low_energy_range2/dct_lowE_counts_source.yaml')
dct_lowE3_counts = dio.loaddict('dct/Low_energy_range3/dct_lowE_counts_source.yaml')
dct_boxes1_counts = dio.loaddict('dct/Low_energy_range1/dct_boxes_counts_source.yaml')
dct_boxes2_counts = dio.loaddict('dct/Low_energy_range2/dct_boxes_counts_source.yaml')
dct_boxes3_counts = dio.loaddict('dct/Low_energy_range3/dct_boxes_counts_source.yaml')

SED_boxes_counts = dct_boxes_counts['6) Differential_flux_profiles']
SED_lowE_counts = dct_lowE_counts['6) Differential_flux_profiles']
SED_GALPROP_counts = dct_GALPROP_counts['6) Differential_flux_profiles']
SED_lowE1_counts = dct_lowE1_counts['6) Differential_flux_profiles']
SED_lowE2_counts = dct_lowE2_counts['6) Differential_flux_profiles']
SED_lowE3_counts = [0,0] + dct_lowE3_counts['6) Differential_flux_profiles']
SED_boxes1_counts = dct_boxes1_counts['6) Differential_flux_profiles']
SED_boxes2_counts = dct_boxes2_counts['6) Differential_flux_profiles']
SED_boxes3_counts = dct_boxes3_counts['6) Differential_flux_profiles']

std_boxes = dct_boxes['7) Standard_deviation_profiles']

total_data_profiles = dio.loaddict('dct/Low_energy_range0/dct_data_counts_source.yaml')['6) Differential_flux_profiles']

Lc = dct_boxes['3) Center_of_lon_bins']
Bc = dct_boxes['4) Center_of_lat_bins']
Es = np.asarray(dct_boxes['5) Energy_bins'])

nB = len(SED_boxes)
nL = len(SED_boxes[0])
nE = len(SED_boxes[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)


expo_dct = dio.loaddict('dct/Low_energy_range0/dct_expo_source.yaml')
exposure_profiles = expo_dct['6) Exposure_profiles'] # shape: (nB, nL, nE)
print "expo_profiles shape: " + str(len(exposure_profiles)) + ", " + str(len(exposure_profiles[0])) + ", " + str(len(exposure_profiles[0][0]))
deltaE = expo_dct['8) deltaE']
dOmega = expo_dct['7) dOmega_profiles']


########################################################################################################################## Define particle-spectra functions

b = latitude

print "Bc[b]: ",  Bc[b]
auxil.setup_figure_pars(plot_type = 'spectrum')
index = 0

    
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

    
for l in [0, 1]:

    V_ROI = dOmega[b][l] * R_GC**2 * line_of_sight
   
    def IC_model(N_0, gamma, Ecut_inv = 0.):
        EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)
        EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
        
        EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
        return lambda E: EdNdE_gamma_IC_vec(Es[E]) * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]

    def pi0_model(N_0, gamma, Ecut_inv = 0.):
        dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
        EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
        EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
        return lambda E: EdNdE_gamma_pi0_vec(Es[E]) * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]
        

########################################################################################################################## Plot SED



    fig = pyplot.figure()
    baseline  = np.asarray(SED_boxes[b][l])
    std = np.asarray(std_boxes[b][l])
    expo_map = np.asarray(exposure_profiles[b][l])
    
    #syst_max = np.maximum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l]])
    #syst_min = np.minimum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l]])
    #syst_min = np.maximum(syst_min, np.zeros_like(syst_min))

    syst_max = np.maximum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], [0,0] + SED_lowE3[b][l], SED_boxes1[b][l], SED_boxes2[b][l], [0,0] + SED_boxes3[b][l]])
    syst_min = np.minimum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], [0,0] + SED_lowE3[b][l], SED_boxes1[b][l], SED_boxes2[b][l], [0,0] + SED_boxes3[b][l]])
    syst_min = np.maximum(syst_min, np.zeros_like(syst_min))

    #syst_max_counts = np.maximum.reduce([SED_boxes_counts[b][l], SED_lowE_counts[b][l], SED_GALPROP_counts[b][l]])
    #syst_min_counts = np.minimum.reduce([SED_boxes_counts[b][l], SED_lowE_counts[b][l], SED_GALPROP_counts[b][l]])
    #syst_min_counts = np.maximum(syst_min_counts, np.zeros_like(syst_min_counts))

    syst_max_counts = np.maximum.reduce([SED_boxes_counts[b][l], SED_lowE_counts[b][l], SED_GALPROP_counts[b][l], SED_lowE1_counts[b][l], SED_lowE2_counts[b][l], [0,0] + SED_lowE3_counts[b][l], SED_boxes1_counts[b][l], SED_boxes2_counts[b][l], [0,0] + SED_boxes3_counts[b][l]])
    syst_min_counts = np.minimum.reduce([SED_boxes_counts[b][l], SED_lowE_counts[b][l], SED_GALPROP_counts[b][l], SED_lowE1_counts[b][l], SED_lowE2_counts[b][l], [0,0] + SED_lowE3_counts[b][l], SED_boxes1_counts[b][l], SED_boxes2_counts[b][l], [0,0] + SED_boxes3_counts[b][l]])
    #syst_min_counts = np.maximum(syst_min_counts, np.zeros_like(syst_min_counts))

    print "syst_max in counts: ", (syst_max * dOmega[b][l] * deltaE * expo_map / Es**2)[fitmin:fitmax]
    print "syst_max_counts: ", syst_max_counts[fitmin:fitmax]

    syst_max_counts = syst_max * dOmega[b][l] * deltaE * expo_map / Es**2
    syst_min_counts = syst_min * dOmega[b][l] * deltaE * expo_map / Es**2
    
    total_data_counts = np.asarray(total_data_profiles[b][l])
    total_data_counts = total_data_counts[len(total_data_counts)-nE:]


    print "b: ", b, ", l: ", l
           
    label = "Rectangles model"

    
    pyplot.errorbar(Es, baseline, std, color=colours[index], marker='s', linestyle = '', label=label)
    pyplot.fill_between(Es, syst_min, syst_max, color = colours[index], alpha = 0.2)

    ls = "-"
    maxmin = "Max: "
    maxmin_label = "max"
    dct = {"x: energies" : Es[fitmin:fitmax]}
    for sig_counts in [syst_max_counts, syst_min_counts]:
        background_counts = total_data_counts - sig_counts



########################################################################################################################## Fit spectra

        
        if fit_plaw:
            def flux_plaw_in_counts(F_0, gamma, Ecut_inv = 0.):  # powerlaw
                return lambda E: (plaw(F_0, gamma, Ecut_inv)(E) * dOmega[b][l] * deltaE[E] * expo_map[E] / Es[E]**2)
            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -      Powerlaw      -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "

            N_0, gamma, Ecut_inv = 6.e-6, 0.1, 0.

            #fit = likelihood1(flux_plaw_in_counts, background_counts, total_data_counts)        # First fit
            #m = Minuit(fit, N_0 = N_0, gamma = gamma, limit_N_0 = (0, 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
            #m.migrad()
            #N_0, gamma  = m.values["N_0"], m.values["gamma"]
            
            fit = likelihood2(flux_plaw_in_counts, background_counts, total_data_counts)        # Second fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, limit_N_0 = (0, 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]
                
            label =  r"PL: $\gamma = %.2f$" %(gamma+2)

            TS_nocut = -2. * np.sum((total_data_counts[E] * np.log(background_counts[E] + flux_plaw_in_counts(N_0, gamma)(E)) - (background_counts[E] + flux_plaw_in_counts(N_0, gamma)(E))) for E in range(fitmin,fitmax))
            TS = TS_nocut
        
            dct_fn = "plot_dct/Min_max_rectangles_source_plaw_l=" + str(Lc[l]) + "_b=" + str(Bc[b]) + dct_fn_ending
            if cutoff:
                while True:
                    Ecut_inv = np.random.random(1) * 1.e-1
                    fit = likelihood_cutoff(flux_plaw_in_counts, background_counts, total_data_counts)     # Third fit
                    m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_gamma = 1., error_Ecut_inv = 1.e-4, limit_N_0 = (0., 1.), errordef = 0.5, limit_Ecut_inv = (0.,1.))
                    m.migrad()
                    N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                    sgm_Ecut = m.errors["Ecut_inv"]
                    
                    dct_fn = "plot_dct/Min_max_rectangles_plaw_source_cutoff_l=" + str(Lc[l]) + "_b=" + str(Bc[b])  + dct_fn_ending
                    dct["E_cut" + maxmin_label] = 1./Ecut_inv
                    dct["sgm_E_cut" + maxmin_label] = - m.errors["Ecut_inv"] / Ecut_inv**2
                    if m.matrix_accurate():
                        break
                    
                print "%"
                print "N_0: ", N_0
                print "gamma: ", (2+gamma)
                print "E_cut: ", (1/Ecut_inv), r" GeV"
                upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
                print "E_95%:  ", (1/upper_bound), r" GeV"
                print "%"

                if (1./Ecut_inv) < 100000.:
                    label = maxmin + r"$\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  %.2f\, \mathrm{TeV}$" %(1./Ecut_inv/1000.) #Ecut in TeV
                    
                else:
                    label = maxmin + r"$\gamma = %.2f$" %(gamma+2)
                TS_cut = -2. * np.sum((total_data_counts[E] * np.log(background_counts[E] + flux_plaw_in_counts(N_0, gamma, Ecut_inv)(E)) - (background_counts[E] + flux_plaw_in_counts(N_0, gamma, Ecut_inv)(E))) for E in range(fitmin,fitmax))
                TS = TS_cut
                print r"$-2\Delta\log(L) = $", TS_nocut, " - ", TS_cut, " = ", (TS_nocut - TS_cut)
                   

            flux_plaw = [plaw(N_0, gamma, Ecut_inv)(E) for E in range(fitmin,fitmax)]


            #print "sig_counts: ", sig_counts[fitmin:fitmax]
            #print "flux_plaw in coutns: ", [flux_plaw_in_counts(N_0, gamma, Ecut_inv)(E) for E in range(fitmin,fitmax)]
            #print "syst_max constchck: ", (syst_max  * dOmega[b][l] * deltaE * expo_map / Es**2)[fitmin:fitmax]
            #print "syst_min constchck: ", (syst_min  * dOmega[b][l] * deltaE * expo_map / Es**2)[fitmin:fitmax]

            
            pyplot.errorbar(Es[fitmin:fitmax], flux_plaw, label = label, color = colours[index], ls = ls)

            if Save_dct:
                      
                dct["y: flux_best_fit_plaw_" + maxmin_label] = np.array(flux_plaw)
                dct["-logL_"+ maxmin_label] = TS

                dct["N_0_" + maxmin_label] = N_0
                dct["gamma_" + maxmin_label] = (gamma + 2)
                dct["-2 Delta logL_" + maxmin_label] = (TS_nocut - TS_cut)
                dct["lower bound E_cut_"+maxmin_label] = (1./upper_bound)
                dct["sgm_N_0_" + maxmin_label] = m.errors["N_0"]
                dct["sgm_gamma_" + maxmin_label] = m.errors["gamma"]
                

            
            ls = "--"
            maxmin = "Min: "
            maxmin_label = "min"
                        


                

        if fit_IC:
            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -  -  -          IC         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "

            N_0, gamma, Ecut_inv = 2.e-13, 0.8, 0.
            
            fit = likelihood1(IC_model, background_counts, total_data_counts)                        # First fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (0.,3.), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]

            fit = likelihood2(IC_model, background_counts, total_data_counts)                        # Second fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = .5, limit_N_0 = (1.e-16, 1.e-7), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]


            
            if cutoff:
                while True:
                    Ecut_inv = np.random.random(1) * 9.e-2
                    fit = likelihood_cutoff(IC_model, background_counts, total_data_counts)            # Third fit
                    m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1.e-13, error_Ecut_inv = 1.e-2, error_gamma = .1, errordef = 0.5, limit_N_0 = (1.e-15, 1.e-7), limit_Ecut_inv = (0.,1.))
                    m.migrad()
                    N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                    if m.matrix_accurate() or no_accurate_matrix:
                        break
                sgm_Ecut = m.errors["Ecut_inv"]
                
                print "%"
                print "N_0 (l.o.s.): ", (N_0*line_of_sight)
                print "gamma: ", (1+gamma)
                print "E_cut: ", (1/Ecut_inv), r" GeV"
                upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
                print "E_95%:  ", (1/upper_bound), r" GeV"
                print "%"
                

            flux_IC = [(IC_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 / dOmega[b][l] / deltaE[E] / expo_map[E]) for E in range(fitmin,fitmax)]

            pyplot.errorbar(Es[fitmin:fitmax], flux_IC, color = colours[index], ls = ':')




        if fit_pi0:

            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -         pi0         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "


            N_0, gamma, Ecut_inv = 2.e-11, 1.6, 0.
            
            fit = likelihood1(pi0_model, background_counts, total_data_counts)              # First fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]

            fit = likelihood2(pi0_model, background_counts, total_data_counts)              # Second fit
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-14, 1.e-7), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]
            
        
            label  = r'$\pi^0:\ \gamma = %.2f$' %gamma #+ r', $-\log L = %.2f$' %TS
               
            
            if cutoff:
                while True:
                    Ecut_inv = np.random.random(1) * 2.e-3
                    fit = likelihood_cutoff(pi0_model, background_counts, total_data_counts)   # Third fit
                    m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1.e-11, error_Ecut_inv = 1., error_gamma = .1, limit_N_0 = (8.e-16, 1.e-7), limit_Ecut_inv = (0.,1.), errordef = 0.5)
                    m.migrad()
                    N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
                    if m.matrix_accurate() or no_accurate_matrix:
                        break
                sgm_Ecut = m.errors["Ecut_inv"]
            
                print "%"
                print "N_0 (l.o.s.): ", (N_0 * line_of_sight)
                print "gamma: ", gamma
                print "E_cut: ", (1/Ecut_inv), r" GeV"
                upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
                print "E_95%:  ", (1/upper_bound), r" GeV"
                print "%"

            flux_pi0 = [(pi0_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 / dOmega[b][l] / deltaE[E] / expo_map[E]) for E in range(fitmin,fitmax)]
        
            pyplot.errorbar(Es[fitmin:fitmax], flux_pi0, color = colours[index], ls = '-.')

      

    
    print "++++++++++++++++++++++++++++++++++++++++++----------------------------+++++++++++++++++++++++++++++++++++++++++"
    if Save_dct:
        dct["Comment: "] = "Min and max model of the FBs for all foreground models considered. Parametric model of min and max described by a powerlaw with spectral index gamma, normalization N_0 and a cutoff E_cut. The corresponding errors output by iminuit are sgm_gamma, sgm_N_0, sgm_E_cut. The 95-percent confidence limit on the cutoff is lower bound E_cut. -2 Delta logL quantifies the difference in likelihood between the powerlaw with and without a cutoff."
        dct["y: flux_min"] = np.array(syst_min)
        dct["y: flux_max"] = np.array(syst_max)
        dio.saveyaml(dct, dct_fn, expand = True)
        


            


########################################################################################################################## Cosmetics, safe plot


    
    ax = fig.add_subplot(111)
    textstr = r'$\ell \in (%i^\circ$' %(Lc[l] - dL/2) + '$,\ %i^\circ)$\n' % (Lc[l] + dL/2) + r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + '$,\ %i^\circ) $' % (Bc[b] + dB[b]/2)
    props = dict( facecolor='white', alpha=1, edgecolor = "white")
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize = 20, verticalalignment='top', bbox=props)


    lg = pyplot.legend(loc='upper right', ncol = 1)
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.xlabel('$E\ \mathrm{[GeV]}$')
    #pyplot.ylabel('Counts')
    pyplot.ylabel(r'$ E^2\frac{\mathrm{d}N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    pyplot.title(titles[l])

    name = 'Summary_SED_b=' + str(int(Bc[b])) + '_l=' + str(int(Lc[l]))
    fn = plot_dir + name + fn_ending
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.ylim((1.e-8,1.e-4))
    pyplot.savefig(fn, format = 'pdf')   

    index += 1

    
    
                    


