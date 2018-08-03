import numpy as np
import pyfits
from matplotlib import pyplot
import healpylib as hlib
import dio
from yaml import load
import auxil
from iminuit import Minuit
from scipy import special
import gamma_spectra

########################################################################################################################## Parameters


latitude = 7# 7 is GP

fn_ending = ".pdf"

fit_plaw = True
fit_IC  = False
fit_pi0 = False

########################################################################################################################## Constants

colours = ["blue", "red"]
lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]
titles = ["West", "East"]


dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

c_light = 2.9979e8                                                                      # m/s speed of light
h_Planck = 4.1357e-15                                                                   # eV * s Planck constant
kB = 8.6173303e-5                                                                       # eV/K
T_CMB = 2.73 * kB                                                                       # CMB temperature
GeV2MeV = 1000.
delta = 0.3837641821164575                                                              # logarithmic distance between two energy bins
kpc2cm = 3.086e21
R_GC = 8.5 * kpc2cm # cm
#V_ROI = 4.e64 #cm^3
line_of_sight = 1.5 * kpc2cm
plot_dir = '../../plots/Plots_9-year/'
dOmega_tot = 0.012   # dOmega of unmasked ROI

fitmin = 3
fitmax = 18                                                    # bins: 0-15 for low-energy range 3, 0-17 else


################################################################################################### Define likelihood class and powerlaw fct

def plaw(N_0, gamma, Ecut_inv = 0.):  # powerlaw
    return lambda E: N_0 * (Es[E])**(-gamma) * np.exp(-Es[E] * Ecut_inv)


class likelihood_1:                                                        # First fit: over 4 energy bins only
    def __init__(self, model_fct, background_map, total_data_map):
        self.model_fct = model_fct
        self.background_map = background_map
        self.total_data_map = total_data_map
    def __call__(self, N_0, gamma):
        background_map  = self.background_map
        model_fct = self.model_fct
        total_data_map = self.total_data_map
        L =  sum(background_map[E] + model_fct(N_0, gamma)(E) - total_data_map[E] * np.log(background_map[E] + model_fct(N_0, gamma)(E)) for E in range(fitmin + 2,fitmin + 4))
        print "N_0, gamma: " + str(N_0) + ", " + str(gamma) + " --> " + str(L)
        return L


class likelihood_2:                                                       # Second fit: without cutoff  
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


########################################################################################################################## Plot


b = latitude
print "Bc[b]: ",  Bc[b]
auxil.setup_figure_pars(plot_type = 'spectrum')

index = 0

E_e = 10.**np.arange(-1., 8.001, 0.1)                                                   # Electron-energies array (0.1 - 10^8 GeV)
p_p = 10.**np.arange(-0.5, 6., 0.1)                                                      # Proton-momenta array

IRFmap_fn = '../../data/ISRF_flux/Standard_0_0_0_Flux.fits.gz'   # Model for the ISRF
hdu = pyfits.open(IRFmap_fn)                                                                # Physical unit of field: 'micron'
wavelengths = hdu[1].data.field('Wavelength') * 1.e-6                                       # in m
E_irf_galaxy = c_light * h_Planck / wavelengths[::-1]                                       # Convert wavelength in eV, invert order
EdNdE_irf_galaxy = hdu[1].data.field('Total')[::-1] / E_irf_galaxy                          # in 1/cm^3. Since unit of 'Total': eV/cm^3
dlogE_irf = 0.0230258509299398                                                              # Wavelength bin size

E_irf = np.e**np.arange(np.log(E_irf_galaxy[len(E_irf_galaxy)-1]), -6.* np.log(10.), -dlogE_irf)[:0:-1] # CMB-energies array with same log bin size as IRF_galaxy in eV
irf_CMB = gamma_spectra.thermal_spectrum(T_CMB)                                             # Use thermal_spectrum from gamma_spectra.py, returns IRF in eV/cm^3
EdNdE_CMB = irf_CMB(E_irf) / E_irf                                                          # in 1/cm^3

EdNdE_irf = EdNdE_CMB + np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), EdNdE_irf_galaxy) # Differential flux in 1/cm^3

#print "deltaE: ", Es/deltaE

print "dOmega: ", dOmega    
for l in xrange(nL):
    
    #print "V/R^2: ", (V_ROI/R_GC**2/line_of_sight)

    V_ROI = dOmega[b][l] * R_GC**2 * line_of_sight
    expo_map = np.asarray(exposure_profiles[b][l])

    def IC_model(N_0, gamma, Ecut_inv = 0.):
        EdNdE_e = N_0 * E_e**(-gamma) * np.exp(-E_e * Ecut_inv)
        EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
        
        EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
        return lambda E: EdNdE_gamma_IC_vec(Es[E]) * V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]
    #* V_ROI / (4. * R_GC**2 * np.pi) * Es[E] / dOmega[b][l]

    def pi0_model(N_0, gamma, Ecut_inv = 0.):
        dNdp_p = N_0 * p_p**(-gamma) * np.exp(-p_p * Ecut_inv)
        EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
        EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
        return lambda E: EdNdE_gamma_pi0_vec(Es[E])* V_ROI * exposure_profiles[b][l][E] / (4. * R_GC**2 * np.pi) * deltaE[E] / Es[E]
    #* V_ROI / (4. * R_GC**2 * np.pi) *  Es[E] / dOmega[b][l]
    
    fig = pyplot.figure()
    baseline  = np.asarray(SED_boxes[b][l])
    std = np.asarray(std_boxes[b][l])
    
    #syst_max = np.maximum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], SED_boxes1[b][l], SED_boxes2[b][l]])
    #syst_min = np.minimum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], SED_boxes1[b][l], SED_boxes2[b][l]])

    syst_max = np.maximum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], [0,0] + SED_lowE3[b][l], SED_boxes1[b][l], SED_boxes2[b][l], [0,0] + SED_boxes3[b][l]])
    syst_min = np.minimum.reduce([SED_boxes[b][l], SED_lowE[b][l], SED_GALPROP[b][l], SED_lowE1[b][l], SED_lowE2[b][l], [0,0] + SED_lowE3[b][l], SED_boxes1[b][l], SED_boxes2[b][l], [0,0] + SED_boxes3[b][l]])
    syst_min = np.maximum(syst_min, np.zeros_like(syst_min))

    syst_max_counts = np.maximum.reduce([SED_boxes_counts[b][l], SED_lowE_counts[b][l], SED_GALPROP_counts[b][l], SED_lowE1_counts[b][l], SED_lowE2_counts[b][l], [0,0] + SED_lowE3_counts[b][l], SED_boxes1_counts[b][l], SED_boxes2_counts[b][l], [0,0] + SED_boxes3_counts[b][l]])
    syst_min_counts = np.minimum.reduce([SED_boxes_counts[b][l], SED_lowE_counts[b][l], SED_GALPROP_counts[b][l], SED_lowE1_counts[b][l], SED_lowE2_counts[b][l], [0,0] + SED_lowE3_counts[b][l], SED_boxes1_counts[b][l], SED_boxes2_counts[b][l], [0,0] + SED_boxes3_counts[b][l]])
    syst_min_counts = np.maximum(syst_min_counts, np.zeros_like(syst_min_counts))

    #syst_max_counts = syst_max * exposure_profiles[b][l] * deltaE /Es**2 * dOmega[b][l]
    #syst_min_counts = syst_min * exposure_profiles[b][l] * deltaE /Es**2 * dOmega[b][l]

    total_data_map = np.asarray(total_data_profiles[b][l])
    total_data_map = total_data_map[len(total_data_map)-nE:]

    print "syst_max_counts: ", syst_max_counts
    print "SED_boxes_counts[b][l]: ", SED_boxes_counts[b][l]
    
    print "b: ", b, ", l: ", l
           
    label = "Rectangles model"

    
    pyplot.errorbar(Es, baseline, std, color=colours[index], marker='s', linestyle = '', label=label)
    pyplot.fill_between(Es, syst_min, syst_max, color = colours[index], alpha = 0.2)

########################################################################################################################## Fit plaw

    if fit_plaw:
        print "---------------------------- PL  ------------------------------------"
        def flux_plaw_in_counts(F_0, gamma, Ecut_inv = 0.):  # powerlaw
            return lambda E: (plaw(F_0, gamma, Ecut_inv)(E) * dOmega[b][l] * deltaE[E] * expo_map[E] / Es[E]**2)

        data_points = syst_max_counts
        background_map = total_data_map - data_points 
 
        N_0, gamma, Ecut_inv = 2.e-5, 0., np.random.random(1)*1.e-3
        fit = likelihood_1(flux_plaw_in_counts, background_map, total_data_map)
        m = Minuit(fit, N_0 = N_0, gamma = gamma, limit_N_0 = (0, 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        #fit = likelihood_2(flux_plaw_in_counts, background_map, total_data_map)    
        #m = Minuit(fit, N_0 = N_0, gamma = gamma, limit_N_0 = (0, 1.), error_N_0 = 1., error_gamma = 1., errordef = 0.5)
        #m.migrad()
        #N_0, gamma  = m.values["N_0"], m.values["gamma"]

        #Ecut_inv =  np.random.random(1) * 1.e-6
        #fit = likelihood_cutoff(flux_plaw_in_counts, background_map, total_data_map)     
        #m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_gamma = 1., error_Ecut_inv = 1.e-4, limit_N_0 = (0., 1.), errordef = 0.5, limit_Ecut_inv = (-0.1,1.))
        #m.migrad()
        #N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
        Ecut_inv = 0.0000000001
            
        #sgm_Ecut = m.errors["Ecut_inv"]
        #upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
        #print "E_cut: ", (1/Ecut_inv)
        #print "Lower bound E_cut: ", (1/upper_bound)
        
        flux_plaw = [plaw(N_0, gamma, Ecut_inv)(E) for E in range(fitmin,fitmax)]

        print "data_points: ", data_points
        print "flux_plaw in counts: ", [flux_plaw_in_counts(N_0, gamma, Ecut_inv)(E) for E in range(fitmin,fitmax)]
        if (1./Ecut_inv) < 100000.:
            label = r"Max: $\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  %.2f\, \mathrm{TeV}$" %(1./Ecut_inv/1000.) #Ecut in TeV
        else:
            label = r"Max: $\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  \infty$"
        pyplot.errorbar(Es[fitmin:fitmax], flux_plaw, color = colours[index], ls = "-", label = label)            



        
        print "---------------------------- PL  ------------------------------------"


        data_points = syst_min_counts
        background_map = total_data_map - data_points
        #print "data_points: ", data_points
        N_0, gamma, Ecut_inv = 6.e-5, 0.1, np.random.random(1)*1.e-3
        fit = likelihood_1(flux_plaw_in_counts, background_map, total_data_map)     
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1.e-15, error_gamma = 0.1, limit_N_0 = (0,1), limit_gamma = (-1,1), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        fit = likelihood_2(flux_plaw_in_counts, background_map, total_data_map)
        m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (0,1), limit_gamma = (-1,1), errordef = 0.5)
        m.migrad()
        N_0, gamma  = m.values["N_0"], m.values["gamma"]

        Ecut_inv =  np.random.random(1) * 1.e-1
        fit = likelihood_cutoff(flux_plaw_in_counts, background_map, total_data_map)
        m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_gamma = 1., error_Ecut_inv = 1.e-4, limit_N_0 = (0,1), limit_gamma = (-1,1), limit_Ecut_inv = (0.,10.), errordef = 0.5)
        m.migrad()
        N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]

            
        sgm_Ecut = m.errors["Ecut_inv"]
        upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
        print "E_cut: ", (1/Ecut_inv)
        print "Lower bound E_cut: ", (1/upper_bound)
        flux_plaw = [plaw(N_0, gamma, Ecut_inv)(E) for E in range(fitmin,fitmax)]
        if (1./Ecut_inv) < 100000.:
            label = r"Min: $\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  %.2f\, \mathrm{TeV}$" %(1./Ecut_inv/1000.) #Ecut in TeV
        else:
            label = r"Min: $\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  \infty$"
        pyplot.errorbar(Es[fitmin:fitmax], flux_plaw, color = colours[index], ls = "--", label = label)  
            
        
        '''
        fit = likelihood_plaw(syst_max)
        N_0 = 1.e-5
        gamma = 0.
        Ecut_inv = 0.
        m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., limit_N_0 = (0., 1.), error_gamma = 1., error_Ecut_inv = 1., errordef = 0.5)
        m.migrad()
        N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
        sgm_Ecut = m.errors["Ecut_inv"]
        upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
        print "E_cut: ", (1/Ecut_inv)
        print "Lower bound E_cut: ", (1/upper_bound)
        label = r"Max: $\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  %.i\, \mathrm{GeV}$" %(1./Ecut_inv)
        pyplot.errorbar(Es[fitmin:fitmax], plaw(N_0, gamma, Ecut_inv)(np.arange(fitmin,fitmax)), color = colours[index], ls = "-", label = label)
        

        fit = likelihood_plaw(syst_min)
        Ecut_inv = 0
        m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., limit_N_0 = (1.e-9, 1.e-4), error_gamma = 1., error_Ecut_inv = 1.e-3, errordef = 0.5)
        m.migrad()
        N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]
        sgm_Ecut = m.errors["Ecut_inv"]
        upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
        print "E_cut: ", (1/Ecut_inv)
        print "Lower bound E_cut: ", (1/upper_bound)
        label = r"Min: $\gamma = %.2f$" %(gamma+2) + r", $E_\mathrm{cut} =  %.i\, \mathrm{GeV}$" %(1./Ecut_inv)
        pyplot.errorbar(Es[fitmin:fitmax], plaw(N_0, gamma, Ecut_inv)(np.arange(fitmin,fitmax)), color = colours[index], ls = "--", label = label)
        '''


########################################################################################################################## IC

    if fit_IC:
        for data_points in (syst_max_counts, syst_min_counts):
            print "---------------------------- IC ------------------------------------"
 
            N_0, gamma, Ecut_inv = 1.e-14, 1.5, 0
            fit = likelihood_1(IC_model, data_points)      
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1.e-15, error_gamma = 0.1, limit_N_0 = (0,1), limit_gamma = (0,5), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]

            fit = likelihood_2(IC_model, data_points)      
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (0,5), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]

            Ecut_inv =  np.random.random(1) * 1.e-4
            fit = likelihood_cutoff(IC_model, data_points)      
            m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_gamma = 1., error_Ecut_inv = 1.e-4, limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (0,5), limit_Ecut_inv = (0.,1.), errordef = 0.5)
            m.migrad()
            N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]

            
            print "N_0: ", (N_0 * line_of_sight)
            sgm_Ecut = m.errors["Ecut_inv"]
            upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
            print "E_cut: ", (1/Ecut_inv)
            print "Lower bound E_cut: ", (1/upper_bound)
            flux_IC = [(IC_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 /dOmega[b][l] /deltaE[E] /exposure_profiles[b][l][E]) for E in range(fitmin,fitmax)]
            pyplot.errorbar(Es[fitmin:fitmax], flux_IC, color = colours[index], ls = "--", label = "IC")
     

########################################################################################################################## pi0

    if fit_pi0:
        for data_points in (syst_max_counts, syst_min_counts):
            print "-----------------------------------------------Pi0--------------------------------------------------"
            N_0, gamma, Ecut_inv = 1.e-9, 2.1, 0
            fit = likelihood_1(pi0_model, data_points)      
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1.e-9, error_gamma = 0.1,  limit_N_0 = (0,1), limit_gamma = (0,5),  errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]

            fit = likelihood_2(pi0_model, data_points)      
            m = Minuit(fit, N_0 = N_0, gamma = gamma, error_N_0 = 1., error_gamma = 1., limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (0,5), errordef = 0.5)
            m.migrad()
            N_0, gamma  = m.values["N_0"], m.values["gamma"]

            Ecut_inv =  np.random.random(1) * 1.e-4
            fit = likelihood_cutoff(pi0_model, data_points)      
            m = Minuit(fit, N_0 = N_0, gamma = gamma, Ecut_inv = Ecut_inv, error_N_0 = 1., error_gamma = 1., error_Ecut_inv = 1.e-4, limit_N_0 = (1.e-16, 1.e-7), limit_gamma = (0,5), limit_Ecut_inv = (0.,1.), errordef = 0.5)
            m.migrad()
            N_0, gamma, Ecut_inv  = m.values["N_0"], m.values["gamma"], m.values["Ecut_inv"]

            
            print "N_0 (l.o.s.): ", (N_0 * line_of_sight)
            sgm_Ecut = m.errors["Ecut_inv"]
            upper_bound = Ecut_inv + np.sqrt(2) * special.erfinv(0.9) * sgm_Ecut
            print "E_cut: ", (1/Ecut_inv)
            print "Lower bound E_cut: ", (1/upper_bound)
            flux_pi0 = [(pi0_model(N_0, gamma, Ecut_inv)(E) * Es[E]**2 /dOmega[b][l] /deltaE[E] /exposure_profiles[b][l][E]) for E in range(fitmin,fitmax)]
            pyplot.errorbar(Es[fitmin:fitmax], flux_pi0, color = colours[index], ls = ":", label = "Pi0")            

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
