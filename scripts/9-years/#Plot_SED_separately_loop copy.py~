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

std_offset = 1.e-7
std_factor = 1.3                                               # Lower for data

input_data = 'lowE'                                            # data, lowE, boxes, GALPROP
plot_diff_leftright = False
data_class = 'source'

fit_plaw = True
fit_IC  = True
fit_pi0 = True

cutoff = False

bin_start_fit = 6                                              # Energy bin where fit starts
binmin = 3
binmax = 16

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

ISFR_heights = [10, 10, 5, 5, 2, 1, 0.5, 0, 0.5, 1, 2, 5, 5, 10, 10]


########################################################################################################################## Load dictionaries

dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) +'/dct_' + input_data + '_' + data_class + '.yaml')

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']

nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = len(diff_profiles[0][0])
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)
#binmax = min(nE-binmin, binmax)

E_e = 10.**np.arange(-1., 8.001, 0.1)                                                   # Electron-energies array (0.1 - 10^8 GeV)
p_p = 10**np.arange(-0.5, 6., 0.1)                                                      # Proton-momenta array
E_g = Es                                                                                # Final photon energies array in GeV


########################################################################################################################## Define powerlaw classes

if fit_plaw: # with chi2 fit with 2 free parameters: spectral index and normalization]
    if cutoff:
        class PowerLawChi2:
            def __init__(self,x,y,sigma, E_zero):
                self.x = x
                self.y = y
                self.sigma = sigma
                self.E_zero = E_zero
            def __call__(self, N_zero, Gamma, E_cut): 
                chi2 = sum((y - N_zero * (x/E_zero)**(-Gamma) * np.exp(-x/E_cut))**2/ sigma**2 for x,y,sigma in zip(self.x, self.y, self.sigma))
                return chi2

    else:
        class PowerLawChi2:
            def __init__(self,x,y,sigma, E_zero):
                self.x = x
                self.y = y
                self.sigma = sigma
                self.E_zero = E_zero
            def __call__(self, N_zero, Gamma): 
                chi2 = sum((y - N_zero * (x/E_zero)**(-Gamma))**2 / sigma**2 for x,y,sigma in zip(self.x, self.y, self.sigma))
                return chi2

if fit_IC:
    if cutoff:
        class IC_chi2:
            def __init__(self,x,y,sigma):
                self.x = x
                self.y = y
                self.sigma = sigma
            def __call__(self, N_0, n, E_cut):
                EdNdE_e = plaw_cut([N_0, n, E_cut])(E_e) # E_cut/c_light???
                EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
                EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
                chi2 = sum((y - x * EdNdE_gamma_IC_vec(x))**2 / sigma**2 for x,y,sigma in zip(self.x, self.y, self.sigma))
                return chi2
    else:
        class IC_chi2: # chi2 fit classes
            def __init__(self,x,y,sigma):
                self.x = x
                self.y = y
                self.sigma = sigma
            def __call__(self, N_0, n):
                EdNdE_e = powerlaw([N_0, n])(E_e)
                EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e) # 1/cm^3s
                EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
                chi2 = sum((y - x * EdNdE_gamma_IC_vec(x))**2 / sigma**2 for x,y,sigma in zip(self.x, self.y, self.sigma))
                return chi2

if fit_pi0:
    if cutoff:
        class pi0_chi2:
            def __init__(self,x,y,sigma):
                self.x = x
                self.y = y
                self.sigma = sigma
            def __call__(self, N_0, n, E_cut):
                dNdp_p = plaw_cut([N_0, n, E_cut])(p_p)
                EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p)
                EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
                chi2 = sum((y - x * EdNdE_gamma_pi0_vec(x))**2 / sigma**2 for x,y,sigma in zip(self.x, self.y, self.sigma))
                return chi2
    else:
        class pi0_chi2:
            def __init__(self,x,y,sigma):
                self.x = x
                self.y = y
                self.sigma = sigma
            def __call__(self, N_0, n):
                dNdp_p = powerlaw([N_0, n])(p_p)
                EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p) # in 1/cm^3/s
                EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
                chi2 = sum((y - x * EdNdE_gamma_pi0_vec(x))**2 / sigma**2 for x,y,sigma in zip(self.x, self.y, self.sigma))
                return chi2

def powerlaw(pars): # power law a * x^b with parameters a,b
    return lambda x: pars[0] * x**pars[1]

def plaw_cut(pars): # power law a * x^b with parameters a,b
    return lambda x: pars[0] * x**pars[1] * np.exp(-x/pars[2])


########################################################################################################################## Load particle data

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
    
    T_CMB = 2.73 * kB                                                                           # CMB temperature
    E_irf = np.e**np.arange(np.log(E_irf_galaxy[len(E_irf_galaxy)-1]), -6.* np.log(10.), -dlogE_irf)[:0:-1] # CMB-energies array with same log bin size as IRF_galaxy in eV
    irf_CMB = gamma_spectra.thermal_spectrum(T_CMB)                                             # Use thermal_spectrum from gamma_spectra.py, returns IRF in eV/cm^3
    EdNdE_CMB = irf_CMB(E_irf) / E_irf                                                          # in 1/cm^3

    EdNdE_irf = EdNdE_CMB + np.append(np.zeros(len(E_irf)-len(E_irf_galaxy)), EdNdE_irf_galaxy) # Differential flux in 1/cm^3 

    
########################################################################################################################## Plot profiles


    
    for l in xrange(nL):
        map  = np.asarray(diff_profiles[b][l])
        std_map = np.asarray(std_profiles[b][l])
        std_first_fit =  np.asarray(std_profiles[b][l])

        for E in xrange(nE):
            std_map[E] *= std_factor # Accounts for modelling uncertainties (otherwise fit does not work due to small errorbars)
            std_first_fit[E] = std_map[E] + std_offset
            if map[E] < 1.e-8:
                std_map[E] = std_offset
                map[E] = 0.
                

        label = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2)
        pyplot.errorbar(Es, map, std_map, color=colours[colour_index], marker='s', markersize=4, markeredgewidth=0.4, linestyle = '', linewidth=0.1, label=label)




########################################################################################################################## Fit spectra



        if fit_plaw:
            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -      Powerlaw      -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "
            
             
            if cutoff:
                #First chi2 fit            
                chi2 = PowerLawChi2(Es[binmin:binmax], map[binmin:binmax], std_first_fit[binmin:binmax], E_zero)
                N_zero = int(len(map)/2) 
                m = Minuit(chi2, N_zero = N_zero, error_N_zero = 0.00001, Gamma = 0., error_Gamma = 0.1, E_cut = 1., limit_E_cut = (0,1.e20), error_E_cut = 10., errordef = 1.)
                m.migrad()
            
                Gamma = m.values['Gamma'] # Spectral index
                N_zero = m.values['N_zero']
                E_cut = m.values['E_cut']

                # Second chi2 fit
                chi2 = PowerLawChi2(Es[binmin:binmax], map[binmin:binmax], std_map[binmin:binmax], E_zero)
                m = Minuit(chi2, N_zero = N_zero, error_N_zero = 0.00001, Gamma = Gamma, error_Gamma = 0.1, E_cut = E_cut, limit_E_cut = (0,1.e20), error_E_cut = 10., errordef = 1.)
                m.migrad()
            
                Gamma = m.values['Gamma'] # Spectral index
                N_zero = m.values['N_zero']
                E_cut = m.values['E_cut']
            
                chi2_value = sum((map[binmin:binmax] - N_zero * (Es[binmin:binmax]/E_zero)**(-Gamma) * np.exp(- Es[binmin:binmax]/E_cut))**2 / std_map[binmin:binmax]**2)
                dof = binmax - binmin - 3
                label = r'$\mathrm{PL}:\ n = %.2f,$ ' %(-2-Gamma) + r'$E_{\mathrm{cut}} = %.1e\ \mathrm{GeV},$ ' %E_cut + r'$\frac{\chi^2}{\mathrm{d.o.f.}} = %.1f$' %(chi2_value / dof)
                pyplot.errorbar(Es[binmin:binmax], [N_zero * (x/E_zero)**(-Gamma) * np.exp(-x/E_cut) for x in Es[binmin:binmax]], label = label, color = colours[colour_index])


            else:
                 
                 #First chi2 fit            
                 chi2 = PowerLawChi2(Es[binmin:binmax], map[binmin:binmax], std_first_fit[binmin:binmax], E_zero)
                 N_zero = int(len(map)/2) 
                 m = Minuit(chi2, N_zero = N_zero, error_N_zero = 0.00001, Gamma = 0., error_Gamma = 0.1, errordef = 1.)
                 m.migrad()
            
                 Gamma = m.values['Gamma'] # Spectral index
                 N_zero = m.values['N_zero']

                 # Second chi2 fit
                 chi2 = PowerLawChi2(Es[binmin:binmax], map[binmin:binmax], std_map[binmin:binmax], E_zero)
                 m = Minuit(chi2, N_zero = N_zero, error_N_zero = 0.00001, Gamma = Gamma, error_Gamma = 0.1, errordef = 1.)
                 m.migrad()
            
                 Gamma = m.values['Gamma'] # Spectral index
                 N_zero = m.values['N_zero']
            
                 chi2_value = sum((map[binmin:binmax] - N_zero * (Es[binmin:binmax]/E_zero)**(-Gamma))**2 / std_map[binmin:binmax]**2)
                 dof = binmax - binmin - 2
                 label = r'$\mathrm{PL}:\ \gamma = %.2f, $' %Gamma + r'$\frac{\chi^2}{\mathrm{d.o.f.}} = %.1f$' %(chi2_value / dof)
                 pyplot.errorbar(Es[binmin:binmax], [N_zero * (x / E_zero)**(-Gamma) for x in Es[binmin:binmax]], label = label, color = colours[colour_index])

                 
                 
                

                 


        if fit_IC:
            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -  -  -          IC         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "
            
            
            if cutoff:
                 # First chi2 fit
                chi2 = IC_chi2(Es[binmin:binmax], map[binmin:binmax], std_first_fit[binmin:binmax])
                m_IC = Minuit(chi2, N_0 = 1.e+9, n= -2., E_cut = 1., error_N_0 = 1000., limit_E_cut = (0.,1.e20), error_n = 1., error_E_cut = 100., errordef = 10.)
                m_IC.migrad()

                N_0 = m_IC.values['N_0']
                n =  m_IC.values['n']
                E_cut = m_IC.values['E_cut']

                # Second chi2 fit
                chi2 = IC_chi2(Es[binmin:binmax], map[binmin:binmax], std_map[binmin:binmax])
                m_IC = Minuit(chi2, N_0 = N_0, n = n, E_cut = E_cut, error_N_0 = 1000., limit_E_cut = (0.,1.e20), error_n = 1., error_E_cut = 100., errordef = 10.)
                m_IC.migrad()

                N_0 = m_IC.values['N_0']
                n =  m_IC.values['n']
                E_cut = m_IC.values['E_cut']
            
                EdNdE_e = plaw_cut([N_0, n, E_cut])(E_e)
                dof_IC = binmax - binmin - 3
                labl_IC = r'$\mathrm{IC}:\ n = %.2f,$ ' %n + r'$E_\mathrm{cut} = %.1e\ \mathrm{GeV},$ ' %E_cut
                

            else:
                # First chi2 fit
                chi2 = IC_chi2(Es[binmin:binmax], map[binmin:binmax], std_first_fit[binmin:binmax])
                m_IC = Minuit(chi2, N_0 = 1.e+9, n= -2., error_N_0 = 1000., error_n = 1.,  errordef = 10.)
                m_IC.migrad()

                N_0 = m_IC.values['N_0']
                n =  m_IC.values['n']

                # Second chi2 fit
                chi2 = IC_chi2(Es[binmin:binmax], map[binmin:binmax], std_map[binmin:binmax])
                m_IC = Minuit(chi2, N_0 = N_0, n = n, error_N_0 = 1000., error_n = 1., errordef = 10.)
                m_IC.migrad()

                N_0 = m_IC.values['N_0']
                n =  m_IC.values['n']
            
                EdNdE_e = powerlaw([N_0, n])(E_e)
                dof_IC = binmax - binmin - 2
                labl_IC = r'$\mathrm{IC}:\ n = %.2f,$ ' %m_IC.values['n']

            EdNdE_gamma_IC =  gamma_spectra.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e) # 1/cm^3/s
            EdNdE_gamma_IC_vec = np.frompyfunc(EdNdE_gamma_IC, 1, 1)
            gamma_spec_IC = E_g * EdNdE_gamma_IC_vec(E_g) # GeV/cm^3/s
            
            chi2_value_IC = sum((map[binmin:binmax] - gamma_spec_IC[binmin:binmax])**2 / std_map[binmin:binmax]**2)

            pyplot.errorbar(E_g[binmin:binmax], gamma_spec_IC[binmin:binmax], color = colours[colour_index], label=labl_IC + r'$\frac{\chi^2}{\mathrm{d.o.f.}} = %.1f$' % (chi2_value_IC/dof_IC), ls = '--')




            

        if fit_pi0:

            print " "
            print " "
            print "-  -  -  -  -  -  -  -  -  -         pi0         -  -  -  -  -  -  -  -  -  -  -  -  -  -"
            print " "

    
            if cutoff:

                 # First chi2 fit           
                chi2 = pi0_chi2(Es[binmin:binmax], map[binmin:binmax], std_first_fit[binmin:binmax])
                m_pi0 = Minuit(chi2, N_0 = 5.e+10, n= -2.5, E_cut = 1000000., limit_E_cut = (0.,1.e20), error_N_0 = 1000., error_n = 0.1, error_E_cut = 100., errordef = 1.)
                #m_pi0 = Minuit(chi2, N_0 = N_zero * 1.e17, n= -2-Gamma, E_cut = 1., limit_E_cut = (0.,1.e20), error_N_0 = 1000., error_n = 0.1, error_E_cut = 100., errordef = 1.)
                m_pi0.migrad()

                N_0 = m_pi0.values['N_0']
                n = m_pi0.values['n']
                E_cut = m_pi0.values['E_cut']
            
    
                
                #Second chi2 fit
                chi2 = pi0_chi2(Es[binmin:binmax], map[binmin:binmax], std_map[binmin:binmax])
                m_pi0 = Minuit(chi2, N_0 = N_0, n= n, E_cut = E_cut, limit_E_cut = (0.,1.e20), error_N_0 = 1000., error_n = 0.1, error_E_cut = 100., errordef = 1.)
                m_pi0.migrad()

                N_0 = m_pi0.values['N_0']
                n= m_pi0.values['n']
                E_cut = m_pi0.values['E_cut']

            
                dNdp_p = plaw_cut([N_0, n, E_cut])(p_p)
                dof_pi0 = binmax - binmin - 3
                labl_pi0 = r'$\pi^0:\ n = %.2f,$ ' %n + r'$p_\mathrm{cut} = %.1e\ \mathrm{GeV},$ ' % E_cut

            else:
               # First chi2 fit           
                chi2 = pi0_chi2(Es[binmin:binmax], map[binmin:binmax], std_first_fit[binmin:binmax])
                m_pi0 = Minuit(chi2, N_0 = 5.e+10, n= -2.5, error_N_0 = 1000., error_n = 0.1, errordef = 1.)
                m_pi0.migrad()

                N_0 = m_pi0.values['N_0']
                n = m_pi0.values['n']

                
                #Second chi2 fit
                chi2 = pi0_chi2(Es[binmin:binmax], map[binmin:binmax], std_map[binmin:binmax])
                m_pi0 = Minuit(chi2, N_0 = N_0, n= n, error_N_0 = 1000., error_n = 0.1, errordef = 1.)
                m_pi0.migrad()

                N_0 = m_pi0.values['N_0']
                n= m_pi0.values['n']
            
                dNdp_p = powerlaw([N_0, n])(p_p)
                dof_pi0 = binmax - binmin - 2
                labl_pi0 = r'$\pi^0:\ n = %.2f,$ ' %m_pi0.values['n']

            EdNdE_gamma_pi0 = gamma_spectra.EdQdE_pp(dNdp_p, p_p) # 1/cm^3/s
            EdNdE_gamma_pi0_vec = np.frompyfunc(EdNdE_gamma_pi0, 1, 1)
            gamma_spec_pi0 = E_g * EdNdE_gamma_pi0_vec(E_g) # GeV/cm^3/s
            
            chi2_value_pi0 = sum((map[binmin:binmax] - gamma_spec_pi0[binmin:binmax])**2 / std_map[binmin:binmax]**2)

            pyplot.errorbar(E_g[binmin:binmax], gamma_spec_pi0[binmin:binmax], color = colours[colour_index], label=labl_pi0 + r'$\frac{\chi^2}{\mathrm{d.o.f.}} = %.1f$' % (chi2_value_pi0/dof_pi0), ls = '-.')

        colour_index += 1
        
########################################################################################################################## Plot difference right - left

        
    if plot_diff_leftright:

        map = [0,0]
        std_map = [0,0]
        
        for ell in xrange(nL):
            map[ell]  = np.asarray(diff_profiles[b][ell])
            std_map[ell] = np.asarray(std_profiles[b][ell])

        difference = map[0] - map[1]
        
        total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)
        label_diff = 'difference right - left'
        
        for reading_point in range(len(difference)):
            if difference[reading_point] < 0:
                ms = 4.
                pyplot.errorbar(Es[reading_point], -difference[reading_point], total_std[reading_point], color='lightgrey', marker='>', markersize=ms, markeredgewidth=0.4, linestyle=':', linewidth=0.1)
            else:
                ms = 6.
                pyplot.errorbar(Es[reading_point], difference[reading_point], total_std[reading_point], color='grey', marker='>', markersize=ms, markeredgewidth=0.4, linestyle=':', linewidth=0.1, label=label_diff)
                label_diff = None


                    
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

