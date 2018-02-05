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
import auxil

########################################################################################################################## Parameters



low_energy_range = 0                                           # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)
lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]

input_data = 'lowE'                                           # data, lowE, boxes, GALPROP
data_class = 'source'

fit_plaw = False
fit_IC  = True
fit_pi0 = True

cutoff = True

fn_ending = '.pdf'
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
p_p = 10.**np.arange(-0.5, 6., 0.1)                                                     # Proton-momenta array


########################################################################################################################## Dima's auxil function

def setup_figure_pars(spectrum=False, plot_type=None):
    if plot_type is None:
        if spectrum:
            plot_type = 'spectrum'
        else:
            plot_type = 'map'
    if plot_type == 'spectrum':
        fig_width = 8  # width in inches
        fig_height = 6    # height in inches
    elif plot_type == 'map':
        fig_width = 9  # width in inches
        fig_height = 6    # height in inches
    elif plot_type == 'zoomin_map':
        fig_width = 4.6  # width in inches
        fig_height = 6    # height in inches

    fig_size =  [fig_width, fig_height]
    params = {'axes.labelsize': 20,
              'axes.titlesize': 20,
              'font.size': 16,
              'legend.fontsize': 14,
              'xtick.labelsize':18,
              'ytick.labelsize':18,
              #'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.minor.size' : 3,
              'ytick.minor.size' : 3,
              'figure.subplot.left' : 0.05,
              'figure.subplot.right' : 0.97,
              'figure.subplot.bottom' : 0.15,
              'figure.subplot.top' : 0.9
                }
    pyplot.rcParams.update(params)
    if plot_type == 'spectrum':
        pyplot.rcParams['figure.subplot.left'] = 0.15
        pyplot.rcParams['figure.subplot.right'] = 0.95
        pyplot.rcParams['figure.subplot.bottom'] = 0.1
    elif plot_type == 'zoomin_map':
        pyplot.rcParams['axes.titlesize'] = 20
        pyplot.rcParams['xtick.labelsize'] = 16
        pyplot.rcParams['ytick.labelsize'] = 16
        pyplot.rcParams['font.size'] = 16
        pyplot.rcParams['axes.labelsize'] = 24
        
        pyplot.rcParams['figure.subplot.left'] = 0.03
        pyplot.rcParams['figure.subplot.right'] = 0.99
        pyplot.rcParams['figure.subplot.bottom'] = 0.12
        pyplot.rcParams['figure.subplot.top'] = 0.9
        #pyplot.rcParams['figure.figsize'][0] *= 2./3.

    #rc('text.latex', preamble=r'\usepackage{amsmath}')


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



expo_dct = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) +'/dct_expo_' + data_class + '.yaml')
exposure_profiles = expo_dct['6) Exposure_profiles'] # shape: (nB, nL, nE)
print "expo_profiles shape: " + str(len(exposure_profiles)) + ", " + str(len(exposure_profiles[0])) + ", " + str(len(exposure_profiles[0][0]))
deltaE = expo_dct['8) deltaE']
dOmega = expo_dct['7) dOmega_profiles']


########################################################################################################################## Read SED from dcts and plot

for b in xrange(nB):
    auxil.setup_figure_pars(plot_type = 'spectrum')
    pyplot.figure()
    colour_index = 0
    
    for l in xrange(nL):
        
        map  = np.asarray(diff_profiles[b][l])
        std_map = np.asarray(std_profiles[b][l])
        expo_map = np.asarray(exposure_profiles[b][l])
                
        label = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + r', $%i^\circ)$' % (Lc[l] + dL/2)
        flux_map = map * Es**2 / dOmega[b][l] / deltaE / expo_map
        flux_std_map = std_map * Es**2 / dOmega[b][l] / deltaE / expo_map

        pyplot.errorbar(Es, flux_map, flux_std_map, color=colours[colour_index], marker='s', markersize=4, markeredgewidth=0.4, linestyle = '', linewidth=0.1, label=label)


########################################################################################################################## Fit spectra



        if fit_plaw:
            if cutoff:
                dct = dio.loaddict('plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_Plaw_cutoff_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml')
                x, y = dct["x"], dct["y"]                
                N_0, gamma, E_cut  = dct["N_0"], dct["gamma"], dct["E_cut"]
                chi2_dof, TS = dct["chi^2/d.o.f."], dct["-logL"]
                label = r'$\mathrm{PL}:\ \gamma = %.2f,$ ' %(gamma+2.) + r'$E_{\mathrm{cut}} = %.1e\ \mathrm{GeV}$ ' %E_cut + ',\n' + r'$-\log L = %.2f$' %TS + r', $\frac{\chi^2}{\mathrm{d.o.f.}} = %.2f$' %(chi2_dof)
                pyplot.errorbar(x, y, label = label, color = colours[colour_index])

            else:
                dct = dio.loaddict('plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_Plaw_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml')
                x, y = dct["x"], dct["y"]                
                N_0, gamma = dct["N_0"], dct["gamma"]
                chi2_dof, TS = dct["chi^2/d.o.f."], dct["-logL"]
                label = r'$\mathrm{PL}:\ \gamma = %.2f,$ ' %(gamma+2.)  + ',\n' + r'$-\log L = %.2f$' %TS + r', $\frac{\chi^2}{\mathrm{d.o.f.}} = %.2f$' %(chi2_dof)
                pyplot.errorbar(x, y, label = label, color = colours[colour_index])
                  
                 


        if fit_IC:
            if cutoff:
                dct = dio.loaddict('plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_IC_cutoff_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml')
                x, y = dct["x"], dct["y"]                
                N_0, gamma, E_cut  = dct["N_0"], dct["gamma"], dct["E_cut"]
                chi2_dof, TS = dct["chi^2/d.o.f."], dct["-logL"]
                label = r'$\mathrm{IC}:\ \gamma = %.2f,$ ' %(gamma+2.) + r'$E_{\mathrm{cut}} = %.1e\ \mathrm{GeV}$ ' %(E_cut) + ',\n' + r'$-\log L = %.2f$' %TS + r', $\frac{\chi^2}{\mathrm{d.o.f.}} = %.2f$' %(chi2_dof)
                pyplot.errorbar(x, y, label = label, color = colours[colour_index], ls = ':')

            else:
                print 'plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_IC_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml'
                IC_dct = dio.loaddict('plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_IC_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml')
                IC_x, IC_y = IC_dct["x"], IC_dct["y"]                
                N_0, gamma = IC_dct["N_0"], IC_dct["gamma"]
                chi2_dof, TS = IC_dct["chi^2/d.o.f."], IC_dct["-logL"]
                label = r'$\mathrm{IC}:\ \gamma = %.2f,$ ' %(gamma+2.)  + ',\n' + r'$-\log L = %.2f$' %TS + r', $\frac{\chi^2}{\mathrm{d.o.f.}} = %.2f$' %(chi2_dof)
                pyplot.errorbar(IC_x, IC_y, label = label, color = colours[colour_index], ls = ':')
            

            


            

        if fit_pi0:
            if cutoff:
                dct = dio.loaddict('plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_pi0_cutoff_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml')
                x, y = dct["x"], dct["y"]                
                N_0, gamma, E_cut  = dct["N_0"], dct["gamma"], dct["E_cut"]
                chi2_dof, TS = dct["chi^2/d.o.f."], dct["-logL"]
                label = r'$\pi^0:\ \gamma = %.2f,$ ' %(gamma+2.) + r'$E_{\mathrm{cut}} = %.1e\ \mathrm{GeV}$ ' %(E_cut) + ',\n' + r'$-\log L = %.2f$' %TS + r', $\frac{\chi^2}{\mathrm{d.o.f.}} = %.2f$' %(chi2_dof)
                pyplot.errorbar(x, y, label = label, color = colours[colour_index], ls = '-.')

            else:
                dct = dio.loaddict('plot_dct/Low_energy_range' + str(low_energy_range) +'/' + input_data + '_'  + data_class + '_pi0_l=' + str(Lc[l]) +'_b=' + str(Bc[b]) + '.yaml')
                x, y = dct["x"], dct["y"]                
                N_0, gamma = dct["N_0"], dct["gamma"]
                chi2_dof, TS = dct["chi^2/d.o.f."], dct["-logL"]
                label = r'$\pi^0:\ \gamma = %.2f,$ ' %(gamma+2.)  + ',\n' + r'$-\log L = %.2f$' %TS + r', $\frac{\chi^2}{\mathrm{d.o.f.}} = %.2f$' %(chi2_dof)
                pyplot.errorbar(x, y, label = label, color = colours[colour_index], ls = '-.')

            

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
    setup_figure_pars(spectrum=True)
    pyplot.savefig(fn, format = 'pdf')


