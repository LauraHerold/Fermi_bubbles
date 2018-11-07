""" Plots the SED of all models in one latitude stripes. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit
import auxil
import dio
from yaml import load

########################################################################################################################## parameters

data_class = 'source'
binmin = 0
binmax = 31

normalized = False
significance_plot = False

fn_ending = '.pdf'
colours = ['magenta', 'gold', 'blue', 'green', 'black', 'red', 'darkorange', 'cyan', 'grey']
#markers = ['.', '.', '.', '.', '.', '.', 'o', 's', 'D', '.', '.', '.', '.', '.', '.']
ls = [':', '-.', '--', '-', '-', '-', '--', '-.', ':']


########################################################################################################################## Constants

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'

########################################################################################################################## Load dictionaries

dct  = dio.loaddict('dct/Low_energy_range0/dct_data_' + data_class + '_lon180.yaml')
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])


nB = len(diff_profiles)
nL = len(diff_profiles[0])
nE = binmax - binmin +1
print 'nB, nL, nE = ' + str(nB) + ', ' + str(nL) + ', ' + str(nE)


########################################################################################################################## Plot spectra

auxil.setup_figure_pars(plot_type = 'spectrum')
fig = pyplot.figure()
index = 0
b = 7 # Galactic plane


for l in range(9,26,2): # nL = 32
     
    map = [0,0]
    std_map = [0,0]
        
    for ell in [0,1]:
        map[ell]  = np.asarray(diff_profiles[b][l+ell])
        std_map[ell] = np.asarray(std_profiles[b][l+ell])
        
    difference = map[0] - map[1]
            
    total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)
    if Lc[l+1] < 0:
        label = r'$(\!%i\!^\circ\!)$' % Lc[l] + ' $ -\ (\!%i\!^\circ\!)$' % Lc[l+1]
    else:
        label = r'$(%i\!^\circ\!)$' % Lc[l] + ' $ -\ (%i\!^\circ\!)$' % Lc[l+1]
    if normalized:
        difference /= (map[0] + map[1])
        total_std /= (map[0] + map[1])
    if significance_plot:
        difference /= total_std
        if difference[0] < 0:
            difference *= -1
            ls = "--"
        else:
            ls = "-"
        pyplot.plot(Es, difference, linewidth=1.3, label = label)
    else:
        pyplot.errorbar(Es, difference, total_std, linewidth=1.3, label = label, color = colours[index], ls = ls[index])
    index += 1
    
       
########################################################################################################################## cosmetics, safe plot
        


pyplot.grid(True)
pyplot.axhline(0, linestyle='-', color='k') # horizontal lines
pyplot.xlabel(r'$E\ [\mathrm{GeV}]$')
#pyplot.title('Difference of data: left - right')


pyplot.axis('tight')
if normalized:
    name = 'Summary_difference_data_lon_normalized'
    pyplot.ylim((-1, 1))
    pyplot.ylabel(r'$(\mathrm{east}-\mathrm{west})/(\mathrm{east}+\mathrm{west})$')
    lg = pyplot.legend(loc='upper left', ncol=2)
    
    
else:
    name = 'Summary_difference_data_lon_'
    pyplot.ylim((-1.5e-5, 2.8e-5))
    pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    lg = pyplot.legend(loc='upper right', ncol=2)
    pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
if significance_plot:
    name = 'Summary_difference_data_lon_sgm'
    pyplot.ylabel(r'$(\mathrm{east}-\mathrm{west})/\sigma$')
    pyplot.ylim((-10, 10))
    lg = pyplot.legend(loc='lower right', ncol=1)

lg.get_frame().set_linewidth(0)
    
fn = plot_dir + name + fn_ending
pyplot.xscale('log')

pyplot.xlim((1., 1.e3))

print 'save figure to file:'
print fn
pyplot.savefig(fn, format = 'pdf')


# Number 2

########################################################################################################################## Plot spectra

fig = pyplot.figure()
index = 0
b = 7 # Galactic plane

auxil.setup_figure_pars(plot_type = 'spectrum')
for l in [1, 3, 5, 7, 27, 29, 31, 33, 35]: # nL = 32
     
    map = [0,0]
    std_map = [0,0]

    if l == 35:
        map[0] = np.asarray(diff_profiles[b][35])
        std_map[0] = np.asarray(std_profiles[b][35])
        map[1]  = np.asarray(diff_profiles[b][0])
        std_map[1] = np.asarray(std_profiles[b][0])

        label = r'$(%i\!^\circ\!)$' % Lc[35] + ' $ -\ (\!%i\!^\circ\!)$' % Lc[0]

        
    else:
        for ell in [0,1]:
            map[ell]  = np.asarray(diff_profiles[b][l+ell])
            std_map[ell] = np.asarray(std_profiles[b][l+ell])

            if Lc[l+1] < 0:
                label = r'$(\!%i\!^\circ\!)$' % Lc[l] + ' $ -\ (\!%i\!^\circ\!)$' % Lc[l+1]
            else:
                label = r'$(%i\!^\circ\!)$' % Lc[l] + ' $ -\ (%i\!^\circ\!)$' % Lc[l+1]
        
    difference = map[0] - map[1]
            
    total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)



    if normalized:
        difference /= (map[0] + map[1])
        total_std /= (map[0] + map[1])
    if significance_plot:
        difference /= total_std
        if difference[0] < 0:
            difference *= -1
            ls = "--"
        else:
            ls = "-"
        pyplot.plot(Es, difference, linewidth=1.3, label = label)
    else:
        pyplot.errorbar(Es, difference, total_std, linewidth=1.3, label = label, color = colours[index], ls = ls[index])
    index += 1
    
       
########################################################################################################################## cosmetics, safe plot
        


pyplot.grid(True)
pyplot.axhline(0, linestyle='-', color='k') # horizontal lines
pyplot.xlabel(r'$E\ [\mathrm{GeV}]$')
#pyplot.title('Difference of data: left - right')


pyplot.axis('tight')
if normalized:
    name = 'Summary_difference_data_lon_normalized_outer_galaxy'
    pyplot.ylim((-1, 1))
    pyplot.ylabel(r'$(\mathrm{east}-\mathrm{west})/(\mathrm{east}+\mathrm{west})$')
    lg = pyplot.legend(loc='upper left', ncol=2)
    
    
else:
    name = 'Summary_difference_data_lon_outer_galaxy'
    pyplot.ylim((-1.5e-5, 2.8e-5))
    pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
    lg = pyplot.legend(loc='upper right', ncol=2)
    pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
if significance_plot:
    name = 'Summary_difference_data_lon_sgm_outer_galaxy'
    pyplot.ylabel(r'$(\mathrm{east}-\mathrm{west})/\sigma$')
    pyplot.ylim((-10, 10))
    lg = pyplot.legend(loc='lower right', ncol=1)

lg.get_frame().set_linewidth(0)
    
fn = plot_dir + name + fn_ending
pyplot.xscale('log')

pyplot.xlim((1., 1.e3))

print 'save figure to file:'
print fn
pyplot.savefig(fn, format = 'pdf')
