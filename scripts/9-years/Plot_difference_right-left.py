""" Plots the SED of all models in one latitude stripes. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

import dio
from yaml import load

########################################################################################################################## parameters

low_energy_range = 0
data_class = 'source'
binmin = 0
binmax = 31

fn_ending = '.pdf'
colours = ['grey', 'grey', 'grey', 'grey', 'grey', 'blue', 'green', 'red', 'orange', 'magenta', 'grey', 'grey', 'grey', 'grey', 'grey']
#markers = ['s', 'o', 'D', '<']


########################################################################################################################## Constants

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'

########################################################################################################################## Load dictionaries

dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) + '/dct_data_' + data_class + '.yaml')
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
colour_index = 0
marker_index = 0
fig = pyplot.figure()

for b in xrange(nB):
        
    map = [0,0]
    std_map = [0,0]
    
    for ell in xrange(nL):
        map[ell]  = np.asarray(diff_profiles[b][ell])
        std_map[ell] = np.asarray(std_profiles[b][ell])

    difference = map[0] - map[1]
            
    total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)

    label = r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2)
    pyplot.errorbar(Es, difference, total_std, linewidth=0.1, label = label, color = colours[colour_index])
    colour_index += 1
    
       
########################################################################################################################## cosmetics, safe plot
        

lg = pyplot.legend(loc='upper right', ncol=3, fontsize = 'small')
lg.get_frame().set_linewidth(0)
pyplot.grid(True)
pyplot.axhline(0, linestyle='-', color='k') # horizontal lines
pyplot.xlabel('$E$ [GeV]')
pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
pyplot.title('Difference of data: left - right')

pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
pyplot.axis('tight')
name = 'Difference_data_for_different_latitudes'
fn = plot_dir + name + fn_ending
pyplot.xscale('log')

pyplot.xlim((1., 2.e3))
#pyplot.ylim((-1.e-5, 1.e-5))
    
pyplot.savefig(fn, format = 'pdf')



