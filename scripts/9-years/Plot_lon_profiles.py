""" Plots the latitude profiles of all models. """

import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit
from optparse import OptionParser
import dio
from yaml import load
import auxil

########################################################################################################################## parameters

input_data = 'lowE'
colours = ['green', 'red', 'blue']
markers = ['s', 'o', 'D']


parser = OptionParser()
parser.add_option("-c", "--data_class", dest = "data_class", default = "source", help="data class (source or ultraclean)")
parser.add_option("-E", "--low_range", dest="lowE_range", default='0', help="There are 3 low-energy ranges: (3,5), (3,3), (4,5), (6,7)")
(options, args) = parser.parse_args()

data_class = str(options.data_class)
low_energy_range = int(options.lowE_range) # 0: baseline, 4: test

########################################################################################################################## Constants

highE_ranges = ((0,5),(6,11),(12,17))        # 1: 0.3-0.5 GeV, 2: 0.5-1.0 GeV, 3: 1.0-2.2 GeV, 0: baseline (0.3-1.0 GeV)

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range) + '/Fine_dct_' + input_data + '_' + data_class + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']
std_profiles = dct['7) Standard_deviation_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])
deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))

 
nB = len(diff_profiles)
nL = len(diff_profiles[0])
    
   
########################################################################################################################## Plot profiles



       

for highE in range(3):
    binmin = highE_ranges[highE][0]
    binmax = highE_ranges[highE][1]
    nE = binmax - binmin + 1
    auxil.setup_figure_pars(plot_type = 'spectrum')
    f = pyplot.figure()
    colour_index = 0
    marker_index = 0


    
    for b in range(6,9):
        profiles = np.zeros(nL)
        std = np.zeros(nL)
        auxil.setup_figure_pars(plot_type = 'spectrum')

        for l in xrange(nL):
            for E in xrange(nE):
                profiles[l] += diff_profiles[b][l][binmin+E] *  deltaE[E] / Es[E]
                std[l] += (std_profiles[b][l][binmin+E] * deltaE[E] / Es[E])**2
        std = np.sqrt(std)
        print profiles

        label = r'$b \in (%.0f^\circ$' % (Bc[b] - 2) + '$,\ %.0f^\circ)$' % (Bc[b] + 2)
        pyplot.errorbar(Lc[::-1], profiles[::-1], std[::-1], color = colours[colour_index], marker=markers[marker_index], linewidth = 1.3,  label=label)
        pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        colour_index += 1
        marker_index += 1


    emin = Es[binmin] * np.exp(-delta/2)
    emax = Es[binmax] * np.exp(delta/2)
    ax = f.add_subplot(111)
    #pyplot.text(0.1, 0.9,'matplotlib', ha='right', va='top')#, transform=ax.transAxes)
    textstr = r'$E\in (%.0f$' %emin + r'$,\ %.0f$' %emax + r'$)\ \mathrm{GeV}$'
    ax.text(0.59, 0.98, textstr, transform=ax.transAxes, fontsize = 20, verticalalignment='top')
    
    lg = pyplot.legend(loc='upper left', ncol=1, fontsize = 'medium')
    lg.get_frame().set_linewidth(0)
    pyplot.grid(True)
    pyplot.axhline(0, linestyle='-', color='k') # horizontal lines
    pyplot.xlabel('$\ell\ [\mathrm{deg}]$')
    pyplot.ylabel(r'$ I\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')

        
           


    name = 'Lon_profiles_'+ input_data
    fn = plot_dir + '/Low_energy_range'+ str(low_energy_range)+ '/' + name + '_' + data_class + '_range_' + str(highE) +'.pdf'
    #pyplot.yscale('log')
    #pyplot.ylim(3.e-7, 1.e-4)  
    pyplot.ylim(-0.5e-5, 2.5e-5)
    pyplot.xlim(20, -20)
    pyplot.savefig(fn, format = 'pdf')
            
