""" Plots the SED of all models in one latitude stripes. """

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

parser = OptionParser()
parser.add_option("-c", "--data_class", dest = "data_class", default = "source", help="data class (source or ultraclean)")

(options, args) = parser.parse_args()

data_class = str(options.data_class)


########################################################################################################################## Constants

#input_data = ['lowE', 'boxes'] 
input = 'boxes'

fn_ending = '.pdf'
colours = ['black', 'blue', 'grey', 'green','black', 'blue', 'grey', 'green']
markers = ['s', 'o', 'D', '<', None, None, None, None]
lss = [":", ":", ":", ":", "-", "-", "-", "-"]
titles = ["West", "East"]

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/'

lowE_ranges = ["0.3-1.0", "0.3-0.5", "0.5-1.0", "1.0-2.2"]

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('dct/Low_energy_range0/dct_' + input + '_' + data_class + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']
Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

nB = len(diff_profiles)
nL = len(diff_profiles[0])



########################################################################################################################## Plot spectra

for b in xrange(nB):
         
    for l in xrange(nL):
        auxil.setup_figure_pars(plot_type = 'spectrum')
        fig = pyplot.figure()
        index = 0
        
        for data_class in ['source', 'ultraclean']:
            for lowE_range in [0,1,2,3]:                
                dct  = dio.loaddict('dct/Low_energy_range' + str(lowE_range) + '/dct_' + input + '_' + data_class + '.yaml')
                diff_profiles = dct['6) Differential_flux_profiles']
                std_profiles = dct['7) Standard_deviation_profiles']
                Es = np.asarray(dct['5) Energy_bins'])
        
                map  = np.asarray(diff_profiles[b][l])
                std_map = np.asarray(std_profiles[b][l])    
            

########################################################################################################################## plot energy flux with error bars


                
                if index < 4:
                    label = lowE_ranges[lowE_range]+ " GeV"
                else:
                    label = None
                pyplot.errorbar(Es, map, std_map, color=colours[index], marker=markers[index], markersize=6, markeredgewidth=0.4, linestyle=lss[index], linewidth=0.5, label=label)
                index += 1

       
########################################################################################################################## cosmetics, safe plot
        

        ax = fig.add_subplot(111)
        textstr = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + ', $%i^\circ)$\n' % (Lc[l] + dL/2) + r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + ', $%i^\circ)$' % (Bc[b] + dB[b]/2)
        props = dict( facecolor='white', alpha=1, edgecolor = "white")
        ax.text(0.15, 0.97, textstr, transform=ax.transAxes, fontsize = 20, verticalalignment='top', bbox=props)
        
        lg = pyplot.legend(loc='lower left', ncol=1)
        lg.get_frame().set_linewidth(0)
        #pyplot.grid(True)
        pyplot.xlabel('$E\ \mathrm{[GeV]}$')
        pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
        pyplot.title(titles[l])
    
        pyplot.axis('tight')
        name = 'SED_different_lowE_ranges_' + input + '_l=' + str(int(Lc[l])) + '_b=' + str(int(Bc[b]))
        fn = plot_dir + name + fn_ending
        pyplot.xscale('log')
        pyplot.yscale('log')

        if dB[b] == 10:
            pyplot.ylim((5.e-8,1.e-5))
        else:
            pyplot.ylim((6.e-7,2.e-5))

        pyplot.xlim((1., 2.e3))

        pyplot.savefig(fn, format = 'pdf')

        pyplot.close(fig)
        print 'plotted '+ str(Bc[b]) + ' deg'

