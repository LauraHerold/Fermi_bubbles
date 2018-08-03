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
parser.add_option("-E", "--lowE_range", dest="lowE_range", default='0', help="There are 3 low-energy ranges: (3,5), (3,3), (4,5), (6,7)")

(options, args) = parser.parse_args()

data_class = str(options.data_class)
low_energy_range = int(options.lowE_range) # 0: baseline


########################################################################################################################## Constants

input_data = ['data', 'lowE', 'boxes', 'GALPROP'] 
labels_dct = {"data":"Data", "lowE": "LowE", "boxes": "Rectangles", "GALPROP":"GALPROP"}

plot_diff_leftright = True

fn_ending = '.pdf'
colours = ['black', 'blue', 'red', 'green']
markers = ['s', 'o', 'D', '<']
titles = ['West', 'East']

dL = 10.
dB = [10., 10., 10., 10., 10., 4., 4., 4., 4., 4., 10., 10., 10., 10., 10.]

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
plot_dir = '../../plots/Plots_9-year/Low_energy_range'+ str(low_energy_range) +'/'

########################################################################################################################## Load dictionaries


dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range)  +'/dct_' + input_data[0] + '_' + data_class + '.yaml')
diff_profiles = dct['6) Differential_flux_profiles']

Lc = dct['3) Center_of_lon_bins']
Bc = dct['4) Center_of_lat_bins']

Es = np.asarray(dct['5) Energy_bins'])


nB = len(diff_profiles)
nL = len(diff_profiles[0])



########################################################################################################################## Plot spectra

for b in xrange(nB):
         
    for l in xrange(nL):
        auxil.setup_figure_pars(plot_type = 'spectrum')
        fig = pyplot.figure()
        colour_index = 0
        marker_index = 0
        
        for input in input_data:    
            print input
            
            dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range)  +'/dct_' + input + '_' + data_class + '.yaml')
            diff_profiles = dct['6) Differential_flux_profiles']
            std_profiles = dct['7) Standard_deviation_profiles']
        
            map  = np.asarray(diff_profiles[b][l])
            std_map = np.asarray(std_profiles[b][l])    
            

########################################################################################################################## plot energy flux with error bars

            label = labels_dct[input]
            #print Es, map, std_map
            pyplot.errorbar(Es, map, std_map, color=colours[colour_index], marker=markers[marker_index], linestyle='-', label=label)
            colour_index += 1
            marker_index += 1
            
########################################################################################################################## Plot difference between right and left

        
        if plot_diff_leftright:
            print 'difference data'
            dct  = dio.loaddict('dct/Low_energy_range' + str(low_energy_range)  +'/dct_' + 'data_' + data_class + '.yaml')
            diff_profiles = dct['6) Differential_flux_profiles']
            std_profiles = dct['7) Standard_deviation_profiles']

            map = [0,0]
            std_map = [0,0]
        
            for ell in xrange(nL):
                map[ell]  = np.asarray(diff_profiles[b][ell])
                std_map[ell] = np.asarray(std_profiles[b][ell])

            difference = map[0] - map[1]

            total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)

            lab0 = lab1 = 'Difference data'
            
            for reading_point in range(len(difference)):

                if difference[reading_point] > 0 and l==0:
                    pyplot.errorbar(Es[reading_point], difference[reading_point], total_std[reading_point], color='grey', marker='>', linestyle=':', label = lab0)
                    lab0 = None
                if difference[reading_point] < 0 and l==1:
                    pyplot.errorbar(Es[reading_point], difference[reading_point], total_std[reading_point], color='grey', marker='>', linestyle=':', label = lab1)
                    lab1 = None
                lab1 = None
                
    
       
########################################################################################################################## cosmetics, safe plot
        

        ax = fig.add_subplot(111)
        textstr = r'$\ell \in (%i^\circ$' % (Lc[l] - dL/2) + '$,\ %i^\circ)$\n' % (Lc[l] + dL/2) + r'$b \in (%i^\circ$' % (Bc[b] - dB[b]/2) + '$, %i^\circ)$' % (Bc[b] + dB[b]/2)
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize = 20, verticalalignment='top')

        lg = pyplot.legend(loc='upper right', ncol=1)
        lg.get_frame().set_linewidth(0)
        #pyplot.grid(True)
        pyplot.xlabel('$E\ [\mathrm{GeV}]$')
        pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
        pyplot.title(titles[l])
        auxil.setup_figure_pars(plot_type = 'spectrum')
        pyplot.axis('tight')
        name = 'SED_all_models_' + data_class + '_l=' + str(int(Lc[l])) + '_b=' + str(int(Bc[b]))
        fn = plot_dir + name + fn_ending
        pyplot.xscale('log')
        pyplot.yscale('log')

        if dB[b] == 10:
            pyplot.ylim((1.e-8,1.e-4))
        else:
            pyplot.ylim((1.e-8,2.e-3))

        pyplot.xlim((1., 2.e3))

        pyplot.savefig(fn, format = 'pdf')

        pyplot.close(fig)
        print 'plotted '+ str(l)

