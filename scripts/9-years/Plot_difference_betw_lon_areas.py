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

normalized = True
significance_plot = False

fn_ending = '.pdf'
colours = ['grey', 'grey', 'grey', 'grey', 'grey', 'darkorange', 'green', 'red', 'blue', 'magenta', 'grey', 'grey', 'grey', 'grey', 'grey']
markers = ['.', '.', '.', '.', '.', '.', 'o', 's', 'D', '.', '.', '.', '.', '.', '.']
lw = ['1', '1', '1', '1', '1', '5', '5', '5', '5', '5', '1', '1', '1', '1', '1']


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
for l in range(nL-1):
    index = 0
    fig = pyplot.figure()

    for b in xrange(nB):#range(5,10):
        
        map = [0,0]
        std_map = [0,0]
        
        for ell in [0,1]:
             map[ell]  = np.asarray(diff_profiles[b][l+ell])
             std_map[ell] = np.asarray(std_profiles[b][l+ell])
        
        difference = map[0] - map[1]
            
        total_std = np.sqrt(std_map[0]**2 + std_map[1]**2)
        label = None
        if colours[index] != "grey":
            label = r'$b \in (%i\!^\circ$' % (Bc[b] - dB[b]/2) + ', $%i\!^\circ\!)$' % (Bc[b] + dB[b]/2)

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
            pyplot.plot(Es, difference, linewidth=1.3, label = label, color = colours[index], marker = markers[index], ls = ls)
        else:
            pyplot.errorbar(Es, difference, total_std, linewidth=1.3, label = label, color = colours[index], marker = markers[index])
        index += 1
    
       
########################################################################################################################## cosmetics, safe plot
        


    pyplot.grid(True)
    pyplot.axhline(0, linestyle='-', color='k') # horizontal lines
    pyplot.xlabel(r'$E\ [\mathrm{GeV}]$')
    #pyplot.title('Difference of data: left - right')


    pyplot.axis('tight')
    if normalized:
        name = 'Difference_data_lon_' + str(Lc[l]) + '_normalized'
        pyplot.ylim((-1, 1))
        pyplot.ylabel(r'$(\mathrm{east}-\mathrm{west})/(\mathrm{east}+\mathrm{west})$')
        lg = pyplot.legend(loc='upper left', ncol=2)
    
    
    else:
        name = 'Difference_data_lon_' + str(Lc[l])
        pyplot.ylim((-1.0e-5, 1.35e-5))
        pyplot.ylabel(r'$ E^2\frac{\mathrm{d} N}{\mathrm{d}E}\ \left[ \frac{\mathrm{GeV}}{\mathrm{cm^2\ s\ sr}} \right]$')
        lg = pyplot.legend(loc='upper left', ncol=2)
        pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    if significance_plot:
        name = 'Difference_data_lon_' + str(Lc[l]) + '_sgm'
        pyplot.ylabel(r'$(\mathrm{east}-\mathrm{west})/\sigma$')
        pyplot.ylim((-10, 10))
        lg = pyplot.legend(loc='lower right', ncol=2)

    lg.get_frame().set_linewidth(0)
    
    fn = plot_dir + name + fn_ending
    pyplot.xscale('log')

    pyplot.xlim((1., 1.e3))

    print 'save figure to file:'
    print fn
    pyplot.savefig(fn, format = 'pdf')


