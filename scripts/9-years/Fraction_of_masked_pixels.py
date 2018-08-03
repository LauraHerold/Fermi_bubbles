""" Calculates differential flux (GeV/(cm**2 s sr)) in 10 deg or 4 deg lat stripes at high or low latitudes, respectively. Values are saved in a dictionary """


import numpy as np
import pyfits
import healpy
import healpylib as hlib
import dio
from matplotlib import pyplot
import auxil


####################################################################################################################### Parameters



###################################################################################################################### Constants

symmetrize_mask = True

dB_dct = {'small': 4., 'large':  10.} # length of bin in latitude
Bmax_dct = {'small': 10., 'large': 60.} # maximal latitude (in deg)
dL = 10. # length of bin in longitudinal
Lmax = 10. # maximal longitude (in deg)

GeV2MeV = 1000.
delta = 0.3837641821164575 # logarithmic distance between two energy bins
npix = 196608
nside = healpy.npix2nside(npix)

###################################################################################################################### Load data and mask


mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'



mask = np.load(mask_fn)
if symmetrize_mask:
    mask *= mask[::-1]

    
###################################################################################################################### Select the region and group together pixels of the same region in the inds_dict

fraction_dct = {}


Bbins = {}
Lbins = np.arange(-Lmax, Lmax + 0.001, dL)
Lc = (Lbins[:-1] + Lbins[1:])/2
nL = len(Lbins)-1


for option in ['small','large']:
    print option
    
    dB = dB_dct[option]
    Bmax = Bmax_dct[option]   
    Bbins[option] = np.arange(-Bmax, Bmax + 0.001, dB)    
    Bc = (Bbins[option][1:] + Bbins[option][:-1])/2
    nB = len(Bbins[option])-1


    print 'calculate indices...'
    inds_dict = hlib.lb_profiles_hinds_dict(nside, Bbins[option], Lbins, mask=np.ones(len(mask)))
    print "total pixels: ", len(inds_dict[(2,0)]), len(inds_dict[(2,1)])
    print "unmasked pixels: ", np.sum(mask[pixel] for pixel in inds_dict[(2,0)]), np.sum(mask[pixel] for pixel in inds_dict[(2,1)])

###################################################################################################################### Calculate differential flux in each pixel, sum over pixels in one lat-lon bin, calculate std


    fraction_dct[option] = np.zeros((nB,nL))
    
    for b in xrange(nB):
        for l in xrange(nL):
            fraction_dct[option][(b,l)] = 1 - (np.sum(mask[pixel] for pixel in inds_dict[(b,l)]) / len(inds_dict[(b,l)]))


        
###################################################################################################################### Save dictionary in YAML format


fraction_profiles = np.asarray(np.append(np.append(fraction_dct['large'][0:5], fraction_dct['small'], axis = 0), fraction_dct['large'][7:13], axis = 0))
fraction_profiles = fraction_profiles.T
Bbins = np.append(np.append(Bbins['large'][0:5], Bbins['small']), Bbins['large'][8:13])
Bc =(Bbins[1:] + Bbins[:-1])/2

auxil.setup_figure_pars(plot_type = 'spectrum')

width = [10,10,10,10,10,4,4,4,4,4,10,10,10,10,10]

pyplot.bar(Bc, fraction_profiles[0], width = width, align='center', label = r"West, $\ell \in (-10^\circ, 0^\circ)$", color = "blue", alpha = 0.5)
pyplot.bar(Bc, fraction_profiles[1], width = width, align='center', label = r"East, $\ell \in (0^\circ, 10^\circ)$", color = "red", alpha = 0.5)

print "West: ", fraction_profiles[0]
print "East: ", fraction_profiles[1]

pyplot.xlabel('$b\ [\mathrm{deg}]$')
pyplot.ylabel('Fraction of masked pixels')
lg = pyplot.legend(loc='upper left', ncol=1)
lg.get_frame().set_linewidth(0)
                                               
plot_dir = '../../plots/Plots_9-year/'
fn = plot_dir + 'Fraction_masked_pixels.pdf'

pyplot.savefig(fn, format = 'pdf')
