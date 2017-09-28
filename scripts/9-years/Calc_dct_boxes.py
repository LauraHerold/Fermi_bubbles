""" Takes differential flux of lowE model (GeV/(cm**2 s sr)) in 10 deg or 4 deg lat stripes at high or low latitudes, respectively, and fits boxes in each latitude with a likelihood fit. The  are saved in a dictionary """


import numpy as np
import pyfits
import healpy
import healpylib as hlib
import dio


####################################################################################################################### Parameters

binmin = 0                                        # range: 0-17
binmax = 17

source_class = False

###################################################################################################################### Constants

dB_dct = {'small': 4., 'large':  10.}             # length of bin in latitude
Bmax_dct = {'small': 10., 'large': 60.}           # maximal latitude (in deg)
dL = 10.                                          # length of bin in longitudinal
Lmax = 10.                                        # maximal longitude (in deg)

GeV2MeV = 1000.
delta = 0.3837641821164575                        # Logarithmic size of one energy bin, in fits header "Step in energy (log)"
npix = 196608
nside = healpy.npix2nside(npix)

###################################################################################################################### Load data

mask_point_sources = True
symmetrize_mask = True

if source_class:
    map_fn = 'fits/Boxes_0.3-1.0GeV_source.fits'
    counts_fn = '../../data/P8_P302_Source_z100_w009_w478/maps/counts_P8_P302_Source_z100_w009_w478_healpix_o7_24bins.fits' # Source class (counts )
    dct_fn ='dct/dct_boxes_source.yaml'
else:
    map_fn = 'fits/Boxes_0.3-1.0GeV_ultraclean.fits'
    counts_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/maps/counts_P8_P302_UltracleanVeto_z90_w009_w478_healpix_o7_24bins.fits' # Ultraclean class (counts)
    dct_fn ='dct/dct_boxes_ultraclean.yaml'

mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'    
    
hdu = pyfits.open(map_fn)
data = hdu[1].data.field('Spectra')[::,binmin:binmax+1]
Es = hdu[2].data.field('GeV')[binmin:binmax+1]

hdu_counts = pyfits.open(counts_fn)                                                                                # Counts are for standard deviation
counts = hdu_counts[1].data.field('Spectra')[::,6+binmin : 6+binmax+1]                                             # The 6th engergy bin in data is the 0th high-energy bin

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmetrize_mask:
        mask *= mask[::-1]

###################################################################################################################### Select the region and group together pixels of the same region in the inds_dict


diff_dct = {}
std_dct = {}

Bbins = {}
Lbins = np.arange(-Lmax, Lmax + 0.001, dL)
Lc = (Lbins[:-1] + Lbins[1:])/2
nL = len(Lbins)-1


nE = binmax - binmin + 1

for option in ['small','large']:
    print option
    
    dB = dB_dct[option]
    Bmax = Bmax_dct[option]   
    Bbins[option] = np.arange(-Bmax, Bmax + 0.001, dB)    
    Bc = (Bbins[option][1:] + Bbins[option][:-1])/2
    nB = len(Bbins[option])-1

    print Bbins[option]
    

    print 'calculate indices...'
    inds_dict = hlib.lb_profiles_hinds_dict(nside, Bbins[option], Lbins, mask=mask)

###################################################################################################################### Calculate differential flux in each pixel, sum over pixels in one lat-lon bin, calculate std


    diff_dct[option] = np.zeros((nB,nL,nE))
    std_dct[option] = np.zeros((nB,nL,nE))
    
    for b in xrange(nB):
        for l in xrange(nL):

            N_gamma = 0
            diff_dct[option][(b,l)] = np.mean([data[pixel] for pixel in inds_dict[(b,l)]], axis = 0)# map = N_gamma / exposure
            N_gamma = np.sum([counts[pixel] for pixel in inds_dict[(b,l)]], axis = 0)           
            
            for i in xrange(len(N_gamma)-1, 0-1, -1): # delete empty lat lon bins
                if np.sqrt(N_gamma[i]) < 0.1:
                    N_gamma[i] = 0.1
                    
            std_dct[option][(b,l)] = diff_dct[option][(b,l)] / np.sqrt(N_gamma) # errors = standard deviation via Gaussian error propagation


        
###################################################################################################################### Save dictionary in YAML format


diff_profiles = np.append(np.append(diff_dct['large'][0:5], diff_dct['small'], axis = 0), diff_dct['large'][7:13], axis = 0)
std_profiles = np.append(np.append(std_dct['large'][0:5], std_dct['small'], axis = 0), std_dct['large'][7:13], axis = 0)
    
Bbins_tot = np.append(np.append(Bbins['large'][0:5], Bbins['small']), Bbins['large'][8:13])
Bc_tot =(Bbins_tot[1:] + Bbins_tot[:-1])/2

if mask_point_sources:
    dct = {'1) Comment':'Latitude-longitude profiles of residual differential flux derived from lowE model and corresponding standard deviation with the shape (lat_bin, lon_bin, energy_bin). Point sources are masked with the small map.'}
dct['6) Differential_flux_profiles'] = diff_profiles
dct['7) Standard_deviation_profiles'] = std_profiles
dct['2) Unit'] = 'GeV / (cm^2 s sr)'
dct['3) Center_of_lon_bins'] = Lc
dct['4) Center_of_lat_bins'] = Bc_tot
dct['5) Energy_bins'] = Es


dio.saveyaml(dct, dct_fn, expand = True)













