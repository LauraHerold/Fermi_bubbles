""" Calculates differential flux (GeV/(cm**2 s sr)) in 10 deg or 4 deg lat stripes at high or low latitudes, respectively. Values are saved in a dictionary """


import numpy as np
import pyfits
import healpy
import healpylib as hlib
import dio


####################################################################################################################### Parameters

binmin = 8 # range: 0 - 23
binmax = 23

source_class = False

###################################################################################################################### Constants

mask_point_sources = True
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

if source_class:
    expo_fn = '../../data/P8_P302_Source_z100_w009_w478/irfs/expcube_P8_P302_Source_z100_w009_w478_P8R2_SOURCE_V6_healpix_o7_24bins.fits'
    dct_fn ='dct/Low_energy_range0/dct_expo_source.yaml'
        
else:
    expo_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/irfs/expcube_P8_P302_UltracleanVeto_z90_w009_w478_P8R2_ULTRACLEANVETO_V6_healpix_o7_24bins.fits'
    dct_fn ='dct/Low_energy_range0/dct_expo_ultraclean.yaml'

mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'


hdu = pyfits.open(expo_fn)
exposure = hdu[1].data.field('Spectra')[::,binmin:binmax+1]
Es = hdu[2].data.field('MeV')[binmin:binmax+1] / GeV2MeV

deltaE = Es * (np.exp(delta/2) - np.exp(-delta/2))

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmetrize_mask:
        mask *= mask[::-1]

    
###################################################################################################################### Select the region and group together pixels of the same region in the inds_dict

expo_dct = {}
dOmega_dct = {}

Bbins = {}
Lbins = np.arange(-Lmax, Lmax + 0.001, dL)
Lc = (Lbins[:-1] + Lbins[1:])/2
nL = len(Lbins)-1


nE = binmax - binmin +1

for option in ['small','large']:
    print option
    
    dB = dB_dct[option]
    Bmax = Bmax_dct[option]   
    Bbins[option] = np.arange(-Bmax, Bmax + 0.001, dB)    
    Bc = (Bbins[option][1:] + Bbins[option][:-1])/2
    nB = len(Bbins[option])-1
    

    print 'calculate indices...'
    inds_dict = hlib.lb_profiles_hinds_dict(nside, Bbins[option], Lbins, mask=mask)

###################################################################################################################### Calculate differential flux in each pixel, sum over pixels in one lat-lon bin, calculate std


    expo_dct[option], dOmega_dct[option] = np.zeros((nB,nL,nE)), np.zeros((nB,nL))
    
    for b in xrange(nB):
        for l in xrange(nL):

            expo_dct[option][(b,l)] = np.mean([exposure[pixel] for pixel in inds_dict[(b,l)]], axis = 0)# map = N_gamma / exposure     
            dOmega_dct[option][(b,l)] = 4. * np.pi * len(inds_dict[(b,l)]) / npix  # calculate solid angle of region

        
###################################################################################################################### Save dictionary in YAML format


expo_profiles = np.append(np.append(expo_dct['large'][0:5], expo_dct['small'], axis = 0), expo_dct['large'][7:13], axis = 0)
dOmega_profiles = np.append(np.append(dOmega_dct['large'][0:5], dOmega_dct['small'], axis = 0), dOmega_dct['large'][7:13], axis = 0)
    
Bbins = np.append(np.append(Bbins['large'][0:5], Bbins['small']), Bbins['large'][8:13])
Bc =(Bbins[1:] + Bbins[:-1])/2

dct = {'1) Comment':'Latitude-longitude profiles of mean exposure (shape: nB, nL, nE), solid angle dOmega (shape: nB, nL) and energy-bin size deltaE (shape: nE). Point sources are masked with the small map in order to calculate the mean exposure.'}
dct['2) Unit'] = '1 / (cm^2 s), sr, GeV'
dct['3) Center_of_lon_bins'] = Lc
dct['4) Center_of_lat_bins'] = Bc
dct['5) Energy_bins'] = Es
dct['6) Exposure_profiles'] = expo_profiles
dct['7) dOmega_profiles'] = dOmega_profiles
dct['8) deltaE'] = deltaE


dio.saveyaml(dct, dct_fn, expand = True)




