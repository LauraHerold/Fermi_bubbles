""" 
Fits low-energy data, isotropic background and rectangles to high-energy data. Saves boxes map as fits file.
Units: GeV/(cm^2 s sr)
Mask: Small, symmetrized
Class: Source or ultraclean (24 energy bins)
Smoothed: High-energy data with 0.4 degree Gaussian kernel
"""

import numpy as np
import pyfits
import healpy
import healpylib as hlib
from iminuit import Minuit


########################################################################################################################## Parameters

source_class = False

binmin_high = 6                     # bin 6 - 23 / 1 GeV - 1 TeV
binmax_high = 23
binmin_low = 3                       # bin 3 - 5 / 0.32 - 1 GeV
binmax_low = 5

############################################################################################################################ Constants

mask_point_sources = True            # Is used in the calculate_indices function
smooth_highE_data = True
symmetrize_mask = True

Lbox = 10.

dL = 10.
dB = 4.
Lmax = 180.
Bmax = 62.

GeV2MeV = 1000.
delta = 0.3837641821164575          # Logarithmic size of one energy bin, in fits header "Step in energy (log)"
npix = 196608                       # Number of Healpix pixels
nside = healpy.npix2nside(npix)     # Side parameter of Healpy projection

########################################################################################################################### Load data from a fits file
if source_class:
    map_fn = '../../data/P8_P302_Source_z100_w009_w478/maps/counts_P8_P302_Source_z100_w009_w478_healpix_o7_24bins.fits'                  # Source class (counts map)
    expo_fn = '../../data/P8_P302_Source_z100_w009_w478/irfs/expcube_P8_P302_Source_z100_w009_w478_P8R2_SOURCE_V6_healpix_o7_24bins.fits' # Exposure
    save_fn_class = '_source.fits'

else:
    map_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/maps/counts_P8_P302_UltracleanVeto_z90_w009_w478_healpix_o7_24bins.fits'                  # Source class (counts map)
    expo_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/irfs/expcube_P8_P302_UltracleanVeto_z90_w009_w478_P8R2_ULTRACLEANVETO_V6_healpix_o7_24bins.fits' # Exposure
    save_fn_class = '_ultraclean.fits'
    
mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'                                                                       # Small mask

hdu = pyfits.open(map_fn)
data = hdu[1].data.field('Spectra')
Es = hdu[2].data.field('MeV') / GeV2MeV
hdu_expo = pyfits.open(expo_fn)
expo = hdu_expo[1].data.field('Spectra')

data = data.T                                  # data.T.shape = (nE, npix)
expo = expo.T

mask = np.ones(npix)
if mask_point_sources:
    mask = np.load(mask_fn)
    if symmetrize_mask:
        mask *= mask[::-1]

########################################################################################################################### Choose the region

Bbins = np.arange(-Bmax, Bmax + 0.001, dB)                                 
Lbins = np.arange(-Lmax, Lmax + 0.001, dL)

Bc = (Bbins[1:] + Bbins[:-1])/2
Lc = (Lbins[:-1] + Lbins[1:])/2

nB = len(Bbins) - 1
nL = len(Lbins) - 1
nE = binmax_high - binmin_high + 1                                                 # number of energy bins in the high-energy range

l_range = xrange(nL)

########################################################################################################################### Define likelihood class

class likelihood:                                                         # fct = sum over pixel in one lat stripe: k * low + c - high * log(k * low + c)
    def __init__(self, x, y, fitbox_l, fitbox_r):
        self.x = x
        self.y = y
        self.fitbox_l = fitbox_l
        self.fitbox_r = fitbox_r
    def __call__(self, k, c, b_l, b_r): # c: isotropic background, k: fit low energies to high energies
        fct = sum(k * y + c + b_l * fitbox_l + b_r * fitbox_r - x * np.log(k * y + c + b_l * fitbox_l + b_r * fitbox_r) for x,y,fitbox_l,fitbox_r in zip(self.x, self.y, self.fitbox_l,self.fitbox_r))
        return fct


########################################################################################################################### Sum over all energy bins


Es_high = Es[binmin_high:binmax_high+1]                                                      # energy bins of the high-energy range
data_high = data[binmin_high:binmax_high+1]
expo_high = expo[binmin_high:binmax_high+1]


expo_mean_high = np.mean(expo[binmin_high:binmax_high+1], axis=0)                            # mean exposure needed for weighting the lowE counts
expo_mean_low =  np.mean(expo[binmin_low:binmax_low+1], axis=0)                              

data_low = np.mean(data[binmin_low:binmax_low+1], axis=0) * expo_mean_high / expo_mean_low   # lowE counts averaged over ebins, weighted with expo


if smooth_highE_data:
    for E in xrange(nE):
        data_high[E] = healpy.smoothing(data_high[E], sigma = np.deg2rad(0.4))    

        
########################################################################################################################### Fit low to high energy data
        

print 'calculate indices...'
inds_dict = hlib.lb_profiles_hinds_dict(nside, Bbins, Lbins, mask=mask)                # generates dictionary of lat-lon bins, masked pixels not included

k = 1.                                                                                 # Initial proportionality factor
k_array = np.zeros((nB, nE))
c = 1.e-8                                                                              # initial isotropic-background factor
c_array = np.zeros((nB, nE))

box_l_map = np.zeros(npix)
box_r_map = np.zeros(npix)

for b in xrange(nB):
    for l in xrange(nL):
        if Lc[l] == -5:
            for pixel in inds_dict[(b,l)]:
                box_r_map[pixel] = 1
        elif Lc[l] == 5:
            for pixel in inds_dict[(b,l)]:
                box_l_map[pixel] = 1
                                

b_l_array = np.zeros((nB, nE)) # b_l: constant box parameter left
b_r_array = np.zeros((nB, nE)) # b_r: constant box parameter right

for E in xrange(nE):
    for b in xrange(nB):                                                               # Concatenate all pixels of one latitude stripe

        x = np.concatenate([np.asarray([data_high[E][pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        y = np.concatenate([np.asarray([data_low[pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        box_l = np.concatenate([np.asarray([box_l_map[pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        box_r = np.concatenate([np.asarray([box_r_map[pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        
        fit = likelihood(x,y,box_l,box_r)                                               # Fit model = (lowE * k + c) to highE
        m = Minuit(fit, k = k, c = c,  b_l = 1., b_r = 1., limit_k = (0,1), limit_c = (0., 10.), limit_b_l = (0., 10.), limit_b_r = (0., 10.), error_k = 0.1, error_c = 0.1, error_b_l = 0.01, error_b_r = 0.01, errordef = 0.1)
        m.migrad()                                                                     # Limits of parameters k and c are important
        
        k_array[b,E] = m.values['k']
        c_array[b,E] = m.values['c']
        b_l_array[b,E] = m.values['b_l']
        b_r_array[b,E] = m.values['b_r']
        
        print 'E = ' + str(E)
        print 'b = ' + str(b)
        print 'x.shape = ' + str(x.shape)


############################################################################################################################ Calculate residual        

boxes_map = np.zeros((nE, npix))        
        
for E in xrange(nE):
    for b in xrange(nB):
        for l in xrange(nL):
            for pixel in inds_dict[(b,l)]:
                boxes_map[E, pixel] = box_l_map[pixel] * b_l_array[b,E] + box_r_map[pixel] * b_r_array[b,E]                            # Calculate model from k and c array
                 

dOmega = 4. * np.pi / npix
deltaE = Es_high * (np.exp(delta/2) - np.exp(-delta/2))

boxes_flux = np.zeros((nE, npix))
for E in xrange(nE):
    for pixel in xrange(npix):
        boxes_flux[E, pixel] = mask[pixel] * (Es_high[E]**2 * boxes_map[E, pixel]) / (deltaE[E] * expo_high[E,pixel] * dOmega)           # Differential flux


############################################################################################################################ Function to save fits files (auxil.py)

def hmap2skymap(values, fn=None, unit=None, kdict=None, comment=None, Es=None, Eunit='GeV'):
    hdulist = [pyfits.PrimaryHDU()]

    clms = []
    npix, nval = values.shape
    fmt = '%iE' % nval
    clm = pyfits.Column(name='Spectra', array=values,
                        format=fmt, unit=unit)
    clms.append(clm)
    dhdu = pyfits.new_table(clms)
    dhdu.name = 'SKYMAP'
    nside = healpy.npix2nside(npix)
    dhdu.header.update('PIXTYPE', 'HEALPIX')
    dhdu.header.update('ORDERING', 'RING')
    dhdu.header.update('NSIDE', nside)
    dhdu.header.update('FIRSTPIX', 0)
    dhdu.header.update('LASTPIX', (npix - 1))
    dhdu.header.update('NBRBINS', nval, 'Number of energy bins')
    dhdu.header.update('EMIN', Es_high[0], 'Minimum energy')
    dhdu.header.update('DELTAE', delta, 'Step in energy (log)')

    hdulist.append(dhdu)

    # create energy HDU
    if Es is not None:       
        clm = pyfits.Column(name=Eunit, array=Es, format='E', unit=Eunit)
        ehdu = pyfits.new_table([clm])
        ehdu.name = 'ENERGIES'
        hdulist.append(ehdu)
    
    hdulist = pyfits.HDUList(hdulist)

    print 'Save skymap to file:'
    print fn
        
    hdulist.writeto(fn)

    return None


############################################################################################################################ Save residual flux as fits file

emax_low = Es[binmax_low] * np.exp(delta/2)
emin_low = Es[binmin_low] * np.exp(-delta/2)

fits_fn = 'fits/Boxes_%.1f' %emin_low + '-%.1fGeV' %emax_low + save_fn_class

skymap = hmap2skymap(boxes_flux.T, fits_fn, unit = 'GeV/(cm^2 s sr)', Es = Es_high)
