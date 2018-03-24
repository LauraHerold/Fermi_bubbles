""" 
Saves differential flux of low-energy model with isotropic background as fits file.
3 different low-energy ranges
Units: GeV/(cm^2 s sr)
Mask: Small, symmetrized
Class: Source or ultraclean(24 energy bins)
Fit: Bubbles excluded
Smoothed: High-energy data 0.4 to 1.4 degree Gaussian kernel (depending on low-energy range)
"""

import numpy as np
import pyfits
import healpy
import healpylib as hlib
from iminuit import Minuit
from optparse import OptionParser
from matplotlib import pyplot
import dio



########################################################################################################################## Parameters

parser = OptionParser()
parser.add_option("-c", "--data_class", dest = "data_class", default = "source", help="data class (source or ultraclean)")
parser.add_option("-E", "--lowE_range", dest="lowE_range", default='0', help="There are 3 low-energy ranges: (3,5), (3,3), (4,5), (6,7)")
(options, args) = parser.parse_args()

data_class = str(options.data_class)
lowE_range = int(options.lowE_range) # 0: baseline, 4: test

############################################################################################################################ Constants

fn_extra = ""
print "new"
if lowE_range ==4:
    fn_extra = "_test"

binmin_low, binmax_low = ((3,5), (3,3), (4,5), (6,7), (3,5))[lowE_range]
binmin_high, binmax_high = ((6,23), (6,23), (6,23), (8,23),(10,11))[lowE_range]
smooth_sigma = (1., 1.41, 0.65, 0.4, 1.)[lowE_range] #  1.25

exclude_bubbles_from_fit = True              # True --> No overcompensation of hard emission --> residual in ROI is larger
mask_point_sources = True                    # Is used in the calculate_indices function
smooth_highE_data = True
symmetrize_mask = True

dL = 20.
dB = 4.
Lmax = 180.
Bmax = 90.

GeV2MeV = 1000.
delta = 0.3837641821164575          # Logarithmic size of one energy bin, in fits header "Step in energy (log)"
npix = 196608                       # Number of Healpix pixels
nside = healpy.npix2nside(npix)     # Side parameter of Healpy projection


########################################################################################################################### Load data from a fits file

if data_class == "source":
    map_fn = '../../data/P8_P302_Source_z100_w009_w478/maps/counts_P8_P302_Source_z100_w009_w478_healpix_o7_24bins.fits' 
    expo_fn = '../../data/P8_P302_Source_z100_w009_w478/irfs/expcube_P8_P302_Source_z100_w009_w478_P8R2_SOURCE_V6_healpix_o7_24bins.fits'
    print "Source class"

else:
    map_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/maps/counts_P8_P302_UltracleanVeto_z90_w009_w478_healpix_o7_24bins.fits'  
    expo_fn = '../../data/P8_P302_UltracleanVeto_z90_w009_w478/irfs/expcube_P8_P302_UltracleanVeto_z90_w009_w478_P8R2_ULTRACLEANVETO_V6_healpix_o7_24bins.fits'
    print "UltracleanVeto class"

mask_fn = '../../data/ps_masks/ps_mask_3FGL_small_nside128.npy'                                                                       # Small mask

hdu = pyfits.open(map_fn)
data = hdu[1].data.field('Spectra').T  # data.T.shape = (nE, npix)
Es = hdu[2].data.field('MeV') / GeV2MeV
hdu_expo = pyfits.open(expo_fn)
expo = hdu_expo[1].data.field('Spectra').T


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

if exclude_bubbles_from_fit:
    l_range = np.array([0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17])                  # 2 energy bins have to be excluded (-20 to 20 deg longitude)
else:
    l_range = np.arange(0, nL+1)
                        

    
########################################################################################################################### Define likelihood class

class likelihood:                                                         # fct = sum over pixel in one lat stripe: k * low + c - high * log(k * low + c), c is proportional to the exposure
    def __init__(self, x, y, expo_c):
        self.x = x
        self.y = y
        self.expo_c = expo_c
    def __call__(self, k, c):                                             # c: isotropic background, k: fit low energies to high energies
        fct = sum(k * dy + c * exp - dx * np.log(k * dy + c * exp) for dx,dy,exp in zip(self.x, self.y, self.expo_c))
        return fct


########################################################################################################################### Sum over all energy bins


Es_high = Es[binmin_high:binmax_high+1]                                                      # Energy bins of the high-energy range
data_high = data[binmin_high:binmax_high+1]
expo_high = expo[binmin_high:binmax_high+1]  

data_low = np.mean(data[binmin_low:binmax_low + 1]/expo[binmin_low:binmax_low + 1], axis=0) * expo_high # shape: (nE, npix)
 
#vmax=500
#rescaling = np.sum(data_low[0])/np.sum(data_high[0])
#healpy.mollview(data_high[0], title='no smoothing', min=0, max=vmax)

if smooth_highE_data:
    for E in xrange(nE):
        data_high[E] = np.array(hlib.heal(data_high[E], mask), dtype = np.float64)
        data_high[E] = healpy.smoothing(data_high[E], sigma = np.deg2rad(smooth_sigma))               # High-energy data is smoothed to compensate for the PSF


#model = data_low[0]/rescaling
#healpy.mollview(model, title='model', min=0, max=vmax)

#healpy.mollview(data_high[0], title='smoothed', min=0, max=vmax)

#healpy.mollview((data_high[0] - model), title='residual', min=-100, max=100)

#pyplot.show()

        
########################################################################################################################### Fit low to high energy data
        

print 'calculate indices...'
inds_dict = hlib.lb_profiles_hinds_dict(nside, Bbins, Lbins, mask=mask) # Generates dictionary of lat-lon bins, masked pixels not included

k = 0.04                                                                # Initial proportionality factor
k_array = np.zeros((nB, nE))
c = 1.e-13                                                              # Initial isotropic-background factor
c_array = np.zeros((nB, nE))

#expo_mean = np.zeros((nB,nE))
#for E in range(nE):
#    for b in range(nB):
#                expo_mean[b,E] += np.mean(np.mean(expo_high[E,pixel] for pixel in inds_dict[(b,l)]) for l in range(nL)) 

for E in xrange(nE):
    for b in xrange(nB):                                                # Concatenate all pixels of one latitude stripe

        x = np.concatenate([np.asarray([data_high[E][pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        y = np.concatenate([np.asarray([data_low[E][pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        expo_c = np.concatenate([np.asarray([expo_high[E][pixel] for pixel in inds_dict[(b,l)]]) for l in l_range])
        
        fit = likelihood(x,y, expo_c)                                                  # Fit model = (lowE * k + c) to highE
        m = Minuit(fit, k = k, c = c, limit_k = (0,1), limit_c = (1e-16,1e-10), error_k = 0.1, error_c = 0.1, errordef = 1.)
        m.migrad()                                                                     # Limits of parameters k and c are important
        
        k_array[b,E] = m.values['k']
        c_array[b,E] = m.values['c']

        print 'E = ' + str(Es_high[E])
        print 'b = ' + str(Bc[b])



############################################################################################################################ Calculate residual flux   

model = np.zeros((nE, npix))        
        
for E in xrange(nE):
    for b in xrange(nB):
        for l in xrange(nL):
            for pixel in inds_dict[(b,l)]:
                model[E, pixel] = k_array[b,E] * data_low[E,pixel] + c_array[b,E] * expo_high[E,pixel]  
                
resid_counts = data_high - model                                                                      


dOmega = 4. * np.pi / npix
deltaE = Es_high * (np.exp(delta/2) - np.exp(-delta/2))

resid_flux = np.zeros((nE, npix))
model_flux = np.zeros((nE, npix))
for E in xrange(nE):
    resid_flux[E] = mask * (Es_high[E]**2 * resid_counts[E]) / (deltaE[E] * expo_high[E] * dOmega)           # Differential flux
    model_flux[E] = mask * (Es_high[E]**2 * model[E]) / (deltaE[E] * expo_high[E] * dOmega)


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

fits_fn_counts = 'fits/LowE_%.1f' %emin_low + '-%.1fGeV' %emax_low + '_counts_' + data_class + fn_extra + '.fits'
skymap_counts = hmap2skymap(resid_counts.T, fits_fn_counts, unit = 'counts', Es = Es_high)

fits_fn_flux = 'fits/LowE_%.1f' %emin_low + '-%.1fGeV' %emax_low + '_flux_' + data_class + fn_extra  + '.fits'
skymap_flux = hmap2skymap(resid_flux.T, fits_fn_flux, unit = 'GeV/(cm^2 s sr)', Es = Es_high)


fits_fn_model = 'fits/LowE_model_%.1f' %emin_low + '-%.1fGeV' %emax_low + '_counts_' + data_class + fn_extra  + '.fits'
skymap_model = hmap2skymap(model.T, fits_fn_model, unit = 'counts', Es = Es_high)

fits_fn_modelflux = 'fits/LowE_model_%.1f' %emin_low + '-%.1fGeV' %emax_low + '_flux_' + data_class + fn_extra  + '.fits'
skymap_modelflux = hmap2skymap(model_flux.T, fits_fn_modelflux, unit = 'GeV/(cm^2 s sr)', Es = Es_high)


############################################################################################################################ Save parameters in dictionary

dct = {'Comment' : 'Parameters of low-energy model'}
dct['c_array'] = c_array
dct['k_array'] = k_array
dct['smooth_sigma'] = smooth_sigma
dct['shape_arrays (nB,nE)'] = c_array.shape
dct['Bc'] = Bc
dct['Es_high'] = Es_high

dct_fn = 'fits/LowE_%.1f' %emin_low + '-%.1fGeV' %emax_low + '_' + data_class + fn_extra + '.yaml'
dio.saveyaml(dct, dct_fn, expand = True)
