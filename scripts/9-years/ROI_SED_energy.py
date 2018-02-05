
import numpy as np
import pyfits
import healpy
from matplotlib import pyplot
import healpylib as hlib
from iminuit import Minuit

#################################################################################################################################### parameters

dist_in_kpc = 8. # guess a distance of the ROI to us

binmin_high = 11 # bin 0 - 9 / 0.0 - 0.97 GeV
binmax_high = 30 # bin 10 - 15 / 1.4 - 7.7 GeV
binmin_low = 8 # bin 16 - 22 / 11 - 87 GeV
binmax_low = 10 # bin 23 - 30 / 124 - 1398 GeV
roi_Bmin = -2
roi_Bmax = 2
roi_Lmin = -10
roi_Lmax = -4
Lmax = 180.
Bmax = 90.
exclude_bubbles_from_fit = False
mask_point_sources = True
show_roi = True
fn_addition = '_1.pdf'

# constants

energy_of_SN = 1e49 # in erg
GeV2MeV = 1000.
erg2GeV = 624.151
cm2kpc = 3.24078e-22
delta = 0.346573590092441
npix = 196608
nside = healpy.npix2nside(npix)
solar_luminosity = 3.828e33  # in erg / s
pp_cross_section = 2.5e-26 # proton-proton scattering cross section in cm^2
speed_of_light = 29979245800 # in cm / s
Crab = 2.4e-8 # in erg / cm^2 / s

# load data from a fits file

plot_dir = '/Users/lauraherold/Bubbles/plots/SED_res/'
map_fn = '/Users/lauraherold/Bubbles/data/counts_P8_P302_Source_z100_healpix_o7_31bins.fits'
expo_fn = '/Users/lauraherold/Bubbles/data/expcube_P8_P302_Source_z100_P8R2_SOURCE_V6_healpix_o7_31bins.fits'
hdu = pyfits.open(map_fn)
data = hdu[1].data.field('Spectra')
Es = hdu[2].data.field('MeV') / GeV2MeV
hdu_expo = pyfits.open(expo_fn)
expo = hdu_expo[1].data.field('Spectra')

# define likelihood class

class likelihood: # fct = sum over longitude: k * low + c - high * log(k * low + c)
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, k, c): # c: isotropic background, k: fit low energies to high energies
        fct = sum(k * y + c - x * np.log(k * y + c) for x,y in zip(self.x, self.y))
        return fct

# sum over all energy bins

nebins = binmax_high - binmin_high + 1
Es_copy = Es[binmin_high:binmax_high+1]
expo_high = np.mean(expo.T[binmin_high:binmax_high+1], axis=0)
print expo_high.shape
expo_low =  np.mean(expo.T[binmin_low:binmax_low+1], axis=0)
print expo_low.shape
map_low = np.mean(data.T[binmin_low:binmax_low+1], axis=0)
data_low = np.zeros((nebins, npix))
for ebin in range(nebins):
    data_low[ebin] = map_low
data_low = data_low.T
print data_low.shape

# find the mask

mask = np.ones(npix)
if mask_point_sources:
    res_fn = '/Users/lauraherold/Bubbles/data/ps_mask_3FGL_nside128.npy'
    mask = np.load(res_fn)

# choose the region

Bbins = np.arange(-Bmax, Bmax + 0.001, 1.)
Lbins = np.arange(-Lmax, Lmax + 0.001, 1.)
nB = len(Bbins) - 1
nL = len(Lbins) - 1
Lc = (Lbins[1::] + Lbins[:-1])/2
Bc = (Bbins[1:] + Bbins[:-1])/2
iB = nB / 2

# create longitudinal profiles 
        
colour_index = 0
print 'calculate indices...'
inds_dict = hlib.lb_profiles_hinds_dict(nside, Bbins, Lbins, mask=mask)
print 'calculate profiles...'
profiles_high = np.zeros((nB, nL, nebins))
profiles_high_T = np.zeros((nB, nebins, nL))
N_gamma_high = np.zeros((nB, nL, nebins))
N_gamma_low = np.zeros((nB, nL, nebins))
profiles_low = np.zeros((nB, nL, nebins))
profiles_low_T = np.zeros((nB, nebins, nL))
expo_profiles = np.zeros((nB, nL, nebins))
# k = profiles_high_T[0][0][int(nL/2)] / (profiles_low_T[0][0][int(nL/2)] + 1)
k = 1
k_array = np.zeros((nB, nebins))
c = 1e-8
c_array = np.zeros((nB, nebins))
for b in range(nB):
    for l in range(nL): # profiles = sum counts over pixels in bin       
        profiles_high[b,l] = np.sum([data[pixel][binmin_high:binmax_high+1] for pixel in inds_dict[(b,l)]], axis=0)
        profiles_low[b,l] = np.sum([data_low[pixel] * expo_high[pixel] / expo_low[pixel] for pixel in inds_dict[(b,l)]], axis=0)
        N_gamma_high[b,l] = np.sum([data[pixel][binmin_high:binmax_high+1] for pixel in inds_dict[(b,l)]], axis=0)
        N_gamma_low[b,l] = np.sum([data_low[pixel] * expo_high[pixel] / expo_low[pixel] for pixel in inds_dict[(b,l)]], axis=0)
        expo_profiles[b,l] = np.mean([expo[pixel][binmin_high:binmax_high+1] for pixel in inds_dict[(b,l)]], axis=0)
        for E in range(nebins):
            if N_gamma_high[b,l,E] == 0:
                N_gamma_high[b,l,E] = 1
            if N_gamma_low[b,l,E] == 0:
                N_gamma_low[b,l,E] = 1
        profiles_high_T[b] = profiles_high[b].T
        profiles_low_T[b] = profiles_low[b].T
            
# fit low-enegy data and isotropic background to high-energy data
   
for b in range(nB):
    for ebin in range(nebins):
        if exclude_bubbles_from_fit:
             x = np.append(profiles_high_T[b][ebin][0 : int(nL/2 - nL/(2 * Lmax) * 20 + 1)], profiles_high[b][ebin][int(nL/2 + nL/(2 * Lmax) * 20) : nL + 1])
             y = np.append(profiles_low_T[b][ebin][0: int(nL/2 - nL/(2 * Lmax) * 20 + 1)], profiles_low[b][ebin][int(nL/2 + nL/(2 * Lmax) * 20) : nL + 1])
        else:
            x = profiles_high_T[b][ebin]
            y = profiles_low_T[b][ebin]
        fit = likelihood(x,y)
        # print 'initial k = ' + str(profiles_high_T[b][ebin][int(nB/2)]) +' / (' + str(profiles_low_T[b][ebin][int(nB/2)]) + ' + 1) = ' + str(k)
        m = Minuit(fit, k = k, c = c, limit_k = (0,1), limit_c = (1e-30,1e-5), error_k = 0.1, error_c = 0.1, errordef = 1.)
        m.migrad() # limits of parameters k and c are important
        k = m.values['k']
        c = m.values['c']
        k_array[b] = k
        c_array[b] = c
        profiles_low_T[b][ebin] = profiles_low_T[b][ebin] * k + c
        sigma_k = m.errors['k']
for b in range(nB):
    profiles_low[b] = profiles_low_T[b].T
    
# convert counts to intensity

for b in range(nB):
    for l in range(nL):
        if inds_dict[(b,l)] == []:
            profiles_high[b,l] = 0
            profiles_low[b,l] = 0
        else:
            profiles_high[b,l] = Es_copy * profiles_high[b,l] / expo_profiles[b,l]
            profiles_low[b,l] = Es_copy * profiles_low[b,l] / expo_profiles[b,l]
            
# integrate over the ROI

b_range = range(int(round(Bmax + roi_Bmin)), int(round(Bmax+roi_Bmax)))
l_range = range(int(round(Lmax + roi_Lmin)),  int(round(Lmax + roi_Lmax)))
counts_high = np.sum(N_gamma_high[b_range][::,l_range], axis = (0,1))
counts_low = np.sum(N_gamma_low[b_range][::,l_range], axis = (0,1))
SED_high = np.sum(profiles_high[b_range][::,l_range], axis = (0,1))
SED_low = np.sum(profiles_low[b_range][::,l_range], axis = (0,1))
std_high = SED_high / np.sqrt(counts_high)
std_low = SED_low / np.sqrt(counts_low)
res = SED_high - SED_low
std = np.sqrt(std_high**2 + std_low**2)

# calculate luminosities

luminosity_per_area = np.sum(res) / erg2GeV # in erg
dist_in_cm = dist_in_kpc / cm2kpc
luminosity = luminosity_per_area * 4 * np.pi * dist_in_cm**2 # in erg
lum_in_sol_lum = luminosity / solar_luminosity
roi_length = np.sin((roi_Lmax - roi_Lmin) * np.pi / 180.) * dist_in_cm # length of ROI in cm
roi_height = np.sin((roi_Bmax - roi_Bmin) * np.pi / 180.) * dist_in_cm
roi_depth = roi_length
volume_of_roi = 4. / 3. * np.pi * (roi_length/2) * (roi_height/2) * (roi_depth/2)
CR_energy_density = 3. * luminosity / pp_cross_section / gas_density / speed_of_light / volume_of_roi # in erg / cm^3

# print the results

for i in xrange(3):
    print "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"
    print " * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * "
print "The luminosity of the ROI per unit detector area is: " + str(luminosity_per_area) + " erg / cm^2 / s = " + str(luminosity_per_area/Crab) + " Crab."
print "Assuming the distance of the ROI is " + str(dist_in_kpc) + " kpc = " + str(dist_in_cm) + " cm, the total luminosity is: "
print str(luminosity) + "erg / s = " + str(lum_in_sol_lum) + " solar luminositiesn."
print "(The distance to the Galactic center is 7.6 - 8.7 kpc.)"
print "For a length / depth of the ROI of " + str(roi_length * dist_in_kpc / dist_in_cm) + " kpc and a height of " + str(roi_height * dist_in_kpc / dist_in_cm) + " kpc,"
print "the cosmic ray energy density necessary for the gamma ray flux is " + str(CR_energy_density) + " erg / cm^3."
print "(Volume of ROI = " + str(volume_of_roi * cm2kpc**3) + " kpc^3 = " + str(volume_of_roi) + " cm^3.)"
print "Total energy stored in CR: " + str(CR_energy_density * volume_of_roi ) + " erg."
print "Assuming an energy release of " + str(energy_of_SN) + " erg per supernova, the number of additional SN necessary"
print "for the excess would be: " + str(CR_energy_density * volume_of_roi/  energy_of_SN)
for i in xrange(3):
    print "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * "
    print " * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * "

# show ROI

if show_roi:
    roi = np.zeros(npix)
    roi_map = np.sum(profiles_high, axis =2) - np.sum(profiles_low, axis =2)
    for b in range(nB):
        for l in range(nL):
            for pixel in inds_dict[(b,l)]:
                roi[pixel] = roi_map[b,l]
    # healpy.mollview(roi, title = "Red indicates the ROI", unit = "relevance")
    # healpy.graticule(dpar = 10., dmer = 10.)
    fn = plot_dir + 'Show_ROI_' + fn_addition
    # pyplot.savefig(fn)
