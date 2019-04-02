# the projected sensitivity for CTA and KM3NeT to detect low latitude bubbles
"""
source /Users/Dmitry/Work/gcfit/gcfit_libs.sh
export PYTHONPATH=/Library/Python/2.7/site-packages:$PYTHONPATH
cd /Users/Dmitry/Work/student_works/github_bubbles/scripts/dima/
python cta_km3net_limits.py -v1 -c0
python cta_km3net_limits.py -v1 -c1
cp plots/low_lat_FB_*.pdf ../../paper/plots/
"""

import numpy as np
import auxil
from matplotlib import pyplot
from scipy import interpolate
from optparse import OptionParser
import pyfits

import gamma_spectra as gs
import numeric as num
import dio


TeV2GeV = 1000.
TeV2erg = 1.6
TeV2MeV = 1.e6
erg2MeV = 1.e6 / 1.6

# functions
def plaw(norm, ind):
    def f(x):
        return norm * x**(-ind)
    return f

def plaw_cut(norm, ind, cutoff):
    def f(x):
        return norm * x**(-ind) * np.exp(-x/cutoff)
    return f


parser = OptionParser()

parser.add_option("-t", "--true", dest="true", default=1,
                  help="use the true max hadronic model")
parser.add_option("-c", "--cygnus", dest="cygnus", default=0,
                  help="add the cygnus spectrum")

parser.add_option("-w", "--show", dest="show", default=0,
                  help="show the plots")
parser.add_option("-v", "--save", dest="save", default=0,
                  help="save the plots")

(options, args) = parser.parse_args()
hadr_model = int(options.true)
add_cygnus = int(options.cygnus)
show_plots = int(options.show)
save_plots = int(options.save)



auxil.setup_figure_pars(spectrum=True)
pyplot.axis('tight')
pyplot.rcParams['figure.subplot.left'] = 0.2
#pyplot.rcParams['figure.subplot.top'] = 0.95
ext = ['pdf']

r = np.deg2rad(2.)
ROI_area = np.pi * r**2

eflux_plot = True

if eflux_plot:
    ymin = 1.e-14
    if add_cygnus:
        ymax = 1.e-7
    else:
        ymax = 1.e-8
    sens_conversion = TeV2erg
    eflux_norm = ROI_area / TeV2GeV * TeV2erg
    ylabel = r'$E^2\frac{dN}{dE}\ \left[{\rm \frac{erg}{cm^2 s}} \right]$'
else:
    sens_conversion = TeV2GeV / ROI_area
    eflux_norm = 1.
    ylabel = r'$E^2\frac{dN}{dE}\ \left[{\rm \frac{GeV}{cm^2 s\, sr}} \right]$'
    ymin = 1.e-9
    ymax = 1.e-3


Es = np.logspace(-1., 2, 100)
if add_cygnus:
    Es = np.logspace(-3., 2, 100)
    # parameters from 3FGL
    # N * (E_TeV * TeV2MeV)^(2 - n) * E0_MeV^n
    cyg_norm = 6.84591e-11 # ph / (MeV cm^2 s) "Flux_Density"
    cyg_ind = 2.175 # "Spectral_Index"
    cyg_E0 = 1000.0 # "Pivot_Energy"
    cyg_cutoff = 10
    cygnus_norm = cyg_norm * cyg_E0**cyg_ind * TeV2MeV**(2 - cyg_ind) / erg2MeV
    # SED in units of erg / (cm^2 s) as a function of E in TeV
    cygnus_sed = plaw_cut(cygnus_norm, cyg_ind - 2, cyg_cutoff)
    
    # Fermi LAT points from the Nature paper
    EsN = np.array([1, 1.78, 3.16, 5.62, 10, 31.6, 100]) # GeV
    EminN = EsN[:-1]
    EmaxN = EsN[1:]
    EcN = np.sqrt(EminN * EmaxN) / TeV2GeV
    FsN = np.array([5.8, 6.2, 6.6, 5.3, 4.3, 3.5]) * 1.e-5 / erg2MeV
    Fs_errN = np.array([1.7, 1.6, 1.1, 1.4, 0.7, 1.0]) * 1.e-5 / erg2MeV
    
    # Milagro point
    if 0:
        mdata = np.loadtxt('data/Milagro_cygnus.txt').T
        EsM = [mdata[0, 0] / TeV2GeV]
        FsM = [mdata[1, 0] / erg2MeV]
        Fs_errM = [np.mean(np.abs(mdata[1, 1:] - FsM)) / erg2MeV]
    EsM = 20.
    FsM = 9.8e-15 * EsM**2 / TeV2erg
    Fs_errM = 2.9e-15 * EsM**2 / TeV2erg
    
    itest = 5
    print 'cygnus SED check at %i GeV: %.3g' % (EcN[itest]*TeV2GeV, cygnus_sed(EcN[itest]))
    print 'cygnus SED from 2011 Nature paper at %i GeV: %.3g' % (EcN[itest]*TeV2GeV, FsN[itest])




if not hadr_model:
    lnormf_min = 1.4e-6 # norm for the model min
    lnormf_max = 7.8e-6 # norm for the model max

    index_max = -2.09
    index_min = -2.01

    cutoff_min = 1.6e2
    E_norm = 1.
else:
    norm_max = 9.5e11 / (4. * np.pi) # norm for the model min and max
    norm_min = 1.1e11 / (4. * np.pi) # norm for the model min and max
    
    index_max = -2.13
    index_min = -1.98
    
    cutoff_min = 1.8e3
    E_norm = 1.

if add_cygnus:
    cutoff_max = 1.e6 # 2.7e5
else:
    cutoff_max = 1.e6

numu_only = True
km3_duty_cycle = 2./3. # fraction of the time the GC is below the horizon
#km3_tratio =  10. / km3_duty_cycle
km3_tratio = 1. / km3_duty_cycle


# interpolation function (for smooth sensitivity curves)
def log_interp(Es, Fs):
    log_func = interpolate.UnivariateSpline(np.log(Es), np.log(Fs))
    #fill_value='extrapolate')
    def func(Es):
        return np.exp(log_func(np.log(Es)))
    return func


# gamma spectra

p_p = np.logspace(-.5, 6., 65)
def get_spectrum(norm=1., index=-2., cutoff=cutoff_max, ID=0):
    dNdp_p = norm * p_p**index * np.exp(-p_p/cutoff)
    return gs.EdQdE_pp(dNdp_p, p_p, n_H=1., ID_PARTICLE=ID)

# max gamma model
def max_pp_gamma(Es):
    if hadr_model:
        dNdp_p = norm_max * p_p**index_max * np.exp(-p_p/cutoff_max)
        res = Es * gs.EdQdE_pp(dNdp_p, p_p, n_H=1., ID_PARTICLE=0)(Es)
    else:
        max_pp_func = get_spectrum(index=index_max, cutoff=cutoff_max)
        max_rescaling = lnormf_max / (E_norm * max_pp_func(E_norm))
        res = np.array(max_rescaling * Es * max_pp_func(Es), dtype=float)
    return res

# min gamma model
def min_pp_gamma(Es):
    if hadr_model:
        dNdp_p = norm_min * p_p**index_min * np.exp(-p_p/cutoff_min)
        res = Es * gs.EdQdE_pp(dNdp_p, p_p, n_H=1., ID_PARTICLE=0)(Es)
    else:
        min_pp_func = get_spectrum(index=index_min, cutoff=cutoff_min)
        min_rescaling = lnormf_min / (E_norm * min_pp_func(E_norm))
        res = np.array(min_rescaling * Es * min_pp_func(Es), dtype=float)
    return res

# max neutrino model
def max_pp_nu(Es):
    max_nu = 0.
    if hadr_model:
        norm = norm_max
        index = index_max
        cutoff = cutoff_max
    else:
        index = index_max
        cutoff = cutoff_max
        max_pp_func = get_spectrum(index=index_max, cutoff=cutoff_max)
        norm = lnormf_max / (E_norm * max_pp_func(E_norm))
    for id in range(3, 7):
        max_nu += Es * get_spectrum(norm=norm, index=index, cutoff=cutoff, ID=id)(Es)
    if numu_only:
        max_nu /= 3.

    return np.array(max_nu, dtype=float)

# min neutrino model
def min_pp_nu(Es):
    min_nu = 0.
    if hadr_model:
        norm = norm_min
        index = index_min
        cutoff = cutoff_min
    else:
        index = index_min
        cutoff = cutoff_min
        min_pp_func = get_spectrum(index=index_min, cutoff=cutoff_min)
        norm = lnormf_min / (E_norm * min_pp_func(E_norm))
    for id in range(3, 7):
        min_nu += Es * get_spectrum(norm=norm, index=index, cutoff=cutoff, ID=id)(Es)
    if numu_only:
        min_nu /= 3.

    return np.array(min_nu, dtype=float)





# load CTA and KM3 sensitivities


cta_sens_data = np.loadtxt('data/CTA_50h_1803.03565.txt').T
cta_sens = log_interp(cta_sens_data[0], sens_conversion * cta_sens_data[1])


km3_years = 10.
km3_sens_data = np.loadtxt('data/KM3NeT_10yr_1803.03565.txt').T
km3_sens = log_interp(km3_sens_data[0], sens_conversion * km3_sens_data[1] * np.sqrt(km3_tratio))


test = 0
if hadr_model and test:
    Es0 = np.logspace(-1., 3, 20)
    # check the normalizations
    Echeck = 10. # GeV
    #ind_check = num.findIndex(Echeck, Es)

    # hadronic model spectrum
    norm_max = 1.0e12 / (4. * np.pi) # norm for the model min and max
    ind_max = 2.15
    dNdp_p = plaw(norm_max, ind_max)(p_p)
    gamma_pi0 = gs.EdQdE_pp(dNdp_p, p_p, n_H=1., ID_PARTICLE=0)
    

    # leptonic model spectrum
    T = 2.73 / gs.eV2K

    c_light = 2.9979e8 # m/s speed of light
    h_Planck = 4.1357e-15 # eV * s Planck constant
    IRFmap_fn = '../../data/ISRF_flux/Standard_0_0_0_Flux.fits.gz' # Model for the ISRF
    hdu = pyfits.open(IRFmap_fn) # Physical unit of field: 'micron'
    wavelengths = hdu[1].data.field('Wavelength') * 1.e-6 # in m
    E_irf = c_light * h_Planck / wavelengths[::-1] # Convert wavelength in eV, invert order
    EdNdE_irf = hdu[1].data.field('Total')[::-1] / E_irf # in 1/cm^3. Since unit of 'Total': eV/cm^3

    EdNdE_irf += gs.thermal_spectrum(T)(E_irf) / E_irf

    norm_max_e = 4.0e10 / (4. * np.pi) # norm for the model min and max
    ind_max_e = 2.71

    E_e = p_p
    EdNdE_e = E_e * plaw(norm_max_e, ind_max_e)(E_e)

    gamma_IC = gs.IC_spectrum(EdNdE_irf, E_irf, EdNdE_e, E_e)
    gamma_IC_vec = np.frompyfunc(gamma_IC, 1, 1)


    # parametric spectrum
    norm = 8.e-6
    ind = 2.1
    gamma_sp = plaw(norm, ind)
    pi0_r = Echeck * gamma_sp(Echeck) / gamma_pi0(Echeck)
    IC_r = Echeck * gamma_sp(Echeck) / gamma_IC(Echeck)
    print
    print 'Check the normalizations at %i GeV' % Echeck
    print 'max model pi0: %.3g' % (Echeck * gamma_pi0(Echeck))
    print 'pi0 rescale:', pi0_r
    print 'max model IC: %.3g' % (Echeck * gamma_IC(Echeck))
    print 'IC rescale:', IC_r
    print 'max model parametric: %.3g' % (Echeck**2 * gamma_sp(Echeck))
    Om_tot = np.deg2rad(10.)*np.deg2rad(4.)
    Om_masked = 0.005496764489
    resc = Om_tot / Om_masked
    print 'ratio of Om', resc
    pyplot.figure()
    pyplot.loglog(Es0, Es0 * gamma_pi0(Es0), label='pi0')
    pyplot.loglog(Es0, Es0 * gamma_IC_vec(Es0), label='IC')
    pyplot.loglog(Es0, Es0**2 * gamma_sp(Es0), label='param')
    pyplot.legend()
    pyplot.show()
    exit()


model_max_gamma = eflux_norm * max_pp_gamma(TeV2GeV * Es)
model_min_gamma = eflux_norm * min_pp_gamma(TeV2GeV * Es)
model_max_nu = eflux_norm * max_pp_nu(TeV2GeV * Es)
model_min_nu = eflux_norm * min_pp_nu(TeV2GeV * Es)



pyplot.figure()
pyplot.loglog(Es, cta_sens(Es), ls='--', label='CTA 50 hours')
pyplot.loglog(Es, model_max_gamma, ls='-', label='Model max')
pyplot.loglog(Es, model_min_gamma, ls='-.', label='Model min')
if add_cygnus:
    pyplot.loglog(Es, cygnus_sed(Es), ls=':', label='Cygnus (3FGL)')
    pyplot.errorbar(EcN, FsN, Fs_errN, ls='', marker='s', label='Cygnus (LAT 2011)')
    pyplot.errorbar(EsM, FsM, Fs_errM, ls='', marker='d', label='Cygnus (Milagro 2007)')

pyplot.xlabel(r'$E\rm\ [TeV]$')
ylabel_loc = ylabel.replace('dN', r'dN_\gamma')
pyplot.ylabel(ylabel_loc)
pyplot.ylim(ymin, ymax)
location = 'upper right'
lg = pyplot.legend(loc=location, ncol=1, numpoints=1, labelspacing=0.4)
lg.get_frame().set_linewidth(0)  #To get rid of the box
plot_fn = 'plots/low_lat_FB_CTA'
if add_cygnus:
    plot_fn += '_cygnus'
auxil.save_figure(plot_fn, ext=ext, save_plots=save_plots)


pyplot.figure()
pyplot.loglog(Es, km3_sens(Es), ls='--', label='KM3NeT 10 years')
pyplot.loglog(Es, model_max_nu, ls='-', label='Model max')
pyplot.loglog(Es, model_min_nu, ls='-.', label='Model min')
pyplot.xlabel(r'$E\rm\ [TeV]$')
ylabel_loc = ylabel.replace('dN', r'dN_\nu')
pyplot.ylabel(ylabel_loc)
pyplot.ylim(ymin, ymax)
location = 'upper right'
lg = pyplot.legend(loc=location, ncol=1, numpoints=1, labelspacing=0.4)
lg.get_frame().set_linewidth(0)  #To get rid of the box
plot_fn = 'plots/low_lat_FB_KM3'
if add_cygnus:
    plot_fn += '_cygnus'
auxil.save_figure(plot_fn, ext=ext, save_plots=save_plots)

if add_cygnus:
    out_dict = {}
    key = 'Energy'
    out_dict[key] = {}
    out_dict[key]['values'] = Es
    out_dict[key]['unit'] = 'TeV'

    key =  'CTA_sensitivity'
    out_dict[key] = {}
    out_dict[key]['values'] = cta_sens(Es)
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = '50 hours CTA sensitivity for a 2 deg source from Ambrogi et al (2018) arxiv:1803.03565'

    key =  'model_max_gamma'
    out_dict[key] = {}
    out_dict[key]['values'] = model_max_gamma
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = 'Model max in gamma rays for hadronic case (exp cutoff at 1 PeV in the protons spectrum)'

    key =  'model_min_gamma'
    out_dict[key] = {}
    out_dict[key]['values'] = model_min_gamma
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = 'Model min in gamma rays for hadronic case'

    key =  'Cygnus_sed'
    out_dict[key] = {}
    out_dict[key]['values'] = cygnus_sed(Es)
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = 'Power law with exp cutoff fit of the Cygnus SED to Fermi LAT and Milagro points'
    out_dict[key]['parameters'] = {'norm':cygnus_norm, 'index':cyg_ind, 'cutoff': cyg_cutoff}


    key =  'KM3NeT_sensitivity'
    out_dict[key] = {}
    out_dict[key]['values'] = km3_sens(Es)
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = '10 years KM3NeT sensitivity for a 2 deg source from Ambrogi et al (2018) arxiv:1803.03565'

    key =  'model_max_nu'
    out_dict[key] = {}
    out_dict[key]['values'] = model_max_nu
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = 'Model max in neutrinos for hadronic case (exp cutoff at 1 PeV in the protons spectrum)'

    key =  'model_min_nu'
    out_dict[key] = {}
    out_dict[key]['values'] = model_min_nu
    out_dict[key]['unit'] = 'erg/cm^2 s sr'
    out_dict[key]['comment'] = 'Model min in neutrinos for hadronic case'

    out_fn = 'results/Fig14_Fig15_max_min_models_CTA_KM3NeT_sensitivities.yaml'
    dio.savedict(out_dict, out_fn)


if show_plots:
    pyplot.show()

