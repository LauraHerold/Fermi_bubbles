# plot the maps of residuals
# resid_signal, resid
# python pyapps/plot_residuals.py -c control_fit_P8_dima.yaml -w1 -v0 -r resid
# python pyapps/plot_residuals.py -c control_fit_P8_dima.yaml -w1 -v0 -r resid_signal -s1
# python pyapps/plot_residuals.py -c fit_cases.yaml -f baseline -m baseline -k0 -w1 -v0 -r resid -s1
# python pyapps/plot_residuals.py -c fit_cases.yaml -f Baseline -w0 -v1 -r data -s1

import numpy as np
import scipy
from optparse import OptionParser
import time
import copy
import os
import pyfits
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot
from matplotlib import cm
#from matplotlib import rc
import healpy
import time

import dio
import auxil
import dmap
import healpylib as hlib
import wcs_plot as wplot
import numeric as num

# constants
epsilon = 1.e-12
GeV2MeV = 1000.
psf_dict = {'all':''}
for i in range(4):
    psf_dict[i] = '_psf%i' %i


def get_model(cdict, key, psf=None, ttype='counts', exposure=None,
              Es=None, gcfit_path=''):
    """
        input E - MeV
    """
    mfolder_dict = cdict['2_models']['folders']
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    irf_base = cdict['1_data']['irf_base']
    irf_select = cdict['1_data']['irf_select']
    
    munit = cdict['2_models'][key]['unit']

    print key
    if not isinstance(cdict['2_models'][key], dict):
        key_in = cdict['2_models'][key]
    else:
        key_in = key
    folder_key = cdict['2_models'][key_in].get('folder')
    fns = cdict['2_models'][key_in]['file']

    if isinstance(fns, str):
        fns = [fns]
    
    folder = auxil.prepend_path(gcfit_path, mfolder_dict[folder_key])
    mdl_out = 0.
    #print 'load %s template %s from file:' % (ttype, key)
    #print fns
    for ifn, fn in enumerate(fns):
        exts = fn.split('.')
        if exts[-1] in ['gz', 'gzip', 'zip']:
            ext = exts[-2]
        else:
            ext = exts[-1]
        
        fn = auxil.prepend_path(folder, fn)
        if ttype == 'counts':
            fn = fn.replace('.%s' % ext, '%s.%s' % (psf_dict[psf], ext))
    
        fn = fn.replace(data_base, data_select)
        fn = fn.replace(irf_base, irf_select)
        

        print 'counts file:', fn
        #print 'check the convert_counts2flux variable 1:', convert_counts2flux
        
        
        if not os.path.isfile(fn):
            fn += '.gz'
            print fn


        if ext == 'npy':
            mdl = np.load(fn)
        elif ext == 'txt':
            mdl = np.loadtxt(fn)
        elif os.path.isfile(fn):
            # the same definition of the flux model as in the fit.py script
            mdl_maps, mdl_Es = auxil.read_skymap(fn)
            inds = [num.findIndex(mdl_Es, EE, nearest=True) for EE in Es]
            #print inds
            mdl = mdl_maps.T[inds]
            #hdu = pyfits.open(fn)
            #mdl = hdu['SKYMAP'].data.field('Spectra')

        if mdl.ndim == 1:
            mdl = healpy.ud_grade(mdl, 2**order)
            mdl = np.outer(mdl, np.ones_like(Es))
        
        if mdl.shape[-1] == max(mdl.shape):
            mdl = mdl.T
        mdl_out += np.array(mdl)

        if munit == 'flux':
            mdl_out = dmap.E2dNdE2counts(mdl_out.T, exposure.T, Es/GeV2MeV).T

    return mdl_out


# options

parser = OptionParser()

parser.add_option("-c", "--control", dest="control",
                  default='control_fit.yaml',
                  help="configuration file")

parser.add_option("-f", "--fit_case", dest="fit_case",
                  default='baseline',
                  help="choice of the fit case")
parser.add_option("-m", "--mc_case", dest="mc_case",
                  default=None,
                  help="MC data case")
parser.add_option("-k", "--kMC", dest="kMC",
                  default=-1,
                  help="MC data realization to use")

parser.add_option("-w", "--show", dest="show", default=0,
                  help="show the plots")
parser.add_option("-v", "--save", dest="save", default=0,
                  help="save the plots")
parser.add_option("-r", "--resid", dest="resid", default='resid',
                  help="type of residual")
parser.add_option("-u", "--unit", dest="unit", default=None,
                  help="type of the units (signif, counts, frac)")
parser.add_option("-p", "--pdf", dest="pdf",
                  default=0,
                  help="Save pdf (together with default png)")
parser.add_option("-t", "--title", dest="title",
                  default=None,
                  help="Plot title")
parser.add_option("-l", "--file", dest="fn",
                  default=None,
                  help="Filename")
parser.add_option("-d", "--output_hdu", dest="output_hdu",
                  default=0,
                  help="Output wcs fits file")


parser.add_option("-s", "--summed", dest="summed", default=0,
                  help="Sum the residuals")
parser.add_option("-g", "--smoothing", dest="smoothing", default=None,
                  help="Smooth the maps")
parser.add_option("-o", "--gnom", dest="gnom", default=0,
                  help="Gnomonic projection")
parser.add_option("-a", "--cmap", dest="cmap", default='jet',
                  help="Color map, e.g., jet, afmhot, afmhot_r")

parser.add_option("-e", "--extra", dest="extra", default=None,
                  help="Additional component key (to add to, e.g., residuals)")


parser.add_option("-x", "--test", dest="test",
                  default=0,
                  help="Test mode")
parser.add_option("-y", "--nomask", dest="nomask",
                  default=0,
                  help="Use all sky map instead of masking PS")


(options, args) = parser.parse_args()

control_fn = options.control
fit_case = options.fit_case
mc_case = options.mc_case
kMC = int(options.kMC)
if options.fn is not None:
    ignore_missing = 1
else:
    ignore_missing = 0
cdict = auxil.get_cdict(control_fn, fit_case=fit_case, mc_case=mc_case, ignore_missing=ignore_missing)
gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'

# supplementary online material
SOM_folder = 'results/SOM/'
SOM_bins = range(3, 30)


# plotting options
ext = cdict['4_output']['resid_plot']['ext']
save_pdf = int(options.pdf)
if save_pdf:
    ext.append('pdf')

show_plots = int(options.show)
save_plots = int(options.save)
resid_type = options.resid
unit = options.unit
gnom = int(options.gnom)
plot_title = options.title
test = int(options.test)
nomask = int(options.nomask)
cmap = options.cmap
if cmap == 'afmhot':
    CMAP = cm.afmhot
elif cmap == 'jet':
    CMAP = cm.jet
elif cmap == 'afmhot_r':
    CMAP = cm.afmhot_r
CMAP.set_bad(color='grey') # sets background to white

if options.extra is None:
    add_mdl_keys = []
else:
    add_mdl_keys = options.extra.split(',')

output_hdu = int(options.output_hdu)

summed = int(options.summed)
smoothing = options.smoothing
if smoothing is not None:
    smoothing = float(smoothing)

dpar = 15.
dmer = 15.

# gnomonic parameters
if gnom:
    #umap_name = cdict['5_umap'].get('umap_name', 'umap_GC_ROI30')
    #umap_name = 'umap_GC_ROI60' # this is a hack
    #umap_name = 'umap_GC_ROI20'  # !!! change back
    umap_name = 'umap_GC_ROI30' #
    print 'ROI:', umap_name
    if umap_name is not None:
        B0 = cdict['5_umap'][umap_name].get('ROI_B0', 0.)
        L0 = cdict['5_umap'][umap_name].get('ROI_L0', 0.)
        #ROI_size = cdict['5_umap'][umap_name].get('ROI_size', 10.)
        B_max = cdict['5_umap'][umap_name].get('B_max', 30.)
        L_max = cdict['5_umap'][umap_name].get('L_max', 30.)
        gLB = [L0, B0]



# data descriptions
use_umap = cdict['1_data'].get('umap', 0)
use_masks = cdict['1_data'].get('use_masks', 0)
use_masks = 0
order = cdict['1_data']['order']
nside = 2**order
npix = healpy.nside2npix(nside)
PSFs = cdict['1_data'].get('PSFs')
if not isinstance(PSFs, list):
    PSFs = [PSFs]
psf_dict = {'all':''}
for i in range(4):
    psf_dict[i] = '_psf%i' %i

if resid_type != 'resid':
    use_umap = 0

data_base = cdict['1_data']['data_base']
data_select = cdict['1_data']['data_select']
irf_base = cdict['1_data']['irf_base']
irf_select = cdict['1_data']['irf_select']

if unit is None:
    unit = cdict['4_output']['resid_plot'].get('type', 'signif')
model_name = cdict['2_models']['model_name']
gadget_compare = (model_name == 'Galprop_5rings_IC_geomLoopI')

wcs_plot = cdict['4_output']['plot_general'].get('wcs_plots', 0)
if CMAP is None:
    CMAP = cdict['4_output']['plot_general']['CMAP']
nticks = int(cdict['4_output']['plot_general'].get('nticks', 5))
print 'nticks', nticks
auxil.setup_figure_pars(spectrum=False, plot_type='map')

# residuals filename
#resid_fn = auxil.output_name(cdict, 'resid')
resid_fn = options.fn
save_SOM = 0
resid_fn_SOM = None
if resid_fn is not None:
    pass
elif resid_type == 'resid':
    resid_fn = auxil.output_name(cdict, 'resid', root_path=gcfit_path, fn='resid.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case)
    if save_SOM:
        resid_fn_SOM = auxil.output_name(cdict, 'resid', root_path=gcfit_path, fn='SOM_resid.fits',
                                     mc_case=mc_case, kMC=kMC, fit_case=fit_case, folder=SOM_folder, simple=True)

elif resid_type == 'resid_signal':
    resid_fn = auxil.output_name(cdict, 'resid', root_path=gcfit_path, fn='resid_signal.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case)
    if save_SOM:
        resid_fn_SOM = auxil.output_name(cdict, 'resid', root_path=gcfit_path, fn='SOM_resid_signal.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case, folder=SOM_folder, simple=True)

elif resid_type == 'flux':
    resid_fn = auxil.output_name(cdict, 'resid_flux', root_path=gcfit_path, fn='flux_resid.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case)
elif resid_type == 'flux_signal':
    resid_fn = auxil.output_name(cdict, 'resid_flux', root_path=gcfit_path, fn='flux_resid_signal.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case)
elif resid_type == 'gas_model':
    resid_fn = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='gas_model.fits',
                                        mc_case=mc_case, kMC=kMC, fit_case=fit_case)
elif resid_type == 'model':
    resid_fn = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='model.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case)
                                 
    if save_SOM:
        resid_fn_SOM = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='SOM_model.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case, folder=SOM_folder, simple=True)

elif resid_type == 'gasPS_model':
    resid_fn = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='gasPS_model.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case)
elif resid_type == 'other_model':
    resid_fn = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='other_model.fits',
                                          mc_case=mc_case, kMC=kMC, fit_case=fit_case)
elif resid_type == 'data':
    resid_fn = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='data.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case, simple=True)

    if save_SOM:
        resid_fn_SOM = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='SOM_data.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case, folder=SOM_folder, simple=True)

elif resid_type == 'bubbles':
    resid_fn = auxil.output_name(cdict, 'model', root_path=gcfit_path, fn='bubbles_model_counts.fits',
                                 mc_case=mc_case, kMC=kMC, fit_case=fit_case, simple=True)



else:
    print "%s not found((" % resid_type



if resid_type.find('flux') > -1:
    power = 0
else:
    power = -2

print 'load residual from file:'
print resid_fn
print


# load exposure
exp_folder = auxil.prepend_path(gcfit_path, cdict['1_data']['exp-folder'])
exp_folder = exp_folder.replace(data_base, data_select)
exp_fn = cdict['1_data']['exp-file']
exp_fn = exp_fn.replace(data_base, data_select)
exp_fn = exp_fn.replace(irf_base, irf_select)
exp_fn = auxil.prepend_path(exp_folder, exp_fn)
exposure = auxil.read_skymap(exp_fn)[0].T

# load the residuals
resid, Es = auxil.read_skymap(resid_fn, order_out=order, power=power)
Es0 = 1. * Es

if resid_fn_SOM is not None:
    #resid_fn_SOM = SOM_folder + resid_fn.split('/')[-1].replace('.fits', '_SOM.fits', )
    print resid_fn_SOM
    print Es0[SOM_bins]
    auxil.hmap2skymap(resid[:,SOM_bins], fn=resid_fn_SOM, unit='counts', Es=Es0[SOM_bins], Eunit='MeV')
    exit()

# add an extra model
if add_mdl_keys:
    spectra_fn = auxil.output_name(cdict, 'spectra', mc_case=mc_case, kMC=kMC, fit_case=fit_case, simple=1)
    sdict = dio.loaddict(spectra_fn)
    Estr = 'lowE'
    sc_dict = sdict['scalings'][Estr]
    extra_string = ''
    for add_mdl_key in add_mdl_keys:
        print 'add model: ', add_mdl_key
        extra_string += '_%s' % add_mdl_key
        extra_mdl = get_model(cdict, add_mdl_key, psf='all', ttype='counts',
                              exposure=exposure.T, Es=Es)
        print 'sum of the template\n', np.sum(extra_mdl, axis=0)
        extra_mdl = sc_dict[add_mdl_key] * extra_mdl
        print sdict.keys()
        print 'scalings\n', np.array(sdict['scalings'][Estr][add_mdl_key])
        print 'spectrum\n', sdict['spectra'][Estr][add_mdl_key]
        print 'mean of the counts model\n', np.mean(extra_mdl, axis=0)
        resid += extra_mdl
    # save the residual with extra components
    resid_fn_new = resid_fn.replace('.fits', '%s.fits' % extra_string)
    print 'save residual with extra components'
    auxil.hmap2skymap(resid, fn=resid_fn_new, unit='counts', Es=Es, Eunit='MeV')

Es /= GeV2MeV
Ebins = auxil.Es2Ebins(Es)
Es0 = 1. * Es

print 'index, Ecenter, Emin'
for i, EE in enumerate(Es):
    print i, EE, Ebins[i]

resid = resid.T

# load mask
if nomask:
    mask = 1.
    inf_map = 0.
else:
    mask = auxil.get_mask(cdict, gcfit_path)
    mask_inds = [i for i in range(npix) if mask[i] == 0]
    inf_map = np.zeros_like(mask)
    inf_map[mask_inds] = np.inf

if 0:
    mask = 1.
    if use_masks:
        test_masking = 0
        masks = cdict['3_fitting']['masks']
        for mask_name in masks:
            fn = auxil.get_mask_fn(cdict, gcfit_path, mask_name)
            print 'load mask from file:'
            print fn
            mask *= np.load(fn)


if nomask:
    nomask_str = '_nomask'
    resid_fn = auxil.get_data_fn_base(cdict, include_folder=1)
else:
    nomask_str = ''

# output filenames

plot_folder = auxil.get_resid_plot_folder(cdict, gcfit_path, mc_case=mc_case, kMC=kMC, fit_case=fit_case)
plot_fn = resid_type #cdict['4_output']['resid_plot']['fn']
res_str = auxil.result_descriptor(cdict, mc_case=mc_case, kMC=kMC, fit_case=fit_case, simple=1)
plot_fn += '_%s%s' % (unit, res_str)
plot_fn = auxil.prepend_path(plot_folder, plot_fn)

if add_mdl_keys:
    add_mdl_str = ''
    for add_mdl_key in add_mdl_keys:
        add_mdl_str += '_%s' % add_mdl_key
    plot_folder = gcfit_path + 'results/%s/%s/' % (data_select, fit_case)
    auxil.mkpath(plot_folder)
    plot_fn = '%s_%s_%s%s' % (fit_case, resid_type, unit, add_mdl_str)
    plot_fn = auxil.prepend_path(plot_folder, plot_fn)
    
    res_fn = '%s_%s%s_counts.fits' % (fit_case, resid_type, add_mdl_str)
    res_fn = auxil.prepend_path(plot_folder, res_fn)
    auxil.hmap2skymap((resid * mask).T, unit='counts', fn=res_fn, Es=Es*GeV2MeV, Eunit='MeV')

    resid_flux = dmap.counts2EpdNdE(resid, exposure, Es) * mask
    res_fn = '%s_%s%s_flux.fits' % (fit_case, resid_type, add_mdl_str)
    res_fn = auxil.prepend_path(plot_folder, res_fn)
    auxil.hmap2skymap(resid_flux.T, unit='1/MeV cm^2 s sr', fn=res_fn, Es=Es*GeV2MeV, Eunit='MeV')

# extra parameters

maxv_dict = {3:100, 7:60, 10:40, 13:15, 15:5, 20:0.5}
if summed is None:
    summed = cdict['4_output']['resid_plot'].get('summed', 0)
else:
    summed = int(summed)

if summed:
    plot_bins = cdict['4_output']['resid_plot'].get('plot_bins', [3, 7, 10, 13, 15])
else:
    imax = min(len(Es), 13) # hack
    plot_bins = range(imax)

print 'plot bins:', plot_bins

#smooth_sigma = np.deg2rad(1.)
if smoothing is not None:
    smooth_sigma = np.deg2rad(smoothing)
else:
    smooth_sigma = np.deg2rad(cdict['4_output']['resid_plot'].get('smooth_sigma', 1))


corr_fr = 1.

ptitles = {'resid': 'Residual', 'resid_signal': '(Residual + GC excess)', 'bubbles': 'Bubbles',
    'gas_model': 'Gas model', 'gasPS_model': 'Gas + PS model', 'other_model': 'Other components', 'data': 'Data', 'model': 'Model'}


#ptitles_add = {'frac': ' / Data', 'signif': ' / Sqrt(Data)', 'counts': ''}
ptitles_add = {'frac': ' / Data', 'signif': ' / $ \\rm \\sqrt{Data}$', 'counts': ''}



punits = {'frac': '$\\rm fraction$', 'signif': '$\\rm significance$', 'counts': '$\\rm counts$', 'log_counts': '$\\rm log_{10}(counts)$',
           'flux': '$\\rm cm^{-2} s^{-1} sr^{-1}$', 'Eflux': '$\\rm E^2 dN/dE\: (10^{-6} GeV cm^{-2} s^{-1} sr^{-1})$'}

if not plot_title:
    plot_title = ptitles[resid_type] + ptitles_add.get(unit, '')
    extra_title = ''
    for add_mdl_key in add_mdl_keys:
        extra_title += ' + %s' % add_mdl_key.replace('_', ' ')
    plot_title = plot_title.replace('Residual', 'Residual%s' % extra_title)


if resid_type.startswith('flux') and unit in ['frac', 'signif']:
    print 'convert flux to counts'
    resid = dmap.E2dNdE2counts(resid, exposure, Es)

if unit in ['flux', 'Eflux']:
    print 'convert counts to flux integrated in energy bins'
    f = np.sqrt(Es[1] / Es[0])
    dE = Es * (f - 1/f)
    resid = dmap.counts2EpdNdE(resid, exposure, Es, p=0).T * dE
    if unit == 'Eflux':
        resid *= Es * 10**6
    resid = resid.T


#exit()

if summed:
    print 'Bins:'
    resid_new = []
    #plot_bins.append(len(Es))
    for i in range(len(plot_bins) - 1):
        rvalues = np.sum(resid[plot_bins[i]:plot_bins[i + 1]], axis=0)
        if unit == 'Eflux':
            print 'inside summed'
            print np.log(Ebins[plot_bins[i + 1]] / Ebins[plot_bins[i]])
            print np.sum(rvalues)
            rvalues /= np.log(Ebins[plot_bins[i + 1]] / Ebins[plot_bins[i]])
        resid_new.append(rvalues)
        print '%i, %.2f - %.2f GeV' % (i, Ebins[plot_bins[i]], Ebins[plot_bins[i + 1]])
    resid = np.array(resid_new)
    Ebins = Ebins[plot_bins]
    Es = np.sqrt(Ebins[1:] * Ebins[:-1])
    plot_bins_old = plot_bins[:]
    plot_bins = range(len(plot_bins) - 1)

# multiply by the mask derived from the residuals
#mask *= 1 - auxil.delta(resid[0])





if gadget_compare:
    # plot Gadget residuals
    fn = 'GaDGET_egbv3_fit_local_dima_0-30_P8_CLEAN_V5_Lorimer_z10kpc_R20kpc_Ts150K_EBV5mag_bugFixed_P8_P301_clean_residual_map.fits'
    hdu = pyfits.open(resid_folder + fn)
    gres = hdu['SKYMAP'].data.field('Spectra').T

    model_folder = resid_folder.replace('/resid/', '/models/')
    fn = 'GaDGET_egbv3_fit_local_dima_0-30_P8_CLEAN_V5_Lorimer_z10kpc_R20kpc_Ts150K_EBV5mag_bugFixed_P8_P301_clean_model_map.fits'
    hdu = pyfits.open(model_folder + fn)
    gmodel = hdu['SKYMAP'].data.field('Spectra').T

    # load the data

    folder = auxil.prepend_path(gcfit_path, cdict['1_data']['data_folder'])
    fn_base = auxil.prepend_path(folder, cdict['1_data']['file_base'])

    data = 0.
    for i, psf in enumerate(PSFs):
        fn_psf = fn_base.replace('.fits', '%s.fits' % psf_dict[psf])
        fn_psf = auxil.prepend_path(folder, fn_psf)
        print 'load data from file:'
        print fn_psf

        mhdu = pyfits.open(fn_psf)
        data += mhdu['SKYMAP'].data.field('Spectra').T

    gres = data - gmodel



if unit in ['frac', 'signif']:
    # load smoothed data
    folder = auxil.prepend_path(gcfit_path, cdict['3_fitting']['metric']['folder'])
    folder = folder.replace(data_base, data_select)

    #fn = cdict['3_fitting']['metric']['file']
    #fn = fn.replace(data_base, data_select)
    if kMC > -1:
        include_folder = 1
    else:
        include_folder = 0

    fn = auxil.get_data_fn_base(cdict, mc_case=mc_case, kMC=kMC, include_folder=include_folder)
    fn = fn.replace('counts', 'counts_smooth')


    data_sm = 0.
    for i, psf in enumerate(PSFs):
        fn_psf = fn.replace('.fits', '%s.fits' % psf_dict[psf])
        fn_psf = auxil.prepend_path(folder, fn_psf)
        print 'load smooth counts from file:'
        print fn_psf

        mhdu = pyfits.open(fn_psf)
        sm_counts_loc = mhdu['SKYMAP'].data.field('Spectra').T
        data_sm += sm_counts_loc
    
    if save_SOM:
        # save the smoothed counts
        print 'save smoothed counts and mask for SOM'
        sm_fn_SOM = SOM_folder + 'SOM_smooth_data_counts.fits'
        auxil.hmap2skymap(data_sm[SOM_bins].T, fn=sm_fn_SOM, unit='counts', Es=Es0[SOM_bins], Eunit='MeV')
        mask_fn_SOM = SOM_folder + 'SOM_mask_200_3FGL_sources.npy'
        print mask_fn_SOM
        np.save(mask_fn_SOM, mask)
        exit()


    if summed:
        data_sm_new = []
        for i in range(len(plot_bins_old) - 1):
            data_sm_new.append(np.sum(data_sm[plot_bins_old[i]:plot_bins_old[i + 1]], axis=0))
        data_sm = np.array(data_sm_new)


if unit in ['frac', 'counts', 'log_counts', 'flux', 'Eflux']:
    if unit == 'frac':
        maxv = 0.3
        minv = -maxv
    else:
        maxv = minv = None

    # smooth the residual
    for i in plot_bins:
        if smooth_sigma > 0.:
            if np.mean(mask) > 0.7:
                if nomask:
                    datah = 1. * resid[i]
                else:
                    datah = np.array(hlib.heal(resid[i], mask), dtype = np.float64)
            else:
                datah = resid[i] * mask + (1 - mask) * np.sum(resid[i]) / np.sum(mask)
            datah = np.array(datah, dtype = np.float64)
            #datah = np.array(hlib.heal(resid[i], mask), dtype = np.float64)
            try:
                datah = healpy.smoothing(datah, sigma=smooth_sigma, regression=False) * corr_fr
            except TypeError:
                datah = healpy.smoothing(datah, sigma=smooth_sigma) * corr_fr
            resid[i] = datah
            if not nomask:
                resid[i] *= mask

        if gadget_compare:
            datah = np.array(hlib.heal(gres[i], mask), dtype = np.float64)
            gres[i] = healpy.smoothing(datah, sigma=smooth_sigma, regression=False) * corr_fr

    if unit == 'frac':
        resid_plot_maps = resid / data_sm
    else:
        resid_plot_maps = resid

    if gadget_compare and unit == 'frac':
        gres /= data_sm


elif unit == 'signif':
    maxv = 10.
    minv = -maxv

    resid_hist = False
    if resid_hist:
        if use_masks:
            inds = [i for i in range(npix) if mask[i] > 0]
        else:
            inds = range(npix)
        nresid = resid[:,inds] / np.sqrt(data_sm[:,inds])
        sigmas = np.std(nresid, axis=1)
        shifts = np.mean(nresid, axis=1)
        np.set_printoptions(precision=3)
        print 'shift times sqrt(npix):'
        print shifts * np.sqrt(npix)
        print 'stds:'
        print sigmas

    n_rnd = 5
    n_eff = hlib.smoothing_neff(nside, smooth_sigma, n_rnd=n_rnd)
    corr_fr = np.sqrt(n_eff)
    for i in plot_bins:
        if np.mean(mask) > 0.7:
            datah = np.array(hlib.heal(resid[i], mask), dtype = np.float64)
        else:
            datah = resid[i] * mask + (1 - mask) * np.sum(resid[i]) / np.sum(mask)
            datah = np.array(datah, dtype = np.float64)
        try:
            resid[i] = healpy.smoothing(datah, sigma=smooth_sigma, regression=False) * corr_fr
        except TypeError:
            resid[i] = healpy.smoothing(datah, sigma=smooth_sigma) * corr_fr
            
        if gadget_compare:
            datah = np.array(hlib.heal(gres[i], mask), dtype = np.float64)
            gres[i] = healpy.smoothing(datah, sigma=smooth_sigma, regression=False) * corr_fr

    resid_plot_maps = resid / np.sqrt(data_sm)

    if gadget_compare:
        gres /= np.sqrt(data_sm)


if not gnom:
    LB_ref = [0., 0.]
    LB_size = [360., 180.]
    LB_npix = np.array(LB_size, dtype=int) * 4 + 1
else:
    Ldeg, Bdeg = gLB
    #reso = 2.
    reso = 5 # !!! change back
    xsize = int(L_max * 120 / reso)
    ysize = int(B_max * 120 / reso)
    LB_ref=[Ldeg, Bdeg]
    LB_size=[2 * L_max, 2 * B_max]
    LB_npix=[xsize, ysize]

if output_hdu:
    if 0:
        wplot.wcs_plot(plot_map * mask + inf_map, title=title, unit=punits[unit], min=minv, max=maxv, out_fname=None, cmap=CMAP,
                   rgn_str=None, allsky=False, add_cbar=True, LB_ref=LB_ref, LB_size=LB_size, LB_npix=LB_npix, proj='CAR',
                   fig_param_dict=fig_param_dict, out_dname=hdu_fn_out)

    hdu_fn_out = plot_fn + '_E%.1f-%.1fGeV' % (Ebins[0], Ebins[-1])
    if gnom:
        hdu_fn_out += '_zoomin'
    hdu_fn_out += '_%ibins.fits' % len(Es)

    wplot.fpixs2wcs_fits(resid_plot_maps, Es, Eunit='GeV',
               LB_ref=LB_ref, LB_size=LB_size, LB_npix=LB_npix,
               xcoord='GLON', ycoord='GLAT', proj='CAR', isint=0, outside_value=0.,
               out_dname=hdu_fn_out,)

for i in plot_bins:
    #    plot_fn_i = plot_fn + '_bin%i' % i
    plot_fn_i = plot_fn + '_E%.1f-%.1fGeV' % (Ebins[i], Ebins[i + 1])
    if Ebins[i + 1] < 1.:
        plot_fn_i = plot_fn + '_E%i-%iMeV' % (Ebins[i] * GeV2MeV, Ebins[i + 1] * GeV2MeV)
    
    if gadget_compare:
        plot_fn_i += '_dfit'
    if gnom:
        plot_fn_i += '_zoomin'
    if test or 1:
        plot_fn_i += '_%s' % cmap
    plot_fn_i += nomask_str

    plot_fn_i = plot_fn_i.replace('.', 'p')
    if output_hdu:
        hdu_fn_out = plot_fn_i + '.fits'
    else:
        hdu_fn_out = None

    if unit.find('log') > -1:
        plot_map = np.log10(auxil.step(resid_plot_maps[i]) * resid_plot_maps[i] + epsilon)
    else:
        plot_map = resid_plot_maps[i]

    if unit.find('counts') > -1:
        minv = None
        maxv = None
        #print 'minv, maxv', minv, maxv
        data_sorted = [v for v in plot_map * mask if np.abs(v) > epsilon]
        data_sorted = np.sort(data_sorted)
        print 'fraction of pixels:', 1. * len(data_sorted) / len(plot_map)
        frac = 0.01
        if resid_type.find('model') > -1 or resid_type.find('data') > -1:
            frac = 0.05
            if unit.find('log') > -1:
                frac = 0.005
                if gnom:
                    frac = 0.001
        ind_max = int((1 - frac) * len(data_sorted))
        ind_min = int(frac * len(data_sorted))
        if summed:
            maxv = np.around(data_sorted[ind_max], decimals=1)
        print 'max value 0:', maxv
        if summed:
            minv = np.around(data_sorted[ind_min], decimals=1)
            if resid_type.find('resid') == -1:
                minv = max(0., minv)
            ndiv = (nticks - 1) / 2
            maxv = (np.round(((maxv - minv) * 10)/ndiv) * ndiv) / 10. + minv
        elif unit.find('log') > -1:
            minv = 0.
            if gnom:
                maxv = np.around(np.max(data_sorted), decimals=1)
            else:
                maxv = np.around(data_sorted[ind_max], decimals=1)
                minv = np.around(data_sorted[ind_min], decimals=1)
                minv = max(minv, 0)
        if 0: #limits for analysis of PS refitting above 30 GeV
            minv = 0.
            maxv = max(5., maxv)
        
        if unit.find('log') > -1 and summed:
            if Es[i] == Es0[10]:
                if gnom:
                    # special values for bubbles ROI plots above between 1 and 6 GeV
                    minv = 0.
                    maxv = 2.4
                else:
                    # special plot limit values for the data and the model between 1 and 6 GeV
                    minv = 0.8
                    maxv = 2.6
            elif Es[i] == Es0[3]:
                if fit_case.find('ASTROGAM') > -1:
                    minv = 1.3
                    maxv = 3.1
                else:
                    minv = 1.6
                    maxv = 3.4
        if maxv == minv:
            maxv = minv + 0.5

        print 'min value:', minv
        print 'max value:', maxv

    elif unit == 'frac' and Ebins[i] > 10:
        maxv = 1.
        minv = -1.
    elif unit.find('flux') > -1:
        # hack
        minv = 0.
        maxv = 4.e-6

    if unit == 'Eflux':
        maxv *= 10**6


    title = '%s, %.1f - %.1f GeV' % (plot_title, Ebins[i], Ebins[i + 1])

    if Ebins[i + 1] < 1.:
        title = '%s, %i - %i MeV' % (plot_title, Ebins[i] * GeV2MeV, Ebins[i + 1] * GeV2MeV)
    if not gnom:
        if wcs_plot:
            fig_param_dict = {key: value for key, value in pyplot.rcParams.iteritems()}
            fig_param_dict['nticks'] = nticks
            
            wplot.wcs_plot(plot_map + inf_map, title=title, unit=punits[unit], min=minv, max=maxv, out_fname=None, cmap=CMAP,
                           rgn_str=None, allsky=True, add_cbar=True, LB_ref=LB_ref, LB_size=LB_size, LB_npix=LB_npix, proj='MOL',
                           fig_param_dict=fig_param_dict, out_dname=None)

        else:
            healpy.mollview(plot_map * mask, min=minv, max=maxv, unit=punits[unit], title=title)
            healpy.graticule(dpar=dpar, dmer=dmer)

    else:
        #zoomin_size = np.minimum(2 * B_max, 30.)
        if not wcs_plot:
            reso = 2
            size_arkmin = reso * 2 * max(B_max, L_max) * 60
            ngpix = size_arkmin / reso

            #minv = np.min(plot_map * mask)
            #maxv = np.max(plot_map * mask)
            #maxv = 30.
            healpy.gnomview(map=plot_map * mask, max=maxv, min=minv, unit=punits[unit], title=title,
                            rot=gLB, xsize=ngpix, reso=reso)

            healpy.graticule(dpar=dpar, dmer=dmer)
        else:
            if 1:
                auxil.setup_figure_pars(spectrum=False, plot_type='zoomin_map')
                if fit_case == 'Sample' or fit_case == 'Bubbles_allsky':
                    pyplot.rcParams['figure.figsize'][0] = pyplot.rcParams['figure.figsize'][1]
                if 0:
                    for key, value in pyplot.rcParams.iteritems():
                        print key, value
                    exit()

            fig_param_dict = {key: value for key, value in pyplot.rcParams.iteritems()}
            fig_param_dict['nticks'] = nticks
            print minv, maxv, title, punits[unit], LB_ref, LB_size, LB_npix, nticks


            wplot.wcs_plot(plot_map * mask + inf_map, title=title, unit=punits[unit], min=minv, max=maxv, out_fname=None, cmap=CMAP,
                           rgn_str=None, allsky=False, add_cbar=True, LB_ref=LB_ref, LB_size=LB_size, LB_npix=LB_npix, proj='CAR',
                           fig_param_dict=fig_param_dict, out_dname=None)



    auxil.save_figure(plot_fn_i, ext=ext, save_plots=save_plots)

    if not show_plots:
        pyplot.close()

    
    if gadget_compare:
        healpy.mollview(gres[i] * mask, min=minv, max=maxv, unit=punits[unit],
                        title='%s Gadget, %.2f - %.2f GeV' % (plot_title, Ebins[i], Ebins[i + 1]))
        healpy.graticule(dpar=dpar, dmer=dmer)
        
        plot_fn_i = plot_fn_i.replace('_dfit', '_gadget')

        auxil.save_figure(plot_fn_i, ext=ext, save_plots=save_plots)
        
        if not show_plots:
            pyplot.close()

    

if show_plots:
    pyplot.show()



