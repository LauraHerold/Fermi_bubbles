# auxiliary functions for the gcfit project

import copy
import os
import numpy as np
#import pywcs
#import pywcsgrid2
#import pyregion
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from pywcsgrid2.allsky_axes import make_allsky_axes_from_header
from matplotlib import pyplot
from matplotlib import rc
from matplotlib import cm
import pyfits
import healpy
from sets import Set


#import upix
#import sca
import numeric as num
import dio
#import constants as const
import healpylib as hlib
#import wcs_plot as wplot

step = lambda x: (1. + np.sign(x)) / 2.
delta = lambda x: (1. + np.sign(x)) * (1. - np.sign(x))

def plaw(pars):
    return lambda x: pars[0] * x**(-pars[1])

def plaw_cut(pars):
    return lambda x: pars[0] * x**(-pars[1]) * np.exp(-x / pars[2])

CMAP = r'afmhot_r'

def setup_figure_pars(spectrum=False, plot_type=None):
    if plot_type is None:
        if spectrum:
            plot_type = 'spectrum'
        else:
            plot_type = 'map'
    if plot_type == 'spectrum':
        fig_width = 8  # width in inches
        fig_height = 6    # height in inches
    elif plot_type == 'map':
        fig_width = 9  # width in inches
        fig_height = 6    # height in inches
    elif plot_type == 'zoomin_map':
        fig_width = 4.6  # width in inches
        fig_height = 6    # height in inches

    fig_size =  [fig_width, fig_height]
    params = {'axes.labelsize': 20,
              'axes.titlesize': 20,
              'font.size': 16,
              'legend.fontsize': 14,
              'xtick.labelsize':18,
              'ytick.labelsize':18,
              #'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.minor.size' : 3,
              'ytick.minor.size' : 3,
              'figure.subplot.left' : 0.05,
              'figure.subplot.right' : 0.97,
              'figure.subplot.bottom' : 0.15,
              'figure.subplot.top' : 0.9
                }
    pyplot.rcParams.update(params)
    if plot_type == 'spectrum':
        pyplot.rcParams['figure.subplot.left'] = 0.15
        pyplot.rcParams['figure.subplot.right'] = 0.95
        pyplot.rcParams['figure.subplot.bottom'] = 0.1
    elif plot_type == 'zoomin_map':
        pyplot.rcParams['axes.titlesize'] = 20
        pyplot.rcParams['xtick.labelsize'] = 16
        pyplot.rcParams['ytick.labelsize'] = 16
        pyplot.rcParams['font.size'] = 16
        pyplot.rcParams['axes.labelsize'] = 24
        
        pyplot.rcParams['figure.subplot.left'] = 0.03
        pyplot.rcParams['figure.subplot.right'] = 0.99
        pyplot.rcParams['figure.subplot.bottom'] = 0.12
        pyplot.rcParams['figure.subplot.top'] = 0.9
        #pyplot.rcParams['figure.figsize'][0] *= 2./3.

    #rc('text.latex', preamble=r'\usepackage{amsmath}')
    return 0

def save_figure(figFn, ext=[], save_plots=False):
    if save_plots:
        for extn in ext:
            print 'save figure to file:'
            fn = '%s.%s' % (figFn, extn)
            print fn
            pyplot.savefig(fn)
    return 0


def plot_map(fmap, unit='', minv=0, maxv=1, title='', nticks=7, LB0=[0., 0.], LB_size=[90., 120.], reso=2., bg_color=None, is_wplot=True, cmap=cm.afmhot):
    def size2npix(size):
        return size * 60 / reso
    if is_wplot:
        LB_ref=LB0
        LB_npix = [int(size2npix(x)) for x in LB_size]
        fig_param_dict = {key: value for key, value in pyplot.rcParams.iteritems()}
        CMAP = cmap
        if cmap == 'afmhot':
            CMAP = cm.afmhot
        elif cmap == 'jet':
            CMAP = cm.jet
        elif cmap == 'afmhot_r':
            CMAP = cm.afmhot_r
        elif type(cmap) is str:
            print 'cmap = %s is not yet implemented in auxli.plot_map' % cmap
        
        if cmap is not None and bg_color is not None:
            CMAP.set_bad(color=bg_color) # sets background to white
        
        fig_param_dict['nticks'] = nticks
        #print minv, maxv, title, unit, LB_ref, LB_size, LB_npix
        wplot.wcs_plot(fmap, min=minv, max=maxv, title=title, unit=unit, out_fname=None, cmap=CMAP,
                       rgn_str=None, allsky=False, add_cbar=True, LB_ref=LB_ref, LB_size=LB_size, LB_npix=LB_npix, proj='CAR',
                       fig_param_dict=fig_param_dict)
    else:
        xsize = int(max(LB_size) * 60. / reso) # number of pixels on the side
        healpy.gnomview(map=fmap, unit=unit, rot=LB0, xsize=xsize, reso=reso, min=minv, max=maxv, title=title)
        dpar = 15.
        dmer = 15.
        healpy.graticule(dpar=dpar, dmer=dmer)
    return 0


def positive(a):
    """
    return  a if a > 0
            0 if a <= 0
    """
    return (np.sign(a) + 1.) * a / 2.


def logPlotData(data, err, epsilon):
    """
    avoid negative values for the down errors on loglog plots
    useful for systematic uncertainty ranges
    INPUT:
        data - data
        err - errors
        epsilon - lower cutoff
    OUTPUT:
        makes data and errors small-positive for plotting in logplots
    """
    err += epsilon
    if err.ndim == 1:
        dnErr = 1. * err
        upErr = 1. * err
    else:
        dnErr = 1. * err[0]
        upErr = 1. * err[1]
    Min = positive(data - dnErr - epsilon) + epsilon
    Med = positive(data - 2 * epsilon) + 2 * epsilon
    Max = positive(data + upErr - 3 * epsilon) + 3 * epsilon
    return Med, [Med - Min, Max - Med]



def plotLimLog(xs, fs, ferrs, sigma=2.0, 
               headw=0.1, arrowl=0.4, **kwargs):
    """ 
    Plot limit when value is less than sigma*error away from zero
    Input:
        xs - array, x-values
        fs - array, y-values
        ferrs - array, y-errors
        sigma - number, plot limit if  f[i] < sigma * ferrs[i]
            Default: 2.
        headw - number, the width of the arrow head relative to xs[i]
                        and the hight of the arrow head relative to ys[i]
            Default: 0.1
        arrowl - number, the length of the arrow line relative to fs[i]
            Default: 0.2
        **kwargs - arguments for pyplot.errobar function
    Output:
        plots errorbars
        plots arrows
        returns errorbars object - the same as output of pyplot.errorbars function

    Comments:
        only works for log-log plots

    Created: Anna Franckowiak and Dmitry Malyshev, SLAC, 03/20/2014

    """

    fs = np.array(fs)
    ferrs = np.array(ferrs)
    xs = np.array(xs)
    upper_limits = fs + ferrs * sigma
    all_inds = range(len(ferrs))

    # find indices where the points are OK
    pt_inds = [i for i in all_inds if ferrs[i] * sigma <= fs[i]]
    
    # find the indices where the points have to be substituted with limits
    lim_inds = [i for i in all_inds if ferrs[i] * sigma > fs[i]]
    
    # plot error bars
    errorbars = pyplot.errorbar(xs[pt_inds], fs[pt_inds],
                                ferrs[pt_inds], ls='', **kwargs)
    color = errorbars[0].get_color()

    # plot arrows
    for i in lim_inds:
        ul = upper_limits[i]
        # parameters of the arrow
        dx = 0
        dy = ul * arrowl
        #al = np.power(10,np.log10(ul[u])-arrowl)-ul[u]
        alh = dy * headw
        alw = xs[i] * headw

        # bar at the top of the arrow
        delta = 0.5 * headw
        xx = xs[i] * np.array([1 - delta, 1 + delta])
        yy = ul * np.ones(2)

        pyplot.arrow(xs[i], ul, dx, -dy, shape='full',
                     length_includes_head=True, head_width=alw, head_length=alh,
                     edgecolor=color, facecolor=color)
        pyplot.plot(xx, yy, c=color)

        pyplot.xscale('log')
        pyplot.yscale('log')
    
    return errorbars

def hmap2skymap(values, fn=None,
                unit=None, kdict=None, comment=None, Es=None, Eunit='MeV'):
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
    dhdu.header.update('EMIN', 42.6776695251465, 'Minimum energy')
    dhdu.header.update('DELTAE', 0.34657359023613, 'Step in energy (log)')

    hdulist.append(dhdu)

    # create energy HDU
    if Es is not None:       
        clm = pyfits.Column(name=Eunit, array=Es, format='E', unit=Eunit)
        ehdu = pyfits.new_table([clm])
        ehdu.name = 'ENERGIES'
        hdulist.append(ehdu)
    
    hdulist = pyfits.HDUList(hdulist)


    if fn is not None:
        print 'save skymap to file:'
        print fn
        
        if os.path.isfile(fn):
            os.remove(fn)
        hdulist.writeto(fn)

        return None
    else:
        return hdulist


def save_skymap(out_fn, data, hdu_example=None, replace=True, comment=None):
    hdu_res = copy.deepcopy(hdu_example)

    if data.shape[-1] == len(hdu_res['SKYMAP'].data['Spectra']):
        data = data.T
    dt_in = hdu_res['SKYMAP'].data['Spectra'][0].dtype
    res = [np.array(data[i], dtype=dt_in) for i in range(len(data))]
    hdu_res['SKYMAP'].data['Spectra'][:] = res[:]
    if comment is not None:
        hdu_res['SKYMAP'].header.add_comment(comment)
    if os.path.isfile(out_fn):
        if replace:
            print 'remove old file'
            os.remove(out_fn)
        else:
            print 'file "%s" already exists' % out_fn
            return None
    print 'save skymap to file:'
    print out_fn
    print
    hdu_res.writeto(out_fn)
    return None

def read_skymap(fn, order_out=None, power=-2):
    hdu = pyfits.open(fn)
    maps = hdu['SKYMAP'].data.field('Spectra')
    if order_out is not None:
        nside = 2**order_out
        npix = healpy.nside2npix(nside)
        if 'Upix' in hdu[1].data.names:
            upixels = hdu[1].data.field('Upix')
            values = hdu[1].data.field('Spectra')
            umap = upix.umap(values=values, upixels=upixels, is_mean=True)
            maps = upix.umap2hmap(umap, order_out=order_out)
        elif maps.shape[0] != npix:
            maps = healpy.ud_grade(maps.T, nside_out=nside, power=power)
            maps = np.array(maps).T
    Es = hdu['ENERGIES'].data.field('MeV')
    return maps, Es

def prepend_path(path, fn):
    '''
    Optional adding of path, if fn doesn't start with '/'

    '''
    if fn.startswith('/') or path is None:
        return fn
    else:
        return path + fn

def mkpath(folder, mode=00775):
    path_list = folder.split('/')
    path_list = [st for st in path_list if len(st) > 0]    
    np = len(path_list)
    path = folder
    nmake = 0
    while not os.path.isdir(path) and nmake < np:
        nmake += 1
        path = folder.split('/%s/' % path_list[np - nmake])[0] + '/'

    for add_dir in path_list[np - nmake:]:
        path = '%s%s/' % (path, add_dir)
        os.mkdir(path)
        os.chmod(path, mode)
    return None

def result_descriptor(cdict=None,
                      skip_MC=False, skip_PSF=False, # depreciated
                      mc_case=None, kMC=0, fit_case=None, alt_case=None, simple=False):

    if cdict is None and fit_case is None:
        raise ValueError, 'cdict or fit_case should be determined'

    if cdict is None or (mc_case is not None) or (alt_case is not None) or simple:
        res = ''
        if mc_case is not None:
            res += '_%s_MC' % mc_case
            if kMC is not None:
                res += '%i' %kMC
        if fit_case is not None:
            res += '_%s_model' % fit_case
        if alt_case is not None:
            res += '_%s_alt_model' % alt_case
        return res

        
    # data pass
    res_str = '_%s' % cdict['1_data']['data_select']
    # description of the map
    if cdict['1_data'].get('umap', 0):
        res_str += '_%s' % cdict['5_umap'].get('umap_name', 'umap')
    else:
        res_str += '_hmap_order%i' % cdict['1_data']['order']
    
    # model description
    res_str += '_%s' % cdict['2_models']['model_name']
    if cdict['2_models'].get('PS') is not None:
        res_str += '_%s' % cdict['2_models'].get('PS')
    if cdict['2_models']['model_base'] != cdict['2_models']['model_select']:
        res_str += '_%s' % cdict['2_models']['model_select']

    if cdict['1_data'].get('use_masks', 0):
        for mask_name in cdict['3_fitting']['masks']:
            res_str += '_%s' % mask_name
        mask_name = cdict['3_fitting']['ps_mask']
        res_str += '_%s' % mask_name
            
    if cdict['2_models'].get('add_dm', 0):
        if not res_str.endswith('DM'):
            res_str += '_DM'
        name = cdict['2_models']['DM'].get('name')
        if name is not None:
            res_str += '_%s' % name
            dm_sp_name = cdict['2_models']['DM']['lowE_sp']
            if dm_sp_name != 'marginal':
                res_str += '_%s' % dm_sp_name
            if name.startswith('wavelet') and 0:
                lmax = cdict['2_models']['DM'][name].get('lmax')
                if lmax is not None:
                    res_str += '_l%i' % (lmax)
                mmax = cdict['2_models']['DM'][name].get('mmax')
                if mmax is not None:
                    res_str += '_m%i' % (mmax)
                kmax = cdict['2_models']['DM'][name].get('kmax')
                if kmax is not None:
                    res_str += '_kmax%i' % (kmax)
            elif name == 'Cusp':
                ind = cdict['2_models']['DM'][name].get('power')
                if ind is not None:
                    ind_str = '%.1f' % ind
                    ind_str = ind_str.replace('.', 'p')
                    res_str += '_n%s' % ind_str
                L0 = cdict['2_models']['DM'][name].get('L0')
                if L0 is not None and L0 != 0.:
                    res_str += '_L%i' % L0
                B0 = cdict['2_models']['DM'][name].get('B0')
                if B0 is not None and B0 != 0.:
                    res_str += '_B%i' % B0


    iso_sp_name = cdict['2_models']['ISO']['lowE_sp']
    if iso_sp_name != 'marginal':
        res_str += '_ISO_%s' % iso_sp_name

    #res_str += '_%s' % cdict['1_data']['irf_select']
    if cdict['1_data'].get('suppress_GC', 0):
        res_str += '_suppressGC'
    if cdict['1_data'].get('gtobssim', 'None') != 'None':
        res_str += '_gtobssim_%s' % cdict['1_data']['gtobssim']
    if cdict['1_data'].get('psrefit', 0):
        nps = cdict['7_psrefit'].get('nps_inner', 'all')
        res_str += '_psrefit_%s_%s' % (nps, cdict['7_psrefit']['cat'])
    if cdict['1_data'].get('positive_norm', 0):
        res_str += '_pnorm'
    if cdict['1_data'].get('edisp', 0):
        res_str += '_edisp'

    
    # psf description
    if not skip_PSF:
        PSFs = cdict['1_data'].get('PSFs', 'all')
        if not isinstance(PSFs, list):
            PSFs = [PSFs]
        psf_str = '_psf'
        for psf in PSFs:
            psf_str += '%s' % psf
        res_str += psf_str

    if cdict['1_data'].get('MC', 0) and not skip_MC:
        res_str += '_' + cdict['1_data'].get('MC_name', 'MC')
    return res_str

def get_resid_plot_folder(cdict, gcfit_path, mc_case=None, kMC=0, fit_case='Baseline'):
    plot_folder = cdict['4_output']['resid_plot']['folder']
    if plot_folder.endswith('/'):
       plot_folder = plot_folder[:-1]
    if mc_case is not None:
        plot_folder += '_mc/plots'
    else:
        plot_folder += '/plots'
    plot_folder = prepend_path(gcfit_path, plot_folder)
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    plot_folder = plot_folder.replace(data_base, data_select)
    print fit_case
    plot_folder += '%s/' % result_descriptor(cdict, mc_case=mc_case, kMC=None, fit_case=fit_case, simple=True)
    mkpath(plot_folder)
    return plot_folder

get_spectra_plot_folder = get_resid_plot_folder

"""
def get_spectra_plot_folder(cdict, gcfit_path, mc_case=None, kMC=0, fit_case='Baseline'):
    plot_folder = cdict['4_output']['spectra_plot']['folder']
    if plot_folder.endswith('/'):
        plot_folder = plot_folder[:-1]
    if mc_case is not None:
        plot_folder += '_mc/plots'
    else:
        plot_folder += '/plots'
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    plot_folder = prepend_path(gcfit_path, plot_folder)
    plot_folder = plot_folder.replace(data_base, data_select)
    plot_folder += '%s/' % result_descriptor(cdict, mc_case=mc_case, kMC=None, fit_case=fit_case, simple=True)
    mkpath(plot_folder)
    return plot_folder
"""

def get_plot_folder(cdict, gcfit_path, folder_name=''):
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    plot_folder = cdict['4_output']['spectra_plot']['folder']
    if plot_folder.find('spectra/')  > -1:
        plot_folder = plot_folder.replace('spectra/', folder_name)
    else:
        plot_folder += folder_name
    
    plot_folder = prepend_path(gcfit_path, plot_folder)
    plot_folder = plot_folder.replace(data_base, data_select)
    
    mkpath(plot_folder)
    return plot_folder

def output_name(cdict, key, root_path=None, fn=None, folder=None, #skip_MC=False
                mc_case=None, kMC=0, fit_case='Baseline', simple=False):
    folder = cdict['4_output'][key].get('folder', folder)
    folder = prepend_path(root_path, folder)
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    folder = folder.replace(data_base, data_select)

    mkpath(folder)
    if fn is None:
        fn = cdict['4_output'][key]['fn']
    extn = fn.split('.')[-1]
    res_str = result_descriptor(cdict, mc_case=mc_case, kMC=kMC, fit_case=fit_case, simple=simple)
    fn = fn.replace('.%s' % extn, '%s.%s' % \
                    (res_str, extn))
    fn = fn.replace(data_base, data_select)
    fn = prepend_path(folder, fn)
    return fn

def profile_plot_fn(cdict, name, mc_case=None):
    data_select = cdict['1_data']['data_select']
    fit_type = cdict.get('6_sca', {}).get('func', 'plaw')
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'
    out_folder = '%s/plots/%s/sca' % (gcfit_path, data_select)
    fn = '%s/profiles_%s' % (out_folder, name)
    if fit_type == 'plaw_cut':
        fn += '_cutoff'
    if mc_case is not None:
        fn += '_MC_%s' % mc_case
    return fn


def wavelets_best_modes_fn(cdict, mc_case=None, kMC=0, fit_case='wavelet'):
    fn = output_name(cdict, 'wavelets', fn='best_modes.txt', mc_case=mc_case, kMC=kMC, fit_case=fit_case)
    if not os.path.isfile(fn):
        folder = fn.split('best_modes')[0]
        fn = folder + 'best_modes_generic.txt'
    return fn

def get_best_modes_fn(cdict):
    return wavelets_best_modes_fn(cdict)


# in the following functions need to implement reading of names from dict
def syst_map_fn(cdict=None):
    return 'systematics/syst_map.fits'

def syst_resid_dir(cdict=None):
    return 'systematics/residuals/'

def syst_sp_dir(cdict=None):
    return 'systematics/spectra/'


def get_mask_fn(cdict, gcfit_path, mask_name):
    dfolder = cdict['1_data'].get('data_base')
    order = int(cdict['1_data']['order'])
    nside = 2**order
    output_folder = '%s/models/%s/mask' % (gcfit_path, dfolder)
    mkpath(output_folder)
#    sky_mask = cdict['3_fitting'][mask_name].get('sky_mask')
#    if sky_mask == 'None':
#        sky_mask = None
#    if sky_mask is not None:
#        mask_name += '_%s' % sky_mask
    return '%s/%s_nside%i.npy' % (output_folder, mask_name, nside)

def get_mask_fns(cdict, gcfit_path):
    fns = []
    use_masks = cdict['1_data'].get('use_masks', 0)
    if use_masks:
        masks = cdict['3_fitting']['masks'][:]
        ps_mask_name = cdict['3_fitting']['ps_mask']
        if ps_mask_name != 'None' and ps_mask_name != 0:
            masks.append(ps_mask_name)
        for mask_name in masks:
            fns.append(get_mask_fn(cdict, gcfit_path, mask_name))
    return fns


def get_mask(cdict, gcfit_path, mask_name=None):
    if mask_name is not None:
        fn = get_mask_fn(cdict, gcfit_path, mask_name)
        return np.load(fn)
    else:
        order = cdict['1_data']['order']
        npix = healpy.nside2npix(2**order)
        mask = np.ones(npix)
        fns = get_mask_fns(cdict, gcfit_path)
        for fn in fns:
            print 'Get mask from file:'
            print fn
            mask *= np.load(fn)
            
        return mask


    


def get_sca_fn(cdict, gcfit_path, comp, mc_case=None, std=False):
    data_select = cdict['1_data']['data_select']
    fit_type = cdict.get('6_sca', {}).get('func', 'plaw')
    outdir_maps = 'results/%s/sca' % data_select
    outdir_maps = prepend_path(gcfit_path, outdir_maps)
    mkpath(outdir_maps)
    fn = '%s/sca_template_%s_%s' % (outdir_maps, comp, fit_type)
    if mc_case is not None:
        fn += '_MC_%s' % mc_case
    if std:
        fn += '_std'
    fn += '.npy'
    return fn


def choose_option(opt1, opt2):
    if opt1 is not None:
        return int(opt1)
    else:
        return opt2

def Es2Ebins(Es):
    Ebins = np.zeros(len(Es) + 1)
    f = np.sqrt(Es[1] / Es[0])
    Ebins[1:] = Es * f
    Ebins[0] = Es[0] / f
    return Ebins

def get_dm_mdl_fn(cdict, key, psf='all', all_sky=0):
    data_select = cdict['1_data']['data_select']
    irf_select = cdict['1_data']['irf_select']
    folder = 'models/%s/DM/' % (data_select)
    mkpath(folder)
    psf_dict = {'all':''}
    for i in range(4):
        psf_dict[i] = '_psf%i' %i

    fn = '%s/%s_%s_%s' % \
         (folder, key, data_select, irf_select)
    
    B0 = cdict['2_models']['DM'][key].get('B0')
    L0 = cdict['2_models']['DM'][key].get('L0')
    power = cdict['2_models']['DM'][key].get('power')
    use_window = cdict['2_models']['DM'][key].get('window', {}).get('use_window', 0)
    ps_mask = cdict['3_fitting']['ps_mask']
    
    if power:
        fn += '_n%.1f' % power
    if L0:
        fn += '_L%i' % L0
    if B0:
        fn += '_B%i' % B0
    if use_window and not all_sky:
        fn += '_truncate_R%i' % cdict['2_models']['DM'][key]['window']['theta']
    if ps_mask != 'None' and not all_sky:
        fn += '_%s' % ps_mask

    fn += '%s.fits' % psf_dict[psf]
    return fn
    
    

def get_tmp_mdl_fn(key, psf='all'):
    mkpath('models/tmp/')
    fn = 'models/tmp/%s_%s.fits' % (key, psf)
    return fn


def save_tmp_umap(umap, key, psf='all', Es=None):
    fn = get_tmp_mdl_fn(key, psf='all')
    print 'save tmp model to file:'
    print fn
    upix.umap2fits(umap, fn=fn,
              unit=None, kdict=None, comment=None, Es=Es, Eunit='MeV')
    return None

def load_tmp_umap(key, psf='all'):
    fn = get_tmp_mdl_fn(key, psf='all')
    if not os.path.isfile(fn):
        return None
    else:
        print 'load tmp model from file:'
        print fn
        return upix.fits2umap(fn)
    

def get_exposure(cdict, silent=True):
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    irf_base = cdict['1_data']['irf_base']
    irf_select = cdict['1_data']['irf_select']
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'
    exp_folder = prepend_path(gcfit_path, cdict['1_data']['exp-folder'])
    exp_folder = exp_folder.replace(data_base, data_select)
    exp_fn = cdict['1_data']['exp-file']
    exp_fn = prepend_path(exp_folder, exp_fn)
    exp_fn = exp_fn.replace(data_base, data_select)
    exp_fn = exp_fn.replace(irf_base, irf_select)
    if not silent:
        print 'get exposure from file:'
        print exp_fn

    return read_skymap(exp_fn)[0]

def get_data(cdict):
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    fn = cdict['1_data']['file_base']
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'
    folder = prepend_path(gcfit_path, cdict['1_data']['data_folder'])
    fn = prepend_path(folder, fn)
    fn = fn.replace(data_base, data_select)
    print 'get data from file:'
    print fn
    return read_skymap(fn)

def get_sm_counts(cdict):
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    folder = prepend_path(gcfit_path, cdict['3_fitting']['metric']['folder'])
    folder = folder.replace(data_base, data_select)
    fn = cdict['3_fitting']['metric']['file']
    fn = fn.replace(data_base, data_select)
    fn = prepend_path(folder, fn)
    return read_skymap(fn)[0]


def get_data_fn_base(cdict, mc_case=None, kMC=0, include_folder=1):
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'
    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    
    folder = prepend_path(gcfit_path, cdict['1_data']['data_folder'])
    if mc_case == 'gtobssim':
        folder_data = cdict['1_data']['data_folder_gtobssim']
        fn_base = cdict['1_data']['file_base_%s' % gtobssim]
    elif mc_case is not None:
        folder = folder.split('/maps/')[0] + '/MC_maps_%s/' % mc_case
        fn = 'counts_%s_MC%i.fits' % (mc_case, kMC)
    else:
        fn = cdict['1_data']['file_base']
    folder = folder.replace(data_base, data_select)
    fn = fn.replace(data_base, data_select)
    if include_folder:
        mkpath(folder)
        fn = prepend_path(folder, fn)
    return fn


def get_model(cdict, key, spectrum=None):
    if key == 'DM':
        name = cdict['2_models']['DM']['name']
        fn = get_dm_mdl_fn(cdict, name, psf='all')
        mdl = read_skymap(fn)[0]
        print 'Get %s model from file:' % key
        print fn
        print

    else:
        data_base = cdict['1_data']['data_base']
        data_select = cdict['1_data']['data_select']
        irf_base = cdict['1_data']['irf_base']
        irf_select = cdict['1_data']['irf_select']
        

        mfolder_dict = cdict['2_models']['folders']
        model_name = cdict['2_models']['model_name']
        model_list = cdict['2_models'][model_name]

        order = cdict['1_data']['order']
        npix = healpy.nside2npix(2**order)

        if not isinstance(cdict['2_models'][key], dict):
            key_in = cdict['2_models'][key]
        else:
            key_in = key
        folder_key = cdict['2_models'][key_in]['folder']
        folder = mfolder_dict[folder_key]
        folder = prepend_path(gcfit_path, folder)
        fn = cdict['2_models'][key_in]['file']
        munit = cdict['2_models'][key_in]['unit']

        fn = prepend_path(folder, fn)

        fn = fn.replace(data_base, data_select)
        fn = fn.replace(irf_base, irf_select)
        if not os.path.isfile(fn):
            fn += '.gz'

        print 'Get %s model from file:' % key
        print fn
        print
                        
        if fn.endswith('npy'):
            mdl = np.load(fn)
        elif fn.endswith('txt'):
            mdl = np.loadtxt(fn)
        else:
            mdl = read_skymap(fn)[0]
        
        if mdl.ndim == 1:
            mdl = healpy.ud_grade(mdl, 2**order)
            
        if mdl.shape[-1] == npix:
            mdl = mdl.T

    if spectrum is not None:
        if mdl.ndim == 1:
            mdl = np.outer(mdl, spectrum)
        else:
            mdl *= spectrum

    return mdl

def get_mask_old(cdict):
    use_masks = cdict['1_data'].get('use_masks', 0)
    order = cdict['1_data']['order']
    npix = healpy.nside2npix(2**order)
    mask = np.ones(npix)
    if use_masks:
        masks = cdict['3_fitting']['masks']
        for mask_name in masks:
            fn = get_mask_fn(cdict, gcfit_path, mask_name)
            print 'load mask from file:'
            print fn
            mask *= np.load(fn)
    return mask




def nan2zero_num(x):
    if np.isnan(x):
        return 0.
    else:
        return x

nan2zero_vec = np.frompyfunc(nan2zero_num, 1, 1)

def nan2zero(x):
    if np.isnan(np.sum(x)):
        print 'substitute nan with zero'
        return np.array(nan2zero_vec(x), dtype=float)
    else:
        return x

def nan2zero_info(x):
    xx = 1. * x
    while xx.ndim > 1:
        xx = np.sum(xx, axis=1)
    nan_inds = [i for i in range(len(xx)) if np.isnan(xx[i])]
    return nan_inds

def nan2zero_old(arr):
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = 0.
    return arr

def get_Es():
    E0 = 0.101505177
    Eb = E0 * 2**np.arange(-1.5, 14.4, 0.5)
    Es = np.sqrt(Eb[1:] * Eb[:-1])
    return Es

def edisp_mat_mean(M, Es=get_Es(), spf=plaw([1,2])):
    nE = len(Es)
    Eb = Es2Ebins(Es)
    nE0 = M.shape[0]
    Eb0 = np.logspace(np.log10(Eb[0]), np.log10(Eb[-1]), nE0 + 1)
    Es0 = np.sqrt(Eb0[1:] * Eb0[:-1])
    if nE0 % nE != 0:
        print 'Problem with dimensions of fine energy dispersion matrix'
        exit()
    k = M.shape[0] / nE
    res = np.zeros((nE, nE))
    spectrum = spf(Es0)
    sp_av = np.mean(spectrum.reshape((nE, k)), axis=1)

    for i in range(nE):
        for j in range(k):
            avg = M[i * k + j] * spectrum
            avg = np.mean(avg.reshape((nE, k)), axis=1) / sp_av
            res[i] += avg
        res[i] /= np.sum(res[i])
    return res

def get_edisp_matrix0(cdict):
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'

    data_base = cdict['1_data']['data_base']
    data_select = cdict['1_data']['data_select']
    irf_base = cdict['1_data']['irf_base']
    irf_select = cdict['1_data']['irf_select']

    irf_folder = prepend_path(gcfit_path, cdict['1_data']['irf_folder'])
    irf_folder = irf_folder.replace(data_base, data_select)


    edisp = cdict['1_data'].get('edisp', 0)
        
    edisp_fn = cdict['1_data'].get('edisp_mix_file_base', None)
    edisp_fn = edisp_fn.replace(data_base, data_select)
    edisp_fn = edisp_fn.replace(irf_base, irf_select)
    edisp_fn = irf_folder + edisp_fn
    res = np.load(edisp_fn)
    return res
    
def get_edisp_matrix(cdict, Es=get_Es(), spf=plaw([1,2])):
    edisp = cdict['1_data'].get('edisp', 0)
    M0 = get_edisp_matrix0(cdict)
    nE_max = len(get_Es())
    factor = len(M0) / nE_max
    if len(Es) != nE_max:
        M0 = M0[:factor * len(Es),:factor * len(Es)]
    res = edisp_mat_mean(M0, Es=Es, spf=spf)
    if edisp:
        return res
    else:
        return np.eye(len(res))

def get_log_index(alpha, EE):
    inds = np.arange(len(alpha))
    res = np.sum(inds * alpha * np.log(EE)**(inds - 1.))
    return res

def fit_poly(xs, ys, metric=1., max_deg=3):
    tmpls = np.array([xs**deg for deg in range(max_deg + 1)])
    fit = sca.lstsq(ys, tmpls, g=metric)
    alpha = fit[0]
    model = np.dot(alpha, tmpls)
    return alpha, model

def fit_log_poly(Es, Fs, metric=1., max_deg=3, Emax=None):
    ys = step(Fs) * np.log(np.abs(Fs))
    xs = np.log(Es)
    g = metric * Fs**2
    if Emax != None:
        imax = num.findIndex(Es, Emax)
        log_model = np.zeros(len(xs))
        alpha, log_model[:imax] = fit_poly(xs[:imax], ys[:imax],
                                           metric=g[:imax], max_deg=max_deg)
        EE = Es[imax - 1]
        index = get_log_index(alpha, EE)
        log_model[imax:] = log_model[imax - 1] + index * np.log(Es[imax:] / EE)
    else:
        alpha, log_model = fit_poly(xs, ys, metric=g, max_deg=max_deg)
    return alpha, np.exp(log_model)

def update_item(dct0, key0, dct1, key1):
    if dct1.get(key1) is not None:
        dct0[key0] = dct1[key1]

def update_cdict(cdict, fdict=None, fit_case='Baseline', mc_case=None, kMC=0):
    if fdict is None:
        return cdict

    base_dict = fdict['Baseline']
    # data and irf selection
    #data = fdict[fit_case].get('data', 'P8_P302_ultraclean_veto_z90')
    #cdict['1_data']['data_select'] = data
    update_item(cdict['1_data'], 'file_base', fdict[fit_case], 'data_file_base')
    update_item(cdict['1_data'], 'exp-file', fdict[fit_case], 'exp_file_base')
    update_item(cdict['1_data'], 'psf-file', fdict[fit_case], 'psf_file_base')
    update_item(cdict['1_data'], 'data_select', fdict[fit_case], 'data')
    update_item(cdict['1_data'], 'PSFs', fdict[fit_case], 'psfs')
    update_item(cdict['1_data'], 'order', fdict[fit_case], 'order')
    update_item(cdict['1_data'], 'umap', fdict[fit_case], 'umap')
    
    data = cdict['1_data']['data_select']
    cdict['1_data']['irf_select'] = fdict['irfs'][data]

    # model selection
    #cdict['2_models']['model_name'] = fdict[fit_case].get('model')
    update_item(cdict['2_models'], 'model_name', fdict[fit_case], 'model')
    if fdict[fit_case].get('extra_components') is not None:
        model_name = cdict['2_models']['model_name']
        comp_list = cdict['2_models'][model_name]
        extra_comps = fdict[fit_case]['extra_components'].get('names')
        if extra_comps is None or len(extra_comps) == 0:
            print 'Warning: extra_components entry has no extra components added'
        else:
            comp_list.extend(extra_comps)
            cdict['2_models'][model_name] = comp_list
            for key in extra_comps:
                if key in fdict[fit_case]['extra_components'].keys():
                    cdict['2_models'][key] = fdict[fit_case]['extra_components'][key]


    #cdict['2_models']['DM']['name'] = IG_tmpl
    update_item(cdict['2_models']['DM'], 'name', fdict[fit_case], 'extra_templates')
    update_item(cdict['2_models']['DM'], 'name', fdict[fit_case], 'IG_template')

    update_item(cdict['2_models'], 'add_dm', fdict[fit_case], 'add_dm')
    update_item(cdict['2_models'], 'DM_PSF', fdict[fit_case], 'DM_PSF')
    update_item(cdict['2_models'], 'reconvolve_psf', fdict[fit_case], 'reconvolve_psf')

    IG_tmpl = cdict['2_models']['DM']['name']
    if IG_tmpl == 'None':
        cdict['2_models']['add_dm'] = 0
    elif IG_tmpl == 'Cusp':
        update_item(cdict['2_models']['DM'][IG_tmpl], 'power', fdict[fit_case], 'IG_scaling')
        update_item(cdict['2_models']['DM'][IG_tmpl], 'L0', fdict[fit_case], 'L0')
        update_item(cdict['2_models']['DM'][IG_tmpl], 'B0', fdict[fit_case], 'B0')
    if cdict['2_models']['DM'].get(IG_tmpl, {}).get('window'):
        update_item(cdict['2_models']['DM'][IG_tmpl]['window'], 'use_window', fdict[fit_case], 'use_window')

    update_item(cdict['2_models']['DM'], 'lowE_sp', fdict[fit_case], 'DM_lowE_sp')
    update_item(cdict['2_models']['ISO'], 'lowE_sp', fdict[fit_case], 'ISO_lowE_sp')
    update_item(cdict['2_models']['IC'], 'lowE_sp', fdict[fit_case], 'ICS_lowE_sp')
    update_item(cdict['2_models']['IC'], 'lowE_sp', fdict[fit_case], 'ICS_lowE_sp')
    extra_tmpl = fdict[fit_case].get('extra_templates')
    if extra_tmpl is not None and fdict[fit_case][extra_tmpl] is not None:
        update_item(cdict['2_models']['DM'][extra_tmpl], 'B0', fdict[fit_case], 'ROI_B0')
        update_item(cdict['2_models']['DM'][extra_tmpl], 'L0', fdict[fit_case], 'ROI_L0')
        update_item(cdict['2_models']['DM'][extra_tmpl], 'theta', fdict[fit_case][extra_tmpl], 'theta')
        update_item(cdict['2_models']['DM'][extra_tmpl], 'dtheta', fdict[fit_case][extra_tmpl], 'dtheta')
        update_item(cdict['2_models']['DM'][extra_tmpl], 'kmax', fdict[fit_case][extra_tmpl], 'kmax')
        update_item(cdict['2_models']['DM'][extra_tmpl], 'all_modes', fdict[fit_case][extra_tmpl], 'all_modes')


    # ROI selection
    if fdict[fit_case].get('umap_name'):
        umap_name = fdict[fit_case].get('umap_name')
    else:
        ROI = fdict[fit_case].get('ROI', base_dict.get('ROI'))
        umap_name = 'umap_%s' % ROI
    cdict['5_umap']['umap_name'] = umap_name
    if cdict['5_umap'].get(umap_name) is None:
        cdict['5_umap'][umap_name] = {}
    update_item(cdict['5_umap'][umap_name], 'ROI_B0', fdict[fit_case], 'ROI_B0')
    update_item(cdict['5_umap'][umap_name], 'ROI_L0', fdict[fit_case], 'ROI_L0')
    update_item(cdict['5_umap'][umap_name], 'ROI_size', fdict[fit_case], 'ROI_size')
    update_item(cdict['5_umap'][umap_name], 'ROI_shape', fdict[fit_case], 'ROI_shape')


    # PS catalog, mask and refitting
    ps_cat = fdict[fit_case].get('ps_catalog', base_dict.get('ps_catalog'))
    if ps_cat is not None:
        cdict['2_models']['PS'] = 'PS_%s_adaptive' % ps_cat
        cdict['7_psrefit']['cat'] = fdict[fit_case].get('ps_refit_catalog', ps_cat)
    update_item(cdict['7_psrefit'], 'nps_inner', fdict[fit_case], 'nps_inner')
    update_item(cdict['7_psrefit'], 'nps_outer', fdict[fit_case], 'nps_outer')
    update_item(cdict['7_psrefit'], 'PSrank_kw', fdict[fit_case], 'PSrank_kw')
    update_item(cdict['7_psrefit'], 'select_value', fdict[fit_case], 'select_value')
    update_item(cdict['7_psrefit'], 'PS_max_value', fdict[fit_case], 'PS_max_value')
    update_item(cdict['7_psrefit'], 'B0', fdict[fit_case], 'B0')
    update_item(cdict['7_psrefit'], 'L0', fdict[fit_case], 'L0')
    update_item(cdict['7_psrefit'], 'R_inner', fdict[fit_case], 'R_inner')
    #print fdict[fit_case]
    #print cdict['7_psrefit']
    #print
    #exit()

    ps_mask = fdict[fit_case].get('ps_mask', base_dict.get('ps_mask'))
    if ps_mask is not None:
        if ps_mask == 'None':
            psm_str = ps_mask
        elif ps_mask in ['all_sky', 'IG', 'OG']:
            psm_str = 'ps_mask_%s' % ps_cat
            if ps_mask in ['IG', 'OG']:
                psm_str += '_%s' % ps_mask
        else:
            psm_str = ps_mask
        cdict['3_fitting']['ps_mask'] = psm_str
    #update_item(cdict['3_fitting'], 'ps_mask', fdict[fit_case], 'ps_mask')

    update_item(cdict['1_data'], 'psrefit', fdict[fit_case], 'ps_refit_IG')
    update_item(cdict['1_data'], 'positive_norm', fdict[fit_case], 'positive_norm')
    update_item(cdict['1_data'], 'positive_ps_norm', fdict[fit_case], 'positive_ps_norm')
    update_item(cdict['3_fitting']['low_energy'], 'E_max', fdict[fit_case], 'E_max')
    update_item(cdict['3_fitting']['low_energy'], 'E_min', fdict[fit_case], 'E_min')
    update_item(cdict['6_sca'], 'func', fdict[fit_case], 'sca_func')
    update_item(cdict['4_output']['plot_general'], 'ext', fdict, 'plot_ext')
    update_item(cdict['4_output']['resid_plot'], 'plot_bins', fdict[fit_case], 'plot_bins')
    update_item(cdict['2_models'], 'model_base', fdict[fit_case], 'model_base')
    update_item(cdict['2_models'], 'model_select', fdict[fit_case], 'model_select')
    update_item(cdict['3_fitting'], 'masks', fdict[fit_case], 'masks')
    update_item(cdict['3_fitting'], 'poisson', fdict[fit_case], 'poisson')


    if mc_case is None:
        update_item(cdict['4_output']['resid'], 'save', fdict[fit_case], 'save_resid_counts')
        update_item(cdict['4_output']['model'], 'save_all', fdict[fit_case], 'save_model_counts')
        update_item(cdict['4_output']['model'], 'save_diffuse', fdict[fit_case], 'save_diff_model_counts')
        update_item(cdict['4_output']['model'], 'save_gas', fdict[fit_case], 'save_gas_model_counts')

    return None
    


def get_cdict(control_fn, fit_case='Baseline', mc_case=None, kMC=0, ignore_missing=False):
    gcfit_path = os.getcwd().split('gcfit')[0] + 'gcfit/'
    control_fn = prepend_path(gcfit_path + 'control/', control_fn)
    cdict = dio.loaddict(control_fn)
    # if the dictionary is the fit cases, then load control from the fit cases file
    if cdict.get('default_control') is not None or cdict.get('low_level_control') is not None:
        if fit_case not in cdict.keys():
            print 'fit case %s is not found in %s' % (fit_case, control_fn)
            if ignore_missing:
                cdict[fit_case] = {0:0}
            else:
                exit()
        control_fn = cdict.get('default_control')
        control_fn = cdict.get('low_level_control', control_fn)
        control_fn = prepend_path(gcfit_path + 'control/', control_fn)
        fdict = copy.deepcopy(cdict)
        cdict = dio.loaddict(control_fn)
        update_cdict(cdict, fdict=fdict, fit_case='Baseline', mc_case=mc_case, kMC=kMC)
        if fit_case != 'Baseline':
            update_cdict(cdict, fdict=fdict, fit_case=fit_case, mc_case=mc_case, kMC=kMC)
    return cdict



def group_spectra(control_fn, Estr='lowE', mc_case=None, kMC=0, fit_case='Baseline', clean=True,
                  local_groups=['GC excess'], try_MC=0, ROI_R10=1, gc_only=False):
    cdict = get_cdict(control_fn, fit_case=fit_case, mc_case=mc_case)
    spectra_fn = output_name(cdict, 'spectra', mc_case=mc_case, kMC=kMC, fit_case=fit_case, simple=True)
    print spectra_fn
    sdict = dio.loaddict(spectra_fn)
    if gc_only:
        groups = {'GC excess': 'DM'}
    else:
        groups = cdict['2_models']['Groups']
    res_sp = {}
    res_err = {}
    sp_dict = sdict['spectra'][Estr]
    sp_dict_MC = {}
    if try_MC:
        sp_dict_MC = sdict.get('spectra_MC', {}).get(Estr, {})


    err_dict = sdict['rel_err'][Estr]
    Es = get_Es()
    Om_ratio = sdict.get('Om_ROI', 4. * np.pi) / (4. * np.pi)

    E_min = cdict['3_fitting']['low_energy']['E_min']
    E_max = cdict['3_fitting']['low_energy']['E_max']
    E_min = max(0.1, E_min)
    E_max = min(E_max, 1000.)
    imin = num.findIndex(Es, E_min)
    imax = num.findIndex(Es, E_max)
    ebins = range(imin, imax)
    chi2 = np.array(sdict['chi2'][Estr])
    if 'mtwo_logL' in sdict.keys():
        poisson = 1
        mtwo_logL = np.array(sdict['mtwo_logL'][Estr])
        mtwo_logL_tot = sum(mtwo_logL[ebins])
    else:
        poisson = 0
        print '\t Poisson fit not found!!!\n\n\n'

    chi2_tot = sum(chi2[ebins])
    PSFs = cdict['1_data'].get('PSFs')
    if not isinstance(PSFs, list):
        PSFs = [PSFs]
    nPSF = len(PSFs)
    ndof = sdict['npix'] * len(ebins) * nPSF

    #print ebins, len(ebins)
    #print chi2_tot, ndof, sdict['npix'], chi2_tot / ndof
    res_sp['chi2'] = chi2_tot
    if poisson:
        res_sp['mtwo_logL'] = mtwo_logL_tot
    res_sp['ndof'] = ndof
    res_sp['npix'] = sdict['npix']
    res_sp['nebins'] = len(ebins)
    res_sp['nPSF'] = nPSF

    fdict = dio.loaddict('control/' + control_fn)
    label = fit_case.replace('_', ' ')
    label = fdict[fit_case].get('label', label)
    res_sp['label'] = label

    for key in groups.keys():
        flux = np.zeros_like(Es)
        err2 = np.zeros_like(Es)
        for mkey in groups[key]:
            if mkey in sp_dict.keys():
                if try_MC and sp_dict_MC.get(mkey):
                    sp_add = np.array(sp_dict_MC[mkey])
                else:
                    sp_add = np.array(sp_dict[mkey])
                flux += sp_add
                err2 += (np.array(err_dict[mkey]) * sp_add)**2
        res_sp[key] = flux
        res_err[key] = np.sqrt(err2)
        if key in local_groups:
            res_sp[key] *= Om_ratio
            res_err[key] *= Om_ratio
        if key == 'GC excess' and ROI_R10:
            res_sp[key] = np.array(sdict['Signal_R10'])
            res_err[key] = np.array(sdict['Signal_R10_err'])

    if clean:
        remove_empty(res_sp, res_err)
    return res_sp, res_err

def MC_group_spectra(control_fn, Estr='lowE', mc_case=None, nMC=1, kMC=0, fit_case='Baseline', clean=True, ROI_R10=1):
    cdict = get_cdict(control_fn, fit_case=fit_case, mc_case=mc_case)
    if mc_case is None:
        return None
    res_sp = {}
    res_err = {}
    groups = cdict['2_models']['Groups']
    for key in groups.keys():
        res_sp[key] = []

    for k in range(kMC, kMC + nMC):
        print 'MC %i' %k
        sp, err = group_spectra(control_fn, Estr=Estr,
                           mc_case=mc_case, kMC=k, fit_case=fit_case, clean=False, ROI_R10=ROI_R10)
        for key in groups.keys():
            res_sp[key].append(sp[key])
            
    for key in res_sp.keys():
        res_err[key] = np.std(res_sp[key], axis=0)
        res_sp[key] = np.mean(res_sp[key], axis=0)
    if clean:
        remove_empty(res_sp, res_err)
    return res_sp, res_err

def syst_group_spectra(control_fn, Estr='lowE', syst_cases=[], mc_case=None, kMC=0, clean=True, ROI_R10=1, gc_only=False):
    cdict = get_cdict(control_fn)
    res_sp = {}
    res_err = {}
    
    extra_keys = ['chi2', 'mtwo_logL', 'ndof', 'npix', 'nebins', 'nPSF', 'label']
    for key in extra_keys:
        res_sp[key] = {}

    if gc_only:
        groups = {'GC excess': 'DM'}
    else:
        groups = cdict['2_models']['Groups']
    for key in groups.keys():
        res_sp[key] = []
        res_err[key] = []

    for fit_case in syst_cases:
        print fit_case
        sp, err = group_spectra(control_fn, Estr=Estr, fit_case=fit_case, mc_case=mc_case, kMC=kMC, clean=False, ROI_R10=ROI_R10, gc_only=gc_only)
        for key in groups.keys():
            res_sp[key].append(sp[key])
            res_err[key].append(err[key])
        for key in extra_keys:
            res_sp[key][fit_case] = sp[key]
        #print sp['chi2'] / sp['ndof']

    for key in groups.keys():
        res_sp[key] = np.array(res_sp[key])
        res_err[key] = np.array(res_err[key])
        

    if clean:
        remove_empty(res_sp, res_err)
    return res_sp, res_err


def remove_empty(sp_dict, err_dict=None):
    for key in sp_dict.keys():
        if isinstance(sp_dict[key], np.ndarray) and np.sum(sp_dict[key]**2) == 0.:
            sp_dict.pop(key)
            if err_dict is not None:
                err_dict.pop(key)
    return None


def get_fit_cases(fdict, cases):
    if cases in fdict.keys():
        cases_list = fdict[cases]
        syst_cases = Set([])
        for key in cases_list:
            if key in fdict['fit_cases'].keys():
                syst_cases.update(fdict['fit_cases'][key])
        syst_cases = list(syst_cases)
    else:
        syst_cases = fdict['fit_cases'].get(cases)
    if syst_cases is None:
        print 'cases %s are not in the list:' % cases
        print fdict['fit_cases'].keys()
        return None
    return syst_cases


def get_excess_name(sdict, cdict):
    ex_names = cdict['2_models']['Groups']['GC excess']
    for name in ex_names:
        if name in sdict.keys():
            return name
    return None



def flux2luminosity(Es, sp, Om_ROI=4.*np.pi):
    sp1 = step(sp) * sp
    Ebins = Es2Ebins(Es)
    dE = Ebins[1:] - Ebins[:-1]
    Rcm = (const.rSun * const.kpc2cm)
    return 4. * np.pi * Om_ROI * Rcm**2 * np.sum(sp1 / Es * dE) / const.erg2GeV

def msp_sp(L, index=1.6, cutoff=4., Om_ROI=4.*np.pi):
    
    # get the spectrum
    Es = get_Es()
    Ebins = Es2Ebins(Es)
    dE = Ebins[1:] - Ebins[:-1]
    pars = [1., index - 2., cutoff,]
    sp = plaw_cut(pars)
    L0 = flux2luminosity(Es, sp(Es), Om_ROI=Om_ROI)
    norm = L / L0
    pars[0] = norm
    sp = plaw_cut(pars)
    return sp


def get_psf_dict():
    psf_dict = {'all':''}
    for i in range(4):
        psf_dict[i] = '_psf%i' %i
    return psf_dict


def intensity2flux(ys, err, Om_ROI, flux_plot):
    if flux_plot:
        return np.array(ys) * Om_ROI, np.array(err) * Om_ROI
    else:
        return ys, err


def select_ebins(Es, cdict, name='low_energy', plotting=0):
    if plotting:
        E_min = cdict['4_output']['plot_general'].get('E_min', 0.1)
        E_max = cdict['4_output']['plot_general'].get('E_max', 1000.)
    else:
        E_min = cdict['3_fitting'][name]['E_min']
        E_max = cdict['3_fitting'][name]['E_max']
    imin = num.findIndex(Es, E_min)
    imax = num.findIndex(Es, E_max)
    return range(imin, imax)

