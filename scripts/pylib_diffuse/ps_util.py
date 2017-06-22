# utility functions to work with point sources
# python pylib/ps_util.py

import numpy as np
import numeric as num
import pyfits
import scipy
from scipy import interpolate
import healpy
from matplotlib import pyplot
from matplotlib import rc
import os
import time

# constants
GeV2MeV = 1000.

import healpylib as hlib
import upix

# new functions

const = lambda x: np.ones_like(x)

vec2theta = lambda vectors, BL: hlib.vec2theta(vectors, BL[0], BL[1])

def separate_inds(nside, theta_max, BL0, inds):
    """
    separate the indices inside and ouside of theta_max from BL0
    INPUT:
        nside - healpix nside parameter
        theta_max - radius (rad)
        BL0 = [B, L] (rad)
        inds - indices of healpix pixels in nest format
    OUTPUT:
        inds_out - list: indices outside of theta_max
        inds_in - list: indices inside theta_max

    """
    if len(inds) == 0:
        return [], []
    mask = hlib.mask_circle(nside, theta_max, BL0, nest=True, inds=inds)
    np = len(mask)
    inds_out = [inds[i] for i in range(np) if mask[i] == 0]
    inds_in = [inds[i] for i in range(np) if mask[i] == 1]
    return inds_out, inds_in

hmap_upgrade = upix.hmap_upgrade

hmap_degrade = upix.hmap_degrade

umap_udgrade = upix.umap_udgrade

def umap_degrade(umap):
    return upix.umap_udgrade(umap)


def ps_umap_tmpl(nside, BL0=[0.,0.], rs=np.deg2rad(np.array([15., 5., 2., 0.5])), orders=[7, 9, 11, 13],):
    theta_max = rs[0]
    #print hlib.__file__
    mask = hlib.mask_circle(nside, theta_max, BL0, nest=True)
    inds_in = [i for i in range(len(mask)) if mask[i] == 1]
    hdict = {}
    for i in range(1, len(rs)):
        order = orders[i - 1]
        order_new = orders[i]

        inds_out, inds_in = separate_inds(2**order, rs[i], BL0, inds_in)
        hdict[order] = inds_out

        inds_in = hlib.get_daughters_list(2**order, inds_in,
                                          nside_new=2**order_new, nest=True)

    hdict[orders[-1]] = inds_in
    return upix.umap(hpix_dict=hdict, is_mean=True)

def pspdf(psf_funcs, order_out='min', theta_max=None,
          B=0., L=0., rs=np.deg2rad(np.array([15., 5., 2., 0.5])), orders=[7, 9, 11, 13],
          output_patch=False, nest=False, print_times=False):
    """
    return the healpix maps with PS pdfs
    INPUT:
        psf_funcs - list of functions: psf[i](theta) where i is the energy index, theta is in rads
        order_out - order of the output pixels
            default: 'min' (smallest order in the adaptive map)
        theta_max (rad) -  max angle where the PSF should be defined
            default: None (determined from the max radius in the adaptive maps)
        B, L (rad) - direction of the PS in Galactic coordinates
            default: B = 0, L = 0
        rs (rad) - array of angles for adaptive map rings
            default: np.deg2rad(np.array([15., 5., 2., 0.5]))
        orders (int) - array: orders of healpix indices
            default: [7, 9, 11, 13]
        output_patch - wether to output a patch of the sky,
            if True then the output is (hinds, values): healpix indices and the values
            if False then the output is the full healpix maps
            default: False
    OUTPUT:
        psf maps - array (npix, nE), where nE = len(psf_funcs), npix - healpix npix parameter
        if output_patch = True, then the output is
            (hinds, values):
            hinds (int) - array, healpix indices in nest format
            values - array (len(hinds), nE), values of the psf
    
    """
    # determine the adaptive resolution map from BL0, rs, and orders
    if print_times:
        t0 = time.time()
        t00 = t0
    out_ordering = 'NESTED'
    if not nest:
        out_ordering = 'RING'
    BL0 = np.array([B, L])
    rs = np.array(rs)
    if theta_max is not None:
        rs[0] = theta_max
    theta_max = rs[0]
    if order_out == 'min':
        order_out = min(orders)
    elif orders[0] < order_out:
        orders[0] = order_out

    nside = 2**order_out
    npix = healpy.nside2npix(nside)

    ps_umap = ps_umap_tmpl(nside, BL0=BL0, rs=rs, orders=orders)
    hdict = ps_umap.hpix_dict()

    if print_times:
        print 'init time: %.3g' % (time.time() - t0)
        t0 = time.time()
    
    # get vectors, thetas and calculate PSF function in pixels
    vectors = upix.hdict2vec(hdict)
    thetas = vec2theta(vectors, BL0)
    nE = len(psf_funcs)
    values = np.zeros((len(thetas), nE))
    for i in range(nE):
        values[:, i] = psf_funcs[i](thetas)

    if print_times:
        print 'values time: %.3g' % (time.time() - t0)
        t0 = time.time()
    
    # create a umap with values in pixels and downgrade to order_out
    umap = upix.umap(values=values, hpix_dict=hdict, is_mean=True)
    umap_d = umap_degrade(umap)

    if print_times:
        print 'umap time: %.3g' % (time.time() - t0)
        print 'total time: %.3g' % (time.time() - t00)

    if output_patch:
        hpix = umap_d.hpixels(order_out)
        if not nest:
            hpix = healpy.nest2ring(2**order_out, hpix)
        return hpix, umap_d.values()
    else:
        return upix.umap2hmap(umap_d, out_ordering=out_ordering)


def get_psf(psf_filen, Es):
    '''
    read the psf table and define a function of ibin and theta
    '''
    hdu = pyfits.open(psf_filen)

    psf_Es = hdu['PSF'].data.field('Energy') / GeV2MeV
    psf_table = hdu['PSF'].data.field('Psf')
    thetas = np.deg2rad(hdu['THETA'].data.field('Theta'))
    psf_funcs = []
    
    for EE in Es:
        psf_bin = num.findIndex(psf_Es, EE, nearest=True)
        func = interpolate.interp1d(thetas, psf_table[psf_bin],
                                    bounds_error=False, fill_value=0.)
        psf_funcs.append(func)

    return psf_funcs, thetas

def get_psf_weighted(fn_psf, Es, nsub=5, ps_ind=2.4):
    nE = len(Es)
    Ebins = np.zeros((nE + 1))
    f = np.sqrt(Es[1] / Es[0])
    Ebins[1:] = Es * f
    Ebins[0] = Es[0] / f
    
    
    Es_fine = np.logspace(np.log10(Ebins[0]), np.log10(Ebins[-1]), (nE * nsub + 1))
    Es_fine = np.sqrt(Es_fine[1:] * Es_fine[:-1])
    psf_funcs_fine, thetas = get_psf(fn_psf, Es_fine)
    thetas_deg = np.rad2deg(thetas)
    
    ps_sp = lambda EE: EE**(-ps_ind)
    def get_psf_loc(ind):
        def fn(theta):
            res = 0.
            norm = 0.
            for i in range(ind * nsub, (ind + 1) * nsub):
                res += ps_sp(Es_fine[i]) * psf_funcs_fine[i](theta)
                norm += ps_sp(Es_fine[i])
            return res / norm
        return fn
    psf_funcs = [get_psf_loc(ind) for ind in range(nE)]
    return psf_funcs, thetas

def get_thetas(v0, vectors):
    '''
    robust calculation of angles between vectors

    '''
    costh = np.dot(v0, vectors)/np.linalg.norm(v0)
    costh /= np.sqrt(np.sum(vectors**2, axis=0))
    costh = np.minimum(costh, np.ones_like(costh))
    costh = np.maximum(costh, -np.ones_like(costh))

    return np.arccos(costh)

def get_theta_thres(psf_funcs, thetas, thres=0.95, ndim=2):
    '''
    calculate theta angles corresponding to flux contaiment equal to thres
    
    '''
    theta_thres = []
    dth = thetas[1:] - thetas[:-1]
    theta_c = (thetas[1:] + thetas[:-1])/2.
    for psf in psf_funcs:

        if ndim == 2:
            diff_distr = np.sin(theta_c) * psf(theta_c) * dth
        elif ndim == 1:
            diff_distr = psf(theta_c) * dth
        diff_distr /= np.sum(diff_distr)

        cumul_distr = 0
        i_thres = -1
        while cumul_distr < thres:
            i_thres += 1
            cumul_distr += diff_distr[i_thres]
            
        theta_thres.append(theta_c[i_thres])
    return np.array(theta_thres)

def get_pssp(ps_ind, thdu, cat_name='3FGL'):
    f = None
    if cat_name.find('FGL') > -1 or cat_name.find('fermipy') > -1:
        sp_name = thdu.data.field('SpectrumType')[ps_ind]
        #print '\nSpectrum:', sp_name
        if sp_name == 'PLExpCutoff' or sp_name == 'PLSuperExpCutoff':
            K = thdu.data.field('Flux_Density')[ps_ind] * GeV2MeV
            Gamma = thdu.data.field('Spectral_Index')[ps_ind]
            Ec = thdu.data.field('Cutoff')[ps_ind] / GeV2MeV
            b = thdu.data.field('Exp_Index')[ps_ind]
            E0 = thdu.data.field('Pivot_Energy')[ps_ind] / GeV2MeV
            def f(E):
                return K * (E / E0)**(-Gamma) * np.exp((E0 / Ec)**b - (E / Ec)**b)

        elif sp_name == 'LogParabola' or sp_name == 'PowerLaw':
            K = thdu.data.field('Flux_Density')[ps_ind] * GeV2MeV
            alpha = thdu.data.field('Spectral_Index')[ps_ind]
            beta = thdu.data.field('beta')[ps_ind]
            E0 = thdu.data.field('Pivot_Energy')[ps_ind] / GeV2MeV
            if sp_name == 'PowerLaw':
                beta = 0.
            #print K, alpha, beta, E0
            def f(E):
                return K * (E / E0)**(-alpha - beta * np.log(E / E0))
        return f
    elif cat_name.find('uw') > -1:
        # looks like a copy from 'FGL' catalogs, may be a bug
        sp_name = thdu.data.field('SpectrumType')[ps_ind]
        #print '\nSpectrum:', sp_name
        if sp_name == 'PLExpCutoff' or sp_name == 'PLSuperExpCutoff':
            K = thdu.data.field('Flux_Density')[ps_ind] * GeV2MeV
            Gamma = thdu.data.field('Spectral_Index')[ps_ind]
            Ec = thdu.data.field('Cutoff_Energy')[ps_ind] / GeV2MeV
            b = thdu.data.field('Exp_Index')[ps_ind]
            E0 = thdu.data.field('Pivot_Energy')[ps_ind] / GeV2MeV
            def f(E):
                return K * (E / E0)**(-Gamma) * np.exp((E0 / Ec)**b - (E / Ec)**b)

        elif sp_name == 'LogParabola' or sp_name == 'PowerLaw':
            K = thdu.data.field('Flux_Density')[ps_ind] * GeV2MeV
            alpha = thdu.data.field('Spectral_Index')[ps_ind]
            beta = thdu.data.field('beta')[ps_ind]
            E0 = thdu.data.field('Pivot_Energy')[ps_ind] / GeV2MeV
            if sp_name == 'PowerLaw':
                beta = 0.
            #print K, alpha, beta, E0
            def f(E):
                return K * (E / E0)**(-alpha - beta * np.log(E / E0))
        return f
    else:
        print "Not configured for catalog type: ", cat_name

    return None

def pssp2counts(pssp, Es, pspdf, exposure):
    f = np.sqrt(Es[1] / Es[0])
    dE = Es * (f - 1. / f)
    maps = pspdf * exposure
    if maps.shape[0] == len(Es):
        maps = maps.T
    dOm = 4. * np.pi / maps.shape[0]
    maps *= pssp(Es) * dE * dOm
    return maps.T

    
def ps_templates(nside, psf_funcs, cat_thdu, cat_name='3FGL',
                 select_kw=None, nps=None, select_value=None, PS_max_value=None, mask=None,
                 Es=None, exposure=None,
                 **kwargs):
    '''
    INPUT:
        nside - healpix nside parameter
        psf_funcs - list of functions of theta
        cat_thdu - catalog table
        select_kw - select a subsample of ps according to the keyward column
        nps - number of PS with largest values that are taken into account
                used only if select_kw is not None
        select_value - only sources above this value are taken into account
                used only if nps is None and select_kw is not None
        PS_max_value - only sources below or equal to this value are taken into account
                used only if nps is None and select_kw is not None
        mask - healpix map, only sources in unmaseked pixels are taken into account
        **kwargs - optional arguments for "pspdf" function: pspdf(psf_funcs, **kwargs)
    OUTPUT:
        tmpls - an array of templates (nPS, nE, npix) array
    '''
    # check the spatial mask
    names = cat_thdu.data.field(0)
    Bs = np.deg2rad(cat_thdu.data.field('GLAT'))
    Ls = np.deg2rad(cat_thdu.data.field('GLON'))
    inds = range(len(Bs))
    if mask is not None:
        hpix = healpy.ang2pix(nside, np.pi/2 - Bs, Ls)
        inds = [i for i in inds if mask[hpix[i]] > 0]
    if nps == 'all':
        nps = len(inds)

    # select point sources above threshold
    vals = cat_thdu.data.field(select_kw)
    #print 'select_value', select_value
    #print 'PS_max_value', PS_max_value
    #print 'nps', nps
    if nps is None:
        if select_value is None and PS_max_value is None:
            print 'nps or select_value or PS_max_value should be defined'
            exit()
        if select_value is not None:
            inds = [i for i in inds if vals[i] > select_value]
        if PS_max_value is not None:
            inds = [i for i in inds if vals[i] <= PS_max_value]
    else:
        vals_sorted = np.sort(vals[inds])
        if nps < len(inds):
            thres = vals_sorted[-nps - 1]
            inds = [i for i in inds if vals[i] > thres]
#print 'Threshold for PS selection:', select_value
#print 'N ps:', nps
#print 'Threshold for PS selection:', thres
#print 'All values:\t', vals[inds]
#print 'Values above threshold:\t', vals[inds]

    # calculate templates for the selected sources
    order_out = int(np.log2(nside))
    tmpls = []
    for k, i in enumerate(inds):
        if isinstance(names[0], str):
            print '%i of %i, ps index: %i, name: %s \t' % (k + 1, len(inds), i, names[i]),
        else:
            print '%i of %i, ps index: %i,' % (k + 1, len(inds), i),
        #if select_value is not None or PS_max_value is not None:
        print 'selected value: %s,' % vals[i],
        print '(B, L) = (%.2f, %.2f)' % tuple(np.rad2deg([Bs[i], Ls[i]]))
        pspdfs = pspdf(psf_funcs, order_out=order_out, B=Bs[i], L=Ls[i]).T
        pssp = get_pssp(i, cat_thdu, cat_name=cat_name)
        tmpl = pssp2counts(pssp, Es, pspdfs, exposure)
        tmpls.append(tmpl)

    return np.array(tmpls)

def bl2ps_tmpls(nside, psf_funcs, Bs, Ls, names=None, vals=None, nps=None, select_value=None,
                mask=None, Es=None, exposure=None):
    inds = range(len(Bs))
    if mask is not None:
        hpix = healpy.ang2pix(nside, np.pi/2 - Bs, Ls)
        inds = [i for i in inds if mask[hpix[i]] > 0]
    
    if select_value is not None and vals is not None:
        thres = select_value
        inds = [i for i in inds if vals[i] > thres]
    elif nps is not None and nps < len(inds) and vals is not None:
        vals_sorted = np.sort(vals[inds])
        thres = vals_sorted[-nps - 1]
        inds = [i for i in inds if vals[i] > thres]
    
    order_out = int(np.log2(nside))
    tmpls = []
    for k, i in enumerate(inds):
        if names is not None:
            print '%i of %i, ps name: %s,' % (k + 1, len(inds), names[i]),
        else:
            print '%i of %i, ps index: %i,' % (k + 1, len(inds), i),
        print '(B, L) = (%.2f, %.2f)' % tuple(np.rad2deg([Bs[i], Ls[i]]))
        pspdfs = pspdf(psf_funcs, order_out=order_out, B=Bs[i], L=Ls[i]).T
        pssp = const
        tmpl = pssp2counts(pssp, Es, pspdfs, exposure)
        tmpls.append(tmpl)
    
    return np.array(tmpls)



# old functions




# functions as defined for the 2FGL

def plawcut_cat(K, E0, G, Ec):
    def func(x):
        return K * np.exp(-G * np.log(x/E0) - (x - E0)/Ec)
    return func

def plaw_cat(K, E0, G):
    def func(x):
        return K * np.exp(-G * np.log(x/E0))
    return func

def logpar_cat(K, E0, G, beta):
    def func(x):
        return K * np.exp(-G * np.log(x/E0) - beta * np.log(x/E0)**2)
    return func

def plaw_supercut_cat(K, E0, G, Ec, Exp_ind):
    def func(x):
        return K * np.exp(-G * np.log(x/E0) - (x/Ec)**Exp_ind)
    return func


def catpars2pars(sp_type, K, E0, G, beta, Ec, Exp_ind):
    res = [K, G]
    if sp_type.startswith('PowerLaw'):
        pass
    elif sp_type.startswith('LogParabola'):
        res.append(beta)
    elif sp_type.startswith('PLExpCutoff'):
        res.append(Ec)
    elif sp_type.startswith('PLSuperExpCutoff'):
        res.append(Ec)
        res.append(Exp_ind)
    return res

def pars2catpars(sp_type, pars):
    K, G, beta, Ec, Exp_ind = -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
    if sp_type.startswith('PowerLaw'):
        K, G = pars
    elif sp_type.startswith('LogParabola'):
        K, G, beta = pars
    elif sp_type.startswith('PLExpCutoff'):
        K, G, Ec = pars
    elif sp_type.startswith('PLSuperExpCutoff'):
        K, G, Ec, Exp_ind = pars
    return K, G, beta, Ec, Exp_ind

def ps_E2spectrum(sp_type, pars, E0=1.):
    KK = pars[0]
    GG = pars[1]
    if sp_type.startswith('PowerLaw'):
        func = plaw_cat(KK, E0, GG)
    elif sp_type.startswith('LogParabola'):
        func = logpar_cat(KK, E0, GG, pars[2])
    elif sp_type.startswith('PLExpCutoff'):
        func = plawcut_cat(KK, E0, GG, pars[2])
    elif sp_type.startswith('LogPoly'):
        func = log_poly(pars)
    elif sp_type.startswith('PLSuperExpCutoff'):
        func = plaw_supercut_cat(KK, E0, GG, pars[2], pars[3])
    else:
        print 'unknown spectral type:', sp_type
        return None
    return lambda x: x**2 * func(x)



def get_psf_old(psf_filen, Es):
    '''
    read the psf table and define a function of ibin and theta
    '''
    hdu = pyfits.open(psf_filen)

    psf_Es = hdu['PSF'].data.field('Energy') / GeV2MeV
    psf_table = hdu['PSF'].data.field('Psf')
    thetas = np.deg2rad(hdu['THETA'].data.field('Theta'))
    psf_funcs = []
    
    for EE in Es:
        psf_bin = num.findIndex(psf_Es, EE, nearest=True)
        min = psf_table[psf_bin][0]
        max = psf_table[psf_bin][-1]
        func = interpolate.interp1d(thetas, psf_table[psf_bin],
                                    bounds_error=False, fill_value=(min, max))
        psf_funcs.append(func)
        if False:
            def fres(theta):
                if theta < thetas[-1]:
                    return func(theta)
                else:
                    return 0.

            fres_vec = np.frompyfunc(fres, 1, 1)
            psf_funcs.append(fres_vec)

    return psf_funcs, thetas


def get_dOm(nside):
    '''
    calculate the area of healpix pixels
    '''
    return 4. * np.pi / healpy.nside2npix(nside)

def get_ps_tmpl(v0, all_vec, psf_funcs, theta_thres, threshold, nside):
    '''
    calculate the templates in energy bins for a point source

    '''

    nbins = len(psf_funcs)


    thetas = get_thetas(v0, all_vec.T)
    nwinpix = len(thetas)
    
    tmpls = np.zeros((nbins, nwinpix))
    weights = np.zeros((nbins, nwinpix))
    for i in range(nbins):
        weights[i] = step(theta_thres[i] - thetas)
        tmpl = psf_funcs[i](thetas) * weights[i]
        norm = np.sum(tmpl) * get_dOm(nside)
        
        # make sure that the integral of the template is equal to threshold value, e.g. 99%
        tmpls[i] = tmpl / norm * threshold
        
    return tmpls, weights





if __name__ == '__main__':
    import auxil
    save_plots = 1
    dpar = 10
    dmer = 10
    ext = ['png']
    order = 7
    order_plot = 11
    nside = 2**order
    print 'ps_map'
    ps_umap = ps_umap_tmpl(nside)
    upixels = ps_umap.upixels()
    orders = upix.upix2order(upixels)
    print 'hpix'
    ord_umap = upix.umap(values=orders, hpix_dict=ps_umap.hpix_dict(), is_mean=True)
    umap_d = umap_udgrade(ord_umap, order_out=order_plot)
    hpix = upix.umap2hmap(umap_d, out_ordering='RING')
    print 'plotting'
    healpy.gnomview(map=hpix, rot=[0., 0.], xsize=2000, min=0, max=13,
                                reso=1.5, title='Orders for adaptive ps resolution')
    healpy.graticule(dpar=dpar, dmer=dmer)
    plot_fn = 'plots/tmp/adaptive_orders'
    auxil.save_figure(plot_fn, ext=ext, save_plots=save_plots)
    pyplot.show()
    exit()

    
    # get theta radii
    thetas = np.arange(0, np.pi, 0.0001)
    theta_thres = get_theta_thres(psf_funcs, thetas, thres=threshold)

    # calculate vectors corresponding to pixels
    vectors = np.array(healpy.pix2vec(nside, range(npix))).T

    # calculation of an index corresponding to b, l
    ind = healpy.ang2pix(nside, np.pi/2 - b, l)

    
