"""
Some utility functions to work with maps in healpix format.

Constants
---------

Functions
-----

Classes
-------

See also
--------

Notes
-----

References
----------

History
----------
Written: Dmitry Malyshev, September 2012, KIPAC, Stanford



"""

__all__ = ['step', 'pix2BL', 'BL2pix', 'func2fpix', 'data2fpix', 'heal', 'gaussBL',
           'pix_size']


import numpy as np
import math
import healpy
#import scipy
#from scipy import optimize
#import copy
from matplotlib import pyplot


#########################################################################
#                                                                       #
#           constants                                                   #
#                                                                       #
#########################################################################

epsilon = 1.e-15
log_eps = np.log(epsilon)

#########################################################################
#                                                                       #
#           general functions                                           #
#                                                                       #
#########################################################################

def step(x):
    '''
    Heaviside step function:
    step(x) = 0   for x < 0
              0.5 for x = 0. or 0 for x = 0
              1   for x > 0
    '''
    return (1 + np.sign(x)) / 2

def delta(x):
    return (1. + np.sign(x)) * (1. - np.sign(x))

def pix_size(nside):
    '''
    calculate an approximate pixel size
    '''
    npix = healpy.nside2npix(nside)
    return math.sqrt(4*math.pi/npix)


def get_root(nside, p, nside_root=None, nest=False):
    '''
    get the index of the larger pixel that p belongs to
    '''
    if nside_root is None or nside == nside_root:
        return p
    if nside < nside_root:
        print 'nside_root is larger than nside'
        return None
    
    ratio = nside / nside_root
    if not nest:
        p = healpy.ring2nest(nside, p)
    if isinstance(p, list):
        p = np.array(p)
    res = p / ratio**2
    if not nest:
        res = healpy.nest2ring(nside_root, res)
    return res
    

def get_daughters(nside, p, nside_new=None, nest=False):
    '''
    return indices of pixels that belong to p
    '''
    if nside_new is None:
        return p
    ratio = int(nside_new/nside)
    if not nest:
        p = healpy.ring2nest(nside, p)
    res = range(p * ratio**2, (p + 1) * ratio**2)
    if not nest:
        return healpy.nest2ring(nside_new, res)
    else:
        return res

def get_daughters_list(nside, p_list, nside_new=None, nest=False):
    return get_daughters_matrix(nside, p_list, nside_new=nside_new, nest=nest).flatten()

def get_daughters_matrix(nside, p, nside_new=None, nest=False):
    '''
    return indices of pixels that belong to p
    assumes nested format
    
    '''
    if nside_new is None:
        return p
    p = np.array(p)
    if not nest:
        p = healpy.ring2nest(nside, p)
    nd = int(nside_new/nside)**2
    res = nd * np.outer(np.array(p), np.ones(nd, dtype=int)) + np.arange(nd)
    if not nest:
        res = np.array(healpy.nest2ring(nside_new, res))
    return res


def map2map(array, transform):
    npix = len(array)
    nside = healpy.npix2nside(npix)
    inds = transform(nside, range(npix))
    return np.array([array[inds[i]] for i in range(npix)])
    
def nest_array2ring_array(array):
    return map2map(array, healpy.ring2nest)

def ring_array2nest_array(array):
    return map2map(array, healpy.nest2ring)


def get_all_neib(nside, p, depth = 0, nest = False):
    """
    get all nearest neighbors with depth = depth
    
    """
    if depth == 0:
        return np.array([p])
    if not nest:
        p = healpy.ring2nest(nside, p)
    S = [p]
    def BFS(p, depth):
        depth -= 1
        neib = healpy.get_all_neighbours(nside, p, nest = True)
        for p in neib:
            if p != -1 and p not in S: S.append(p)
        for p in neib:
            if p != -1 and depth != 0:
                BFS(p, depth)
    BFS(p, depth)
    if nest:
        return np.array(S)
    else:
        return healpy.nest2ring(nside, np.array(S))
    



#########################################################################
#                                                                       #
#           pixelation of data and functions                            #
#                                                                       #
#########################################################################


def pix2BL_old(nside, inds=None, nest=False):
    """
        transform an array of indices
        to arrays of Galactic latitudes and longitudes 
    INPUT:
        nside - integer: healpix parameter
        inds - array_like, shape (n,): list of indices
        nest - boolean optional:
            if True, then the healpix map is in nested format
            DEFAULT: False
            
    OUTPUT:
        Bs - array_like, shape (n,): Galactic latitudes (rad)
        Ls - array_like, shape (n,): Galactic longitudes (rad)
    HISTORY:
        Sept 13, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    """
    if inds is None:
        inds = range(healpy.nside2npix(nside))
    pix2angf = lambda ind: healpy.pix2ang(nside, int(ind), nest=nest)
    pix2ang_vec = np.frompyfunc(pix2angf, 1, 2)
    ths, Ls = pix2ang_vec(inds)
    Bs = 0.5 * np.pi - ths
    return np.array(Bs, dtype=float), np.array(Ls, dtype=float)

def pix2BL_new(nside, inds=None, nest=False):
    if inds == None:
        npix = healpy.nside2npix(nside)
        inds = range(npix)
    ths, Ls = healpy.pix2ang(nside, inds, nest=nest)
    Bs = 0.5 * np.pi - ths
    return np.array(Bs, dtype=float), np.array(Ls, dtype=float)

pix2BL = pix2BL_new


def BL2pix(nside, Bs, Ls, nest=False):
    """
        transform arrays of Galactic latitudes and longitudes 
        to an array of indices
    INPUT:
        nside - integer: healpix parameter
        Bs - array_like, shape (n,): Galactic latitudes (rad)
        Ls - array_like, shape (n,): Galactic longitudes (rad)
        nest - boolean optional:
            if True, then the healpix map is in nested format
            DEFAULT: False
            
    OUTPUT:
        inds - array_like, shape (n,): list of indices
        
    HISTORY:
        Sept 13, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    """
    
    ang2pixf = lambda theta, phi: healpy.ang2pix(nside, theta, phi, nest=nest)
    ang2pix_vec = np.frompyfunc(ang2pixf, 2, 1)

    thetas = 0.5 * np.pi - Bs
    inds = ang2pix_vec(thetas, Ls)

    return np.array(inds, dtype=int)


def func2fpix(nside, func, inds=None, nest=False):
    """
        pixelization of a continuous function
    INPUT:
        nside - integer: healpix parameter
        func - a function of galactic coords (B, L) in radians
        inds - list of pixel indices where the function should be computed
            if None, then inds = range(npix)
            DEFAULT: None
        nest - boolean optional:
            if True, then the healpix map is in nested format
            DEFAULT: False
    OUTPUT:
        fpix - array_like, shape (npix,): pixelized map of func
    HISTORY:
        Sept 13, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    """
    #npix = healpy.nside2npix(nside)

    Bs, Ls = pix2BL(nside, inds, nest=nest)
    func_vec = np.frompyfunc(func, 2, 1)
    
    #fpix = func_vec(Bs, Ls)

    return func_vec(Bs, Ls)


def data2fpix(nside, Bs, Ls, nest=False):
    """
        pixelization of an array of descrete events
    INPUT:
        nside - integer: healpix parameter
        Bs - array_like, shape (n,): Galactic latitudes of the events (rad)
        Ls - array_like, shape (n,): Galactic longitudes of the events (rad)
                where n - number of events
        nest - boolean optional:
            if True, then the healpix map is in nested format
            DEFAULT: False
    OUTPUT:
        fpix - array_like, shape (npix,): total counts of events in pixels
    HISTORY:
        Sept 13, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    """
    npix = healpy.nside2npix(nside)
    
    fpix = np.zeros(npix)
    inds = BL2pix(nside, Bs, Ls, nest=nest)
    
    for ind in inds:
       fpix[ind] += 1.

    return fpix


#########################################################################
#                                                                       #
#           Lat - Lon profile plots                                     #
#                                                                       #
#########################################################################



# create a dictionary of Healpix indices in L and B bins
def lb_profiles_hinds_dict(nside, Bbins, Lbins, mask=None):
    '''
        create a dictionary of Healpix indices in L and B bins
        UNPUT:
        Bbins - list, B bins in deg
        Lbins - list, L bins in deg
        mask - healpix mask map - if mask[i] == 0, then the pixel "i" is omitted
        OUTPUT:
        a dictionary {(lon_index, lat_index): healpix_inds}, where healpix_inds are the indices of healpix pixels inside the lat-lon bin,
        example:
        ind_dict = make_inds(nside, Bbins, Lbins, mask=mask)
        inds = ind_dict[(0, 0)] - gives healpix indices in the first lat-lon bin
        
        '''
    npix = healpy.nside2npix(nside)
    bbins = np.deg2rad(Bbins)
    lbins = np.deg2rad(Lbins)
    
    inds = {}
    for i in range(len(bbins)-1):
        for j in range(len(lbins)-1):
            inds[(i, j)] = []
    
    for n in range(npix):
        if mask is None or mask[n] > 0:
            theta, ll = healpy.pix2ang(nside, n)
            bb = 0.5 * np.pi - theta
            if ll > np.pi:
                ll -= 2 * np.pi
            if bb > bbins[0] and bb < bbins[-1] and ll > lbins[0] and ll < lbins[-1]:
                il = findIndex(lbins, ll) - 1
                ib = findIndex(bbins, bb) - 1
                inds[(ib, il)].append(n)
    return inds


def fpix_lb_profiles(fpix, inds_dict, nB, nL, std=0):
    '''
        put fpix in lat-lon bins
        INPUT:
        fpix - healpix map or an array with dimensions (nmaps, npix)
        inds_dict - dictionary of healpix indices in lat-lon bins, {(lon_index, lat_index): healpix_inds}
        nB - number of lat bins
        nL - number of lon bins
        std - if 0, the value is found by averaging over pixels in the bin (intensity average)
        if 1, then the value is found as sqrt(sum f_i**2)) / n, which is the std of the average
        OUTPUT:
        nB by nL array of average values in lat-lon bins
        
        '''
    if fpix.ndim == 1:
        res = np.zeros((nB, nL))
    else:
        res = np.zeros((len(fpix), nB, nL))
    if std:
        power = 2
    else:
        power = 1
    for i in range(nB):
        for j in range(nL):
            nn = len(inds_dict[(i, j)])
            for n in inds_dict[(i, j)]:
                if fpix.ndim == 1:
                    res[i, j] += (fpix[n] / nn)**power
                else:
                    res[:, i, j] += (fpix[:, n] / nn)**power
    if std:
        res = np.sqrt(res)
    return res


#########################################################################
#                                                                       #
#           working with masked data                                    #
#                                                                       #
#########################################################################


def heal(fpix, mask, nest=False, outsteps=False):
    """
        fill masked pixels with an average over nearest neighbour pixels
        (up to 8 pixels on sides and diagonals)
    INPUT:
        fpix - array_like, shape (npix,): input map
        mask - array_like, shape (npix,): mask, 0 = masked, 1 = unmasked
        nest - boolean optional:
            if True, then the healpix map is in nested format
            DEFAULT: False
        outsteps - boolean optional:
            if True, then output the steps of filling in the mask
            DEFAULT: False

    OUTPUT:
        fpix - array_like, shape (npix,): the map with filled in masked pixels
    HISTORY:
        Sept 13, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    """
    if not isinstance(mask, np.ndarray) and mask == 1.:
        return fpix

    fpix_new = fpix * mask
    fpix_new = fpix_new.T
    #mask_new = np.ceil(np.array(step(mask), dtype=float))
    mask_new = 1. * mask
    npix = fpix.shape[-1]
    nside = healpy.npix2nside(npix)

    run = 0
    nzeros = npix - sum(mask_new)
    while nzeros > 0.:
        run += 1
        if outsteps:
            print 'number of zeros in the mask: ', nzeros
            print 'fill step ', run
            
        checked = []
        for i in range(npix):
            if mask_new[i] == 0.:    
                neib = healpy.get_all_neighbours(nside, i, nest=False)
                n = 0.
                for j in neib:
                    if mask_new[j] > 0:
                        n += 1.
                        fpix_new[i] += fpix_new[j]
                if n > 0:
                    fpix_new[i] /= n
                    checked.append(i)
        for i in checked:
            mask_new[i] = 1
            nzeros -= 1
    return np.array(fpix_new.T, dtype=np.float64)


def ps2maskpix(nside, bs, ls, dist, nest=False):
    """
    mask all the pixels that either contain (bs[i], ls[i])
    or the distance from the point to the center
    of the pixel is less than dist
    INPUT:
        nside - healpix nside parameter
        bs - array of Gal latitudes (rad)
        ls - array of Gal longitudes (rad)
        dist - array of distances (rad)
        nest - output map is in nested format
            DEFAULT: False
    OUTPUT:
        fpix - array_like, shape (npix,): mask map (0 if masked)
    HISTORY:
        Dec 26, 2014 - Written - D.Malyshev (KIPAC, Stanford)
        
    """
    nestin = True
    npix = healpy.nside2npix(nside)
    mask = np.ones(npix)
    pixel_size = pix_size(nside)
    if not isinstance(dist, np.ndarray):
        dists = np.ones(len(bs)) * dist
    else:
        dists = dist

    depth_min = min(dists / pixel_size)
    if depth_min < 2.:

        vp = np.array(BL2xyz(bs, ls))
        vec2pix = lambda x, y, z: healpy.vec2pix(nside, x, y, z, nest=nestin)
        vec2pix_vec = np.frompyfunc(vec2pix, 3, 1)
        pixs = np.array(vec2pix_vec(vp[0], vp[1], vp[2]), dtype=int)
        mask[pixs] = 0.

        for i in range(len(bs)):
            if i % 100 == 0 and i > 0:
                print i
            
            depth = np.ceil(dists[i] / pixel_size)
            neib = get_all_neib(nside, pixs[i], depth=depth, nest=nestin)
            for pn in neib:
                vpn = healpy.pix2vec(nside, pn, nest=nestin)
                if np.arccos(np.dot(vp[:,i], vpn)) < dists[i]:
                    mask[pn] = 0.
        if nest:
            return mask
        else:
            return nest_array2ring_array(mask)


    else:
        inds = range(npix)
        vecs = np.array(healpy.pix2vec(nside, inds, nest=False)).T
        for i in range(len(bs)):
            if i % 100 == 0 and i > 0:
                print i
            BL0 = (bs[i], ls[i])
            mask *= 1. - mask_circle(nside, dists[i], BL0, inds=inds,
                                nest=nest, vecs=vecs)
            
        return mask


#########################################################################
#                                                                       #
#           some functions on the sphere                                #
#                                                                       #
#########################################################################

_min_values = lambda a, b: a * step(b - a) + b * step(a - b)


def _Blimit(B):
    '''
    constrain the values of B to interval (-pi/2, pi/2)
    '''
    return np.sign(B) * _min_values(np.abs(B), np.pi/2)
    

def _Llimit(L):
    '''
    constrain the values of L to interval (-pi, pi)
    '''
    res = L % (2 * np.pi)
    return L * step(np.pi - L) + (L - 2 * np.pi) * step(L - np.pi)



def gaussBL(BL0, BLsigma, Bs, Ls):
    '''
        bivariate gaussian on the sphere
    INPUT:
        BL0 = [B0, L0] - array_like: center of the gaussian (rad)
            limits: B0 in (-90, 90), L0 in (-180, 180)
        sigmas = [sigma_B, sigma_L] - array_like: standard deviations (rad)
        Bs - array_like, shape (npix,): values of Bs in pixels (rad)
            limit: Bs[i] in (-pi/2, pi/2)
        Ls - array_like, shape (npix,): values of Ls in pixels (rad)
            limit: Ls[i] in (0, 2 pi)
    OUTPUT:
        array_like, shape (npix,): the map of the bivariate gaussian
    HISTORY:
        Sept 13, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    '''
    cosB = np.cos(Bs)
    B0, L0 = BL0
    sB, sL = BLsigma

    B0 = _Blimit(B0)
    L0 = _Llimit(L0)


    # check if offset by (2 pi) in L has a smaller distance
    deltaL = _min_values(np.abs(Ls - L0),  np.abs(Ls - 2 * np.pi - L0))
    
    r2 = (Bs - B0)**2 / (2 * sB**2) + deltaL**2 * cosB**2 / (2 * sL**2)
    
    # cutoff at numerical precision
    arg = _min_values(-log_eps, r2)

    pre_exp = cosB / (2 * np.pi * sB * sL)

    return pre_exp * np.exp(-arg)


def smoothing_neff(nside, sigma, n_rnd=10):
    '''
        Monte Carlo calculation of effective number of pixels covered
        by a Gaussian kernel
    INPUT:
        nside - integer: healpix parameter
        sigma - float: gaussian kernel sigma (rad)
        n_rnd - integer: number of trials in MC calculation
    OUTPUT:
        n_eff - float: effective number of pixels
            n_eff = (std_0 / std_smooth)**2
            std_0 is the st. dev. of a gaussian field
            std_smooth is the st. dev. of the smoothed gaussian field
    HISTORY:
        Sept 14, 2012 - Written - D.Malyshev (KIPAC, Stanford)
    '''
    npix = healpy.nside2npix(nside)
    std_sm = np.zeros(n_rnd)
    for i in range(n_rnd):
        fpix_rnd = np.random.normal(size=npix)
        try:
            fpix_sm = healpy.smoothing(fpix_rnd, sigma=sigma, regression=False)
        except TypeError:
            fpix_sm = healpy.smoothing(fpix_rnd, sigma=sigma)
        std_sm[i] = np.std(fpix_sm)

    return 1/np.mean(std_sm)**2

def uind2lk(uind):
    ll = np.array(np.floor(np.sqrt(uind)), dtype=int)
    kk = uind - ll**2
    return ll, kk

def lk2uind(ll, kk):
    return ll**2 + kk

def lm2hind(lmax, ll, mm):
    '''
    hind - healpix ordering of spherical harmonics: m = 0, l = 0... lmax, m = 1, l = 1...lmax etc.

    '''
    return mm * (lmax + 1) - (mm - 1) * mm / 2 + ll - mm

def hind2lm(hind, lmax):
    if np.max(hind) > lmax * (lmax + 3) / 2:
        raise Exception('Index is too large compared to lmax')
    mp = np.array(np.floor(-0.5 + np.sqrt(0.25 + 2. * hind)), dtype=int)
    p = hind - mp * (mp + 1) / 2
    m = lmax - mp[::-1]
    l = lmax - p[::-1]
    return l, m

def k2m(ll, kk):
    return kk % (ll + 1) + kk / (ll + 1)

def lk2hind(lmax, ll, kk):
    #mm = kk - ll * step(kk - ll)
    return lm2hind(lmax, ll, k2m(ll, kk))


def uind2hind(lmax, uind):
    ll, kk = uind2lk(uind)
    return lk2hind(lmax, ll, kk)
    

def sph_basis(nside, lmax=None, mmax=None, lks=None, output_inds=False):
    '''
        Basis of spherical harmonics functions
    INPUT:
        nside - integer: healpix parameter
        lmax - integer: maximal degree of spherical functions
        mmax - integer: maximal angular number of spherical harmonics
    OUTPUT:
        Ylm maps with l <= lmax, m <= l
            ordering of the maps: l = 0, 1, ..., lmax, k = 0, ..., mmax, l, ..., l + mmax
        The normalization is such that the integral of f_i^2
        over the sphere is 1.
    HISTORY:
        Jan 30, 2015 - Written - D.Malyshev (KIPAC, Stanford)
    '''
    # output maps template
    npix = healpy.nside2npix(nside)
    if lks is None:
        if lmax is None:
            print 'either lks or lmax should be defined'
            exit()
        nsph = (lmax + 1)**2
        lks = np.zeros((nsph, 2), dtype=int)
        index = 0
        for i in range(nsph):
            hind = uind2hind(lmax, i)
            ll, kk = uind2lk(i)
            mm = k2m(ll, kk)
            if mmax is None or mm <= mmax:
                lks[index] = ll, kk
                index += 1
        lks = lks[:index]
    else:
        if lks.ndim == 1:
            lks = np.array([lks])
        lmax = max(lks[:, 0])

    nsph = len(lks)
    res = np.zeros((nsph, npix))
    uinds = lk2uind(lks[:, 0], lks[:, 1])

    nalm_max = (lmax + 1) * (lmax + 2) / 2
    #alms = np.eye(nalm_max, dtype=complex)

    for i in range(nsph):
        ll, kk = lks[i]
        hind = lk2hind(lmax, ll, kk)
        factor = 1.
        if kk > 0:
            factor /= np.sqrt(2.)
        if kk > ll:
            factor *= 1J
        alm = np.zeros(nalm_max, dtype=complex)
        alm[hind] = 1.
        res[i] = healpy.alm2map(factor * alm, nside)

    if output_inds:
        return res, uinds
    else:
        return res


#########################################################################
#                                                                       #
#           general functions                                           #
#                                                                       #
#########################################################################


def BL2xyz(B, L):
    """
        transform between Galactic coordinates and xyz-coordinates
    INPUT:
       B, L - angles in galactic coordinates in radians
    OUTPUT:
       x, y, z - unit vector in the direction of (B, L)
    """
    x = np.cos(B) * np.cos(L)
    y = np.cos(B) * np.sin(L)
    z = np.sin(B)
    return x, y, z

def ang2vec(theta, phi):
    return np.array(BL2xyz(0.5 * np.pi - theta, phi))

def BL2xyz_grid(bs, ls):
    res = np.zeros((len(bs), len(ls), 3))
    for i in range(len(bs)):
        res[i] = np.array([np.cos(bs[i]) * np.cos(ls),
                           np.cos(bs[i]) * np.sin(ls),
                           np.sin(bs[i]) * np.ones(len(ls))]).T
    return res

def xyz2BL(x, y, z):
    """
       transformation from (x, y, z) to galactic (b, l) in radians
    INPUT:
       x, y, z - a vector
    OUTPUT:
       b, l - angles in rads such that:
           x = r cos(b)cos(l)
           y = r cos(b)sin(l)
           z = r sin(b)
    """
    r = np.sqrt(x*x + y*y + z*z)
    b = np.arcsin(z / r)
    
    rho = np.sqrt(x*x + y*y)
    if    rho == 0.: l = 0.
    elif  x == 0.:   l = np.pi / 2 * np.sign(y)
    elif  x > 0. :   l = np.arcsin(y / rho)
    else:            l = np.pi - np.arcsin(y / rho)
    if l < 0: l += 2 * np.pi
    
    return b, l


xyz2BL_vec = np.frompyfunc(xyz2BL, 3, 2)

def vec2ang(v):
    B, phi = np.array(xyz2BL_vec(v[0], v[1], v[2]), dtype=float)
    theta = np.pi / 2 - B
    return theta, phi


def pix2vec(nside, inds=None, nest=False):
    if inds == None:
        inds = range(healpy.nside2npix(nside))
    return np.array(healpy.pix2vec(nside, inds, nest=nest))

def vec2theta(vectors, B0, L0):
    """
    calculate angles between the direction [B0, L0]
    and vectors
    INPUT:
        B0, L0 (rad)
        vectors - array with shape (3, nvec): vectors
    OUTPUT:
        theta - array with shape (nvec): angles
    
    """
    v0 = np.array(BL2xyz(B0, L0))
    # normalize
    costh = np.dot(v0, vectors)/np.linalg.norm(v0)
    costh /= np.sqrt(np.sum(vectors**2, axis=0))
    # make sure that costh is within +- 1
    costh = np.minimum(costh, np.ones_like(costh))
    costh = np.maximum(costh, -np.ones_like(costh))
    return np.arccos(costh)   

def pix2theta(nside, B0, L0, inds=None, nest=False):
    return vec2theta(pix2vec(nside, inds=inds, nest=nest), B0, L0)

def findIndex(array, number, nearest=False):
    """
        return the min index i such that number < array[i]
        return len(array) if array[-1] < number
        if nearest = True, then return the index of the closet
        array entry to the number
    """
    if array[0] > number:
        return 0
    elif array[-1] < number:
        if nearest:
            return len(array) - 1
        else:
            return len(array)
    else:
        imin = 0
        imax = len(array)
        while imax > imin + 1:
            imed = (imax + imin)/2
            if array[imed] < number:
                imin = imed
            else:
                imax = imed
        
        if nearest and number < (array[imax] + array[imax - 1])/2:
            return imax - 1
        else:
            return imax


#########################################################################
#                                                                       #
#           masking functions                                           #
#                                                                       #
#########################################################################


def mask_circle(nside, phi_max, BL0, nest=False, inds=None, vecs=None):
    '''
    mask = 0 outside phi_max
    BL0 = (B0, L0) - coordinates of the center
    '''
    
    rs = 2*np.sin(phi_max/2)
    rsInv = 1/rs

    B0, L0 = BL0
    n0 = np.array(BL2xyz(B0, L0))
    if inds is None:
        npix = healpy.nside2npix(nside)
        inds = range(npix)
    if vecs is not None:
        n1 = vecs
    else:
        n1 = np.array(healpy.pix2vec(nside, inds, nest=nest)).T

    return np.floor(step(1 - np.sum((rsInv*(n1 - n0))**2, axis=1)))


def mask_tanh(nside, BL0=np.deg2rad([0.,0.]), theta=np.deg2rad(10.), dtheta=np.deg2rad(3.),
              theta_max=None, nest=False):
    """
    create a circlular mask (window function) with smooth boundaries
    INPUT:
        nside - healpix parameter
        BL0 - [B0, L0] - the center of the mask (radians), default: [0., 0.]
        theta - radius of the mask (radians), default: np.deg2rad(10.)
        dtheta - half width of the boundary (radians), default: np.deg2rad(3.)
        theta_max - max radius behind which the mask is set to zero,
                    default: None (in this case theta_mas = theta + 3 * dtheta
        nest - healpix pixelation type, default: False
    USAGE:
        import numpy as np
        import healpylib
        hmap = healpylib.mask_tanh(nside, BL0=np.deg2rad([0., 0.]), theta=np.deg2rad(10.),
                                   dtheta=np.deg2rad(3.), theta_max=None, nest=False)
    OUTPUT:
        healpix map
        
    """
    if theta_max is None:
        theta_max = theta + 3 * dtheta
    npix = healpy.nside2npix(nside)

    # get the pixels
    mask = mask_circle(nside, theta_max, BL0, nest=nest)
    inds = [i for i in range(npix) if mask[i] > 0]

    v = pix2vec(nside, inds, nest=nest)
    B, L = BL0
    v0 =  BL2xyz(B, L)
    thetas = np.arccos(np.dot(v0, v))
    window = (1. - np.tanh((thetas - theta) / dtheta)) / 2.
    res = np.zeros(npix)
    mask[inds] = window
    return mask



    
def mask_lat_stripe(nside, bmin=-10, bmax=10, lmin=None, lmax=None, rad=False,
                    nest=False):
    '''
    if bmin < B < bmax, then mask = 0.
    otherwise mask = 1.
    '''

    if not rad:
        bmin = np.deg2rad(bmin)
        bmax = np.deg2rad(bmax)
        if lmin is not None and lmax is not None:
            lmin = np.deg2rad(lmin)
            lmax = np.deg2rad(lmax)
            
    npix = healpy.nside2npix(nside)

    thetas, phis = np.array(healpy.pix2ang(nside, range(npix), nest=nest))
    
    Bs = np.pi / 2. - thetas
    Ls = phis
    
    maskB = np.floor(step(bmin - Bs) + step(Bs - bmax))
    if lmin is not None and lmax is not None:
        lmin = lmin % (2. * np.pi)
        lmax = lmax % (2. * np.pi)    
        if lmin < lmax:
            maskL = 1. - step(lmax - Ls) * step(Ls - lmin)
        else:
            maskL = step(lmin - Ls) * step(Ls - lmax)
        #maskL = np.floor(step(lmin - Ls) + step(Ls - lmax))
    else:
        maskL = 0.

    return np.floor(step(maskB + maskL))

def jet_template(nside, length, width, angle):
    '''
    make a stripe template starting at GC
    INPUT:
        nside
        length (deg)
        width (deg)
        angle (deg) - angle to positive longitudes (counterclockwise)
    
    '''
    
    R, w, phi = np.deg2rad([length, width, angle])
    npix = healpy.nside2npix(nside)
    jet_tmpl = np.zeros(npix)

    # define a grid of vectors in pisitive lon direction
    pix = 1. / nside
    step = pix / 2.
    nl = np.floor(R / step)
    nb = np.floor(w / step)
    ls = np.arange(0., R + epsilon, R / nl)
    bs = np.arange(-w/2., w/2 + epsilon, w/nb)
    vectors = BL2xyz_grid(bs, ls)

    # define the rotation matrix (multiplication from the right)
    sn = np.sin(phi)
    cs = np.cos(phi)
    rotM = np.array([[1., 0., 0.],
                     [0., cs, -sn],
                     [0., sn, cs]])

    vectors = np.dot(vectors, rotM)
    vectors = vectors.T

    
    # find pixels corresponding to the vectors

    func = lambda x, y, z: healpy.vec2pix(nside, x, y, z)
    func_vec = np.frompyfunc(func, 3, 1)
    
    pixels = func_vec(vectors[0], vectors[1], vectors[2])
    pixels = pixels.flatten()
    pixels = list(pixels)

    jet_tmpl[pixels] = 1.

    return jet_tmpl
    
    

def mask_ellipse(nside, rsInv, n0, mask=None):
    npix = healpy.nside2npix(nside)
    if mask is None:
        rmask = np.ones(npix)
    else:
        rmask = 1. * mask
    for i in range(npix):
        n1 = healpy.pix2vec(nside, i, nest = False)
        if np.sum((rsInv*(n1 - n0))**2) > 1.:
            rmask[i] = 0
    return rmask


if __name__ == '__main__':
    import time
    #from matplotlib import pyplot

    lmax = 5
    hind = np.arange((lmax + 2) * (lmax + 1) / 2)
    l, m = hind2lm(hind, lmax)

    nside = 64
    nside_new = 128
    npix = healpy.nside2npix(nside)
    t0 = time.time()
    res1 = get_daughters_list(nside, range(npix), nside_new=nside_new, nest=False, new=True)
    print 'mat time: %.3g s' % (time.time() - t0)
    t0 = time.time()
    res2 = get_daughters_list(nside, range(npix), nside_new=nside_new, nest=False, new=False)
    #res2 = get_daughters_matrix(nside, range(npix), nside_new=nside_new, nest=False)
    print 'list time: %.3g s' % (time.time() - t0)
    print np.sum((res1 - res2)**2)
