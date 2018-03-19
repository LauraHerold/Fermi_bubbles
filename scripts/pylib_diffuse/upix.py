"""
Universal pixels class definition. 


Constants
---------

-----

Classes
-------
umap

See also
--------

Notes
-----

References
----------

History
----------
Written: Dmitry Malyshev, December 2014, KIPAC, Stanford


"""

__all__ = []

import numpy as np
import healpy
import pyfits
import os
import time

from matplotlib import pyplot
#from matplotlib import rc
import healpy


import healpylib as hlib

step = lambda x: (1. + np.sign(x)) / 2.
delta  = lambda x: (1 + np.sign(x)) * (1 + np.sign(-x))

def upix2order(upixels):
    res = np.floor(0.5 * np.log2(1 + (1. * upixels) / 4.))
    return np.array(res, dtype=int)

def upix2order_set(upixels):
    orders = upix2order(upixels)
    order_set = list(set(orders))
    order_set.sort()
    return order_set

def upix2hpix_dict(upixels):
    '''
    transform universal pixels indices to healpix pixels
    INPUT:
        upixels - array of universal pixels
    OUTPUT:
        hpix_dict - dictionary (order: hpixels) of healpix pixels

    '''
    orders = upix2order(upixels)
    order_set = list(set(orders))
    order_set.sort()
    order_lengths = [np.sum(delta(orders - order)) for order in order_set]
    hpix = upixels - 4 * (4**orders - 1)
    res = {}
    start = 0
    for i, order in enumerate(order_set):
        finish = start + order_lengths[i]
        res[order] = np.array(hpix[start:finish], dtype=int)
        start = finish
    return res

def hpix2upix(order, hpix=None):
    '''
    transform healpix pixels indices to universal pixels
    INPUT:
        hpix - healpix pixels
    OUTPUT:
        upixels - array of universal pixels
        
    '''
    if hpix is None:
        nside = 2**order
        npix = healpy.nside2npix(nside)
        hpix = np.arange(npix, dtype=int)
    res = 4 * (4**order - 1) + hpix
    return np.array(res, dtype=int)

def npixels(hpix_dict):
    if hpix_dict is None:
        return None
    else:
        return sum([len(hpix_dict[k]) for k in hpix_dict.keys()])

def hpix_dict2upix(hpix_dict):
    length = npixels(hpix_dict)
    res = np.zeros(length, dtype=int)
    orders = hpix_dict.keys()
    orders.sort()
    start = 0
    for k in orders:
        finish = start + len(hpix_dict[k])
        res[start:finish] = hpix2upix(k, hpix_dict[k])
        start = finish
    return res

def umap2fits(umap, fn=None,
              unit=None, kdict=None, comment=None, Es=None, Eunit='MeV'):

    # create primary HDU
    hdulist = [pyfits.PrimaryHDU()]

    # create data HDU
    clms = []
    upixels = umap.upixels()
    values = umap.values()

    if upixels is not None:
        clm = pyfits.Column(name='Upix', array=upixels, format='J')
        clms.append(clm)

    if values is not None:
        if values.ndim > 2 or values.shape[0] != upixels.shape[0]:
            print "dimensions of umap.values should be (npix, nval)"
            return None
        nval = values.shape[1]
        fmt = '%iE' % nval
        clm = pyfits.Column(name='Spectra', array=values,
                            format=fmt, unit=unit)
        clms.append(clm)


    dhdu = pyfits.new_table(clms)
    dhdu.name = 'SKYMAP'

    if umap.is_mean:
        dhdu.header.update('is_mean', 1)
    else:
        dhdu.header.update('is_mean', 0)

    if kdict is not None:
        for key in kdict.keys():
            dhdu.header.update(key, kdict[key])
    if comment is not None:
        dhdu.header.add_comment(comment)

    hdulist.append(dhdu)
    
    # create energy HDU
    if Es is not None:
        clm = pyfits.Column(name=Eunit, array=Es, format='E', unit=Eunit)
        ehdu = pyfits.new_table([clm])
        ehdu.name = 'ENERGIES'
        hdulist.append(ehdu)
    
    hdulist = pyfits.HDUList(hdulist)

    if fn is not None:
        print 'save Umap to file:'
        print fn
        
        if os.path.isfile(fn):
            os.remove(fn)
        hdulist.writeto(fn)

        return None
    else:
        return hdulist

def fits2umap(fn=None, hdu=None):
    if hdu is None and fn is not None:
        hdu = pyfits.open(fn)

    if 'Spectra' in hdu['SKYMAP'].header.values():
        values = np.array(hdu['SKYMAP'].data.field('Spectra'), dtype=float)
    else:
        values = None
    if 'Upix' in hdu['SKYMAP'].header.values():
        upixels = hdu['SKYMAP'].data.field('Upix')
    else:
        upixels = None

    is_mean = 1
    if 'IS_MEAN' in hdu['SKYMAP'].header.keys():
        is_mean = hdu['SKYMAP'].header['IS_MEAN']

    return umap(values=values, upixels=upixels, is_mean=is_mean)

def hmap2umap(hpixmap, upixels, inmap_nest=False, is_mean=True):
    '''
    create a umap in a set of upixels from a healpix map

    '''
    hpix_dict = upix2hpix_dict(upixels)
    npix = hpixmap.shape[0]
    #if hpixmap.ndim == 2:
    #    nval = hpixmap.shape[1]
    
    nside = healpy.npix2nside(npix)
    inmap_order = int(np.log2(nside))
    
    orders = hpix_dict.keys()
    orders.sort()
    #if not inmap_nest:
    #    hpixmap = hlib.ring_array2nest_array(hpixmap)
    #if hpixmap.ndim == 1:
    #    res = np.zeros(len(upixels))
    #else:
    #    res = np.zeros((len(upixels), nval))
    dims = [len(upixels)]
    dims.extend(hpixmap.shape[1:])
    res = np.zeros(dims)
    start = 0
    #t0 = time.time()
    for k in orders:
        nside_new = 2**k
        inds = np.array(hpix_dict[k])
        finish = start + len(inds)

        if k < inmap_order:
            if not inmap_nest:
                inds = healpy.nest2ring(nside_new, inds)
            inds_daugh = hlib.get_daughters_matrix(nside_new, inds, nside_new=nside,
                                              nest=inmap_nest)
            
            nd = inds_daugh.shape[1]
            for i in range(nd):
                res[start:finish] += hpixmap[inds_daugh[:, i]] / nd
            
        else:
            inds = hlib.get_root(nside_new, hpix_dict[k],
                                 nside_root=nside, nest=True)
            if not inmap_nest:
                inds = healpy.nest2ring(nside, inds)
            res[start:finish] = 1. * hpixmap[inds]

        if not is_mean:
            res[start:finish] *= (1. * nside / nside_new)**2
        start = finish

    #print (time.time() - t0)        
    return umap(values=res, upixels=upixels, is_mean=is_mean)

hpixmap2umap = hmap2umap

def mask_umap(umap_in, mask, nest=False, strict_masking=False):
    upixels = umap_in.upixels()
    mask_umap = hmap2umap(mask, upixels, inmap_nest=nest, is_mean=True)
    mask_values = mask_umap.values()
    if strict_masking:
        nomask_inds = [i for i in range(len(upixels)) if mask_values[i] == 1]
    else:
        nomask_inds = [i for i in range(len(upixels)) if mask_values[i] > 0]
    upixels_new = upixels[nomask_inds]
    if umap_in.values() is not None:
        values_new = umap_in.values()[nomask_inds]
    else:
        values_new = None
    return umap(values=values_new, upixels=upixels_new,
                is_mean=umap_in.is_mean, default_value=umap_in.default_value)

def make_dict(inds, values):
    """
    create a dictionary: {i:values[i] for i in range(len(inds))}
    
    """
    return {int(inds[i]): values[i] for i in range(len(inds))}

def merge_dicts(dict1, dict2):
    """
    merge two dictionaries into one
    
    """
    if dict1 is None:
        return dict2
    if dict2 is None:
        return dict1
    res = {}
    keys = set.union(set(dict1.keys()), set(dict2.keys()))
    for key in keys:
        res[key] = dict1.get(key, 0.) + dict2.get(key, 0.)
    return res

def hmap_upgrade(order0, map0_dict, order1, map1_dict=None, power=0):
    """
    upgrade order0 map to order1 and merge with an order1 map
    NOTATIONS:
        hind - healpix index
        order = log2(nside) - order of the healpix map
    INPUT:
        order1 - order of the higher resolution map
        map1_dict - dictionary: values in healpix indices at order1 {hind: value}
        order0 - order of the lower resolution map
        map0_dict - dictionary: values in healpix indices at order0 {hind: value}
            default: None
        power - healpix power parameter in ud_grade:
            default: 0
            power = 0 - density
            power = -2 - counts
    OUTPUT:
        map1_dict_merged - dictionary {hind, value} healpix map at order1:
            combination of upgraded map0 and map1 (sum for power = -2, average for power = 0)
    
    """
    hpixs0 = map0_dict.keys()
    hpixs0.sort()
    daughters = hlib.get_daughters_list(2**order0, hpixs0, nside_new=2**order1, nest=True)
    dims = [len(daughters)]
    key = map0_dict.keys()[0]
    if isinstance(map0_dict[key], np.ndarray):
        dims.extend(map0_dict[key].shape)

    map01 = np.zeros(dims)
    nsub = 4**(order1 - order0)
    for i0, pix0 in enumerate(hpixs0):
        map01[i0 * nsub:(i0 + 1) * nsub] += map0_dict[pix0]
    map01 *= 2.**((order1 - order0) * power)


    #print 'rescale', 2.**((order0 - order1) * (2 + power))
    #exit()
    map01_dict = make_dict(daughters, map01)
    res = merge_dicts(map01_dict, map1_dict)

    return res


def hmap_degrade(order1, map1_dict, order0, map0_dict=None, power=0):
    """
    degrade order1 map to order0 and merge with an order0 map
    NOTATIONS:
        hind - healpix index
        order = log2(nside) - order of the healpix map
    INPUT:
        order1 - order of the higher resolution map
        map1_dict - dictionary: values in healpix indices at order1 {hind: value}
        order0 - order of the lower resolution map
        map0_dict - dictionary: values in healpix indices at order0 {hind: value}
            default: None
        power - healpix power parameter in ud_grade:
            default: 0
            power = 0 - density
            power = -2 - counts
    OUTPUT:
        map0_dict_merged - dictionary {hind, value} healpix map at order0:
            combination of degraded map1 and map0 (sum for power = -2, average for power = 0)
    
    """
    hpixs1 = map1_dict.keys()
    hpixs1.sort()
    root1 = hlib.get_root(2**order1, hpixs1, nside_root=2**order0, nest=True)
    root1_set = list(set(root1))
    root1_set.sort()
    dims = [len(root1_set)]
    key = map1_dict.keys()[0]
    if isinstance(map1_dict[key], np.ndarray):
        dims.extend(map1_dict[key].shape)
    map10 = np.zeros(dims)
    for i0, pix0 in enumerate(root1_set):
        dht = hlib.get_daughters(2**order0, pix0, 2**order1, nest=True)
        for pix1 in dht:
            map10[i0] += map1_dict[pix1]

    map10 *= 2.**((order0 - order1) * (2 + power))
    #print 'rescale', 2.**((order0 - order1) * (2 + power))
    #exit()
    map10_dict = make_dict(root1_set, map10)
    return merge_dicts(map10_dict, map0_dict)

def umap_udgrade(umap_in, order_out='min', hmap_out=False, out_time=False):
    '''
    if umap_in.values() is None, then assume values = 1
    
    '''
    if out_time:
        t0 = time.time()
    orders = umap_in.orders()
    power = umap_in.hpower()
    if umap_in.values() is None:
        upixels = umap_in.upixels()
        values = np.ones_like(upixels)
        umap_in_new = umap(values=values, upixels=upixels)
    else:
        umap_in_new = umap_in
    hmap_dicts = umap2hmap_dicts(umap_in_new)
    if order_out == 'min':
        order_out = orders[0]
    elif order_out == 'max':
        order_out = orders[-1]
    if order_out not in orders:
        hmap_dicts[order_out] = {}
        orders.append(order_out)
        orders.sort()
    kdiv = orders.index(order_out)
    


    # degrade the maps
    if kdiv < len(orders) - 1:
        hmap_dict_d = hmap_dicts[orders[-1]]
        for k in range(len(orders) - 2, kdiv - 1, -1):
            hmap_dict_d = hmap_degrade(orders[k + 1], hmap_dict_d, orders[k],
                                     map0_dict=hmap_dicts[orders[k]], power=power)
        hmap_dicts[orders[kdiv]] = hmap_dict_d


    
    # upgrade the maps
    if kdiv > 0:
        hmap_dict_u = hmap_dicts[orders[0]]
        for k in range(1, kdiv + 1):
            hmap_dict_u = hmap_upgrade(orders[k - 1], hmap_dict_u, orders[k],
                                     map1_dict=hmap_dicts[orders[k]], power=power)
        hmap_dicts[orders[kdiv]] = hmap_dict_u

    if out_time:
        print 'hmap_udgrade time: %.3g sec' % (time.time() - t0)
        t0 = time.time()
    hmap_dict = hmap_dicts[order_out]
    hpix = hmap_dict.keys()
    hpix.sort()
    hpix_dict = {order_out: hpix}
    t0 = time.time()
    dims = [len(hpix)]
    if isinstance(hmap_dict[hpix[0]], np.ndarray):
        dims.extend(hmap_dict[hpix[0]].shape)
    values = np.zeros(dims)
    for i, pix in enumerate(hpix):
        values[i] = hmap_dict[pix]
    
    umap_out = umap(values=values, hpix_dict=hpix_dict)
    if out_time:
        print 'sorting values time: %.3g sec' % (time.time() - t0)
    return umap_out



def umap2hmap_new(umap, order_out='max', power=None, out_ordering='RING', out_time=False):
    umap_d = umap_udgrade(umap, order_out=order_out, out_time=out_time)
    if out_time:
        t0 = time.time()
    order = umap_d.orders()[0]
    npix = healpy.nside2npix(2**order)
    vals = umap_d.values()
    dims = list(vals.shape)
    dims[0] = npix
    res = np.zeros(dims, dtype=vals.dtype)
    hpix = umap_d.hpixels(order)
    if out_time:
        print 'nest to ring time: %.3g sec' % (time.time() - t0)
        t0 = time.time()
    if out_ordering == 'RING':
        hpix = healpy.nest2ring(2**order, hpix)
    if out_time:
        print 'nest to ring time: %.3g sec' % (time.time() - t0)
        t0 = time.time()
    res[hpix] = vals
    if out_time:
        print 'final array time: %.3g sec' % (time.time() - t0)
    return res


def umap2hmap_old(umap, order_out='max', power=None, out_ordering='RING'):
    dims = list(umap.values().shape)

    # single order umap
    if len(umap.orders()) == 1:
        order = umap.orders()[0]
        npix = healpy.nside2npix(2**order)
        dims[0] = npix
        vals = umap.values()
        res = np.zeros(dims, dtype=vals.dtype)
        hpix = umap.hpixels(order)
        if out_ordering == 'RING':
            hpix = healpy.nest2ring(2**order, hpix)
        res[hpix] = vals
        return res

    # umap with multiple orders
    hpix_dict = umap.hpix_dict()
    if power is None:
        power = umap.hpower()

    if order_out == 'max':
        order_out = max(umap.orders())
    elif order_out == 'min':
        order_out = min(umap.orders())
    else:
        order_out = int(order_out)
    nside_out = 2**order_out
    npix_out = healpy.nside2npix(nside_out)
    dims_out = [npix_out]
    dims_out.extend(dims[1:])
    hpix = np.zeros(dims_out)
    
    for order in umap.orders():
        dims_loc = dims_out[:]
        dims_loc[0] = healpy.nside2npix(2**order)
        hpix_loc = np.zeros(dims_loc)
        inds = hpix_dict[order]
        hpix_loc[inds] = umap.values(order)
        if len(dims_loc) < 3:
            res = healpy.ud_grade(hpix_loc.T, nside_out, power=power,
                                  order_in='NESTED', order_out=out_ordering)
            hpix += np.array(res).T
        elif len(dims_loc) == 3:
            for i in range(dims_out[-1]):
                print i, power
                res = healpy.ud_grade(hpix_loc.T[i], nside_out, power=power,
                                      order_in='NESTED', order_out=out_ordering)
                hpix[:,:,i] += np.array(res).T
    return hpix


umap2hmap = umap2hmap_new

def get_dict(inds, values):
    res = {}
    for i in range(len(inds)):
        res[int(inds[i])] = values[i]
    return res


def umap2umap_dict(umap):
    return get_dict(umap.upixels(), umap.values())
    #umap_dict = {}
    #upixels = umap.upixels()
    #values = umap.values()
    #for i in range(len(upixels)):
    #    umap_dict[int(upixels[i])] = values[i]
    #return umap_dict

def umap2hmap_dicts(umap):
    hmap_dicts = {}
    orders = umap.orders()
    for order in orders:
        inds = umap.hpix_dict()[order]
        values = umap.values(order)
        hmap_dicts[order] = get_dict(inds, values)
    return hmap_dicts


def upix_arr2values(umap, upix_arr):
    umap_dict = umap2umap_dict(umap)
    if 0:
        # cleaner syntax but a little slower
        def func(pix):
            return umap_dict.get(pix, umap.default_value)
    else:
        # a little faster function
        def func(pix):
            if not umap_dict.has_key(pix):
                return umap.default_value
            else:
                return umap_dict[pix]
    func_vec = np.frompyfunc(func, 1, 1)
    res = np.array(func_vec(upix_arr), dtype=float)
    return res

def vec2values(umap, x, y, z):
    res = 0.
    for order in umap.orders():
        hpix_arr = healpy.vec2pix(2**order, x, y, z, nest=True)
        upix_arr = hpix2upix(order, hpix_arr)
        res += upix_arr2values(umap, upix_arr)
    return res

def ang2values(umap, theta, phi):
    res = 0.
    for order in umap.orders():
        hpix_arr = healpy.ang2pix(2**order, theta, phi, nest=True)
        upix_arr = hpix2upix(order, hpix_arr)
        res += upix_arr2values(umap, upix_arr)
    return res

def hdict2vec(hdict):
    """
    returns 3 by npix matrix of vectors
    """
    npix = npixels(hdict)
    res = np.zeros((3, npix))
    start = 0
    orders = hdict.keys()
    orders.sort()
    for order in orders:
        hpix = hdict[order]
        res[:, start:start + len(hpix)] = healpy.pix2vec(2**order, hpix, nest=True)
        start += len(hpix)
    return res

def orders2npix(orders):
    return 12 * 4**orders

def dOm(order):
    return 4. * np.pi / orders2npix(order)



class umap:
    '''
    A map with values in unversal pixels
    Attributes (internal):
        self.vals
        self.hdict
        self.ords
        self.is_mean
        self.default_value
    
    '''
    def __init__(self, values=None, upixels=None, hpix_dict=None,
                 is_mean=True, default_value=0.):
        if values is not None:
            self.vals = np.array(values)
        else:
            self.vals = None
        if upixels is not None:
            self.hdict = upix2hpix_dict(upixels)
        elif hpix_dict is not None:
            self.hdict = {order:hpix_dict[order]
                          for order in hpix_dict.keys()
                          if len(hpix_dict[order]) > 0}
            for key in self.hdict.keys():
                self.hdict[key ] = np.array(hpix_dict[key])
        elif values is not None and healpy.isnpixok(len(values)):
            npix = len(values)
            nside = healpy.npix2nside(npix)
            order = int(np.log2(nside))
            self.hdict = {}
            self.hdict[order] = range(npix)
            self.ords = [order,]
        else:
            self.hdict = None

            
        self.is_mean = is_mean

        if self.hdict is not None:
            self.ords = self.hdict.keys()
            self.ords.sort()
        else:
            self.ords = None
        if self.vals is not None:
            self.default_value = default_value * np.ones_like(self.vals[0])
        else:
            self.default_value = default_value

    def upixels(self, order=None):
        if order is None:
            return hpix_dict2upix(self.hdict)
        elif order not in self.ords:
            return None
        else:
            return hpix2upix(order, hpix=self.hdict[order])

    def hpix_dict(self):
        return self.hdict

    def hpixels(self, order):
        return self.hdict.get(order)

    def dOm(self):
        return dOm(upix2order(self.upixels()))
        

    def npixels(self):
        return npixels(self.hdict)

    def orders(self):
        return self.ords

    def update_values(self, new_values, new_default=0.):
        self.vals = new_values
        self.default_value = new_default * np.ones_like(self.vals[0])
        return None

    def hpower(self):
        if self.is_mean:
            return 0.
        else:
            return -2.
        
    def values(self, order=None):
        if order is None or self.vals is None:
            return self.vals
        elif order not in self.ords:
            return None
        else:
            start = 0
            i = 0
            while self.ords[i] < order:
                start += len(self.hdict[self.ords[i]])
                i += 1
            finish = start + len(self.hdict[order])
            return self.vals[start:finish]
