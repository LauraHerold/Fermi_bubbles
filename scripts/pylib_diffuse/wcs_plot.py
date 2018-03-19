import os
import sys
import numpy as np
#import scipy
import pyfits
#import matplotlib.patheffects
from matplotlib import pyplot
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
import healpy
#from optparse import OptionParser
import time

import pywcs
import pywcsgrid2
from pywcsgrid2.allsky_axes import make_allsky_axes_from_header

#import pyregion
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import upix

CMAP = r'afmhot_r'
#CMAP = 'CMRmap'
#CMAP = r'YlGnBu_r'

step = lambda x: (1. + np.sign(x)) / 2.
delta = lambda x: 4 * step(x) * step(-x)


def setup_figure_pars():
    fig_width = 12  # width in inches
    fig_height = 8    # height in inches
    fig_size =  [fig_width, fig_height]
    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'text.fontsize': 14,
              'legend.fontsize': 14,
              'xtick.labelsize':14,
              'ytick.labelsize':14,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : 6,
              'ytick.major.size' : 6,
              'xtick.minor.size' : 3,
              'ytick.minor.size' : 3,
              'figure.subplot.left' : 0.07,
              'figure.subplot.right' : 0.97,
              'figure.subplot.bottom' : 0.15,
              'figure.subplot.top' : 0.9
                }
    pyplot.rcParams.update(params)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    return 0

# number of pixels (in latitude)
# the number of longitude pixels is twice larger
nLBpix = 360

def fpix2wcs_fits(fpix,
                  LB_ref=[0., 0.], LB_size=[360., 180.], LB_npix=[2*nLBpix+1, nLBpix+1],
                  xcoord='GLON', ycoord='GLAT', proj='MOL', isint=0):
    '''
    transform a healpix map into maps in WCS coordinates
    this is an old function, keep here as a reference

    '''
    ndim = 2                        # number of output axis
    wcs = pywcs.WCS(naxis=ndim)     # wcs class

    dl_LB = np.array(LB_size, dtype=float) / (np.array(LB_npix) - 1)
    p0 = (np.array(LB_npix) + 1.) / 2

    wcs.wcs.crpix = [p0[0], p0[1]]
    wcs.wcs.cdelt = [-dl_LB[0], dl_LB[1]]
    wcs.wcs.crval = [LB_ref[0], LB_ref[1]] # reference point in (L, B) (deg, deg)
    wcs.wcs.ctype = ["%s-%s" % (xcoord, proj), "%s-%s" % (ycoord, proj)]
    wcs.wcs.cunit = ["deg", "deg"]

    # pixel grid
    Lpix = np.arange(1, LB_npix[0] + 1)
    Bpix = np.arange(1, LB_npix[1] + 1)
    pix_crds = np.array([[Lpix[i], Bpix[j]]
                         for i in range(LB_npix[0])
                         for j in range(LB_npix[1])])

    sky_crds = wcs.wcs_pix2sky(pix_crds, 1)
    sky_crds2D_rad = np.deg2rad(sky_crds.reshape((LB_npix[0], LB_npix[1], ndim)))

    
    # calculate healpix indices corresponding to the grid points
    nside = healpy.npix2nside(len(fpix))
    bl2pix = lambda L, B: healpy.ang2pix(nside, np.pi/2 - B, L)
    #bl2pix_vec = np.frompyfunc(bl2pix, 2, 1)

    indices = np.zeros((LB_npix[0], LB_npix[1]))    
    for i in range(LB_npix[0]):
        Ls = sky_crds2D_rad[i][:,0]
        Bs = sky_crds2D_rad[i][:,1]
        #indices[i] = bl2pix_vec(Ls, Bs)
        indices[i] = bl2pix(Ls, Bs)
    indices = np.array(indices, dtype = int)
    
    # write the values in the grid points from the healpix values
    sky_map = np.zeros((LB_npix[0], LB_npix[1]))
    for i in range(LB_npix[0]):
        sky_map[i] = fpix[indices[i]]
        #for j in range(LB_npix[1]):
        #    sky_map[i,j] = fpix[indices[i, j]]
        
    if isint:
        sky_map = sky_map.astype(int)
        format = 'I'
    else:
        format = 'E'

    header = wcs.to_header()
    hdu = pyfits.PrimaryHDU(header=header, data=sky_map.T)
    
    return hdu

def fpixs2wcs_fits(fpix, Es, Eunit='GeV',
                   LB_ref=[0., 0.], LB_size=[360., 180.], LB_npix=[2*nLBpix+1, nLBpix+1],
                   xcoord='GLON', ycoord='GLAT', proj='MOL', isint=0, outside_value=0.,
                   out_dname=None,):
    '''
    transform healpix maps into maps in WCS coordinates

    '''
    ndim = 3                        # number of output axis
    if not isinstance(fpix, upix.umap):
        if fpix.ndim == 1:
            fpix = np.array([fpix])
        elif fpix.ndim != 2:
            print 'wrong dimensions of fpix input to fpixs2wcs_fits'
            exit()
        
    wcs = pywcs.WCS(naxis=ndim)     # wcs class
    
    nLs = LB_npix[0]
    nBs = LB_npix[1]
    dl_LB = np.array(LB_size) / (np.array(LB_npix) - 1)
    #print 'dl_LB', dl_LB
    p0 = (np.array(LB_npix) + 1.) / 2
    if len(Es) == 1:
        dlogE = 0
    else:
        dlogE = np.log10(Es[1] / Es[0])

    wcs.wcs.crpix = [p0[0], p0[1], 1]
    wcs.wcs.cdelt = [-dl_LB[0], dl_LB[1], dlogE]
    wcs.wcs.crval = [LB_ref[0], LB_ref[1], np.log10(Es[0])] # reference point in (L, B) (deg, deg)
    wcs.wcs.ctype = ["%s-%s" % (xcoord, proj), "%s-%s" % (ycoord, proj), 'log10 E']
    wcs.wcs.cunit = ["deg", "deg", Eunit]

    # pixel grid
    Lpix = np.arange(1, LB_npix[0] + 1)
    Bpix = np.arange(1, LB_npix[1] + 1)
    if LB_size[0] == 360 and LB_size[1] == 180:
        allsky = True
    else:
        allsky = False

    
    if 0:
        # old and very slow calculation
        pix_crds = np.array([[Lpix[i], Bpix[j], 1]
                             for i in range(LB_npix[0])
                             for j in range(LB_npix[1])])
    else:
        # new and quick calculation
        pix_crds = np.ones((LB_npix[0] * LB_npix[1], 3), dtype=int)
        for i in range(LB_npix[0]):
            pix_crds[i * LB_npix[1]:(i + 1) * LB_npix[1], 0] = Lpix[i]
            pix_crds[i * LB_npix[1]:(i + 1) * LB_npix[1], 1] = Bpix[:]

    sky_crds = wcs.wcs_pix2sky(pix_crds, 1)
    sky_crds2D_rad = np.deg2rad(sky_crds.reshape((LB_npix[0], LB_npix[1], ndim)))

    #t0 = time.time()
    if isinstance(fpix, upix.umap):
        phi = sky_crds2D_rad[:, :, 0]
        theta = np.pi/2 - sky_crds2D_rad[:, :, 1]
        sky_map = upix.ang2values(fpix, theta, phi)
        print sky_map.shape
    else:
        # calculate healpix indices corresponding to the grid points
        nside = healpy.npix2nside(len(fpix[0]))
        bl2pix = lambda L, B: healpy.ang2pix(nside, np.pi/2 - B, L)
        #bl2pix_vec = np.frompyfunc(bl2pix, 2, 1)

        indices = np.zeros((LB_npix[0], LB_npix[1]))
        mask = np.ones((LB_npix[0], LB_npix[1]))
        for i in range(LB_npix[0]):
            Ls = sky_crds2D_rad[i][:,0]
            Bs = sky_crds2D_rad[i][:,1]
            if allsky:
                # mask in latitudes
                imin = 0
                imax = len(Bs) - 1
                while Bs[imin] == 0. and imin < imax:
                    imin += 1
                while Bs[imax] == 0. and imax > imin:
                    imax -= 1
                mask[i, :imin] = 0.
                mask[i, imax + 1:] = 0.
                # mask in longitudes
                imed = (nBs - 1) / 2
                Lsdeg = np.rad2deg(Ls)
                if i > (nLs - 1)/ 2:
                    Lsdeg -= 360
                L0 = Lsdeg[imed]
                #print i
                #print L0
                if L0 > 180 or L0 < -180:
                    mask[i] = 0.
                else:
                    imaxL = imed
                    while Lsdeg[imaxL] < 180 and Lsdeg[imaxL] > -180 and imaxL < imax + 1:
                        imaxL += 1
                    mask[i, imaxL:] = 0.
                    iminL = imed
                    while Lsdeg[iminL] < 180 and Lsdeg[iminL] > -180  and iminL > imin - 1:
                        iminL -= 1
                    mask[i, :iminL + 1] = 0.

                        
                if i == 0 and 0:
                    print 'i = 0'
                    print L0
                    print Lsdeg
                    #print np.rad2deg(Bs)
                    print mask[i]
                    print
                if i == 90 and 0:
                    print 'i =', i
                    print L0
                    print Lsdeg
                    #print np.rad2deg(Bs)
                    print mask[i]
                    print

                if i == (nLs - 1)/2 and 0:
                    print 'i = %i out of %i' % (i, nLs)
                    print L0
                    print Lsdeg
                    #print np.rad2deg(Bs)
                    print mask[i]
                    print
                if i == (nLs - 1)/2 + 90 and 0:
                    print 'i = %i out of %i' % (i, nLs)
                    print L0
                    print Lsdeg
                    #print np.rad2deg(Bs)
                    print mask[i]
                    print

            #mask[i] *= step(Bs + np.pi/2) * step(np.pi/2 - Bs)
            #mask[i] *= step(Ls) * step(2. * np.pi - Ls)
            #indices[i] = bl2pix_vec(Ls, Bs)
            if xcoord == 'RA':
                r = healpy.rotator.Rotator(coord=['C', 'G'])
                theta, Ls = r(np.pi/2 - Bs, Ls)
                Bs = np.pi/2 - theta
            indices[i] = bl2pix(Ls, Bs)
        indices = np.array(indices, dtype = int)

        
        # write the values in the grid points from the healpix values
        sky_map = np.zeros((LB_npix[0], LB_npix[1], len(Es)))
        
        for i in range(LB_npix[0]):
            # try multiplication by a mask
            sky_map[i] = (fpix[:, indices[i]] * mask[i] + outside_value * (1. - mask[i])).T
            #sky_map[i] = fpix[:, indices[i]].T
            #for j in range(LB_npix[1]):
            #    sky_map[i,j] = fpix[indices[i, j]]

    #print 'sky map time: %.3g' % (time.time() - t0)
    if allsky and sky_map.shape[-1] == 1 and 0:
        nx, ny = sky_map.shape[:2]
        proxi_value = sky_map[0, 0, 0]
        proxi_mask = delta(proxi_value - sky_map)
        #sky_map += 2 * proxi_mask * np.max(sky_map)
        print sky_map[p0[0], p0[1]]
        print sky_map[p0[0], p0[1] + 1]
        print p0
        print proxi_value
        #sky_map[p0[0], p0[1]] = proxi_value
    
    if isint:
        sky_map = sky_map.astype(int)
        format = 'I'
    else:
        format = 'E'

    header = wcs.to_header()
    hdu = pyfits.PrimaryHDU(header=header, data=sky_map.T)

    if out_dname is not None:
        hdulist = pyfits.HDUList([hdu])
        print 'save hdu to file:'
        print out_dname
        if os.path.isfile(out_dname):
            os.remove(out_dname)
        hdulist.writeto(out_dname)

    return hdu



def wcs_plot(indata, title=None, unit=None, min=None, max=None, out_fname=None, cmap=CMAP,
             rgn_str=None, allsky=False, add_cbar=True, fig_param_dict={},
             out_dname=None, **kwargs):
    """
    plot a healpix array fpix using pywcsgrid2 package

    """
    #print "inside wcs_plot"
    
    # transform healpix array to WCS hdu table
    if type(indata) == pyfits.hdu.hdulist.HDUList:
        hdu = indata[0]        
    elif type(indata) == pyfits.hdu.image.PrimaryHDU:
        hdu = indata
    else:
        hdu = fpix2wcs_fits(indata, **kwargs)

    if out_dname is not None:
        hdulist = pyfits.HDUList([hdu])
        print 'save hdu to file:'
        print out_dname
        
        if os.path.isfile(out_dname):
            os.remove(out_dname)
        hdulist.writeto(out_dname)


    header, data0 = hdu.header, hdu.data

    # can only plot 2D arrays
    if header['NAXIS'] == 2:
        data0 = [data0]

    if not isinstance(title, list):
        title = [title] * len(data0)

    setup_figure_pars()
    pyplot.rcParams.update(fig_param_dict)
    #if not allsky:
        #pyplot.rcParams['figure.figsize'][0] *= 2./3.


    for i in range(len(data0)):
        data = data0[i]
        
        #print 'inside', pyplot.rcParams['axes.labelsize']
        pyplot.rcParams['axes.labelsize'] = fig_param_dict.get('axes.labelsize', 12)
        fig = pyplot.figure()

        

        # define the plot from the hdu header and the hdu data
        if allsky:
            ax = make_allsky_axes_from_header(fig, rect=111, header=header, lon_center=0.)
        else:
            ax = pywcsgrid2.subplot(111, header=header)

        if data.ndim == 3:
            plot_data = data[:, :, 0]
        else:
            plot_data = data
        
        im = ax.imshow(plot_data,
                       #norm=LogNorm(vmin=min, vmax=max),
                       origin='lower', interpolation="nearest", cmap=cmap)
        im.set_clip_path(ax.patch)

        if min is not None and max is not None:
            im.set_clim(min, max)

        # color bar
        if add_cbar:
            if min is not None and max is not None:
                ticks = np.linspace(min, max, fig_param_dict.get('nticks', 5))
            else:
                ticks = None
            if allsky:
                cbar = fig.colorbar(im, orientation='horizontal', pad=.03, fraction=0.06, aspect=30, shrink=.8, ticks=ticks)
            else:
                cbar = fig.colorbar(im, orientation='horizontal', pad=.15, fraction=0.04, aspect=20, shrink=1., ticks=ticks)
            if 1:
                #cbar.set_ticklabels([r'$0$', r'$1.5 \times 10^{-6}$', r'$3 \times 10^{-6}$',
                #                     r'$4.5 \times 10^{-6}$', r'$6 \times 10^{-6}$'], update_ticks=True)
                cbar.set_ticklabels([r'$0$', r'$1$', r'$2$',
                                      r'$3$', r'$4$'], update_ticks=True)


            #cbar.ax.xaxis.set_label_coords(0.9,-1.8)
            if unit is not None:
                cbar.set_label(unit)


        if allsky:
            # setup the appearance of the axes of grid lines
            axis = ax.axis["lat=0"]
            axis.line.set_linestyle(':')
            axis.line.set_linewidth(.5)

            # get rid of the numbers on the grid lines
            
            for key in ax.axis.keys():
                ax.axis[key].major_ticklabels.set_visible(False)
                ax.axis[key].set_label('')

        ax.grid()

        if title[i] is not None:
            pyplot.title(title[i])

        if rgn_str is not None:
	    rgn_str1=pyregion.open(reg_str)
            rgn = pyregion.parse(rgn_str1).as_imagecoord(header)
            patch_list, artist_list = rgn.get_mpl_patches_texts()


            for p in patch_list:
                ax.add_patch(p)
            for t in artist_list:
                ax.add_artist(t)

        if out_fname is not None:
            print 'save figure to file:'
            print out_fname
            pyplot.savefig(out_fname)

    return 0

def save_figure(figFn, png=False, pdf=False):
    if png or pdf:
        endings = []
        if png: endings.append('.png')
        if pdf: endings.append('.pdf')
        for ending in endings:
            print 'save figure to file:'
            print figFn + ending
            pyplot.savefig(figFn + ending)
    return 0



if __name__ == '__main__':
    '''
    test the plotting function

    '''
    nside = 64
    npix = healpy.nside2npix(nside)
    pixels = range(npix)
    
    theta, Ls = healpy.pix2ang(nside, pixels)
    Bs = np.pi/2. - theta
    Ls = Ls - step(Ls - np.pi) * 2. * np.pi
    
    fpix = 1. + np.exp(-16*(Bs/np.pi)**2) + np.cos(Ls)
    unit = r'$E^2\frac{dN}{dE}\; \left({\rm 10^{-7}\; \frac{GeV}{cm^2\, s\, sr}} \right)$'
    title = r'$f = 1 + \exp(- 16 b^2/\pi^2) + \cos(\ell)$'
    if 1:
        file_name = 'wcs_plot_test_%s.pdf' % CMAP
    else:
        file_name = None
    wcs_plot(fpix, title=title, unit=unit, min=0., max=3., out_fname=file_name)
    pyplot.show()
