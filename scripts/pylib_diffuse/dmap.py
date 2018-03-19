# manipulations with maps


import numpy as np
import healpy


def Es2Ebins(Es):
    f = np.sqrt(Es[1] / Es[0])
    Ebins = np.zeros(len(Es) + 1)
    Ebins[1:] = Es * f
    Ebins[0] = Es[0] / f
    return Ebins

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

def counts2EpdNdE(counts, exposures, Es, dOm=None, p=0):
    '''
    counts map to E^p dN/dE map conversion

    '''
    nbins = len(Es)
    Ebins = Es2Ebins(Es)
    dE = Ebins[1:] - Ebins[:-1]
    if dOm is None:
        npix = counts.shape[-1]
        dOm = 4. * np.pi / npix
    res = np.zeros_like(counts)
    for i in range(nbins):
        res[i] = counts[i] / (exposures[i] * dE[i] * dOm) * Es[i]**p
    return res


def counts2E2dNdE(counts, exposures, Es, dOm=None):
    '''
    counts map to E^2 dN/dE map conversion

    '''
    if 1:
        return counts2EpdNdE(counts, exposures, Es, dOm=dOm, p=2)
    else:
        nbins = len(Es)
        Ebins = Es2Ebins(Es)
        dE = Ebins[1:] - Ebins[:-1]
        if dOm is None:
            npix = counts.shape[-1]
            dOm = 4. * np.pi / npix
        res = np.zeros_like(counts)
        for i in range(nbins):
            res[i] = counts[i] / (exposures[i] * dE[i] * dOm) * Es[i]**2
        return res

def counts2E2dNdE_spectrum(counts, exposures, Es, dOm=None):
    '''
    counts map to E^2 dN/dE spectrum conversion
    spectrum weighted with the size of pixels:
        s = sum_i f_i * dOm_i / sum_i dOm_i

    '''
    fpix = counts2E2dNdE(counts, exposures, Es, dOm=1.)
    npix = counts.shape[-1]
    if dOm is None:
        dOm = 4. * np.pi / npix
    if isinstance(dOm, np.ndarray):
        sum_dOm = np.sum(dOm)
    else:
        sum_dOm = dOm * npix
    #print 'step 0', fpix.shape
    fpix = np.sum(fpix, axis=-1) / sum_dOm
    #print 'step 1', fpix.shape
    while fpix.ndim > 1:
        #print fpix
        fpix = np.mean(fpix, axis=-1)
        #print 'step 2', fpix.shape
    #print fpix
    return fpix

def EpdNdE2counts(flux, exposures, Es, dOm=None, p=0):
    '''
    E^p dN/dE to number counts conversion

    '''
    nbins = len(Es)
    Ebins = Es2Ebins(Es)
    dE = Ebins[1:] - Ebins[:-1]
    if dOm is None:
        npix = flux.shape[-1]
        dOm = 4. * np.pi / npix
    res = np.zeros_like(flux)
    for i in range(nbins):
        res[i] = flux[i] * exposures[i] * dE[i] * dOm / Es[i]**p
    return res


def E2dNdE2counts(flux, exposures, Es, dOm=None):
    '''
    E^2 dN/dE to number counts conversion

    '''
    if 1:
        return EpdNdE2counts(flux, exposures, Es, dOm=None, p=2)
    else:
        nbins = len(Es)
        Ebins = Es2Ebins(Es)
        dE = Ebins[1:] - Ebins[:-1]
        if dOm is None:
            npix = flux.shape[-1]
            dOm = 4. * np.pi / npix
        res = np.zeros_like(flux)
        for i in range(nbins):
            res[i] = flux[i] * exposures[i] * dE[i] * dOm / Es[i]**2
        return res


    
intF = lambda E0, E1: 1./E0 - 1./E1

def pl_int(index):
    """
    integral of x**(-index) from x0 to x1

    """
    def f(x0, x1):
        if index != 1:
            return (x0**(1 - index) - x1**(1 - index)) / (index - 1)
        else:
            return np.log(x1 / x0)
    return f

def integr_flux(E2dNdE, Emin, Emax, Es, ifstd=False):
    """
    Takes E2dN/dE and returns flux integrated between Emin and Emax

    """
    Ebins = Es2Ebins(Es)
    dE = Ebins[1:] - Ebins[:-1]
    imin = findIndex(Ebins, Emin) - 1
    imax = findIndex(Ebins, Emax)
    
    corrF = intF(Emin, Ebins[imin + 1]) / intF(Ebins[imin], Ebins[imin + 1])
    

    FF = np.array([dE[i] / Es[i]**2 * E2dNdE[i] for i in range(imin, imax)])
    if ifstd:
        res = np.sum(FF[1:]**2, axis=0) + corrF**2 * FF[0]**2
        res = np.sqrt(res)
    else:
        res = np.sum(FF[1:], axis=0) + corrF * FF[0]
    return res

def fraction(EE0, EE1, E0, E1, index=2):
    return pl_int(index)(EE0, EE1) / pl_int(index)(E0, E1)

def add_counts(counts, Emin, Emax, Es, ifstd=False, index=2):
    """
    Add counts between Emin and Emax

    """
    Ebins = Es2Ebins(Es)
    nE = len(Es)
    dE = Ebins[1:] - Ebins[:-1]
    imin = findIndex(Ebins, Emin) - 1
    imax = findIndex(Ebins, Emax)
    if (imin == -1 and imax == 0) or (imin == nE and imax == nE + 1):
        return 0
    if imin == -1:
        imin = 0
        Emin = Ebins[0]
    if imax == nE + 1:
        imax = nE
        Emax = Ebins[nE]
    print 'Energy bins %i to %i, Emin_boundary=%.1e GeV, Emax_boundary=%.1e GeV' \
        % (imin, imax - 1, Ebins[imin], Ebins[imax])

    if imax == imin + 1:
        corrF = fraction(Emin, Emax, Ebins[imin], Ebins[imax],
                      index=index)
        return corrF * counts[imin]
            
    else:
        corrF0 = fraction(Emin, Ebins[imin + 1], Ebins[imin], Ebins[imin + 1],
                      index=index)
        corrF1 = fraction(Ebins[imax - 1], Emax, Ebins[imax - 1], Ebins[imax],
                      index=index)
    
    FF = counts[imin: imax]
    if ifstd:
        res = np.sum(FF[1:-1]**2, axis=0) + corrF0**2 * FF[0]**2 \
              + corrF1**2 * FF[-1]**2
        res = np.sqrt(res)
    else:
        #print FF.shape
        res = np.sum(FF[1:-1], axis=0) + corrF0 * FF[0] + corrF1 * FF[-1]
    return res


def smoothing(fpix, **kwargs):
    return healpy.smoothing(fpix, regression=False, **kwargs)
