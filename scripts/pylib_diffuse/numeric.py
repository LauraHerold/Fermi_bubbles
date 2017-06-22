# a library of general numeric algorithms


#import healpy
import numpy as np
import scipy
from scipy import optimize
import time
import copy
#import math
#from scipy import special
#import scipy.integrate as integr
#import warnings


#import wish

step = lambda x: (1 + np.sign(x)) / 2


epsilon = 1.e-15

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


def interpolate_linear1d(xs, zs):
    '''
    linear interpolation
    INPUT:
        xs - array_like, shape (n,)
        zs - array_like, shape (n,):
            values of the function at x
    OUTPUT:
        f - function of x that interpolates zs
    '''
    def f(x):
        xind = findIndex(xs, x, nearest=True)

        dx = (x - xs[xind])
        if xind == len(xs) - 1 or (dx < 0 and xind > 0):
            imax = xind
            imin = xind - 1
        else:
            imax = xind + 1
            imin = xind

        vx = (zs[imax] - zs[imin]) / (xs[imax] - xs[imin])

        v0 = zs[xind]
        vs = v0 + dx * vx

        return vs

    f_vec = np.frompyfunc(f, 1, 1)
    
    return f_vec



def interpolate_linear2d(xs, ys, zs):
    '''
    linear interpolation
    INPUT:
        xs - array_like, shape (n,)
        ys - array_like, shape (k,)
        zs - array_like, shape (n, k,):
            values of the function at x and y
    OUTPUT:
        f - function of (x, y) that interpolates zs
    '''
    def f(x, y):
        xind = findIndex(xs, x, nearest=True)
        yind = findIndex(ys, y, nearest=True)


        dx = (x - xs[xind])
        if xind == len(xs) - 1 or (dx < 0 and xind > 0):
            imax = xind
            imin = xind - 1
        else:
            imax = xind + 1
            imin = xind

        vx = (zs[imax, yind] - zs[imin, yind]) / (xs[imax] - xs[imin])

        dy = (y - ys[yind])
        if yind == len(ys) - 1 or (dy < 0 and yind > 0):
            imax = yind
            imin = yind - 1
        else:
            imax = yind + 1
            imin = yind

        vy = (zs[xind, imax] - zs[xind, imin]) / (ys[imax] - ys[imin])

        v0 = zs[xind, yind]

        
        vs = v0 + dx * vx + dy * vy

        return vs
    
    return f


def _multiply(mats):
    """
    multiply the matricies in the sequence "mats"
    
    """
    res = 1.
    for i in range(len(mats)):
        if i < len(mats) - 1 and mats[i].ndim < 2 or np.array(res).ndim == 0:
            res = res * mats[i]
        else:
            res = np.dot(res, mats[i])
    return res


def _pack_pars(pars, inds):
    '''
    pack parameters into an array

    '''
    return np.array([pars[j][i] for j in range(len(pars)) for i in inds[j]])

def _unpack_pars(p, pars, inds):
    '''
    unpack parameters from an array

    '''
    k = 0
    pars = copy.deepcopy(pars)
    for i in range(len(pars)):
        for j in inds[i]:
            pars[i][j] = p[k]
            k += 1
    return pars

def _lstsq(d, m, g):
    '''
    calculate the least squared

    '''
    
    H = _multiply((m, g, m.conj().T))
    nmax = m.shape[0]
    B = np.zeros((nmax, nmax))
    while(np.linalg.det(H[:nmax, :nmax]) == 0):
        nmax -= 1
    B[:nmax, :nmax] = np.linalg.inv(H[:nmax, :nmax])
    q = _multiply((d, g, m.conj().T, B))
    resid = d - np.dot(q, m)
    chi2 = _multiply((resid, g, resid.conj().T))
    return q, chi2, B, resid


def _chi2(d, m, g, lstsq=True):
    '''
    calculate the minimum of chi2
    
    '''
    if lstsq:
        return _lstsq(d, m, g)[1]
    else:
        if d.ndim == m.ndim:
            resid = d - m
        else:
            resid = d - np.sum(m, axis=0)
        return _multiply((resid, g, resid.conj().T))

def get_model(xs, f_data, lin_pars):
    
    fs, pars, inds = f_data
    
    model = np.zeros_like(xs)
    for i in range(len(fs)):
        model += lin_pars[0][i] * fs[i](pars[i])(xs)
        
    return model


def fit_params(xs, data, metric, f_data, logpriors=None, mask=None, out_steps=False,
               lstsq=True, **kwargs):
    '''
    logpriors - list of "- 2 log likelihood" of parameters
    
    '''
    fs, pars0, inds = f_data
    p0 = _pack_pars(pars0, inds)
    
    if mask is None:
        mask = np.ones_like(data)

    if logpriors is None:
        logpriorf = lambda x: 0.
    else:
        def logpriorf(pars):
            res = 0
            for ind in range(len(logpriors)):
                if logpriors[ind] is not None:
                    res += logpriors[ind](pars[ind])
            return res

    if len(p0) > 0:
        
        def objf(p):
            pars = _unpack_pars(p, pars0, inds)
            
            models = np.array([fs[i](pars[i])(xs) * mask
                               for i in range(len(fs))])
            
            chi2 = _chi2(data, models, metric, lstsq=lstsq) + logpriorf(pars)
            if out_steps:
                print p, chi2
            return chi2

        # fit the non-linear parameters
        fit_param = scipy.optimize.fmin(objf, p0, full_output=True, **kwargs)

        fit_pt = fit_param[0]
        nonlin_pars = _unpack_pars(fit_pt, pars0, inds)
    else:
        nonlin_pars = pars0

    # find the linear parameters        
    models = np.array([fs[i](nonlin_pars[i])(xs) * mask
                       for i in range(len(fs))])

    if lstsq:
        lstsq_fit = _lstsq(data, models, metric)
    else:
        chi2 = _chi2(data, models, metric, lstsq=False)
        lstsq_fit = (None, chi2, None, None)
    #lstsq_fit = metric * (data - np.sum(models, axis=0))**2

    return nonlin_pars, lstsq_fit

def get_sigmas(xs, data, metric, f_data, chi20,
               mask=None, sigma=1., factor=1.2, out_steps=False, limits=False, **kwargs):
    '''
    find marginalized uncertainties
    
    '''
    fs, pars0, inds0 = f_data
    if limits:
        up_sigma = copy.deepcopy(pars0)
        down_sigma = copy.deepcopy(pars0)
    else:
        sigmas = copy.deepcopy(pars0)
    for ip in range(len(pars0)):
        for k in range(len(pars0[ip])):
            if limits:
                up_sigma[ip][k] = 0.
                down_sigma[ip][k] = 0.
            else:
                sigmas[ip][k] = 0.
            
    inds = copy.deepcopy(inds0)

    
    for ip in range(len(inds0)):
        pinds = inds0[ip][:]
        for k in range(len(pinds)):
            sind = pinds.pop(0) # sigma index - variable for which sigma is calculated
            inds[ip] = pinds # the rest of indices

            pars = copy.deepcopy(pars0)

            if sind == 0:
                lstsq = False
            else:
                print 'least sq fitting'
                print 
                lstsq = True
                        
            
            def objf(par):
                pars[ip][sind] = par
                if out_steps:
                    print 'starting point:', pars
                f_data_var = [fs, pars, inds]
                nonlin_pars, lstsq_fit = fit_params(xs, data, metric, f_data_var,
                                                    mask=mask, out_steps=out_steps,
                                                    lstsq=lstsq, **kwargs)

                chi2 = lstsq_fit[1]
                print 'chi2 = ', chi2
                print 'chi20 = ', chi20
                print 'test parameters', nonlin_pars
                print 'baseline pars:', pars0
                #print lstsq_fit[0]
                print
                print 
                return np.abs(chi2 - chi20 - sigma)

            par0 = pars0[ip][sind]

            # upper limit calculation
            n_steps = 10
            def get_dpar(par0, upper=True, thres=0.1):
                test = 0
                if test:
                    return 0.
                
                if upper:
                    direction_factor = 1
                else:
                    direction_factor = -1
                    
                delta_factor = 1
                delta = direction_factor * (factor - 1) * abs(par0)
                dchi2_min = np.inf
                par1 = par0
                print 'initial parameter: %.3e' % par0
                print 'initial chi2:', chi20
                for i in range(n_steps):
                    par1 += delta

                    print
                    print 'iteration number:', (i + 1)
                    print 'running parameter: %.3e' % par1
                    pars[ip][sind] = par1
                    f_data_var = [fs, pars, inds]
                    nonlin_pars, lstsq_fit = fit_params(xs, data, metric, f_data_var,
                                                            mask=mask, out_steps=out_steps,
                                                            lstsq=lstsq, **kwargs)
                    

                    chi21 = lstsq_fit[1]
                    dchi2 = chi21 - chi20 - sigma

                    print 'initial chi2:', chi20
                    print 'running chi2:', chi21
                    print 'running dchi2 - sigma:', dchi2
                
                    
                    if dchi2 < 0:
                        delta *= delta_factor
                        delta = direction_factor * abs(delta)
                    else:
                        delta_factor = 0.5
                        delta *= delta_factor
                        delta = -direction_factor * abs(delta)

                    print 'delta = %.3e' % delta
                    if abs(dchi2) < dchi2_min:
                        dchi2_min = abs(dchi2)
                        par1_best = par1
                        if dchi2_min < thres:
                            break
                    
                return par1_best

            par_minus = get_dpar(par0, upper=False)
            print 'lower limit: %.3e' % par_minus
    
            par_plus = get_dpar(par0, upper=True)
            print 'upper limit: %.3e' % par_plus
            
            if 0:
                par_plus = par0 - 1
                par_plus_test = par0
                while par_plus < par0:
                    print
                    print 
                    print 'upper limit calculation'
                    print
                    print 
                    par_plus_test += abs(par0) * (factor - 1)
                    if par0 == 0.:
                        par_plus_test += (factor - 1)

                    par_plus = scipy.optimize.fmin(objf, [par_plus_test,], **kwargs)[0]
                    
                    print 'center point: ', par0
                    print 'upper limit: ', par_plus

                par_minus = par0 + 1
                par_minus_test = par0
                while par_minus > par0:
                    print
                    print 
                    print 'lower limit calculation'
                    print
                    print 
                    par_minus_test -= abs(par0) * (factor - 1)
                    if par0 == 0.:
                        par_minus_test -= (factor - 1)
                        
                    par_minus = scipy.optimize.fmin(objf, [par_minus_test,], **kwargs)[0]
                    
                    print 'center point: ', par0
                    print 'lower limit: ', par_minus
            
            if limits:
                up_sigma[ip][sind] = par_plus - par0
                down_sigma[ip][sind] = par0 - par_minus
            else:
                sigmas[ip][sind] = abs(par_plus - par_minus) / 2

            pinds.append(sind) # return sind to the rest of indicies
            
    if limits:
        return down_sigma, up_sigma
    else:
        return sigmas



def MCMH(P, x0, sigmas = 1, Nsteps = 1, print_steps = False, positive = False, return_logL = False,\
        jump_frac = 0.35, tolerance = 0.15, check_point = 50, relax_ind = 50, sigma_max = None):
    """
       Metropilis Hastings Monte Carlo sampler
    INPUT:
        P:      probability distribution P(x)
        x0:     initial point
        sigmax: standard deviation of delta x for the next step
            if sigmax[i] == 0, then x[i] = x0[i] - it is fixed
        Nsteps: number of steps in the sample
    OUTPUT:
       Nsteps by len(x0) array of points sampled by the algorithm
    HISTORY:
       2010-Dec-07 - Written - D.Malyshev (NYU)
    """
    output = np.zeros(((Nsteps - relax_ind), len(x0)))
    logL = np.zeros(Nsteps - relax_ind)
    ntries = np.zeros_like(x0)
    njumps = np.zeros_like(x0)
    n_jump = 0.
    t0 = time.clock()
    
    goal_jump_frac = jump_frac * np.ones_like(x0)

    if np.array(sigmas).ndim == 0:
        sigmas = sigmas*np.abs(x0)

    
    variables = [i for i in range(len(x0)) if sigmas[i] > 0]
    x = x0.copy()
    px = P(x)
    #output[0] = x0.copy()
    for i in range(Nsteps):
        y = x.copy()
        n = np.random.random_integers(0, len(variables) - 1)
        ind = variables[n]
        dy = sigmas[ind] * np.random.standard_normal()
        if positive:
            while y[ind] + dy <= 0:
                dy = sigmas[ind] * np.random.standard_normal()
        y[ind] += dy
        a = np.random.uniform()

        if print_steps:
            print '\nStep\t%i\t time = %i sec' % (i + 1, time.clock() - t0)
            print 'Try index:\t%i' % ind
            print 'Initial point:\t', x, np.log(px)
        

        py = P(y)
        if print_steps: print 'Trial point:\t', y, np.log(py)

        if i >= relax_ind: ntries[ind] += 1.
        if py / px > a:
            x = y
            px = py
            if print_steps: print 'Jump!!!'
            if i >= relax_ind: 
                njumps[ind] += 1.

        if i >= relax_ind:
            output[i - relax_ind] = x
            logL[i - relax_ind] = np.log(px)

        if i >= relax_ind and print_steps: 
            print 'jump fracs:\t', njumps / (ntries + epsilon)
            

        if i >= relax_ind + check_point and i%check_point == 0:
            if print_steps:
                print 'old sigmas:\t', sigmas
            # adjust sigmas if needed
            for i in range(len(sigmas)):
                if abs(goal_jump_frac[i] - njumps[i])/(ntries[i] + epsilon) > tolerance:
                    sigmas[i] = sigmas[i]*(1 - goal_jump_frac[i] + njumps[i] / (ntries[i] + epsilon))
            if sigma_max != None:
                sigmas = np.array([min(sigmas[i], sigma_max[i]) for i in range(len(sigmas))])    
            if print_steps:
                print 'new sigmas:\t', sigmas
        
        
    jfrac = njumps / (ntries + epsilon)
    if return_logL:
        return output, jfrac, logL
    else:
        return output, jfrac







