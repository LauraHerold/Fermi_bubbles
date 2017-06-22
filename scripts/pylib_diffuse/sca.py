"""
Spectral components analysis (SCA) of data vectors.

This module provides a function that can be used to
do a linear regression of the data and find
components associated with certain energy spectra.
The vectors associated with spectra are modeled
as linear combinations of data vectors
(Internal linear combinations, or ILC vectors).


Constants
---------

Functions useful to interprete the results of fitting
-----
'marginalize'
'f_vec'
'standard_models'

Classes
-------

See also
--------

Notes
-----

References
----------
arxiv: paper

History
----------
Written: Dmitry Malyshev, December 2011, KIPAC, Stanford


"""

__all__ = ['lstsq', 'least_squares', 'multiply', 'marginalize', 'f_vec', 'standard_models',
           'sca_fit', 'get_spectrum_filter', 'get_map_filter', 'get_smoothing_filter',
           'ridge_smooth_filter', 'get_gauge_filter', 'get_data_kernels',
           'kernel_pca', 'em_pca']

import numpy as np
import scipy
from scipy import optimize
import copy
#from matplotlib import pyplot




def multiply(mats):
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





def lstsq(d, m, g = None, full_output=False):
    """
    fit vector d (data) wrt vectors m (models) in space with metric g
    INPUT:  d - an N-vector: N - dimension
            m - (r, N)-matrix or an N-vector: r - number of models
            g - (N, N)-matrix or an N-vector
    OUTPUT:
            q - (k, r)-matrix: projections of d on m
            chi2 - k-vector: length of the residual vectors squared
            B = H^(-1) - (r, r)-matrix: inverse of hessian (H = f'')
            res - (k, N)-matrix: residual vectors
            
    """
    if g is None:
        if m.ndim == 1:   g = np.eye(len(m))
        elif m.ndim == 2: g = np.ones(len(m[0]))
        else:
            print 'm.ndim > 2 in lstsq function'
            return None
    if m.ndim == 1:
        H = np.array([[multiply((m, g, m.conj().T))]])
    else:
        H = multiply((m, g, m.conj().T))
    if np.linalg.det(H) == 0.:
        print 'determinant of the hessian matrix in lstsq is zero:'
        print '\t check that the models are non zero'
        return None
    
    B = np.linalg.inv(H)
    if m.ndim == 1 or m.shape[0] == 1:
        q = multiply((d, g, m.conj().T)) * B[0, 0]
    else:
        q = multiply((d, g, m.conj().T, B))
    resid = d - np.dot(q, m)
    chi2 = multiply((resid, g, resid.conj().T))
    if full_output:
        return q, chi2.real, B, resid
    else:
        return q, chi2.real, B

def least_squares(d, m, g = None, full_output = False):
    """
    fit vectors d (data) wrt vectors m (models) in space with metric g
    Constants:
        k - number of data vectors,
        N - number of components in the vectors (number of pixels)
        r - number of models
    
    INPUT:  d - array, shape (k, N)
            m - array, shape (r, N) or (r, k, N)
            g - array, shape (k, N, N) or (k, N)
    OUTPUT:
            q - array, shape (k, r): projections of d on m
            chi2 - k-vector: length of the residual vectors squared
            B = H^(-1) - (k, r, r)-matrix: inverse of hessian (H = f'')
            res - array, shape (k, N): residual vectors
            
    """
    k, N = d.shape
    r = m.shape[0]
    q = np.zeros((k, r))
    chi2 = np.zeros(k)
    B = np.zeros((k, r, r))
    resid = np.zeros((k, N))

    if m.ndim == 2:
        models = m
    for i in range(k):
        if m.ndim == 3:
            models = m[:,i]
        fit_res = lstsq(d[i], models, g[i], full_output=True)
        if fit_res is None:
            return None
        else:
            q[i], chi2[i], B[i], resid[i] = fit_res

    if full_output:
        return q, chi2.real, B, resid
    else:
        return q, chi2.real, B


    

def _d2v0(data):
    """
    Default data to ILC space vectors conversion.
    Returns a copy of the data itself with one dimension added.
        
    """
    return np.array([copy.deepcopy(data) for i in range(data.shape[0])])       

def _standard_data(data):
    '''
    return data in the standard form with data.ndim = 2
    '''
    if data is None or data.ndim > 2 or data.ndim < 1:
        print 'sca_fit: check dimensions of the data'
        exit()
    elif data.ndim == 1:
        return np.array([data])
    else:
        return data

def standard_models(models, k = None):
    '''
    return models in the standard form with models.ndim = 3
    
    '''
    if models is None or np.prod(models.shape) == 0:
        return None
    elif models.ndim < 1 or models.ndim > 3:
        print 'scq_fit: check dimensions of template models'
        exit()
    elif models.ndim == 1:
        return np.array([[models]])
    elif models.ndim == 2:
        d1 = models.shape[0]
        d2 = models.shape[1]
        res = np.zeros((k, d1, d2), dtype = models.dtype)
        for i in range(k):
            res[i] = copy.deepcopy(models)
        return res
    else:
        return models


def marginalize(data, model, metric):
    """
    project data to space perpendicular to model
    
    """
    if data is None or data == [] or np.prod(data.shape) == 0:
        return None
    if model is None or model == [] or np.prod(model.shape) == 0:
        return data


    if data.ndim == 1: data = np.array([data])
    k = data.shape[0]
    datap = np.zeros_like(data)


    for i in range(k):
        if model.ndim < 3:  mdls = model
        else:               mdls = model[i]

        if (metric.ndim == 2 and metric.shape[0] == metric.shape[1]) \
           or metric.ndim == 1:
            gg = metric
        else:
            gg = metric[i]

        if data.ndim == 2:
            datap[i] = lstsq(data[i], mdls, gg, full_output = True)[3]
        elif data.ndim == 3:
            for j in range(data.shape[1]):
                datap[i, j] = lstsq(data[i, j], mdls, gg,
                                     full_output = True)[3]
        else:
            print 'marginalize: unacceptable shape of data \
                   or value of ndim_out'
            print 'data.shape = ', data.shape
    return datap

def _param2pt(w_data = None, u_data = None, f_data = None):
    """
    transform parameters to a point in fitting function

    """
    pt = []
    if w_data:
        p, p_ind = w_data[1:]
        pt.extend([p[i][n] for i in range(len(p)) for n in p_ind[i]])
    if u_data:
        q, q_ind = u_data[1:]
        pt.extend([q[i][n] for i in range(len(q)) for n in q_ind[i]])
    if f_data:
        f, gauge_ind = f_data
        free_ind = [n for n in range(f.shape[1]) if n not in gauge_ind]
        nf = f.shape[0]*(f.shape[1] - f.shape[0])
        pt.extend([f[i][n] for i in range(f.shape[0]) for n in free_ind])
    return np.array(pt).real

def _pt2param(pt, w_data = None, u_data = None, f_data = None):
    """
    transform point to parameters

    """
    if pt is None:
        return w_data, u_data, f_data
    k = 0
    _w_data = _u_data = _f_data = None
    if w_data is not None:
        p = w_data[1]
        p_ind = w_data[2]
        for i in range(len(p)):
            for n in range(len(p[i])):
                if n in p_ind[i]:
                    p[i][n] = pt[k]
                    k += 1
        w_data = (w_data[0], p, p_ind)
    if u_data is not None:
        q = u_data[1]
        q_ind = u_data[2]
        for i in range(len(q)):
            for n in range(len(q[i])):
                if n in q_ind[i]:
                    q[i][n] = pt[k]
                    k += 1
        u_data = (u_data[0], q, q_ind)
    if f_data is not None:
        f = f_data[0]
        gauge_ind = f_data[1]
        free_ind = [n for n in range(f.shape[1]) if n not in gauge_ind]
        for i in range(f.shape[0]):
            for n in free_ind:
                f[i][n] = pt[k]
                k += 1
        f_data = (f, gauge_ind)
    return w_data, u_data, f_data

def _ptStd2parStd(pt, w_data = None, u_data = None, f_data = None):
    """
    transform point STD to parameters STD
    """
    k = 0
    _w_data = _u_data = _f_data = None
    if w_data:
        p = copy.deepcopy(w_data[1])
        p_ind = copy.deepcopy(w_data[2])
        for i in range(len(p)):
            for n in range(len(p[i])):
                if n in p_ind[i] and pt is not None:
                    p[i][n] = pt[k]
                    k += 1
                else:
                    p[i][n] = 0.
        _w_data = (w_data[0], p, p_ind)
    if u_data is not None:
        q = copy.deepcopy(u_data[1])
        q_ind = copy.deepcopy(u_data[2])
        for i in range(len(q)):
            for n in range(len(q[i])):
                if n in q_ind[i] and pt is not None:
                    q[i][n] = pt[k]
                    k += 1
                else:
                    q[i][n] = 0.
        _u_data = (u_data[0], q, q_ind)
    if f_data:
        f = np.zeros_like(f_data[0])
        gauge_ind = copy.deepcopy(f_data[1])
        free_ind = [n for n in range(f.shape[1]) if n not in gauge_ind]
        if pt is not None:
            for i in range(f.shape[0]):
                for n in free_ind:                
                    f[i][n] = pt[k]
                k += 1
        _f_data = (f, gauge_ind)
    return _w_data, _u_data, _f_data


# helper functions for lstsq_ILC
def _vv_mat(vectors, metric):
    return multiply((vectors, metric, vectors.conj().T))
    
def _dv_vec(d, v, metric):
    return multiply((d, metric, v.conj().T))

def f_vec(fs, ps, Ener):
    return np.array([fs[i](ps[i])(Ener) for i in range(len(fs))])


def _f_mat(fs, ps, Ener):
    vec = f_vec(fs, ps, Ener)
    return np.outer(vec, vec.conj())


def _LHS_mat_inv(fs, ps, Es, v, g, check = False):

    res = np.kron(_f_mat(fs, ps, Es[0]), _vv_mat(v[0], g[0]))
    
    if check:
        print 'vv\n', _vv_mat(v[0], g[0]).shape
        print 'f mat\n', _f_mat(fs, ps, Es[0]).shape
        print 'kron vv f\n', res.shape

    for i in range(1, len(Es)):        
        res += np.kron(_f_mat(fs, ps, Es[i]), _vv_mat(v[i], g[i]))
    if np.linalg.det(res) == 0.:
        return np.inf
    else:
        if check:
            print 'LHS inv \n', np.linalg.inv(res)
        return np.linalg.inv(res)

def _RHS_vec(fs, ps, Es, d, v, g, check = False):

    res = np.kron(f_vec(fs, ps, Es[0]).conj(), _dv_vec(d[0], v[0], g[0]))
    if check:
        print 'dv\n', _dv_vec(d[0], v[0], g[0])
        print 'f vec\n', f_vec(fs, ps, Es[0]).conj()
        print 'kron dv f\n', res
    
    for i in range(1, len(Es)):
        res += np.kron(f_vec(fs, ps, Es[i]).conj(),
                       _dv_vec(d[i], v[i], g[i]))
    return res

def _f_mu(fs, ps, Es, d, v, g, check = False):
    
    res = np.dot(_RHS_vec(fs, ps, Es, d, v, g),
                 _LHS_mat_inv(fs, ps, Es, v, g)).real
    dim0 = v.shape[1]
    if check:
        print 'rhs times lhs inv\n', res
        print 'rhs times lhs inv reshape\n', np.reshape(res, (len(fs), dim0))
    return np.reshape(res, (len(fs), dim0))

def _fB(fs, ps, Es, v, g, check = False):
    d0 = _f_mat(fs, ps, Es[0]).shape[0]
    d1 = _vv_mat(v[0], g[0]).shape[0]
    mat = _LHS_mat_inv(fs, ps, Es, v, g, check = check)
    res = np.array([mat[d1*i:d1*(i+1), d1*i:d1*(i+1)] for i in range(d0)])
    if check:
        print d0, d1
        print mat
        print res
    return res





        

        

        


def sca_fit(d, g = None, # data
            m_mrg = None, # marginal templates
            m_sp = None, w_data = None, # templates with functional spectra
            v = None, u_data = None, gauge_ind = None, # ILC vectors
            d2v = _d2v0, # data to ILC vectors conversion
            Es = None, # energies
            fit_mode = 'fmin', # fitting options
            collin_punish = None, outsteps = False, 
            maxiter = 5000, maxfun = 5000, # fit function options
            chi2_in_bins = False, # output chi2 for each bin
            project = False,  nm_perp = True # depreciated parameters
            ):
    """
    find least squared and ILC vectors together with spectrum

    Constants:
        k - number of energy bins,
        N - number of components in the vectors (number of pixels)
        r - number of vectors to form linear combinations, usually r = k
        nm_mrg - number of vectors with marginal spectra
        nm_sp - number of vectors with functional spectra
        nv_mrg - number of ILC vectors with marginal spectra
        nv_sp - number of ILC vectors with functional spectra
        
    INPUT:
        d - array_like, shape (k, N)
            data vectors
        g - array_like, shape (k, N, N) or (k, N), optional
            metric, the shape (k, N) is for diagonal metric
            DEFAULT: None
            if g is None, then g equal to unit matrix is assumed
        m_mrg - array_like, shape (k, nm_mrg, N) or (nm_mrg, N), optional
            fixed model vectors with marginal spectra:
            the coefficients in every energy bin are fitted independently
            DEFAULT: None
        m_sp - array_like, shape (k, nm_sp, N) or (nm_sp, N), optional
            fixed model vectors with functional spectra
            DEFAULT: None
            
        w_data - tuple, shape (3,), optional
            parametrization of spectra for the components with known
                spatial distribution
            contains three parameters:
                functions - tuple, length = nm_sp
                    function names to be used in fitting
                    functions should have one parameter: an array of variables
                parameters - tuple of lists, length = nm_sp
                    arguments for the functions specifying
                    initial point in fitting
                variable indices - tuple of lists, length = nm_sp
                    indices of parameters that one wants to vary
                    if an index of a variable doesn't appear,
                    this variable remains fixed at the initial value
            DEFAULT: None
                    
                EXAMPLE: functions = (plaw, plaw)
                         parameters = ([1., 0.], [1., 1.])
                         var indices = ([1], [0, 1])
                         w_data = (functions, parameters, variable indices)
                    where plaw([A, n]) = lambda x: A * x**n
                    for the first function only the index is varied,
                    while for the second function both the normalization
                    and the index are varied
                    
        v - array_like, shape (k, r, N) or (r, N), optional
            vectors spanning ILC space
            DEFAULT: None
                if v is None, then r = k and v = d
        gauge_ind - array_like, shape (nv_mrg,), optional
            list of gauge indices that specify a submatrix which is fixed
            to be the unit matrix
            In order to break the change of basis degeneracy,
            nv_mrg is determined as len(gauge_ind)
            DEFAULT: None
        u_data - tuple, shape (3,), optional
            similarly to w_data this tuple has three fields:
            (functions, parameters, variable indices)
            that determine the functional form, the initial point
            and the indices to be varied
            for the ILC vectors with functional spectra
            the overall normalization should not be included
            in variable indices,
            since it is degenerate with the normalization of ILC vectors
            nv_sp is determined as len(functions)
            
        d2v - function that transforms data vectors into ILC vectors v
            INPUT: array_like, shape = d.shape
            OUTPU: array_like, shape = v.shape

        Es - array_like, shape (k,), optional
            energies of the bins
            
        fit_mode - string, optional
            optimization function to use in fitting
            DEFAULT: 'fmin' - scipy.optimize.fmin
            other options: 'fmin_bfgs' - scipy.optimize.fmin_bfgs

        collin_punish - function, optional
            a function that increases chi2 when some vectors are collinear
            DEFAULT: None
            INPUT: f - array_like, shape (nf, N),
                       where nf is the number of vectors
            OUTPUT: float
        outsteps - boolean, optional
            if True, then each step of the fitting is printed
            DEFAULT: False
        maxiter, maxfun - integers, optional
            parameters passed to minimizing functions, 'fmin' and 'fmin_bfgs'
            DEFAULT: maxiter = maxfun = 5000
        chi2_in_bins - boolean, optional
            output chi2 for each bin
            DEFAULT: False
            
        project, nm_perp - depreciated
        
            
    OUTPUT:
        a tuple: (data_fit, chi2, B_fit)
        data_fit = (w_td, p, f_td, u_td, f, q)
            w_td - spectrum for marginalized templates
            p - parameters for templates with spectra
            f_td - marginalized ILC vectors
            u_td - spectrum for marginalized ILC vectors
            f - ILC vectors with functional spectrum
            q - spectrum parameters for ILC vectors 'f'
        chi2 - best fit chi2
        B_fit = (w_td_B, p_std, f_td_std, u_td_B, f_B, q_std)
            w_td_B - inverse hessian for w_td (covariance matrix)
            p_std - std of parameters 'p' for templates with spectra
            f_td_std - std of marginalized ILC vectors
            u_td_B - inverse hessian for the spectrum of marginalized
                     ILC vectors
            f_B - inverse hessian for ILC vectors with functional spectrum
            q_std - std of spectrum parameters for ILC vectors 'f'
            
    """
    # initialize the output parameters
    w_td = p = f_td = u_td = f = q = None
    w_td_B = p_std = f_td_std = u_td_B = f_B = q_std = None

    

    # make sure that data.ndim = 2
    d = _standard_data(d)
    k = d.shape[0]
    N = d.shape[1]

    chi2 = np.zeros(k)

    if g is None:
        g = np.ones(k)

    # make sure that models.ndim = 3 or models are None
    m_mrg = standard_models(m_mrg, k = k)
    m_sp = standard_models(m_sp, k = k)

    nm = nm_sp = nm_mrg = 0
    if m_mrg is not None:
        nm_mrg = m_mrg.shape[1]
    if m_sp is not None:
        nm_sp = m_sp.shape[1]
        if w_data is None or len(w_data[0]) != nm_sp:
            print 'sca_fit: the parameters of w_data \
                   do not match the dimensions of m_sp'
            return None
    nm = nm_sp + nm_mrg
        

    if gauge_ind == []:
        gauge_ind = None

    # make sure that v.ndim = 3 or v is None
    v = standard_models(v, k = k)

    nv = nv_sp = nv_mrg = 0
    
    if u_data is not None:
        nv_sp = len(u_data[0])
    if gauge_ind is not None:
        nv_mrg = len(gauge_ind)
    nv = nv_sp + nv_mrg

    if nv > 0 and v is None:
        v = _d2v0(d)

    if v is not None:
        r = v.shape[1]
    else:
        r = 0
        
    if nv > r:
        print 'sca_fit: nv > r - number of ILC vectors is larger \
               than the ILC space dimension'
        return None

    if Es is None:
        Es = np.arange(1., k+1)

    if collin_punish is not None:
        f_all = np.zeros((nv, r))

    
    # find the spectra of m_mrg and their covariance matrix B
    # in case when only marginal templates are present
    
    if nm_mrg > 0:
        w_td = np.zeros((k, nm_mrg), dtype = d.dtype)
        w_td_B = np.zeros((k, nm_mrg, nm_mrg), dtype = d.dtype)
        if nm_sp == 0 and nv == 0:
            for i in range(k):
                w_fit = lstsq(d[i], m_mrg[i], g[i], full_output = True)
                w_td[i] = w_fit[0]
                chi2[i] = w_fit[1]
                w_td_B[i] = w_fit[2]
                #if nv == 0 and outsteps:
                #    print 'bin %i chi2 inside sca: %i' % (i, chi2[i])

            data_res = (w_td, p, f_td, u_td, f, q)
            B_res = (w_td_B, p_std, f_td_std, u_td_B, f_B, q_std)

            if chi2_in_bins:
                return data_res, chi2.real, B_res
            else:
                return data_res, sum(chi2).real, B_res

        
    
    # The main fitting
    
    # 1. project all vectors to the space perpendicular
    #    to fixed marginal models: m_mrg
    
    dp = marginalize(d, m_mrg, g)
    m_spp = marginalize(m_sp, m_mrg, g)
    vp = marginalize(v, m_mrg, g)

    
    
    # initial values for ILC vectors
    if nv_mrg > 0:
        #f_td0 = np.zeros((nv_mrg, nv), dtype = dp.dtype)
        f_td0 = np.random.normal(loc=.0, scale=.1, size=(nv_mrg, r))
        for i in range(len(gauge_ind)):
            for j in range(nv_mrg):
                if i == j:  f_td0[j, gauge_ind[i]] = 1.
                else:       f_td0[j, gauge_ind[i]] = 0.
        
        f_data = (f_td0, gauge_ind)
    else:
        f_data = None

    # initial point in non-linear fit
    pt0 = _param2pt(w_data = w_data, u_data = u_data, f_data = f_data)

    
    # 2 - 4. Minimize wrt p, q, and f tilde
    def objf(pt, w_data = w_data, u_data = u_data, f_data = f_data,
             Es = Es, outsteps = outsteps, full_output = False,
             p_std = None, q_std = None, f_td_std = None,
             collin_punish = collin_punish,
             chi2_in_bins = False,
             nm_perp = True # depreciated
             ):
        """
        non-linear fit object function
        """
        # define non-linear parameters: p, q, f_td from input point
        p = f_td = u_td = f = q = None
        u_td_B = f_B = None

        # local w, u, and f data
        _w_data, _u_data, _f_data = \
            _pt2param(pt, w_data = w_data, u_data = u_data, f_data = f_data)
        if w_data is not None: w, p, p_ind = _w_data
        if u_data is not None: u, q, q_ind = _u_data
        if f_data is not None: f_td, gauge_ind = _f_data


        # 2. Subtract fixed models with spectra
        resid = copy.deepcopy(dp)
        vpm = copy.deepcopy(vp)
        
        if w_data is not None:
            # subtract templates with spectra from data
            for i in range(k):
                resid[i] -= np.dot(f_vec(w, p, Es[i]), m_spp[i])
        
            # subtract templates with spectra from ILC vectors
            if nv > 0:
                mvecs = np.array([multiply((f_vec(w, p, Es[i]), m_spp[i])) 
                                  for i in range(k)])
                vpm -= d2v(mvecs)


        # 3. Marginalize over f_td*v
        if f_data is not None:
            f_tdv = np.zeros((k, nv_mrg, N), dtype = f_td.dtype)
            if full_output:
                u_td = np.zeros((k, nv_mrg), dtype = d.dtype)
                u_td_B = np.zeros((k, nv_mrg, nv_mrg), dtype = d.dtype)

            for i in range(k):
                f_tdv[i] = np.dot(f_td, vpm[i])
                
                if full_output:
                    # find u_td and its covariance matrix B
                    u_fit = lstsq(resid[i], f_tdv[i], g[i],
                                   full_output = True)
                    u_td[i] = u_fit[0]
                    u_td_B[i] = u_fit[2]

            resid = marginalize(resid, f_tdv, g)

        
        # 4. Subtract ILC vectors with spectra
        if u_data is not None:
            f = _f_mu(u, q, Es, resid, vpm, g, check = False)
            for i in range(k):
                resid[i] -= multiply((f_vec(u, q, Es[i]), f, vpm[i]))

            if full_output:
                f_B = _fB(u, q, Es, vpm, g, check = False)

            # subtracting ILC vectors with spectra may introduce
            # components along f_td*v
            # here we remarginilize wrt f_td*v
            # doesn't seem to be necessary
            if f_data and False:
                if full_output:
                    for i in range(k):
                        u_fit = lstsq(resid[i], f_tdv[i], g[i],
                                       full_output = True)
                        u_td[i] += u_fit[0]

                resid = marginalize(resid, f_tdv, g)

                    
        
        # calculate the length of residual vector squared
        chi2 = np.zeros(k)
        #chi2_c = 0
        for i in range(k):
            dchi2 = multiply((resid[i], g[i], resid[i].conj().T)).real
            if dchi2 < 0:
                print '\tnegative norm of the vector in objf of lstsq_ILC, ',
                print 'bin ', i
                print '\t|v|^2 = ', dchi2
                #print resid[i], g[i]
                exit()
            chi2[i] = dchi2

        
        if collin_punish is not None:
            # add a term to chi2 to punish collinearity of ILC vectors
            f_all[:nv_mrg] = f_td[:]
            f_all[nv_mrg:] = f[:]
            col_punish = collin_punish(f_all)
            chi2 += col_punish
            if outsteps:
                print 'collin_punish ', col_punish

        
        if outsteps:
            print _param2pt(w_data = _w_data, u_data = _u_data,
                            f_data = _f_data), np.sum(chi2)

        if full_output:
            if nm_mrg > 0:
                # adjust the marginal template spectra
                resid = copy.deepcopy(d)
                if w_data:
                    for i in range(k):
                        resid[i] -= multiply((f_vec(w, p, Es[i]), m_sp[i]))

                for i in range(k):
                    w_fit = lstsq(resid[i], m_mrg[i], g[i],
                                   full_output = True)            
                    w_td[i] = w_fit[0]
                    w_td_B[i] = w_fit[2]                        
                        
                
            data_res = (w_td, p, f_td, u_td, f, q)
            B_res = (w_td_B, p_std, f_td_std, u_td_B, f_B, q_std)

            if chi2_in_bins:
                return data_res, chi2, B_res
            else:
                return data_res, sum(chi2), B_res
        else:
            return sum(chi2)
    

    # find best fit (p, q, f_td) using the obj function
    if fit_mode == 'fmin' and len(pt0) > 0:
        fit_pars = scipy.optimize.fmin(objf, pt0, full_output = True,
                                       maxiter = maxiter, maxfun = maxfun, ftol=1.e-2)
        fit_pt = fit_pars[0]

        w_data_std, u_data_std, f_data_std = \
                    _ptStd2parStd(None, w_data = w_data, u_data = u_data,
                                f_data = f_data)
        
    elif fit_mode == 'fmin_bfgs' and len(pt0) > 0:
        fit_pars = scipy.optimize.fmin_bfgs(objf, pt0, full_output = True, \
                                   maxiter = maxiter)

        fit_pt = fit_pars[0]

        std = np.sqrt(np.diagonal(fit_pars[3]))
        w_data_std, u_data_std, f_data_std = \
                    _ptStd2parStd(std, w_data = w_data, u_data = u_data,
                                  f_data = f_data)
    elif len(pt0) == 0:
        fit_pt = None
        
        

    w_data, u_data, f_data = \
            _pt2param(fit_pt, w_data = w_data, u_data = u_data,
                      f_data = f_data)


    # standard deviation around best fit point
    if fit_pt is not None:
        if w_data is not None: p_std = w_data_std[1]
        if u_data is not None: q_std = u_data_std[1]
        if f_data is not None: f_td_std = f_data_std[0]
        
    return objf(fit_pt, w_data = w_data, u_data = u_data, f_data = f_data,
                Es = Es, outsteps = outsteps, full_output = True,
                p_std = p_std, q_std = q_std, f_td_std = f_td_std,
                nm_perp = nm_perp, chi2_in_bins = chi2_in_bins)


def snapshot_pca(data, model=None, metric=None, \
               chi2_precision=0.1, outsteps=False):
    """
    find the principal component vector given the data vectors d,
    model vectors m and metric g

    Constants:
        k - number of energy bins,
        N - number of components in the vectors (number of pixels)
        m - number of vectors to form linear combinations, for usual PCA m = k 
        
    INPUT:
        data - array_like, shape (k, N)
            data vectors
        model - array_like, shape (k, m, N) or (m, N), optional
            model vectors to form linear combinations
            DEFAULT: None
            if model is none, then the usual PCA model is used:
            model[i] = data for i = 0..(k - 1)
            
        g - array_like, shape (k, N, N) or (k, N), optional
            metric, the shape (k, N) is for diagonal metric
            DEFAULT: None
            if g is None, then g equal to unit matrix is assumed
    OUTPUT:
        a tuple: (pars, chi2, stds, residual)
        pars = (v, f)
            v - array_like, shape (k, m),
                spectrum for the principal components
            f - array_like, shape (m, k),
                principal component vectors
        chi2 - best fit chi2
        stds = (v_std, f_std)
            v_std - array_like, shape (k, m),
                st dev of spectra
            f_std - array_like, shape (m,),
                st dev of vectors (sqrt of diagonal of the inverse hessian for f)
                
        residual - array_like, shape (k, N)
            data projected onto space prependicular
            to the principal component
            
    """

    #print metric[:3,:10]
    #print data[:3,:10]
    #print model[:3,:3,:10]
    #exit()
    
    # initialization
    if data.ndim != 2:
        print 'in snapshot_pca data.ndim != 2'
        exit()
  
    k, N = data.shape

    if model is None:
        m = k
        model = np.zeros((k, m, N), dtype=data.dtype)
        for i in range(k):
            model[i] = data[:]
    elif model.ndim != 2 or model.ndim != 3:
        print 'in snapshot_pca model.ndim != 2 or 3'
        exit()
    else:
        m = model.shape[1]

    if metric is None:
        metric = np.ones_like(data)
    elif metric.ndim != 2 and metric.ndim != 3:
        print 'in snapshot_pca metric.ndim != 2 or 3'
        exit()

    # the two steps of minimization

    def lstsq_spectrum(f):
        v = np.zeros(k)
        for i in range(k):
            lmodel = np.dot(f, model[i])
            v[i] = multiply((data[i], metric[i], lmodel.T))
            v[i] /= multiply((lmodel, metric[i], lmodel.T))
        return v

    def _inv_model_mat(modeltd):
        lhs_mat = np.zeros((m, m))
        for j1 in range(m):
            for j2 in range(m):
                for i in range(k):
                    lhs_mat[j1, j2] += multiply((modeltd[i, j1], metric[i], modeltd[i, j2]))
        
        return np.linalg.inv(lhs_mat)
        

    def lstsq_vector(v):
        modeltd = np.array([v[i]*model[i] for i in range(k)])

        rhs_vec = np.zeros(m)
        for j in range(m):
            for i in range(k):
                rhs_vec[j] += multiply((data[i], metric[i], modeltd[i, j]))
        
        lhs_mat_inv = _inv_model_mat(modeltd)

        f = np.dot(lhs_mat_inv, rhs_vec)
        f /= np.linalg.norm(f)
        return f
        

    
    # minimize chi2

    end = False
    chi2 = np.inf
    f = np.random.random(m)
    f /= np.linalg.norm(f)
    v = lstsq_spectrum(f)
    if outsteps:
            print 'f0: ', f
            print 'v0: ', v
    
    while not end:
        # find the vector
        f = lstsq_vector(v)
        v = lstsq_spectrum(f)

        dmodel = np.array([v[i] * np.dot(f, model[i]) for i in range(k)])

        resid = data - dmodel

        chi2_new = 0.
        for i in range(k):
            chi2_new += multiply((resid[i], metric[i], resid[i].T))

        if chi2 - chi2_new < chi2_precision:
            end = True

        chi2 = chi2_new

        if outsteps:
            print 'f: ', f
            print 'v: ', v
            print 'chi2 = ', chi2

    pars = (v, f)

    # find the standard deviations of the parameters

    # std of v
    lmodel = np.array([np.dot(f, model[i]) for i in range(k)])
    v_std = np.array([1./multiply((lmodel[i], metric[i], lmodel[i].T))
                         for i in range(k)])
    v_std = np.sqrt(v_std)

    # std of f
    modeltd = np.array([v[i]*model[i] for i in range(k)])
    f_std = np.diag(_inv_model_mat(modeltd))
    f_std = np.sqrt(f_std)
    
    stds = (v_std, f_std)

    
    return (pars, chi2, stds, resid)


#########################################################################
#                                                                       #
#           kernel PCA and related functions                            #
#                                                                       #
#########################################################################

def _get_sp_filter(spectrum, stds, marginal=True):
    '''
    compute the spectrum kernel

    Constants:
        k - number of energy bins
    
    INPUT:
        spectrum - array, shape (k,)
        stds - float or array with shape (k,)
            uncertainties in the prior spectrum
            if std = np.inf, then the prior is flat (unconstrained)
    OUTPUT:
        R - list of arrays [R2, R1, R0]
            R2 - array, shape (k, k): quadratic part of the kernel
            R1 - array, shape (k,): linear part of the kernel
            R0 - float: constant part of the kernel
            
    '''

    # check the dimensions of the spectrum
    if spectrum.ndim != 1:
        raise ValueError, "Wrong dimension of spectrum."
    k = len(spectrum)
    
    if not isinstance(stds, float):
        if stds.ndim != 1:
            raise ValueError, "Wrong dimensions of spectrum."

    # initialize an array of var = std**2
    var = np.ones_like(spectrum)*stds**2
    
    sp_norm = spectrum / var

    
    res2 = np.zeros((k, k))
    inv_var = 1 / var
    if sum(inv_var) > 0:
        res2 = np.diag(inv_var)        
        if marginal:
            num = np.outer(sp_norm, sp_norm)
            den = np.dot(sp_norm, spectrum)
            res2 -= num / den
            res1 = np.zeros(k)
            res0 = 0.
        else:
            res1 = -2 * sp_norm
            res0 = np.dot(sp_norm, spectrum)
    
    return [res2, res1, res0]


def get_spectrum_filter(spectra, stds, marginal=True):
    '''
    compute the spectrum kernel

    Constants:
        k - number of energy bins,
        m - number of principal components
    
    INPUT:
        spectra - array, shape (m, k)
        stds - float or array with shape (m,) or (m, k)
            uncertainties in the prior spectra,
            if std = np.inf, then the prior is flat (unconstrained)
        marginal - bool or a list of bool
            if True the spectrum is marginalized wrt overall normalizations
    OUTPUT:
        R - list of arrays [R2, R1, R0]
            R2 - array, shape (k, k, m, m): quadratic part of the kernel
            R1 - array, shape (k, m): linear part of the kernel
            R0 - float: constant part of the kernel
            
    '''

    # check the dimensions of the spectra
    if spectra.ndim != 1 and spectra.ndim != 2:
        raise ValueError, "Wrong dimensions of spectra."

    if spectra.ndim == 1:
        k = len(spectra)
        m = 1
    else:
        m, k = spectra.shape

    if isinstance(stds, float):
        stds = np.ones(m)*stds
    elif stds.ndim != 1 and stds.ndim != 2:
        raise ValueError, "Wrong dimensions of spectra."

    if type(marginal) is not list:
        mrg = [marginal] * m
    else:
        mrg = marginal

    res0 = 0
    res1 = np.zeros((k, m))
    res2 = np.zeros((k, k, m, m))
    for i in range(m):
        sp_filter = _get_sp_filter(spectra[i], stds[i], marginal=mrg[i])
        res2[:,:,i,i] = sp_filter[0]
        res1[:,i] = sp_filter[1]
        res0 += sp_filter[2]

    return [res2, res1, res0]


def get_gauge_filter(stds, r, m):
    '''
    calculate the gauge filter

    Constants:
        r - number of templates
        m - number of principal components
    
    INPUT:
        stds - float or array with shape (m,)
            uncertainties in gauge kernel
            usually, stds = 1 / sqrt(n), where n is the total number of photons

    OUTPUT:
        R - list of arrays [R2, R1, R0]
            R2 - array, shape (m, m, r, r): quadratic part of the kernel
            R1 - array, shape (m, r): linear part of the kernel
            R0 - float: constant part of the kernel
    
    '''
    if type(stds) is not np.ndarray:
        stds = np.ones(m) * stds

    invvar = 1/stds**2
    res2 = np.outer(np.diag(invvar), np.ones((r, r)))
    res2 = res2.reshape((m, m, r, r))
    res1 = -2 * np.outer(invvar, np.ones(r)).reshape((m, r))
    res0 = np.sum(invvar)

    return [res2, res1, res0]


def get_map_filter(templates, models, stds, marginal=True):
    '''
    compute the model map kernel

    Constants:
        r - number of templates
        N - number of pixels
        m - number of principal components
    
    INPUT:
        templates - array, shape (r, N)
        models - array, shape (N,) or (m, N)
        stds - float or array with shape (m,) or (m, N)
            uncertainties in the prior maps of emission components,
            if std = np.inf, then the prior is flat (unconstrained)
    OUTPUT:
        R - list of arrays [R2, R1, R0]
            R2 - array, shape (m, m, r, r): quadratic part of the kernel
            R1 - array, shape (m, r): linear part of the kernel
            R0 - float: constant part of the kernel
    
    '''
    r, N = templates.shape
    
    if models.ndim != 1 and models.ndim != 2:
        raise ValueError, "Wrong dimensions of models."
    
    if models.ndim == 1:
        m = 1
    else:
        m = models.shape[0]    
        
    if isinstance(stds, float):
        metric = np.ones_like(models) / stds**2
    elif stds.ndim == 1:
        if len(stds) != m:
            raise ValueError, "The dimensions of stds and models do not agree."

        metric = np.outer(1. / stds**2, np.ones(N))
        
    else:
        if stds.shape != models.shape:
            raise ValueError, "The dimensions of stds and models do not agree."
        metric = 1. / stds**2

    res2 = np.zeros((m, m, r, r))
    res1 = np.zeros((m, r))
    res0 = 0.
    for mu in range(m):
        if np.sum(metric[mu]) > 0:
            TT = np.array([[multiply((templates[i], metric[mu], templates[j]))
                            for i in range(r)]
                           for j in range(r)])

            TM = np.array([multiply((templates[i], metric[mu], models[mu]))
                           for i in range(r)])

            MM = multiply((models[mu], metric[mu], models[mu]))

            res2[mu,mu,:,:] = TT
            if marginal:
                res2[mu,mu,:,:] -= np.outer(TM, TM) / MM
            else:
                res1[mu,:] = -2 * TM
                res0 += MM
    return [res2, res1, res0]

def ridge_smooth_filter(templates, counts, stds, m):
    '''
    quadratic shrinking of linear coefficients

    Constants:
        r - number of templates
        N - number of pixels
        m - number of principal components
    
    INPUT:
        templates - array, shape (r, N)
        counts - array, shape (r, N)
        stds - float or array with shape (N,)
    OUTPUT:
        K - array, shape (m, m, r, r)
            kernel for the smoothing filter 
        
    '''
    r = templates.shape[0]

    if type(stds) is np.ndarray and stds.ndim == 2:
        res = np.zeros((m,m,r,r))
        for i in range(m):
            kernel = np.diag(np.sum(templates**2 / counts / stds[i]**2, axis=1))
            res[i, i] = kernel
            
    else:
        kernel = np.diag(np.sum(templates**2 / counts / stds**2, axis=1))
        res = np.outer(np.eye(m), kernel).reshape((m, m, r, r))
    
    return res

    

def get_smoothing_filter(diff_templates, stds, m=1):
    '''
    compute the model map kernel

    Constants:
        r - number of energy templates
        N - number of pixels
        m - number of principal components
    
    INPUT:
        diff_templates - array, shape (r, N)
        stds - float or array with shape (m,) or (m, N)
            uncertainties in the prior maps of emission components,
            if std = np.inf, then the prior is flat (unconstrained)
    OUTPUT:
        K - array, shape (m, m, r, r)
            kernel for the smoothing filter 
    
    
    '''

    r, N = diff_templates.shape  
        
    if isinstance(stds, float):
        metric = np.ones_like(diff_templates) / stds**2
    elif stds.ndim == 1:
        if len(stds) != m:
            raise ValueError, "The dimensions of stds and models do not agree."

        metric = np.outer(1. / stds**2, np.ones(N))
        
    else:
        #if stds.shape != models.shape:
        #    raise ValueError, "The dimensions of stds and models do not agree."
        metric = 1. / stds**2

    res = np.zeros((m, m, r, r))
    for mu in range(m):
        if np.sum(metric[mu]) > 0:
            res[mu,mu,:,:] = multiply((diff_templates, metric[mu], diff_templates.T))

    return res

def get_data_kernels(data, metric, templates):
    '''
    calculate scalar products of data with itself and with templates

    Constants:
        k - number of data vectors
        N - number of pixels
        r - number of templates, r = k for usual PCA
        
    INPUT:
        data - array, shape (k, N)
        metric - array, shape (k, N)
        templates - array, shape (r, N)

    OUTPUT:
        kernels - list of three arrays [(DD), (DT), (TT)]
        DD - array, shape (k, k)
        DT - array, shape (k, r)
        TT - array, shape (k, r, r)

    '''
    k, N = data.shape
    r = templates.shape[0]
    
    DD = np.array([multiply((data[i1], metric[i1], data[i1]))
               for i1 in range(k)])

    DT = np.array([[multiply((data[i1], metric[i1], templates[i2]))
                    for i2 in range(r)]
                   for i1 in range(k)])

    TT = np.array([[[multiply((templates[i2], metric[i1], templates[i3]))
                     for i3 in range(r)]
                    for i2 in range(r)]
                   for i1 in range(k)])

    return [DD, DT, TT]
    

def gf0(vec):
    return vec / np.linalg.norm(vec)

def kernel_pca(kernels,
               m=1,
               gauge_function=gf0,
               gauge_filter=None,
               map_filter=None,
               smoothing_filter=None,
               spectrum_filter=None,
               chi2_precision=0.1,
               outsteps=False,
               maxiter=100,
               full_output=False):
    """
    find the principal component vector given the data vectors d,
    model vectors m and metric g

    Constants:
        k - number of data vectors
        m - number of principal components
        r - number of templates, r = k for usual PCA
        
    INPUT:
        kernels - array (DD), shape (k, k) or (k, k, k)
                or list of three arrays [(DD), (DT), (TT)]
                with shapes [(k), (k, r), (k, r, r)]

        m - int, number of principal components
        
        gauge_filter, map_filter, smoothing_filter - lists of arrays [K2, K1, K0],
            K2 - array, shape (m, m, r, r) quadratic part
            K1 - array, shape (m, r) linear part
            K0 - float, constant part
        spectrum_filter - list of arrays [R2, R1, R0],
            R2 - array, shape (k, k, m, m) quadratic part
            R1 - array, shape (k, m) linear part
            R0 - float, constant part
            
        chi2_precision - float, precision of chi^2 calculation
        outsteps - if True: output the steps of calculation
        
    OUTPUT:
        a tuple: (pars, chi2, stds)
        pars = (U, F)
            U - array_like, shape (k, m),
                spectrum for the principal components
            F - array_like, shape (m, r),
                principal component vectors
        chi2 - best fit chi2
        stds = (U_std, F_std)
            U_std - array_like, shape (k, m),
                st dev of spectra
            F_std - array_like, shape (m, r),
                st dev of vectors

            
    """
    
    #########################################################################
    #                                                                       #
    #           initializations and checks                                  #
    #                                                                       #
    #########################################################################

    
    def _get_kernels(kernels):
        '''
        Get kernels out of TT in the case when the templates are the data vectors.
        '''
        if kernels.ndim != 3:
            raise ValueError, "Wrong number of dimensions in kernel."
        for i in kernels.shape[1:]:
            if i != kernels.shape[0]:
                raise ValueError, "Wrong dimensions of kernel."
            
        k = kernels.shape[0]
        r = k
    
        DD = kernels.diagonal().diagonal()
        DT = np.array([kernels[i,i] for i in range(k)])
        TT = kernels
        return DD, DT, TT
                
    # check the input data and assign kernels
    if isinstance(kernels, np.ndarray):
        DD, DT, TT = _get_kernels(kernels)
    elif isinstance(kernels, list):
        if len(kernels) != 1 and len(kernels) != 3:
            raise ValueError, "Wrong number of kernels."
        elif len(kernels) == 1:
            DD, DT, TT = _get_kernels(kernels[0])
        else:
            DD, DT, TT = kernels
            if DD.ndim != 1:
                raise ValueError, "Wrong number of dimensions of DD = kernels[0]."
            if DT.ndim != 2:
                raise ValueError, "Wrong number of dimensions of DT = kernels[1]."
            if TT.ndim != 3:
                raise ValueError, "Wrong number of dimensions of TT = kernels[2]."



    k = DD.shape[0]
    r = DT.shape[1]
    
    # check rest of dimensions
    if DT.shape[0] != k or TT.shape[0] != k or TT.shape[1] != r or TT.shape[2] != r:
        raise ValueError, "Wrong dimensions of kernels"

    # check the dimensions of the filters
    def _check_filter(xfilter, shape):
        if xfilter is not None:
            for i in range(len(xfilter)):
                if xfilter[i].shape != shape[i]:
                    raise ValueError, "The shape of the filter should be %s." % shape[i]
        return None


    #map_filter_shape = [(m, m, r, r), (m, r), (m,)]
    #_check_filter(map_filter, map_filter_shape)

    def reshape_filter(in_filter, d1=m, d2=r):
        '''
        reshape the filters in tensor form to matrices and vectors
        
        '''
        out_filter2 = np.zeros((d1 * d2, d1 * d2))
        out_filter1 = np.zeros(d1 * d2)
        out_filter0 = 0.
        if in_filter is not None:
            if type(in_filter) is list:
                out_filter0 = in_filter[2]
                out_filter1 = in_filter[1].reshape(d1 * d2)
                in_filter2 = in_filter[0]
            else:
                in_filter2 = in_filter
            
            for i in range(d1):
                for j in range(d1):
                    out_filter2[i*d2:(i+1)*d2, j*d2:(j+1)*d2] = in_filter2[i, j, :, :]
                    
        return out_filter2, out_filter1, out_filter0

    G_filter = reshape_filter(gauge_filter)
    F_filter = reshape_filter(map_filter)
    S_filter = reshape_filter(smoothing_filter)
    
    U_filter = reshape_filter(spectrum_filter, d1=k, d2=m)
    

        
    #########################################################################
    #                                                                       #
    #           minimization functions                                      #
    #                                                                       #
    #########################################################################

    

    # the two steps of minimization

    def lstsq_spectra(F, std_out=False):
        LHS_mat = np.zeros((k*m, k*m))
        if False and spectrum_filter is None:
            U = np.zeros((k, m))
            for i in range(k):
                LHS_mat_block = multiply((F, TT[i], F.T))
                RHS_vec = np.dot(DT[i], F.T)
                U[i] = np.dot(RHS_vec, np.linalg.inv(LHS_mat_block))
                LHS_mat[i*m:(i+1)*m, i*m:(i+1)*m] = LHS_mat_block[:]
        else:
            for i in range(k):
                LHS_mat[i*m:(i+1)*m, i*m:(i+1)*m] = multiply((F, TT[i], F.T))
            LHS_mat += U_filter[0]
            RHS_vec = np.array([np.dot(DT[i], F.T) for i in range(k)])
            RHS_vec = RHS_vec.flatten()

            RHS_vec -= 0.5 * U_filter[1]
            
            U = np.dot(RHS_vec, np.linalg.inv(LHS_mat)).reshape((k, m))

        if std_out:
            std = np.diag(np.linalg.inv(LHS_mat)).reshape((k, m))
            std = np.sqrt(std)
            return std
            
        else:
            return U
        

    def lstsq_vectors(U, std_out=False):
        LHS_mat = np.zeros((m*r, m*r))
        RHS_vec = np.zeros(m*r)
        for i in range(k):
            mat = np.kron(U[i], U[i]).reshape((m, m))
            LHS_mat += np.kron(mat, TT[i])
            
        LHS_mat += G_filter[0] + F_filter[0] + S_filter[0]

        for i in range(k):
            RHS_vec += np.kron(U[i], DT[i])

        RHS_vec -= 0.5 * (G_filter[1] + F_filter[1] + S_filter[1])
        
        

        LHS_mat_inv = np.linalg.inv(LHS_mat)
        F = np.dot(RHS_vec, LHS_mat_inv)
        F = F.reshape((m, r))
        #print np.sum(F, axis=1)

        # if there is no gauge filter, then apply the gauge condition 
        if gauge_filter is None:
            for i in range(m):
                F[i] = gauge_function(F[i])
                
        if std_out:
            std = np.diag(LHS_mat_inv).reshape((m, r))
            std = np.sqrt(std)
            return std
        else:
            return F
        
    def get_chi2(U, F, full_output=False):
        chi2_dict = {}
        
        chi2 = np.sum(DD)
        chi2 -= 2 * np.trace(multiply((DT, F.T, U.T)))
        for i in range(k):
            chi2 += multiply((U[i], F, TT[i], F.T, U[i]))
            
        chi2_dict['data'] = chi2
        
        def filter_chi2(vec, filters, in_filter):
            if in_filter is None:
                return 0.
            else:
                res = multiply((vec, filters[0], vec))
                res += np.dot(vec, filters[1])
                res += filters[2]
                return res

        F_flat = F.flatten()
        U_flat = U.flatten()
        
        chi2_dict['guage'] = filter_chi2(F_flat, G_filter, gauge_filter)
        chi2_dict['map'] = filter_chi2(F_flat, F_filter, map_filter)
        chi2_dict['smooth'] = filter_chi2(F_flat, S_filter, smoothing_filter)
        chi2_dict['spec'] = filter_chi2(U_flat, U_filter, spectrum_filter)
            
        if full_output:
            return chi2_dict
        else:
            return sum(chi2_dict.values())

    
    #########################################################################
    #                                                                       #
    #           minimization steps                                          #
    #                                                                       #
    #########################################################################


    end = False
    step_number = 0
    F = np.random.random((m, r))
    for i in range(m):
        F[i] = gauge_function(F[i])
    U = lstsq_spectra(F)


    chi2_dict = get_chi2(U, F, full_output=True)
    if outsteps:
        print 
        #print 'Kernel PCA fitting'
        #print
        print 'step ', step_number
        #print 'F: ', F
        #print 'U: ', U.T
        print chi2_dict
        print
        
    chi2 = sum(chi2_dict.values())
        


    while not end and step_number < maxiter:
        step_number += 1
        # find the vector
        F = lstsq_vectors(U)
        U = lstsq_spectra(F)

        chi2_new_dict = get_chi2(U, F, full_output=True)
        if outsteps:
            print 'step ', step_number
            #print 'F: ', F
            #print 'U: ', U.T
            print chi2_new_dict
            print
        chi2_new = sum(chi2_new_dict.values())
        
        if abs(chi2 - chi2_new) < chi2_precision:
            end = True

        chi2 = chi2_new

        

    pars = (U, F)

    all_chi2_dict = get_chi2(U, F, full_output=True)
    #print all_chi2_dict

    # find the standard deviations of the parameters

    # std of v
    F_std = lstsq_vectors(U, std_out=True)
    U_std = lstsq_spectra(F, std_out=True)
    
    stds = (U_std, F_std)

    if full_output:
        chi2 = all_chi2_dict
        
    return (pars, chi2, stds)


def em_pca(data, nc=1, metric=None, \
               chi2_precision=0.1, outsteps=False):
    """
    find the principal component vector given the data vectors d and metric g

    Constants:
        k - number of energy bins
        N - number of components in the vectors (number of pixels)
        
    INPUT:
        data - array_like, shape (k, N)
            data vectors

        nc - integer, optional
            number of principal components to compute
            DEFALT:  1
            
        g - array_like, shape (k, N, N) or (k, N), optional
            metric, the shape (k, N) is for diagonal metric
            DEFAULT: None
            if g is None, then g equal to unit matrix is assumed
    OUTPUT:
        a tuple: (pars, chi2, stds, residual)
        pars = (u, v)
            u - array_like, shape (k, nc),
                spectrum for the principal components
            v - array_like, shape (nc, N),
                principal component vectors
        chi2 - best fit chi2
        stds = (u_std, v_std)
            u_std - array_like, shape (k, nc),
                st dev of spectra
            v_std - array_like, shape (nc, N),
                st dev of vectors (sqrt of diagonal of the inverse hessian)
                
        residual - array_like, shape (k, N)
            data projected onto space prependicular
            to the principal component
            
    """

    #print metric[:3,:10]
    #print data[:3,:10]
    #print model[:3,:3,:10]
    #exit()
    
    # initialization
    if data.ndim != 2:
        print 'in em_pca data.ndim != 2'
        exit()
  
    k, N = data.shape

    if metric is None:
        metric = np.ones_like(data)
    elif metric.ndim != 2 and metric.ndim != 3:
        print 'in em_pca metric.ndim != 2, 3'
        exit()

    # the two steps of minimization

    def lstsq_spectra(v):
        u = np.zeros((k, nc))
        for i in range(k):
            DdotV = multiply((data[i], metric[i], v.T))
            VdotV = multiply((v, metric[i], v.T))
            u[i] = np.dot(np.linalg.inv(VdotV), DdotV)
        for i in range(nc):
            u[:,i] /= np.linalg.norm(u[:,i])
        return u

    
    def lstsq_vectors(u):
        v = np.zeros((nc, N))
        for i in range(N):
            UdotD = multiply((data[:,i], metric[:,i], u))
            UdotU = multiply((u.T, metric[:,i], u))
            v[:, i] = np.dot(np.linalg.inv(UdotU), UdotD)
        return v


    
    # minimize chi2
    if outsteps:
        print 
        print 'EM PCA fitting'
        print
        

    end = False
    chi2 = np.inf
    step_number = 0
    v = np.random.random((nc, N))
    
    while not end:
        step_number += 1
        # find the vector
        u = lstsq_spectra(v)
        v = lstsq_vectors(u)

        dmodel = np.dot(u, v)

        resid = data - dmodel

        chi2_new = 0.
        for i in range(k):
            chi2_new += multiply((resid[i], metric[i], resid[i].T))

        if abs(chi2 - chi2_new) < chi2_precision:
            end = True

        chi2 = chi2_new

        if outsteps:
            print 'step ', step_number
            print 'u: ', u.T
            print 'chi2 = %.3e' % chi2
            print

    pars = (u, v)

    # find the standard deviations of the parameters

    # std of v
    v_std = np.zeros((nc, N))
    if nc == 1:
        UdotU = 0.
        for i in range(k):
            UdotU += u[i, 0]**2 * metric[i]
        v_std[0] = 1 / np.sqrt(UdotU)
    else:
        for i in range(N):
            UdotU = multiply((u.T, metric[:,i], u))
            v_std[:, i] = np.sqrt(np.diag(np.linalg.inv(UdotU)))

    # std of u
    u_std = np.zeros((k, nc))
    for i in range(k):
        VdotV = multiply((v, metric[i], v.T))
        u_std[i] = np.sqrt(np.diag(np.linalg.inv(VdotV)))
    
    stds = (u_std, v_std)

    return (pars, chi2, stds, resid)



if __name__ == '__main__':
    """
    tests

    """
    pass
