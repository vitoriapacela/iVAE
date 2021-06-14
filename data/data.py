"""
Script for generating piece-wise stationary data.

Each component of the independent latents is comprised of `ns` segments, and each segment has different parameters.\
Each segment has `nps` data points (measurements).

The latent components are then mixed by an MLP into observations (not necessarily of the same dimension).
It is possible to add noise to the observations
"""

import os

import numpy as np
import scipy
import torch
from scipy.stats import hypsecant
from torch.utils.data import Dataset

torch.set_default_dtype(torch.double)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        
def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
        """
        one dimensional implementation of leaky ReLU
        """
        if _x > 0:
            return _x
        else:
            return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


# def generate_mixing_matrix(d_sources: int, d_data=None, lin_type='uniform', cond_threshold=25, n_iter_4_cond=None,
#                            dtype=np.double, staircase=False):
#     # old: dtype=np.float32
#     """
#     Ilyes' code, do not use! 
#     Generate square linear mixing matrix. Not used in simple iVAEs (not used by Vitoria).
#     @param d_sources: dimension of the latent sources
#     @param d_data: dimension of the mixed data
#     @param lin_type: specifies the type of matrix entries; either `uniform` or `orthogonal`.
#     @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
#     @param n_iter_4_cond: or instead, number of iteration to compute condition threshold of the mixing matrix.
#         cond_threshold is ignored in this case/
#     @param dtype: data type for data
#     @param staircase: if True, generate mixing that preserves staircase form of sources
#     @return:
#         A: mixing matrix
#     @rtype: np.ndarray
#     """

#     def _gen_matrix(ds, dd, dtype):
#         A = (np.random.uniform(0, 2, (ds, dd)) - 1).astype(dtype)
#         for i in range(dd):
#             A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#         return A

#     def _gen_matrix_staircase(ds, dd, dtype, sq=None):
#         if sq is None:
#             sq = dd > 2
#         A1 = np.zeros((ds, 1))  # first row of A should be e_1
#         A1[0, 0] = 1
#         A2 = np.random.uniform(0, 2, (ds, dd - 1)) - 1
#         if sq:
#             A2[0] = 0
#         A = np.concatenate([A1, A2], axis=1).astype(dtype)
#         for i in range(dd):
#             A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#         return A

#     if d_data is None:
#         d_data = d_sources

#     if lin_type == 'orthogonal':
#         A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))[0]).astype(dtype)

#     elif lin_type == 'uniform':
#         if n_iter_4_cond is None:
#             cond_thresh = cond_threshold
#         else:
#             cond_list = []
#             for _ in range(int(n_iter_4_cond)):
#                 A = np.random.uniform(-1, 1, (d_sources, d_data)).astype(dtype)
#                 for i in range(d_data):
#                     A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#                 cond_list.append(np.linalg.cond(A))

#             cond_thresh = np.percentile(cond_list, 25)  # only accept those below 25% percentile

#         gen_mat = _gen_matrix if not staircase else _gen_matrix_staircase
#         A = gen_mat(d_sources, d_data, dtype)
#         while np.linalg.cond(A) > cond_thresh:
#             A = gen_mat(d_sources, d_data, dtype)

#     else:
#         raise ValueError('incorrect method')
#     return A


def generate_simple_mixing_matrix_by_percentile(d_sources: int, d_data=None, n_iter_4_cond=1e4, dtype=np.double, mix_bounds=np.array([-1, 1]), percentile=25):
    '''
    Generate a mixing matrix by first estimating the condition number threshold.
    
    @param int d_sources: dimension of the latent sources
    @param int d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param np.ndarray mix_bounds: upper and lower bounds of the Uniform distribution to sample the mixing matrix
    (only active when simple_mixing == True)
    
    @return:
        mixing matrix
    @rtype: np.ndarray
    '''
    def _gen_matrix(ds, dd, dtype):
        '''
        Auxiliary function to generate a mixing matrix.
        '''
        A = (np.random.uniform(0, 2, (dd, ds)) -1).astype(dtype)
        return A
    
    if d_data is None:
        d_data = d_sources
        
    cond_list = []
    for _ in range(int(n_iter_4_cond)):
        A = np.random.uniform(mix_bounds[0], mix_bounds[1], (d_data, d_sources)).astype(dtype)
        cond_list.append(np.linalg.cond(A))

    # First, find the condition number threshold
    # by generating n_iter_4_cond matrices and selecting the lowest 25% percentile
    cond_thresh = np.percentile(cond_list, percentile)  # only accept those below 25% percentile
    
    # Then generate a matrix that has condition number below such threshold
    A = _gen_matrix(d_sources, d_data, dtype)
    while np.linalg.cond(A) > cond_thresh:
        A = _gen_matrix(d_sources, d_data, dtype)
    return A
    

def generate_segment_mean_var(n_per_seg: int, n_seg: int, d: int, std_bounds=np.array([0.5, 3]), dtype=np.double, uncentered=False, centers=None, staircase=False, m_bounds=np.array([-5,5]), same_var=False):
    # old: dtype=np.float32
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n_per_seg: (int) number of points per segment
    @param n_seg: (int) number of segments
    @param d: (int) dimension of the sources
    @param std_bounds: (numpy.ndarray, dim=2) optional, upper and lower bounds for the modulation parameter - standard deviation from U(lower, upper)
    @param dtype: (type) data type for data
    @param uncentered: (bool) True to generate uncentered data
    @param centers: (np.ndarray) if uncentered, pass the desired centers to this parameter. If None, the centers will be drawn
                    at random
    @param staircase: (bool) if True, s_1 will have a staircase form, used to break TCL.
    @param m_bounds: (numpy.ndarray, dim=2) optional, lower and upper bounds for the means sampled from U(lower, upper)
    @param same_var: (bool) if True, the modulation parameter, or variance of each segment, is fixed to 1.
    @return:
        m: mean of each component
        L: modulation parameter of each component
    @rtype: (np.ndarray, np.ndarray)
    """
    var_lb = std_bounds[0]
    var_ub = std_bounds[1]

    # variances
    if not same_var:
        L = np.random.uniform(var_lb, var_ub, (n_seg, d))
    else:
        L = np.ones((n_seg, d))
    
    # means
    if uncentered:
        if centers is not None:
            assert centers.shape == (n_seg, d)
            m = centers
        else:
            m = np.random.uniform(m_bounds[0], m_bounds[1], (n_seg, d))
    else:
        m = np.zeros((n_seg, d))

    if staircase:
        m1 = 3 * np.arange(n_seg).reshape((-1, 1))
        a = np.random.permutation(n_seg)
        m1 = m1[a]
        # L[:, 0] = .2
        if uncentered:
            m2 = np.random.uniform(-1, 1, (n_seg, d - 1))
        else:
            m2 = np.zeros((n_seg, d - 1))
        m = np.concatenate([m1, m2], axis=1)

    return m, L


def generate_nonstationary_sources(n_per_seg: int, n_seg: int, d: int, m, L, prior='gauss', dtype=np.double):
    # old: dtype=np.float32
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n_per_seg: (int) number of points per segment
    @param n_seg: (int) number of segments
    @param d: (int) dimension of the sources, same as data
    @param prior: (str) distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    @param dtype: (type) data type for data
    @param m: (np.ndarray) segment means of each component
    @param L: (np.ndarray) segment variances of each component

    @return:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
    @rtype: (np.ndarray, np.ndarray)
    """
    n = n_per_seg * n_seg

    # initial sources
    if prior == 'lap':
        sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
        sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
    elif prior == 'gauss':
        sources = np.random.randn(n, d).astype(dtype)
    else:
        raise ValueError('incorrect dist')

    # segments U
    labels = np.zeros(n, dtype=dtype)
    
    for seg in range(n_seg):
        segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
        sources[segID] *= L[seg]
        sources[segID] += m[seg]
        labels[segID] = seg

    return sources, labels


# def generate_nonstationary_sources(n_per_seg: int, n_seg: int, d: int, prior='gauss', var_bounds=np.array([0.5, 3]), dtype=np.double, uncentered=False, centers=None, staircase=False, m_bounds=np.array([-5,5]), same_var=False):
#     # old: dtype=np.float32
#     """
#     Generate source signal following a TCL distribution. Within each segment, sources are independent.
#     The distribution withing each segment is given by the keyword `dist`
#     @param n_per_seg: (int) number of points per segment
#     @param n_seg: (int) number of segments
#     @param d: (int) dimension of the sources, same as data
#     @param prior: (str) distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
#     @param var_bounds: (numpy.ndarray, dim=2) optional, upper and lower bounds for the modulation parameter - variance U(lower, upper)
#     @param dtype: (type) data type for data
#     @param uncentered: (bool) True to generate uncentered data
#     @param centers: (np.ndarray) if uncentered, pass the desired centers to this parameter. If None, the centers will be drawn
#                     at random
#     @param staircase: (bool) if True, s_1 will have a staircase form, used to break TCL.
#     @param m_bounds: (numpy.ndarray, dim=2) optional, lower and upper bounds for the means sampled from U(lower, upper)
#     @param same_var: (bool) if True, the modulation parameter, or variance of each segment, is fixed to 1.
#     @return:
#         sources: output source array of shape (n, d)
#         labels: label for each point; the label is the component
#         m: mean of each component
#         L: modulation parameter of each component
#     @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
#     """
#     var_lb = var_bounds[0]
#     var_ub = var_bounds[1]
#     n = n_per_seg * n_seg

#     # initial sources
#     if prior == 'lap':
#         sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
#     elif prior == 'hs':
#         sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
#     elif prior == 'gauss':
#         sources = np.random.randn(n, d).astype(dtype)
#     else:
#         raise ValueError('incorrect dist')

#     # variances
#     if not same_var:
#         L = np.random.uniform(var_lb, var_ub, (n_seg, d))
#     else:
#         L = np.ones((n_seg, d))
    
#     # means
#     if uncentered:
#         if centers is not None:
#             assert centers.shape == (n_seg, d)
#             m = centers
#         else:
#             m = np.random.uniform(m_bounds[0], m_bounds[1], (n_seg, d))
#     else:
#         m = np.zeros((n_seg, d))

#     if staircase:
#         m1 = 3 * np.arange(n_seg).reshape((-1, 1))
#         a = np.random.permutation(n_seg)
#         m1 = m1[a]
#         # L[:, 0] = .2
#         if uncentered:
#             m2 = np.random.uniform(-1, 1, (n_seg, d - 1))
#         else:
#             m2 = np.zeros((n_seg, d - 1))
#         m = np.concatenate([m1, m2], axis=1)

#     # segments U
#     labels = np.zeros(n, dtype=dtype)
    
#     for seg in range(n_seg):
#         segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
#         sources[segID] *= L[seg]
#         sources[segID] += m[seg]
#         labels[segID] = seg

#     return sources, labels, m, L


def generate_data(n_per_seg, n_seg, d_sources, d_data=None, n_layers=3, prior='gauss', activation='lrelu', batch_size=0,
                  seed=10, slope=.1, std_bounds=np.array([0.5, 3]), lin_type='uniform', n_iter_4_cond=1e4,
                  dtype=np.double, noisy=0, uncentered=False, centers=None, staircase=False, discrete=False,
                  one_hot_labels=True, repeat_linearity=False, simple_mixing=True, mix_bounds=np.array([-1, 1]), m_bounds=np.array([-5,5]), identity=False, cond_thresh=None, same_var=False, norm_A_data=False, c=1.0, norm_logl=False, norm_prior_mean=False, diag=False, percentile=25, sparse=False):
    # old: dtype=np.float32
    """
    Generate artificial data with arbitrary mixing
    @param int n_per_seg: number of observations per segment
    @param int n_seg: number of segments
    @param int d_sources: dimension of the latent sources
    @param int or None d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param str activation: activation function for the mixing MLP; can be `none, `lrelu`, `xtanh` or `sigmoid`
    @param str prior: prior distribution of the sources; can be `lap` for Laplace or `hs` for Hypersecant
    @param int batch_size: batch size if data is to be returned as batches. 0 for a single batch of size n
    @param int seed: random seed
    @param np.ndarray std_bounds: upper and lower bounds for the modulation parameter
    @param float slope: slope parameter for `lrelu` or `xtanh`
    @param str lin_type: specifies the type of matrix entries; can be `uniform` or `orthogonal`
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param float noisy: if non-zero, controls the level of noise added to observations
    @param bool uncentered: True to generate uncentered data
    @param np.ndarray centers: array of centers if uncentered == True
    @param bool staircase: if True, generate staircase data
    @param bool one_hot_labels: if True, transform labels into one-hot vectors
    @param bool simple_mixing: if True, have elements of mixing matrix from a Uniform distribution \
    and skip all the other mixing code
    @param np.ndarray mix_bounds: upper and lower bounds of the Uniform distribution to sample the mixing matrix
    (only active when simple_mixing == True)
    @param np.ndarray m_bounds: upper and lower bounds of the Uniform distribution to sample the means
    @param bool identity: if True, the mixing matrix A=I (for debugging purposes)
    @param int cond_thresh: condition number threshold. Generate a new mixing matrix until its condition number is <= cond_thresh.
    @param bool same_var: if True, the modulation parameter, or variance of each segment, is fixed to 1.
    @param bool norm_A_data: if True, normalize the mixing matrix in the data generation.
    @param double c: multiplying constant for mixing matrix.
    @param bool norm_logl: if True, normalize the prior log-variance (logl) in the data generation.
    @param bool norm_prior_mean: if True, normalize the prior means in the data generation.
    @param bool diag: if True, create a diagonal mixing matrix.
    
    @return:
        tuple of batches of generated (sources, data, auxiliary variables, mean, variance, mixing matrix)
    @rtype: tuple

    By default, estimate a good condition number by sampling multiple matrices.
    """
    if seed is not None:
        np.random.seed(seed)

    if d_data is None:
        d_data = d_sources

    # moved the mixing matrix sampling up (before generating sources)
    # so that the random seed is comparable when different means are used
    if simple_mixing:
        if identity:
            if d_data != d_sources:
                raise ValueError("Cannot have identity if the matrix is not square.")
            A = np.identity(d_sources).astype(dtype)
            
        elif diag:
            if d_data != d_sources:
                raise ValueError("We want to create a square diagonal matrix.") # symmetric
            d = np.random.randn(d_sources) # sample diagonal elements from standard normal
            A = np.diag(d).astype(dtype)
            
        elif sparse:
            # to be used exceptionally
            A = np.identity(5)
            A[0,4] = 1
            A[4,0] = 1
            A[1,0] = 1
            A[0,1] = 1
            A = A.astype(dtype)
            
        else:
            if cond_thresh is not None:
                A = np.random.uniform(mix_bounds[0], mix_bounds[1], (d_data, d_sources)).astype(dtype)
                while np.linalg.cond(A) > cond_thresh:
                    A = np.random.uniform(mix_bounds[0], mix_bounds[1], (d_data, d_sources)).astype(dtype)
            else:
                A = generate_simple_mixing_matrix_by_percentile(d_sources=d_sources, d_data=d_data, n_iter_4_cond=n_iter_4_cond, dtype=dtype, mix_bounds=mix_bounds, percentile=percentile)
    
    if norm_A_data:
        # Normalize mixing matrix (divide by norm column-wise)
        A = A / np.linalg.norm(A, axis=0) 
        A = c * A
    
    # Segment means and variances
    M, L = generate_segment_mean_var(n_per_seg=n_per_seg, n_seg=n_seg, d=d_sources, std_bounds=std_bounds, dtype=dtype, uncentered=uncentered, centers=centers, staircase=staircase, m_bounds=m_bounds, same_var=same_var)
    
    if norm_logl:
        # careful, this normalizes the stds, not the log-variances (logl)
#         L = L / np.linalg.norm(L, axis=0) # column-wise. Do NOT use!
#         L = (L.T / np.linalg.norm(L, axis=1)).T # row-wise
        
#         # to normalize the variances (row-wise):
#         V = L**2
#         V = (V.T / np.linalg.norm(V, axis=1)).T
#         L = np.sqrt(V)
        
        ## Usually use this!
        # to normalize log-variances (logl), row-wise: 
        logl = np.log(L**2)
        logl = (logl.T / np.linalg.norm(logl, axis=1)).T
        L = np.sqrt(np.exp(logl))
        
#         # Frobenius norm (all matrix):
#         logl = np.log(L**2)
#         logl = logl / np.linalg.norm(logl)
#         L = np.sqrt(np.exp(logl))
        
    if norm_prior_mean:
#         M = M / np.linalg.norm(M, axis=0) # column-wise. Do not use.
        M = (M.T / np.linalg.norm(M, axis=1)).T # row-wise
    
#         # Frobenius norm (all matrix):
#         M = M / np.linalg.norm(M)
    
    # sources
    S, U = generate_nonstationary_sources(n_per_seg=n_per_seg, n_seg=n_seg, d=d_sources, m=M, L=L, dtype=dtype)
    n = n_per_seg * n_seg

# Notice: we don't want to standarize (subtract the mean). Normalize the variance could be fine.
#     if norm_z:
#         # Standardize sources before mixing
#         S = S/np.std(S, axis=0)
    
    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        act_f = lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    # Mixing time!

    if simple_mixing:
        # linear mixing, mixes only once
#         X = np.dot(S, A) # this does the transpose of A !!!

        X = A @ S.T
        X = X.T
        
#     else:
#         if not repeat_linearity:
#             X = S.copy()
#             for nl in range(n_layers):
#                 A = generate_mixing_matrix(X.shape[1], d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype, staircase=staircase)
#                 if nl == n_layers - 1:
#                     X = np.dot(X, A)
#                 else:
#                     X = act_f(np.dot(X, A))

#         else:
#             assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
#             A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
#             X = act_f(np.dot(S, A))
#             if d_sources != d_data:
#                 B = generate_mixing_matrix(d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
#             else:
#                 B = A
#             for nl in range(1, n_layers):
#                 if nl == n_layers - 1:
#                     X = np.dot(X, B)
#                 else:
#                     X = act_f(np.dot(X, B))
                    

    # add noise:
    if noisy:
        X += noisy * np.random.randn(*X.shape)

    if discrete:
        X = np.random.binomial(1, sigmoid(X))
        
#         # make labels -1 and 1 instead of 0 and 1
#         X[np.where(X==0)] = -1
        
    if not batch_size:
        if one_hot_labels:
            U = to_one_hot([U], m=n_seg)[0]
            
        # if U is a vector, transform it in a matrix, so that aux_dim=1
        try:
            U.shape[1]
        except:
            U = np.expand_dims(U, axis=1)
        return S, X, U, M, L, A
    
    else:
        idx = np.random.permutation(n)
        Xb, Sb, Ub, Mb, Lb = [], [], [], [], []
        n_batches = int(n / batch_size)
        for c in range(n_batches):
            Sb += [S[idx][c * batch_size:(c + 1) * batch_size]]
            Xb += [X[idx][c * batch_size:(c + 1) * batch_size]]
            Ub += [U[idx][c * batch_size:(c + 1) * batch_size]]
            Mb += [M[idx][c * batch_size:(c + 1) * batch_size]]
            Lb += [L[idx][c * batch_size:(c + 1) * batch_size]]
        if one_hot_labels:
            Ub = to_one_hot(Ub, m=n_seg)
            
    # if U is a vector, transform it in a matrix, so that aux_dim=1
    try:
        U.shape[1]
    except:
        U = np.expand_dims(U, axis=1)
        
        return Sb, Xb, Ub, Mb, Lb, A


def save_data(path, dtype=np.double, *args, **kwargs):
    # old: dtype=np.float32
    """
    Generate data and save it.
    :param str path: path where to save the data.
    """
    kwargs['batch_size'] = 0  # leave batch creation to torch DataLoader
    S, X, U, M, L, A = generate_data(*args, **kwargs)
    
    print('Creating dataset {} ...'.format(path))
    
    dir_path = '/'.join(path.split('/')[:-1])
    
    if not os.path.exists(dir_path):
        os.makedirs('/'.join(path.split('/')[:-1]))
    
    # Split datasets for training, validation, and testing
    len_data = len(S)
    len_seg = len(U)
    len_m = len(M)

    S_train = S
    S_val, U_val = generate_nonstationary_sources(n_per_seg=kwargs["n_per_seg"], n_seg=kwargs["n_seg"], d=kwargs["d_sources"], m=M, L=L, dtype=dtype)
    S_test, U_test = generate_nonstationary_sources(n_per_seg=kwargs["n_per_seg"], n_seg=kwargs["n_seg"], d=kwargs["d_sources"], m=M, L=L, dtype=dtype)
    
    X = X.astype(dtype)
    X_train = X
    
    U_train = U
    
    # We need to have the same mean and variance as in the training set
    M_train = M
    
    L_train = L
    
    X_val = A @ S_val.T
    X_val = X_val.T
    
    X_test = A @ S_test.T
    X_test = X_test.T
    
    if kwargs['noisy']:
        X_val += kwargs['noisy'] * np.random.randn(*X_val.shape)
        X_test += kwargs['noisy'] * np.random.randn(*X_test.shape)

    if kwargs['discrete']:
        X_val = np.random.binomial(1, sigmoid(X_val))
        X_test = np.random.binomial(1, sigmoid(X_test))
        
        X_val = X_val.astype(dtype)
        X_test = X_test.astype(dtype)
    
#     np.savez_compressed(path, s=S, x=X, u=U, m=M, L=L, A=A)
        
    path_train = path+'_train'
    path_val = path+'_val'
    path_test = path+'_test'
        
    # U_train = U_val = U_test should hold when all the sets have the same size
    # this avoids dtype and shape operations
    np.savez_compressed(path_train+'.npz', s=S_train, x=X_train, u=U_train, m=M_train, L=L_train, A=A)
    np.savez_compressed(path_val+'.npz', s=S_val, x=X_val, u=U_train, m=M_train, L=L_train, A=A)
    np.savez_compressed(path_test+'.npz', s=S_test, x=X_test, u=U_train, m=M_train, L=L_train, A=A)
    
    print(' ... done')


class SyntheticDataset(Dataset):
    def __init__(self, root, nps, ns, dl, dd, nl, s, p, a, uncentered=False, noisy=False, centers=None, double=False, one_hot_labels=False, simple_mixing=True, which='train', discrete=False, identity=False, m_bounds=np.array([-5,5]), cond_thresh=None, same_var=False, norm_A_data=False, norm_logl=False, norm_prior_mean=False, std_bounds=np.array([0.5, 3]), diag=False, percentile=25):
        '''
        :param which: (str) 'train' or 'val' or 'test'
        '''
        self.root = root
        
        data = self.load_tcl_data(root, nps, ns, dl, dd, nl, s, p, a, uncentered, noisy, centers, one_hot_labels, which, discrete, identity, m_bounds, cond_thresh, same_var, norm_A_data, norm_logl, norm_prior_mean, std_bounds, diag, percentile)
        
        self.data = data
        self.s = torch.from_numpy(data['s'])
        self.x = torch.from_numpy(data['x'])
        self.u = torch.from_numpy(data['u'])
        self.l = data['L']
        self.m = data['m']
        self.A_mix = data['A']
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.prior = p
        self.activation = a
        self.seed = s
        self.n_layers = nl
        self.uncentered = uncentered
        self.noisy = noisy
        self.double = double
        self.one_hot_labels = one_hot_labels
        self.cond_thresh = cond_thresh
        self.same_var=same_var
        self.norm_A_data = norm_A_data
        self.norm_logl = norm_logl
        self.norm_prior_mean = norm_prior_mean
        self.std_bounds = std_bounds
        self.diag = diag
        self.percentile = 25
#         self.norm_z = norm_z

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.double:
            return self.x[index], self.u[index], self.s[index]
        else:
            indices = range(len(self))
            index2 = np.random.choice(indices)
            return self.x[index], self.x[index2], self.u[index], self.s[index]

    @staticmethod
    def load_tcl_data(root, nps, ns, dl, dd, nl, s, p, a, uncentered, noisy, centers, one_hot_labels, which, discrete, identity, m_bounds, cond_thresh, same_var, norm_A_data, norm_logl, norm_prior_mean, std_bounds, diag, percentile):
        '''
        Try to load data (tcl data simply refers to the nonstationarity, but is used in the iVAE) if it exists.
        If not, create a new dataset and save it.
        '''
        path_to_dataset = root + 'tcl_' + '_'.join(
            [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
        if uncentered:
            path_to_dataset += '_u'
        if noisy:
            path_to_dataset += '_noisy'
        if one_hot_labels:
            path_to_dataset += '_one_hot'

        if identity:
            path_to_dataset = path_to_dataset + '_identity'
            
        if diag:
            path_to_dataset = path_to_dataset + '_diag'
            
        if uncentered:    
            path_to_dataset = path_to_dataset + '_m' + str(m_bounds[1])
            
        # std_bounds:
        path_to_dataset = path_to_dataset + '_v' + str(std_bounds[0]) + '_' + str(std_bounds[1])
        
        if discrete:
            path_to_dataset = path_to_dataset + '_discrete'
            
        if same_var:
            path_to_dataset = path_to_dataset + '_var1'
            
        if norm_A_data:
            path_to_dataset = path_to_dataset + '_norm_A'
            
        if norm_logl:
            path_to_dataset = path_to_dataset + '_norm_logl'
            
        if norm_prior_mean:
            path_to_dataset = path_to_dataset + '_norm_prior_mean'
            
#         if norm_z:
#             path_to_dataset = path_to_dataset + '_normz'
        
        # train or val or test
        file_name = path_to_dataset+'_'+which
            
        file_name = file_name+'.npz'
        
        if not os.path.exists(file_name) or s is None:
            # if the path is not found or if the seed is not defined, create a new dataset
            kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                      "activation": a, "seed": s, "batch_size": 0, "uncentered": uncentered, "noisy": noisy,
                      "centers": centers, "repeat_linearity": True, "one_hot_labels": one_hot_labels, "discrete": discrete, "identity": identity, "m_bounds": m_bounds, "cond_thresh":cond_thresh, "same_var":same_var, "norm_A_data":norm_A_data, "norm_logl": norm_logl, "norm_prior_mean": norm_prior_mean, "std_bounds": std_bounds, "diag": diag, "percentile": percentile}
            
            save_data(path_to_dataset, **kwargs)
        
        print('loading data from {}'.format(file_name))
        
        return np.load(file_name)

    def get_test_sample(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.randint(max(0, self.len - batch_size))
        return self.x[idx:idx + batch_size], self.u[idx:idx + batch_size], self.s[idx:idx + batch_size]

    
# Do not use! Not needed so far (except for TCL) and configs would need to be updated
class CustomSyntheticDataset(Dataset):
    def __init__(self, X, U, S=None, A=None, m=None, L=None, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(self.device)
        self.u = torch.from_numpy(U).to(self.device)
        if S is not None:
            self.s = torch.from_numpy(S).to(self.device)
        else:
            self.s = self.x
        if A is not None:
            self.A_mix = torch.from_numpy(A).to(self.device)
        if m is not None:
            self.m = torch.from_numpy(m).to(self.device)
        if L is not None:
            self.l = torch.from_numpy(m).to(self.device)
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.nps = int(self.len / self.aux_dim)

        print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.u[index], self.s[index]
    
#     def __getitem__(self, index):
#         if not self.double:
#             return self.x[index], self.u[index], self.s[index]
#         else:
#             indices = range(len(self))
#             index2 = np.random.choice(indices)
#             return self.x[index], self.x[index2], self.u[index], self.s[index]
    

    def get_metadata(self):
        return {'nps': self.nps,
                'ns': self.aux_dim,
                'n': self.len,
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'aux_dim': self.aux_dim,
                }
    
    def get_test_sample(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.randint(max(0, self.len - batch_size))
        return self.x[idx:idx + batch_size], self.u[idx:idx + batch_size], self.s[idx:idx + batch_size]

# Do not use! Not needed so far (except for TCL) and configs would need to be updated
def create_if_not_exist_dataset(root='data/', nps=1000, ns=40, dl=2, dd=4, nl=3, s=1, p='gauss', a='xtanh',
                                uncentered=False, noisy=False, arg_str=None):
    """
    Create a dataset if it doesn't exist.
    This is useful as a setup step when running multiple jobs in parallel, to avoid having many scripts attempting
    to create the dataset when non-existent.
    This is called in `cmd_utils.create_dataset_before`
    """
    if arg_str is not None:
        # overwrites all other arg values
        # arg_str should be of this form: nps_ns_dl_dd_nl_s_p_a_u_n
        arg_list = arg_str.split('\n')[0].split('_')
        print(arg_list)
        assert len(arg_list) == 10
        nps, ns, dl, dd, nl = map(int, arg_list[0:5])
        p, a = arg_list[6:8]
        if arg_list[5] == 'n':
            s = None
        else:
            s = int(arg_list[5])
        if arg_list[-2] == 'f':
            uncentered = False
        else:
            uncentered = True
        if arg_list[-1] == 'f':
            noisy = False
        else:
            noisy = True

    path_to_dataset = root + 'tcl_' + '_'.join(
        [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
    if uncentered:
        path_to_dataset += '_u'
    if noisy:
        path_to_dataset += '_n'
    path_to_dataset += '.npz'

    if not os.path.exists(path_to_dataset) or s is None:
        kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                  "activation": a, "seed": s, "batch_size": 0, "uncentered": uncentered, "noisy": noisy}
        save_data(path_to_dataset, **kwargs)
    return path_to_dataset
