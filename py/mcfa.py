# pylint: disable=invalid-name
"""Multiset Correlation and Factor Analysis

Contains functions for calculating correlated and private factors for
multiple high-dimensional datasets.

Usage:
  TODO(brielin): Add usage example
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Union
from sklearn import preprocessing
from MPCCA.py import em


@dataclass
class PCARes:
    """Simple dataclass for storing PCA results."""
    pcs: torch.Tensor
    var_exp: torch.Tensor
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor
    k: int
    n: int
    d: int
    mp_dim: int


@dataclass
class MCFARes:
    """Simple dataclass for storing MCFA results."""
    Z: torch.tensor
    X: List[torch.Tensor]
    W: List[torch.Tensor]  # Or the concatenation?
    L: List[torch.Tensor]  # ^
    Phi: List[torch.Tensor]  # ^
    lev_W: torch.Tensor
    lev_L: torch.Tensor
    rho: torch.Tensor
    lam: torch.Tensor
    l: List[float]
    cd: List[float]


def pca_transform(X: torch.Tensor, k: int = 'infer', center: bool = True,
        scale: bool = True):
    """Transforms data to k-dimensional space with identity covariance.

    Use this if you have a wide (p > N) matrix and you only need whitened
    points in the reduces space (an N by k matrix). For the full
    inference procedure use mccfa.pca().

    Args:
        X: Two-dimesional (N x p) torch.tensor.
        k: Integer or 'infer'. Number of pcs to keep. Default is to
          infer using the marchenko pasteur cutoff.
        center: bool. True to mean-center columns of X.
        scale: bool. True to variance-scale the columns of X to 1.0.
    Returns:
        An N by k matrix with X.T @ X / N = I_k.
    """
    N, D = X.shape
    if N > D: raise ValueError('Matrix is tall.')

    if center | scale:
        X = torch.from_numpy(preprocessing.scale(
            X, with_mean=center, with_std=scale))
    mp_lower_bound = 1 + np.sqrt(D / N)

    gram_mat = (X @ X.T) / N
    vals, vecs = torch.linalg.eigh(gram_mat)
    S = torch.sqrt(torch.abs(torch.flip(vals, dims = [0])))
    A = torch.flip(vecs, dims=[1])
    mp_dim = sum(S > mp_lower_bound)

    if k == 'infer': k = mp_dim
    U = A[:, 0:k]
    return np.sqrt(N) * U


def pca(X: torch.tensor, k: int = 'infer', center: bool = True,
        scale: bool = True, calc_V = True) -> PCARes:
    """Basic PCA implementation.

    This is a basic PCA implementation which is particularly efficient for
    top-k PCA by doing the eigendecomposition of either X X.T / N or
    X.T X / N.

    Args:
        X: Two-dimesional (n x d) torch.tensor.
        k: Integer, 'infer' or 'all'. Number of pcs to keep. Default is to
          infer using the marchenko pasteur cutoff.
        center: bool. True to mean-center columns of X.
        scale: bool. True to variance-scale the columns of X to 1.0.
        calc_V: True to track the PC loadings (right singular vectors)
          of X. Setting to False can save substantial memory if X is very
          wide.
    Returns:
        A PCARes instance.
    """
    if center | scale:
        X = torch.from_numpy(preprocessing.scale(
            X, with_mean=center, with_std=scale))
    N, D = X.shape
    mp_lower_bound = 1 + np.sqrt(D / N)

    gram_mat = (X @ X.T) / N if D > N else X.T @ X / N
    vals, vecs = torch.linalg.eigh(gram_mat)
    S = torch.sqrt(torch.abs(torch.flip(vals, dims = [0])))
    A = torch.flip(vecs, dims=[1])
    mp_dim = sum(S > mp_lower_bound)

    if k in ('infer', 'all'):
        k = mp_dim if k == 'infer' else min(N, D)
    S_k = S[0:k]

    if D > N:
        U = A[:, 0:k]
        V = X.T @ U / (S_k * np.sqrt(N)) if calc_V else torch.eye(k, dtype=torch.double)
    else:
        V = A[:, 0:k]
        U = X @ V / (S_k * np.sqrt(N))
        V = V if calc_V else torch.eye(k, dtype=torch.double)

    pcs = U * S_k * np.sqrt(N)
    var_exp = S_k**2 / torch.sum(S**2)
    return PCARes(pcs, var_exp, U, S_k, V, k, N, D, mp_dim)


def _ppca(X, d):
    """Produces a facorization X = W @ W.T + s2*I for symmetric X.

    Args:
      X: symmetric p x p matrix.
      d: number of dimensions to keep.
    """
    vals, vecs = torch.linalg.eigh(X)
    vals = torch.abs(torch.flip(vals, dims = [0]))
    vecs = torch.flip(vecs, dims=[1])
    W = vecs[:, 0:d] * torch.sqrt(vals[0:d])
    s2 = torch.mean(vals[d:])
    return W, s2, vals


def _init_var_W(Y_pcs, psum, d, informative):
    """Initializes W using sumcor with avgvar constraint.

    Args:
      Y_pcs: List of PCARes objects.
      psum: List of break indices for inidivudal datasets.
      d: Number of components to keep.
      informative: True to keep W in PC spaces, False to return
        to original data space.
    """
    U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
    UTU = U_all.T @ U_all

    W, _, vals = _ppca(UTU, d)
    W = [(W[i:j, :].T * pc.S).T for i, j, pc in zip(psum[:-1], psum[1:], Y_pcs)]
    if informative is False: W = [pc.V @ W for pc, W in zip(Y_pcs, W)]
    return W, vals


def _init_norm_W(Sigma_hat, psum, d, M):
    """Initializes W using sumcor with avgnorm constraint.

    Args:
      Sigma_hat: Cross correlation matrix to model.
      d: Number of components to keep.
      M: Number of datasets.
    """
    W, _, _ = _ppca(Sigma_hat, d)
    W = [W[i:j, :] for i, j in zip(psum[:-1], psum[1:])]
    rho = sum([(W_m**2).sum(0) for W_m in W])
    return W, rho


def _init_L_Phi(Sigma_hat, W, psum, p, k):
    """Initializes L and Phi for a given W, Sigma_hat.

    Args:
      Sigma_hat: Cross correlation matrix to model.
      W: LIST
      psum: List of break indices for inidivudal datasets.
      p: List of integers, dimensions of datasets.
      k: List of integers or None, dimensions of private spaces.
    """
    Phi = [Sigma_hat[i:j, i:j] - W_m @ W_m.T
           for W_m, i, j in zip(W, psum[:-1], psum[1:])]

    L = None
    if k is not None:
        resid_pcas = [_ppca(Phi_m, k_m) for k_m, Phi_m in zip(k, Phi)]
        L, s2s, _ = list(map(list, zip(*resid_pcas)))
        Phi = [torch.diag(torch.tensor([s2]*p_m)) for s2, p_m in zip(s2s, p)]
    return L, Phi


def _calc_var_exp(Y, Z):
    """Calculates the variance in Y explained by Z.

    Args:
      Y: N by p data matrix.
      Z: N by k data matrix, columns have unit variance.
    Returns:
      A k-vector with the variance in Y explained by each feature in Z.
    """
    N, _ = Y.shape
    beta = Z.T @ Y
    varY = sum(torch.var(Y, 0))
    varB = torch.sum(beta**2, 1)/N**2
    return varB/varY


def _rho_mp_sim(N: int, p: List[int], nsims=100, device='cpu'):
    """Calculates the MCCA (Parra) solution to random data.

    Args:
      N: Integer. Sample size.
      p: List of integers. Dimensions of the datasets to simulate.
      nsims: Number of simulation iterations.
      device: Device to run on.
    """
    sim_res = []
    for _ in range(nsims):
        Y = [torch.randn(N, p_m) for p_m in p]
        Y_pcs = [pca(Y_m, 'all') for Y_m in Y]
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho = torch.linalg.eigvalsh(UTU)
        sim_res.append(torch.max(rho))
    sim_res = torch.Tensor(sim_res)
    return sim_res.mean(), np.sqrt(sim_res.var()/nsims)


def mcfa(Y: List[torch.Tensor], n_pcs: Union[str, List[int]] = 'infer',
         d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
         center: bool = True, scale: bool = True, init: str = 'avgvar',
         calc_leverages: bool = True, maxit: int = 1000, delta: float = 1e-6,
         device = 'cpu', rcond: float = 1e-8, verbose: bool = True):
    """Interface function to the MCFA estimators.

    Args:
        Y: List of N (samples) by p_m (features) torch tensors, the
          M=len(Y) datasets to analyze.
        center: Bool. True to mean-center columns of Y.
        scale: Bool. True to variance-scale columns of Y to 1.0.
        n_pcs: 'infer', 'all' or a list of length M of integers. The number
          of PCs of each dataset to keep. 'all' does not whiten/PCA data
          prior to modeling, 'infer' uses the Marchenko-Pasteur
          cutoff to choose PCs. A list of integers
          specifies the number of PCs to keep from each dataset.
        d: 'infer', 'all' or integer. Dimensionality of the hidden space.
          If 'infer' a simulation will be done to determine the number
          of correlated components to keep.
        k: 'infer', None, or list of integers. Number of private components
          to model per dataset. If 'infer', k is set to n_pcs - d. If None,
          no private components will be modeled.
        init: Either 'avgvar', 'avgnorm' or 'random'. Initialization
          strategy. 'avgvar' (default) maximizes the sum of correlations
          with a global mahalalanobis constraint (eg Parra 2019), 'avgnorm'
          does the same with a global euclidean constraint (eg Seurat),
          and random samples W from a std normal while setting Phi to I. Note
          that if n_pcs is not 'all', the model is fit explicitly to the PCs
          of each dataset and therefore the 'avgvar' and 'avgnorm'
          initializations are equivalent.
        calc_leverages: bool or list of bools, one per dataset. True to
          propagate inference back to the original data space, rather than
          remaining in PC space. Setting this to False for particularly
          wide datasets that you aren't interested in the individual features
          loadings for can save substantial memory.
        maxit: Maximum number of iterations of the pgm to run. Set to 0
          to run none and return only the initial solution.
        delta: Float, convergance tolerance for EM.
        rcond: Float, zero tolerance for least squares routines.
        verbose: Bool. True to print progress.
    Returns:
        an MCFARes instance.
    Raises:
        NotImplementedError: if a TODO feature is called.
        ValueError: if the input data matrices have a different number
          of rows.
    """
    if isinstance(n_pcs, str) & (n_pcs not in ['infer', 'all']):
        raise NotImplementedError(
            'n_pcs must be "infer", "all" or a list of integers.')

    if isinstance(d, str) & (d not in ['infer', 'all']):
        raise NotImplementedError(
            'd must be "infer", "all" or an integer.')

    if isinstance(k, str) & (k not in ['infer', 'all']):
        raise NotImplementedError(
            'k must be "infer", "all" or a list of integers.')

    if init not in ['avgvar', 'avgnorm', 'random']:
        raise NotImplementedError(
            'Implemented initializers are avgnorm, avgvar, and random.')

    N = Y[0].shape[0]
    if any(Y_m.shape[0] != N for Y_m in Y):
        raise ValueError(
            'Input data matrices must have an equal number of samples (rows).')

    if isinstance(n_pcs, List):
        if len(n_pcs) != len(Y):
            raise ValueError(
                'Length of PC list does not match number of datasets.')

    if isinstance(k, List):
        if len(n_pcs) != len(Y):
            raise ValueError(
                'Length of private list does not match number of datasets.')

    # TODO(brielin): this needs to be a list if n_pcs is allowed to have
    #   individual entries be 'all' so only some datasets are processed
    #   with informative PCs.
    informative = (n_pcs == 'infer')  | isinstance(n_pcs, List)

    M = len(Y)
    if n_pcs == 'all':
        n_pcs = ['all']*M
    elif n_pcs == 'infer':
        n_pcs = ['infer']*M

    if calc_leverages is True:
        calc_leverages = [True]*M
    elif calc_leverages is False:
        calc_leverages = [False]*M

    if verbose: print('Calculating data PCs.')

    Y_pcs = [pca(Y_m, n_pc_m, center, scale, lev_m)
             for Y_m, n_pc_m, lev_m in zip(Y, n_pcs, calc_leverages)]

    if informative:
        p = [pc.k for pc in Y_pcs]
        Y_all = torch.cat([pc.pcs for pc in Y_pcs], axis=1)
        # Y_all = torch.cat([np.sqrt(N) * pc.U for pc in Y_pcs], axis=1)
    else:
        p = [Y_m.shape[1] for Y_m in Y]
        if center | scale:
            Y_all = torch.cat(
                [torch.from_numpy(preprocessing.scale(
                    Y_m, with_mean=center, with_std=scale)) for Y_m in Y],
                axis = 1)
        else:
            Y_all = torch.cat(Y, axis=1)
    if verbose: print('Calculating exmpirical covariance.')
    Sigma_hat = Y_all.T @ Y_all / N
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    p_all = sum(p)

    # TODO(brielin): This is doing a little extra work/memory if d == 'infer'
    #   and init = 'avgvar' (the default). Also note that _init_ methods may
    #   no longer neeed to return rho and ppca may no longer ned to return vals.
    if verbose: print('Initialzing model.')
    if d == 'all': d = p_all
    elif d == 'infer':
        if verbose: print('Inferring the shared dimensionality.')
        rho_min, _ = _rho_mp_sim(N, p)
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho0 = torch.linalg.eigvalsh(UTU)
        d = sum(rho0 > rho_min)
        if verbose:
            print(('There are {} components above rho' +
                   ' inclusion threshold {}.').format(d, rho_min))

    if init == 'random':
        W0 = [torch.randn((p_m, d)).double() for p_m in p]
    elif init == 'avgnorm':
        W0, _ = _init_norm_W(Sigma_hat, psum, d, M)
    elif init == 'avgvar':
        W0, _  = _init_var_W(Y_pcs, psum, d, informative)

    if k == 'infer':
        k = [pca_m.mp_dim - d for pca_m in Y_pcs]

    if init == 'random':
        L0 = None if k is None else [
            torch.randn((p_m, k_m)).double() for k_m, p_m in zip(k, p)]
        Phi0 = [torch.eye(p_m) for p_m in p]
    else:
        L0, Phi0 = _init_L_Phi(Sigma_hat, W0, psum, p, k)

    # TODO(brielin): consider orthogonalizing W, L and/or reordering from
    #   initial order.
    if verbose: print('Fitting the model.')
    W, L, Phi, l, cd = em.fit_EM_iter(
        Y_all, Sigma_hat, W0, L0, Phi0, maxit, device, rcond, delta, verbose)
    Z, X = em.get_latent(W, L, Phi, Y_all, device, rcond)

    if verbose: print('Calculating feature importance and leverage scores.')
    if informative:
        W = [pc_m.V @ W_m for W_m, pc_m in zip(W, Y_pcs)]
        L = None if L is None else [pc_m.V @ L_m for L_m, pc_m in zip(L, Y_pcs)]
        Phi = [pc_m.V @ Phi_m @ pc_m.V.T for Phi_m, pc_m in zip(Phi, Y_pcs)]

    lev_W = [(W_m**2).sum(1) for W_m in W]
    lev_L = None if L is None else [(L_m**2).sum(1) for L_m in L]

    rho = sum([(W_m**2).sum(0) for W_m in W])
    lam = None if L is None else [(L_m**2).sum(0) for L_m in L]

    return MCFARes(Z, X, W, L, Phi, lev_W, lev_L, rho, lam, l, cd)
