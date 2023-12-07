# pylint: disable=invalid-name
"""Multiset Correlation and Factor Analysis

Contains functions for calculating correlated and private factors for
multiple high-dimensional datasets.

Usage:
  TODO(brielin): Add usage example
"""

import numpy as np
import pandas as pd
import torch
from torch import multiprocessing
from dataclasses import dataclass
from typing import List, Union, Iterable
from sklearn import model_selection
from sklearn import preprocessing
from multicor_fa import _em


# This is required or the pool code below hands on .join().
# force=True is required or I get an error that the context
# is already set. I have absolutely no idea why this works.
# https://pythonspeed.com/articles/python-multiprocessing/
multiprocessing.set_start_method('spawn', force=True)

@dataclass
class PCARes:
    """Simple dataclass for storing PCA results."""
    pcs: pd.DataFrame
    var_exp: pd.Series
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
    data_pcs: List[PCARes]
    Z: pd.DataFrame
    X: Iterable[pd.DataFrame]
    W: Iterable[pd.DataFrame]
    L: Iterable[pd.DataFrame]
    Phi: Iterable[pd.DataFrame]
    rho: pd.Series
    lam: Iterable[pd.Series]
    var_exp_Z: Iterable[pd.Series]
    var_exp_X: Iterable[pd.Series]
    l: List[float]
    cd: List[float]
    n_pcs: List[int]
    d: int
    k: List[int]
    center: bool
    scale: bool
    init: str
    maxit: int
    delta: float
    device: str
    rcond: float


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


def pca(X: pd.DataFrame, k: int = 'infer', center: bool = True,
        scale: bool = True, calc_V = True) -> PCARes:
    """Basic PCA implementation.

    This is a basic PCA implementation which is particularly efficient for
    top-k PCA by doing the eigendecomposition of either X X.T / N or
    X.T X / N.

    Args:
        X: n (samples) by d (features) pandas DataFrame.
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
    sample_names = X.index

    if center | scale:
        X = torch.from_numpy(preprocessing.scale(
            X, with_mean=center, with_std=scale))
    else:
        X = torch.from_numpy(X.values)
    N, D = X.shape
    mp_lower_bound = 1 + np.sqrt(D / N)

    gram_mat = (X @ X.T) / N if D > N else X.T @ X / N
    vals, vecs = torch.linalg.eigh(gram_mat)
    S = torch.sqrt(torch.abs(torch.flip(vals, dims = [0])))
    A = torch.flip(vecs, dims=[1])
    mp_dim = sum(S > mp_lower_bound)

    # TODO(brielin): Throw error if infer and not center/scale.
    if k in ('infer', 'all'):
        k = mp_dim if k == 'infer' else min(N, D)
    S_k = S[0:k]

    if D > N:
        U = A[:, 0:k]
        V = torch.eye(k, dtype=torch.double)
        if calc_V:
            V = X.T @ U / (S_k * np.sqrt(N))
    else:
        V = A[:, 0:k]
        U = X @ V / (S_k * np.sqrt(N))
        V = V if calc_V else torch.eye(k, dtype=torch.double)

    pc_names = ['PC' + str(i+1) for i in range(k)]
    pcs = pd.DataFrame((U * S_k * np.sqrt(N)).numpy(), index=sample_names,
                       columns=pc_names)
    var_exp = pd.Series(S_k**2 / torch.sum(S**2), index=pc_names)
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


# TODO(brielin): Double check that
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


def _orthogonalize(X):
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    return U*S


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
        Y = [pd.DataFrame(np.random.normal(size=(N, p_m))) for p_m in p]
        Y_pcs = [pca(Y_m, 'all') for Y_m in Y]
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho = torch.linalg.eigvalsh(UTU)
        sim_res.append(torch.max(rho))
    sim_res = torch.Tensor(sim_res)
    return sim_res.mean(), np.sqrt(sim_res.var()/nsims)

def score(data, Z, transform = True):
    """Calculates (transformed) correlations between model factors and data.

    Args:
      data: A pandas DataFrame with features as columns. The original data.
      Z: The factor set used to calculate gene statistics. Usually mcfa_res.Z.
      transform: bool. True to transform correlations to Z-scores.
    Returns:
      A pd.DataFrame with rows as data features and columns as factors, entires
      are (optionally Z-transformed) correlations.
    """
    n = Z.shape[0]
    # Note: this cannot be done in pytorch because it does not get
    #  along with concurrent.futures.
    cors = preprocessing.scale(data).T.dot(preprocessing.scale(Z)) / n
    if transform:
        cors = np.sqrt(n-3) * np.arctanh(cors)
    cors = pd.DataFrame(cors, index=data.columns)
    return cors


def _cv_one_iter(
        indices, Y: Iterable[pd.DataFrame],
        Z: pd.DataFrame, X: Iterable[pd.DataFrame],
        n_pcs: Union[str, List[int]] = 'infer',
        d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
        center: bool = True, scale: bool = True, init: str = 'avgvar',
        maxit: int = 1000, delta: float = 1e-6,
        device = 'cpu', rcond: float = 1e-8, verbose: bool = False):
    # TODO(brielin): At the moment, this is assuming that we're doing
    #   pc-based analysis. Some modification is required if we aren't.
    (it, (train_idx, test_idx)) = indices
    n_train = len(train_idx)
    n_test = len(test_idx)
    if verbose: print(it, n_train, n_test)

    if isinstance(Y, dict):
        ds_names = Y.keys()
        Y = list(Y.values())
        X = list(X.values())

    Y_train = [Y_m.iloc[train_idx] for Y_m in Y]
    Y_test = [Y_m.iloc[test_idx] for Y_m in Y]
    if center | scale:
        scalers = [preprocessing.StandardScaler(with_mean=center, with_std=scale)
                   for _ in Y_train]
        scalers = [scaler.fit(Y_train_m)
                   for scaler, Y_train_m in zip(scalers, Y_train)]
        Y_train = [pd.DataFrame(scaler.transform(Y_tr_m),
                                index=Y_tr_m.index, columns=Y_tr_m.columns)
                   for scaler, Y_tr_m in zip(scalers, Y_train)]
        Y_test = [pd.DataFrame(scaler.transform(Y_te_m),
                                index=Y_te_m.index, columns=Y_te_m.columns)
                   for scaler, Y_te_m in zip(scalers, Y_test)]

    cv_res = fit(Y_train, n_pcs=n_pcs, d=d, k=k,
                 center=True, scale=True,
                 init=init, maxit=maxit,
                 delta=delta, device=device,
                 rcond=rcond, result_space = 'pc', verbose=verbose)
    Z_tr = torch.from_numpy(cv_res.Z.values)
    X_tr = [torch.from_numpy(X_m_tr.values) for X_m_tr in cv_res.X]


    # The held out data needs to be projected into the PC space learned
    #   from the training data.
    # TODO(brielin): does this work with non-informative analysis?
    Y_test_pcs = [torch.from_numpy(Y_m.values) @ pca_m.V
                  for Y_m, pca_m in zip(Y_test, cv_res.data_pcs)]
    W_tens = [torch.from_numpy(W_m.values) for W_m in cv_res.W]
    L_tens = [torch.from_numpy(L_m.values) for L_m in cv_res.L]
    Phi_tens = [torch.from_numpy(Phi_m.values) for Phi_m in cv_res.Phi]
    Z_te, X_te = _em.get_latent(W_tens, L_tens, Phi_tens, torch.cat(Y_test_pcs, axis=1),
                               device, rcond)
    Y_test_hat = [(Z_te @ W_m.T @ pca_m.V.T + X_m_te @ L_m.T @ pca_m.V.T).numpy()
                  for X_m_te, L_m, W_m, pca_m in zip(X_te, L_tens, W_tens, cv_res.data_pcs)]

    # Y_train = [torch.from_numpy(pca_m.pcs.values) for pca_m in cv_res.data_pcs]
    Y_train_pcs = [torch.from_numpy(Y_m.values) @ pca_m.V
                   for Y_m, pca_m in zip(Y_train, cv_res.data_pcs)]
    Y_train_hat = [(Z_tr @ W_m.T @ pca_m.V.T + X_m_tr @ L_m.T  @ pca_m.V.T).numpy()
                   for X_m_tr, L_m, W_m, pca_m in zip(X_tr, L_tens, W_tens, cv_res.data_pcs)]

    nrmse_tr = [np.sqrt(((Y_m - Y_m_hat)**2/Y_m.var(0)).values.mean())
                for Y_m, Y_m_hat in zip(Y_train, Y_train_hat)]
    nrmse_te = [np.sqrt(((Y_m - Y_m_hat)**2/Y_m_tr.var(0)).values.mean())
                for Y_m, Y_m_hat, Y_m_tr in zip(Y_test, Y_test_hat, Y_train)]

    if verbose:
        print(nrmse_tr, nrmse_te)

    # The CV res Z and X signs might not be aligned with the original.
    Z_signs = np.sign(np.corrcoef(Z.iloc[train_idx].T, cv_res.Z.T).diagonal(Z.shape[1]))
    X_signs = [np.sign(np.corrcoef(X_m.iloc[train_idx].T, X_m_cv.T).diagonal(X_m.shape[1]))
               for X_m, X_m_cv in zip(X, cv_res.X)]
    Z_te = Z_te * Z_signs
    X_te = [X_m_te * X_m_signs for X_m_te, X_m_signs in zip(X_te, X_signs)]
    return Z_te, X_te, nrmse_tr, nrmse_te



def cv(Y: Iterable[pd.DataFrame], mcfa_res: MCFARes,
       folds: Union[str, int] = 10, threads: [int] = 1,
       verbose: bool = False):
    """Checks for over-fitting using k-fold cross validation.

    Args:
      Y: Iterable of N (samples) by p_m (features) pandas DataFrames, the
        M=len(Y) datasets to analyze.
      mcfa_res: An MCFAres dataclass fit from Y.
      folds: Integer, number of folds. Or 'loo' for leave-one-out.
      threads: Integer, number of paralell folds to run.
    Returns:
      A tuple Z, X with each entry formed by removing that row, fitting the
      MCFA model, and projecting that sample's Y into the space fit without
      it.
    """
    if folds == 'loo':
        folds = mcfa_res.Z.shape[0]
    elif isinstance(folds, str):
        raise NotImplementedError

    cv_iter = model_selection.KFold(n_splits=folds)
    results = []
    X_hat = []
    Z_hat = []
    nrmse_tr = []
    nrmse_te = []
    if threads > 0:
        raise NotImplementedError
        # with multiprocessing.get_context('spawn').Pool(threads) as pool:
        #     for indices in enumerate(cv_iter.split(mcfa_res.Z)):
        #         results.append(pool.apply_async(_cv_one_iter, (
        #             indices, Y, mcfa_res.Z, mcfa_res.X, mcfa_res.n_pcs,
        #             mcfa_res.d, mcfa_res.k, mcfa_res.center,
        #             mcfa_res.scale, mcfa_res.init, mcfa_res.maxit, mcfa_res.delta,
        #             mcfa_res.device, mcfa_res.rcond, verbose)))
        #     pool.close()
        #     pool.join()
    else:
        for indices in enumerate(cv_iter.split(mcfa_res.Z)):
            results.append(_cv_one_iter(
                indices, Y, mcfa_res.Z, mcfa_res.X, mcfa_res.n_pcs,
                mcfa_res.d, mcfa_res.k, mcfa_res.center,
                mcfa_res.scale, mcfa_res.init, mcfa_res.maxit, mcfa_res.delta,
                mcfa_res.device, mcfa_res.rcond, verbose))

    for res in results:
        Z_res, X_res, nrmse_tr_res, nrmse_te_res = res.get() if threads > 0 else res
        X_hat.append(X_res)
        Z_hat.append(Z_res)
        nrmse_tr.append(nrmse_tr_res)
        nrmse_te.append(nrmse_te_res)

    X_hat = [np.concatenate(X_m, 0) for X_m in map(list, zip(*X_hat))]
    Z_hat = np.concatenate(Z_hat, 0)

    ds_names = None
    if isinstance(Y, dict):
        ds_names = Y.keys()

    Z_names = ['Z' + str(i+1) for i in range(mcfa_res.d)]
    if ds_names is not None:
        X_names = [['X' + str(i+1) + '_' + name for i in range(k_m)]
                   for name, k_m in zip(ds_names, mcfa_res.k)]
    else:
        X_names = [['X' + str(i+1) + '_' + str(m+1) for i in range(k_m)]
                   for m, k_m in enumerate(mcfa_res.k)]
    Z_hat = pd.DataFrame(Z_hat, index=mcfa_res.Z.index, columns=Z_names)
    X_hat = [pd.DataFrame(X_m, index=mcfa_res.Z.index, columns=names)
         for X_m, names in zip(X_hat, X_names)]
    nrmse_tr = pd.DataFrame(
        np.array(nrmse_tr), index=['fold_' + str(i) for i in range(folds)])
    nrmse_te = pd.DataFrame(
        np.array(nrmse_te), index=['fold_' + str(i) for i in range(folds)])

    if ds_names is not None:
        X_hat = dict(zip(ds_names, X_hat))
        nrmse_tr.columns = ds_names
        nrmse_te.columns = ds_names

    return Z_hat, X_hat, nrmse_tr, nrmse_te

# TODO(brielin): set delta=NULL to just use maxit.
# TODO(brielin): EM is broken if you don't center.
def fit(Y: Iterable[pd.DataFrame], n_pcs: Union[str, List[int]] = 'infer',
        d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
        center: bool = True, scale: bool = True, init: str = 'avgvar',
        result_space: str = 'full', maxit: int = 1000, delta: float = 1e-6,
        device = 'cpu', rcond: float = 1e-8, verbose: bool = True):
    """Interface function to the MCFA estimators.

    Args:
        Y: Iterable of N (samples) by p_m (features) pandas DataFrames, the
          M=len(Y) datasets to analyze. If a dictionary, keys will be used
          as names in the results.
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
        result_space: Either 'full' or 'pc' (for informative analyses only).
          If 'full', weight matrices (W, L) will be transformed back to
          the observed space (gene features). If 'pc', weight matrices will
          remain in pc space (pc features). Note that noise matrices (phi)
          are left untransformed because the full pxp noise matrix can be
          enormous.
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

    if isinstance(result_space, str) & (result_space not in ['pc', 'full']):
        raise NotImplementedError(
            'n_pcs must be "infer", "all" or a list of integers.')


    ds_names = None
    if isinstance(Y, dict):
        ds_names = Y.keys()
        Y = list(Y.values())
    sample_names = [Y_m.index for Y_m in Y]
    feature_names = [Y_m.columns for Y_m in Y]
    common_samples = sample_names[0]
    for names in sample_names[1:]:
        common_samples = common_samples.intersection(names)
    N = len(common_samples)
    if any(Y_m.shape[0] > N for Y_m in Y):
        print('WARNING: there are {0:d} samples in common across the datasets.'
              ' Data will be filtered to just these samples'.format(N))
        Y = [Y_m.loc[common_samples] for Y_m in Y]

    if isinstance(n_pcs, List):
        if len(n_pcs) != len(Y):
            raise ValueError(
                'Length of PC list does not match number of datasets.')

    if isinstance(k, List):
        if len(k) != len(Y):
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

    if verbose: print('Calculating data PCs.')

    Y_pcs = [pca(Y_m, n_pc_m, center, scale)
             for Y_m, n_pc_m in zip(Y, n_pcs)]
    if informative and result_space == 'pc':
        feature_names = [['pc_' + str(k+1) for k in range(Y_pc.k)]
                         for Y_pc in Y_pcs]

    if informative:
        p = [pc.k for pc in Y_pcs]
        n_pcs = p
        Y_all = torch.cat([torch.from_numpy(pc.pcs.values)
                           for pc in Y_pcs], axis=1)
    else:
        p = [Y_m.shape[1] for Y_m in Y]
        n_pcs = None
        if center | scale:
            Y_all = torch.cat(
                [torch.from_numpy(preprocessing.scale(
                    Y_m, with_mean=center, with_std=scale)) for Y_m in Y],
                axis = 1)
        else:
            Y_all = torch.cat([torch.from_numpy(Y_m.values) for Y_m in Y],
                              axis=1)
    if verbose: print('Calculating exmpirical covariance.')
    Sigma_hat = Y_all.T @ Y_all / N
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    p_all = sum(p)

    # TODO(brielin): This is doing a little extra work/memory if d == 'infer'
    #   and init = 'avgvar' (the default). Also note that _init_ methods may
    #   no longer need to return rho and ppca may no longer need to return vals.
    if verbose: print('Initialzing model.')
    if d == 'all':
        d = p_all
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

    if verbose: print('Fitting the model.')
    W, L, Phi, l, cd = _em.fit_EM_iter(
        Y_all, Sigma_hat, W0, L0, Phi0, maxit, device, rcond, delta, verbose)
    rho = _em.calculate_rho(W, L, Phi, Y_all, device, rcond, 'genvar')
    rho, order = torch.sort(rho, descending=True)

    W = [W_m[:, order] for W_m in W]
    Z, X = _em.get_latent(W, L, Phi, Y_all, device, rcond)

    if verbose: print('Calculating feature importance.')
    if informative and (result_space == 'full'):
        W = [pc_m.V @ W_m for W_m, pc_m in zip(W, Y_pcs)]
        L = None if L is None else [pc_m.V @ L_m for L_m, pc_m in zip(L, Y_pcs)]

    Z_names = ['Z' + str(i+1) for i in range(d)]
    if ds_names is not None:
        X_names = [['X' + str(i+1) + '_' + name for i in range(k_m)]
                   for name, k_m in zip(ds_names, k)]
    else:
        X_names = [['X' + str(i+1) + '_' + str(m+1) for i in range(k_m)]
                   for m, k_m in enumerate(k)]
    Z = pd.DataFrame(Z.numpy(), index=common_samples, columns=Z_names)
    X = [pd.DataFrame(X_m.numpy(), index=common_samples, columns=names)
         for X_m, names in zip(X, X_names)]
    W = [pd.DataFrame(W_m.numpy(), index=names, columns=Z_names)
         for W_m, names in zip(W, feature_names)]
    rho = pd.Series(rho, index=Z_names)
    Phi = [pd.DataFrame(phi.numpy()) for phi in Phi]

    if scale:
        var_exp_Z = [(W_m**2).sum(0)/W_m.shape[0] for W_m in W]
    else:
        var_exp_Z = [(W_m**2).sum(0)/sum(Y_m.var(0))
                     for W_m, Y_m in zip(W, Y)]
    var_exp_Z = pd.concat(var_exp_Z, axis=1)

    var_exp_X = None
    lam = None
    if L is not None:
        L = [pd.DataFrame(L_m.numpy(), index=ind_names, columns=col_names)
             for L_m, ind_names, col_names in zip(L, feature_names, X_names)]
        lam = [(L_m**2).sum(0) for L_m in L]
        if scale:
            var_exp_X = [l/L_m.shape[0] for l, L_m in zip(lam, L)]
        else:
            var_exp_X = [l/sum(Y_m.var(0)) for l, Y_m in zip(lam, Y)]

    if ds_names is not None:
        X = dict(zip(ds_names, X))
        Y_pcs = dict(zip(ds_names, Y_pcs))
        W = dict(zip(ds_names, W))
        L = dict(zip(ds_names, L))
        Phi = dict(zip(ds_names, Phi))
        lam = dict(zip(ds_names, lam))
        var_exp_Z.columns = ds_names
        var_exp_X = dict(zip(ds_names, var_exp_X))

    return MCFARes(Y_pcs, Z, X, W, L, Phi, rho, lam, var_exp_Z, var_exp_X, l,
                   cd, n_pcs, d, k, center, scale, init, maxit, delta,
                   device, rcond)
