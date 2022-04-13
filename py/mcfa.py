# pylint: disable=invalid-name
"""Multiset Correlation and Factor Analysis

Contains functions for calculating correlated and private factors for
multiple high-dimensional datasets.

Usage:
  TODO(brielin): Add usage example
"""

import concurrent.futures
import functools
import numpy as np
import time
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Union
from scipy import stats
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
    var_exp_Z: List[torch.Tensor]
    var_exp_L: List[torch.Tensor]
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
        Y = [torch.randn(N, p_m) for p_m in p]
        Y_pcs = [pca(Y_m, 'all') for Y_m in Y]
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho = torch.linalg.eigvalsh(UTU)
        sim_res.append(torch.max(rho))
    sim_res = torch.Tensor(sim_res)
    return sim_res.mean(), np.sqrt(sim_res.var()/nsims)


def load_annot(gmt_file):
    """Loads a GMT annotation file into a dict mapping annot to gene list."""
    annot = {}
    with open(gmt_file) as f:
        for line in f:
            line_split = line.split()
            annot[line_split[0]]  = line_split[2:]
    return annot


def gsea_parametric(W: pd.DataFrame, annot: dict,
                    ref_sample: pd.DataFrame = None, sign = True,
                    min_genes = 5, shrink = 0):
    """Performs gene set enrichment analysis (GSEA).

    Peforms GSEA using the parametric PCGSE approach of Frost 2015 BDM.

    Args:
      W: A p (features) by k (factors) DataFrame with index corresponding
        to the values in annot.
      annot: A dictionary of gene set annotations mapping annotation
        names to a list of gene ids.
      ref_sample: An N (samples) by p (features) DataFrame with reference
        samples to use as feature correlations
      sign: True to consider the sign of W, otherwise compute test statistics
        using abs(W).
      min_genes: The minimum number of genes in an annotation to include.
    Returns:
      A tuple of len(annot) x k DataFrames, the first containing the
      enrichment test statistics as entries, the second containing p-values.
    """
    p = W.shape[0]
    sigma_p = np.sqrt(W.var())
    score_df = {}
    pv_df = {}
    for category, gene_set in annot.items():
        if ref_sample is not None:
            genes_in_set = ref_sample.columns.intersection(gene_set)
            genes_not_in_set = ref_sample.columns.difference(gene_set)
            m_k = len(genes_in_set)
            if m_k < min_genes:
                continue
            rho_k = (np.triu(ref_sample[genes_in_set].corr()*(1-shrink), k=1).sum()) / m_k
            df = len(ref_sample.index) - 2
        else:
            genes_in_set = W.index.intersection(gene_set)
            genes_not_in_set = W.index.difference(gene_set)
            m_k = len(genes_in_set)
            if m_k < min_genes:
                continue
            rho_k = 0
            df = p - 2
        VIF = 1 + rho_k
        if sign:
            in_set_mean = W.loc[genes_in_set].mean()
            out_set_mean = W.loc[genes_not_in_set].mean()
        else:
            in_set_mean = abs(W).loc[genes_in_set].mean()
            out_set_mean = abs(W).loc[genes_not_in_set].mean()
        num = (in_set_mean - out_set_mean)
        denom = (sigma_p * np.sqrt( VIF / m_k + 1 / (p - m_k)))

        Z_k = num/denom
        score_df[category] = Z_k
        if sign:
            pv_df[category] = 2 * (1 - stats.t.cdf(abs(Z_k), df=df))
        else:
            pv_df[category] = 1 - stats.t.cdf(Z_k, df=df)
    return pd.DataFrame(score_df).T, pd.DataFrame(pv_df).T


def _calc_gsea_scores(data, Z, transform):
    n = Z.shape[0]
    # cors = torch.from_numpy(
    #     preprocessing.scale(data).T) @ torch.from_numpy(preprocessing.scale(Z)) / n
    cors = preprocessing.scale(data).T.dot(preprocessing.scale(Z)) / n
    if transform:
        # cors = np.sqrt(n-3) * torch.atanh(cors)
        cors = np.sqrt(n-3) * np.arctanh(cors)
    # cors = pd.DataFrame(cors.numpy(), index=data.columns)
    cors = pd.DataFrame(cors, index=data.columns)
    return cors


def _gsea_one_perm(it, start_time, data, Z, transform, annot, min_genes, sign):
    if it%100 == 0:
        print('Permutation {0:d}. Time {1:.2f}s'.format(it, time.time()-start_time))
    perm_data = data.sample(frac=1)
    perm_scores = _calc_gsea_scores(perm_data, Z, transform)
    perm_stats = gsea_parametric(
        W=perm_scores, annot=annot, ref_sample=None,
        sign=sign, min_genes=min_genes, shrink=0)[0].values
    return perm_stats


def gsea_permutation(data: pd.DataFrame, Z: pd.DataFrame,
                     annot: dict, n_perm = 1000, sign = True, min_genes = 5,
                     transform = True, threads = 1):
    """Performs GSEA using the permutation approach of Frost 2015.

    Args:
      data: A pandas DataFrame with column names matching entires in annot.
        The original data.
      Z: The feature set used to calculate gene statistics. Usually mcfa_res.Z.
      annot: A dictionary of gene set annotations mapping annotation
        names to a list of gene ids.
      n_perm: Number of permutations.
      sign: True to consider the sign of W, otherwise compute test statistics
        using abs(W).
      min_genes: The minimum number of genes in an annotation to include.
      transform: bool, true to transform correlation coefficients to Z-scores.
    Returns:
      A tuple of len(annot) x k DataFrames, the first containing the
      enrichment test statistics as entries, the second containing p-values.
    """
    start = time.time()
    f = functools.partial(_gsea_one_perm, start_time=start, data=data, Z=Z,
                          transform=transform, annot=annot, min_genes=min_genes, sign=sign)
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        all_perm_stats = executor.map(f, range(n_perm))
    perm_stats = np.stack(all_perm_stats)
    print('Finished in {0:f}'.format(time.time()-start))
    true_scores = _calc_gsea_scores(data, Z, transform)
    true_stats, _ = gsea_parametric(W=true_scores, annot=annot, ref_sample=None,
                                 sign=sign, min_genes=min_genes, shrink=0)
    print(perm_stats.shape)
    print(true_stats.shape)
    if sign:
        p_vals = 1 - np.sum(true_stats.values**2 > perm_stats**2, 0) / n_perm
    else:
        p_vals = 1 - np.sum(true_stats.values > perm_stats, 0) / n_perm
    p_vals = pd.DataFrame(p_vals, index=true_stats.index)
    return true_stats, p_vals



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
    #   no longer neeed to return rho and ppca may no longer need to return vals.
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
        W0, rho0  = _init_var_W(Y_pcs, psum, d, informative)
        print(rho0)

    if k == 'infer':
        k = [pca_m.mp_dim - d for pca_m in Y_pcs]

    if init == 'random':
        L0 = None if k is None else [
            torch.randn((p_m, k_m)).double() for k_m, p_m in zip(k, p)]
        Phi0 = [torch.eye(p_m) for p_m in p]
    else:
        L0, Phi0 = _init_L_Phi(Sigma_hat, W0, psum, p, k)

    # TODO(brielin): consider reordering from initial order.
    if verbose: print('Fitting the model.')
    W, L, Phi, l, cd = em.fit_EM_iter(
        Y_all, Sigma_hat, W0, L0, Phi0, maxit, device, rcond, delta, verbose)
    rho = em.calculate_rho(W, L, Phi, Y_all, device, rcond, 'genvar')
    rho, order = torch.sort(rho, descending=True)

    W = [W_m[:, order] for W_m in W]
    Z, X = em.get_latent(W, L, Phi, Y_all, device, rcond)

    if verbose: print('Calculating feature importance and leverage scores.')
    if informative:
        W = [pc_m.V @ W_m for W_m, pc_m in zip(W, Y_pcs)]
        L = None if L is None else [pc_m.V @ L_m for L_m, pc_m in zip(L, Y_pcs)]
        Phi = [pc_m.V @ Phi_m @ pc_m.V.T for Phi_m, pc_m in zip(Phi, Y_pcs)]

    lev_W = [(W_m**2).sum(1) for W_m in W]
    lev_L = None if L is None else [(L_m**2).sum(1) for L_m in L]

    W2_sum = [(W_m**2).sum(0) for W_m in W]
    L2_sum = None if L is None else [(L_m**2).sum(0) for L_m in L]
    var_exp_Z = [sum_m/Y_m.shape[1] if scale else sum_m/sum(torch.var(Y_m, 0))
                 for sum_m, Y_m in zip(W2_sum, Y)]
    var_exp_L = None
    if L is not None:
        var_exp_L = [sum_m/Y_m.shape[1] if scale else sum_m/sum(torch.var(Y_m, 0))
                     for sum_m, Y_m in zip(L2_sum, Y)]
    lam = None if L is None else [(L_m**2).sum(0) for L_m in L]

    return MCFARes(Z, X, W, L, Phi, lev_W, lev_L, rho, lam, var_exp_Z, var_exp_L, l, cd)
