# pylint: disable=invalid-name
"""Multiset Correlation and Factor Analysis

Contains functions for calculating correlated and private factors for
multiple high-dimensional datasets.

Usage:
  TODO(brielin): Add usage example
"""

import functools
import numpy as np
import time
import torch
import pdb
import pandas as pd
from concurrent import futures
from torch import multiprocessing
from dataclasses import dataclass
from typing import List, Union, Iterable
from scipy import stats
from sklearn import model_selection
from sklearn import preprocessing
from MPCCA.py import em
from torch.distributions.multivariate_normal import MultivariateNormal

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
    X: List[pd.DataFrame]
    W: List[pd.DataFrame]
    L: List[pd.DataFrame]
    Phi: List[pd.DataFrame]
    rho: pd.Series
    lam: List[pd.Series]
    var_exp_Z: List[pd.Series]
    var_exp_L: List[pd.Series]
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


def _sparse_mp_sim(X: torch.Tensor, S: torch.Tensor, quantile=0.75, nsims=50,
    device='cpu', sparsity_thresh: float = 0.95, debug=True, use_cov=True):
    """ Does a sparse simulation for selecting the number of PCs to include
    in dataset X.
    Arguments:
        X: torch.tensor. A N x p matrix containing centered + scaled data.
        S: torch.tensor. Pre-computed eigenvalues using the gram matrix.
        quantile: Float. Quantile of eigenvalue to estimate to determine
          eigenvalue threshold.
        nsims: Integer. Number of simulations to run.
        sparsity_thresh: Float. Proportion of entries to zero-out in the sparse
          simulation.
        use_cov: Boolean. Whether to use the gene-gene covariance matrix in
          the MP simulation.
    Returns:
        An integer denoting the number of PCs to keep in the data.
    """
    if debug:
        print(f"Simulation settings: (quantile, {quantile}), (nsims, {nsims})"\
        f"(sparsity_thresh, {sparsity_thresh})")
    eps = 1e-4
    N, p_m = X.shape
    cov_mat = X.cov() if use_cov else torch.eye(N)
    normal_dist = MultivariateNormal(torch.zeros(N), cov_mat + eps * \
        torch.eye(N).float())
    all_max_eigenvalues = []

    for i in range(nsims):
        if debug:
            print(f"On MP sparse simulation {i}")
        sim_mat = normal_dist.sample((p_m,)).T
        sim_mat *= torch.bernoulli((1-sparsity_thresh) * torch.ones(N, p_m))
        sim_mat -= torch.mean(sim_mat, axis=0, keepdim=True)
        sim_mat /= torch.std(sim_mat, axis=0, keepdim=True)
        cov = sim_mat.T @ sim_mat / N
        cur_lambda = torch.quantile(torch.linalg.eigvalsh(cov), quantile, dim=0)
        all_max_eigenvalues.append(np.sqrt(cur_lambda))

    all_max_eigenvalues = torch.tensor(all_max_eigenvalues)
    dim =  sum(S > all_max_eigenvalues.mean())
    if debug:
        print(f"{S.max()}, {all_max_eigenvalues.mean() - all_max_eigenvalues.std()}")
        print(f"{dim} PCs Inferred using Sparse Simulation")
    return dim


def pca_transform(X: torch.Tensor, k: int = 'infer', center: bool = True,
        scale: bool = True, quantile=0.75, nsims=50, sparsity_thresh=0.75,
        use_cov=True):
    """Transforms data to k-dimensional space with identity covariance.

    Use this if you have a wide (p > N) matrix and you only need whitened
    points in the reduces space (an N by k matrix). For the full
    inference procedure use mccfa.pca().

    Args:
        X: Two-dimesional (N x p) torch.tensor.
        k: Integer or 'infer' or 'sparse'. Number of pcs to keep. Default is to
          infer using the marchenko pasteur cutoff. 'sparse' uses a sparse MP
          simulation.
        center: bool. True to mean-center columns of X.
        scale: bool. True to variance-scale the columns of X to 1.0.
        quantile: Float. Ignored if k != 'sparse'. Quantile of eigenvalue
          to estimate.
        nsims: Integer.  Ignored if k != 'sparse'. Number of simulations to run.
        sparsity_thresh: Float. Ignored if k != 'sparse'. Proportion of
          entries to zero-out in sparse simulation.
        use_cov: Boolean. Ignored if k != 'sparse'. Whether to use the
          gene-gene covariance structure when generating data.
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

    if k == 'infer':
        k = mp_dim
    elif k == 'sparse':
        k = _sparse_mp_sim(X, S, quantile=quantile, nsims=nsims,
            sparsity_thresh=sparsity_thresh, use_cov=use_cov)

    U = A[:, 0:k]
    return np.sqrt(N) * U


def pca(X: pd.DataFrame, k: int = 'infer', center: bool = True,
        scale: bool = True, calc_V = True, quantile=0.75, nsims=50,
        sparsity_thresh=0.75, use_cov=True) -> PCARes:
    """Basic PCA implementation.

    This is a basic PCA implementation which is particularly efficient for
    top-k PCA by doing the eigendecomposition of either X X.T / N or
    X.T X / N.

    Args:
        X: n (samples) by d (features) pandas DataFrame.
        k: Integer, 'infer' or 'all' or 'sparse'. Number of pcs to keep. Default
          is to infer using the marchenko pasteur cutoff. 'sparse' uses a
          simulation of sparse noise to select the PC dimension.
        center: bool. True to mean-center columns of X.
        scale: bool. True to variance-scale the columns of X to 1.0.
        calc_V: True to track the PC loadings (right singular vectors)
          of X. Setting to False can save substantial memory if X is very
          wide.
        quantile: Float. Ignored if k != 'sparse'. Quantile of eigenvalue
          to estimate.
        nsims: Integer.  Ignored if k != 'sparse'. Number of simulations to run.
        sparsity_thresh: Float. Ignored if k != 'sparse'. Proportion of
          entries to zero-out in sparse simulation.
        use_cov: Boolean. Ignored if k != 'sparse'. Whether to use the
          gene-gene covariance structure when generating data.
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

    if k in ('infer', 'all', 'sparse'):
        if k == 'infer':
            k = mp_dim
        elif k == 'all':
            k = min(N, D)
        else:
            k = _sparse_mp_sim(X, S, quantile=quantile, nsims=nsims,
            sparsity_thresh=sparsity_thresh, use_cov=use_cov)

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


def _sparse_rho_mp_sim(N: int, p: List[int], quantile: float = 0.90,
    nsims: int = 50, sparsity_thresh: float = 0.80, debug: bool = True,
    device: str ='cpu'):
    """ Does a simulation to infer the shared dimensionality of the data.
    Simulates sparse white noise, centers + standardizes, then runs the
    Parra solution, returning either the maximum or a quantile based
    eigenvalue. Note that this is for data where the samples are *genes*.
    Therefore, when we do our simulation, we sample p_m feature vectors of
    dimension N (rather than N feature vectors of dimension p_m).

    Arguments:
        N: Integer. Number of samples.
        p: List[Integer]. Dimensionality of the feature space.
        quantile: Float. Quantile of eigenvalue to estimate.
        nsims: Integer. Number of simulations to do.
        sparsity_thresh: Float. Proportion of entries in the data to zero-out
            when doing the simulation.
    Returns:
        Estimated quantile eigenvalue and sample variance.
    """
    noise_distributions = []

    if debug:
        print("Inferring sparse shared dimensionality")

    sim_res = []
    for i in range(nsims):
        Y = []
        if debug:
            print(f"On simulation {i}.")

        for k, p_m in enumerate(p):
            sim_mat = torch.randn(N, p_m)
            sim_mat *= torch.bernoulli((1-sparsity_thresh) * torch.ones(N, p_m))
            sim_mat -= torch.mean(sim_mat, axis=0, keepdim=True)
            sim_mat /= torch.std(sim_mat, axis=0, keepdim=True)
            sim_df = pd.DataFrame(sim_mat.numpy())
            assert sim_df.shape[0] == N and sim_df.shape[1] == p_m
            Y.append(sim_df)
        if debug:
            print("Calculating Parra Solution")
        # calculate the Parra CCA solution, picking a quantile
        Y_pcs = [pca(Y_m, 'all') for Y_m in Y]
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho = torch.linalg.eigvalsh(UTU)
        sim_res.append(torch.quantile(rho, quantile, dim=0))

    sim_res = torch.Tensor(sim_res)
    return sim_res.mean(), np.sqrt(sim_res.var()/nsims)


def _rho_mp_sim(N: int, p: List[int], quantile=0.90, nsims=50, device='cpu',
                sparse=True, sparsity_thresh=0.95, debug=True):
    """Calculates the MCCA (Parra) solution to random data.

    Args:
      N: Integer. Sample size.
      p: List of integers. Dimensions of the datasets to simulate.
      nsims: Number of simulation iterations.
      device: Device to run on.
      sparse: Boolean. Whether to use the sparse simulation method or not.
      sparsity_thresh: Float. Number of entries to zero-out in the sparse
        simulation. Ignored if sparse=False.
    """
    if sparse:
        return _sparse_rho_mp_sim(N, p, quantile=quantile, nsims=nsims,
            sparsity_thresh=sparsity_thresh, device=device, debug=debug)

    sim_res = []
    for _ in range(nsims):
        Y = [pd.DataFrame(np.random.normal(size=(N, p_m))) for p_m in p]
        Y_pcs = [pca(Y_m, 'all') for Y_m in Y]
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho = torch.linalg.eigvalsh(UTU)
        sim_res.append(torch.quantile(rho, quantile, dim=0))
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


def gene_score(W: pd.DataFrame, mapping: dict, method: str = 'mean',
               min_features: int = 5):
    """Creates gene scores for non-gene-centric data.

    Args:
      W: Weight matrix (features x factors). Rows will be converted
        to gene scores via method.
      mapping: A dictionary mapping gene names to rows of W.
      method: Approach for gene score calculation. 'mean' or
        'meansq' for mean of squares.
      min_features: Minimum number of features to calculate score.
    """
    if method not in ['mean', 'meansq']:
        raise(NotImplementedError)
    scores = {}
    for gene, features in mapping.items():
        avail_features = W.index.intersection(features)
        if len(avail_features) > min_features:
            if method == 'mean':
                score = W.loc[avail_features].mean()
            elif method == 'meansq':
                score = (W.loc[avail_features]**2).mean()
            scores[gene] = score
    return pd.DataFrame(scores).T


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
    # Note: this cannot be done in pytorch because it does not get
    #  along with concurrent.futures.
    cors = preprocessing.scale(data).T.dot(preprocessing.scale(Z)) / n
    if transform:
        cors = np.sqrt(n-3) * np.arctanh(cors)
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
    f = functools.partial(
        _gsea_one_perm, start_time=start, data=data, Z=Z,
        transform=transform, annot=annot, min_genes=min_genes, sign=sign)
    with futures.ProcessPoolExecutor(max_workers=threads) as executor:
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


def _cv_one_iter(
        indices, Y: Iterable[pd.DataFrame],
        Z: pd.DataFrame, X: Iterable[pd.DataFrame],
        n_pcs: Union[str, List[int]] = 'infer',
        d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
        center: bool = True, scale: bool = True, init: str = 'avgvar',
        maxit: int = 1000, delta: float = 1e-6,
        device = 'cpu', rcond: float = 1e-8, verbose: bool = True):
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

    cv_res = mcfa(Y_train, n_pcs=n_pcs, d=d, k=k,
                  center=True, scale=True,
                  init=init, maxit=maxit,
                  delta=delta, device=device,
                  rcond=rcond, result_space = 'pc', verbose=False)
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
    Z_te, X_te = em.get_latent(W_tens, L_tens, Phi_tens, torch.cat(Y_test_pcs, axis=1),
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


def mcfa_cv(Y: Iterable[pd.DataFrame], mcfa_res: MCFARes,
            folds: Union[str, int] = 10, threads: [int] = 1,
            verbose: bool = True):
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
        with multiprocessing.get_context('spawn').Pool(threads) as pool:
            for indices in enumerate(cv_iter.split(mcfa_res.Z)):
                results.append(pool.apply_async(_cv_one_iter, (
                    indices, Y, mcfa_res.Z, mcfa_res.X, mcfa_res.n_pcs,
                    mcfa_res.d, mcfa_res.k, mcfa_res.center,
                    mcfa_res.scale, mcfa_res.init, mcfa_res.maxit, mcfa_res.delta,
                    mcfa_res.device, mcfa_res.rcond, verbose)))
            pool.close()
            pool.join()
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
    # TODO(brielin): add indices and column labels to X and Z.
    X_hat = [np.concatenate(X_m, 0) for X_m in map(list, zip(*X_hat))]
    Z_hat = np.concatenate(Z_hat, 0)
    return Z_hat, X_hat, np.array(nrmse_tr), np.array(nrmse_te)


def mcfa(Y: Iterable[pd.DataFrame], n_pcs: Union[str, List[int]] = 'infer',
         d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
         center: bool = True, scale: bool = True, init: str = 'avgvar',
         result_space: str = 'full', sparsity_d: float = 0.95,
         quantile_d: float = 0.90, sparsity_pc: float=0.95, 
         quantile_pc: float = 0.90, maxit: int = 1000, delta: float = 1e-6,
         device = 'cpu', rcond: float = 1e-8, verbose: bool = True):
    """Interface function to the MCFA estimators.
    Args:
        Y: Iterable of N (samples) by p_m (features) pandas DataFrames, the
          M=len(Y) datasets to analyze. If a dictionary, keys will be used
          as names in the results.
        center: Bool. True to mean-center columns of Y.
        scale: Bool. True to variance-scale columns of Y to 1.0.
        n_pcs: 'infer', 'all', 'sparse', or a list of length M of integers.
          The number of PCs of each dataset to keep. 'all' does not whiten/PCA
          data prior to modeling, 'infer' uses the Marchenko-Pasteur
          cutoff to choose PCs, 'sparse' uses a sparse simulation to choose PCs,
          a list of integers specifies the number of PCs to keep from each
          dataset.
        d: 'infer', 'all', 'sparse', or integer. Dimensionality of the hidden
          space. If 'infer' a simulation will be done to determine the number
          of correlated components to keep. If 'sparse', a sparse simulation
          will be done to determine the number of correlated components to keep.
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
        sparsity_d: Float. If d is set to 'sparse', then this is the sparsity
          of the simulation that is done. This parameter is ignored when
          d != 'sparse'.
        sparsity_pc: Float. If n_pcs is set to 'sparse', then this is the
          sparsity of the simulation that is done. This parameter is ignored
          when d != 'sparse'.
        quantile_d: Float. Quantile of the eigenvalues to use when estimating
          the rho cutoff when inferring the number of CCs.
        quantile_pc: Float. Quantile of the eigenvalues to use when estimating
          the eigenvalue cutoff for inferring the number of PCs.
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
    if isinstance(n_pcs, str) & (n_pcs not in ['infer', 'all', 'sparse']):
        raise NotImplementedError(
            'n_pcs must be "infer", "all", "sparse", or a list of integers.')

    if isinstance(d, str) & (d not in ['infer', 'all', 'sparse']):
        raise NotImplementedError(
            'd must be "infer", "all", "sparse", or an integer.')

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
    elif n_pcs == 'sparse':
        n_pcs = ['sparse']*M

    if verbose: print('Calculating data PCs.')

    Y_pcs = [pca(Y_m, n_pc_m, center, scale, sparsity_thresh=sparsity_pc,
            quantile=quantile_pc, nsims=10, use_cov=True)
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
    if verbose: print('Initializing model.')
    if d == 'all':
        d = p_all
    elif d == 'infer' or d == 'sparse':
        use_sparse = (d == 'sparse')
        if verbose: print('Inferring the shared dimensionality. Simulation' \
        f' params: (use_sparse, {use_sparse}), (quantile, {quantile_d})' \
        f' (sparsity, {sparsity_d})')
        rho_min, _ = _rho_mp_sim(N, p, quantile=quantile_d, nsims=10,
            sparse=use_sparse, sparsity_thresh=sparsity_d, debug=verbose)
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
    W, L, Phi, l, cd = em.fit_EM_iter(
        Y_all, Sigma_hat, W0, L0, Phi0, maxit, device, rcond, delta, verbose)
    rho = em.calculate_rho(W, L, Phi, Y_all, device, rcond, 'genvar')
    rho, order = torch.sort(rho, descending=True)

    W = [W_m[:, order] for W_m in W]
    Z, X = em.get_latent(W, L, Phi, Y_all, device, rcond)

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

    var_exp_L = None
    lam = None
    if L is not None:
        L = [pd.DataFrame(L_m.numpy(), index=ind_names, columns=col_names)
             for L_m, ind_names, col_names in zip(L, feature_names, X_names)]
        lam = [(L_m**2).sum(0) for L_m in L]
        if scale:
            var_exp_L = [l/L_m.shape[0] for l, L_m in zip(lam, L)]
        else:
            var_exp_L = [l/sum(Y_m.var(0)) for l, Y_m in zip(lam, Y)]

    if ds_names is not None:
        X = {name: X_m for name, X_m in zip(ds_names, X)}
        Y_pcs = {name: Y_m for name, Y_m in zip(ds_names, Y_pcs)}
    return MCFARes(Y_pcs, Z, X, W, L, Phi, rho, lam, var_exp_Z, var_exp_L, l,
                   cd, n_pcs, d, k, center, scale, init, maxit, delta,
                   device, rcond)
