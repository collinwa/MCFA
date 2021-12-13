# pylint: disable=invalid-name
"""Multiset Informative Canonical Correlation Analysis model.

A basic implementation of multiset canonical correlation analysis as described
in Para bioRxiv 2018. Input datasets are optionally projected onto their top
principal components for "informative" analysis as described in Asendorf 2015.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List
from sklearn import preprocessing


@dataclass
class PCARes:
    """Simple dataclass for storing PCA results."""
    pcs: torch.tensor
    var_exp: torch.tensor
    U: torch.tensor
    lam: torch.tensor
    k: int
    n: int
    d: int
    mp_dim: int


@dataclass
class MICCARes:
    """Simple dataclass for storing MICCA results."""
    C_all: torch.tensor
    C_each: List[torch.tensor]
    rho: torch.Tensor
    C_private: List[torch.tensor]
    lam_private: torch.Tensor


def pca(X: torch.tensor, k: int = None, center: bool = True,
        scale: bool = True) -> PCARes:
    """Basic PCA implementation.

    Args:
        X: Two-dimesional (n x d) torch.tensor.
        k: Integer. Number of pcs to keep. Default is min(n ,d).
        center: bool. True to mean-center columns of X.
        scale: bool. True to variance-scale the columns of X to 1.0.
    Returns:
        A PCARes instance.
    """
    if center | scale:
        X = torch.from_numpy(preprocessing.scale(
            X, with_mean=center, with_std=scale, copy=False))
    n, d = X.shape
    if d > n:
        gram_mat = (X @ X.T) / n
        values, vectors = torch.linalg.eigh(gram_mat)
        values = torch.sqrt(torch.abs(torch.flip(values, dims = [0])))
        vectors = torch.flip(vectors, dims=[1])
    else:
        vectors, values, _ = torch.linalg.svd(X, full_matrices=False)
        values = values/np.sqrt(n)

    if k is None:
        k = min(n, d)

    mp_lower_bound = 1 + np.sqrt(d / n)
    keep = values > mp_lower_bound
    mp_dim = sum(keep)

    pcs = vectors[:, 0:k] * values[0:k]
    var_exp = values[0:k] / torch.sum(values[0:k])
    return PCARes(pcs, var_exp, vectors, values, k, n, d, mp_dim)


def _make_P_mats(
        A_list: List[torch.Tensor]) -> (List[torch.Tensor], List[torch.Tensor]):
    P_pars = [A @ torch.linalg.lstsq(A.T @ A, A.T)[0] for A in A_list]
    P_perps = [torch.eye(P.shape[0]) - P for P in P_pars]
    return P_pars, P_perps


# TODO(brielin): fix to match ML_params (.T missing)
def micca_nopc(data_pcs: List[PCARes],
               dimensions: List[int] = None,
               c_shared: int = None,
               c_private: List[int] = None) -> MICCARes:
    """Implementation of multiset informative CCA when PCs have been computed.

    Args:
        data_pcs: A list of PCARes objects, one per dataset.
        dimensions: A list of integers. Dimension for each dataset to use
            if different from the one in the PCARes.
        c_shared: Integer, number of shared components to keep. None for
            sum of  dimensions in data_pcs.
        c_private: List of integers, number of residual private components
            to keep for each dataset.
    """
    n_sets = len(data_pcs)

    if dimensions is None:
        dimensions = [data_pc.k for data_pc in data_pcs]

    if c_shared is None:
        c_shared = sum(dimensions)

    if c_private is None:
        c_private = [max(d - c_shared, 0) for d in dimensions]

    U_splits = [data_pc.U[:, 0:k] for data_pc, k in zip(data_pcs, dimensions)]
    lam_splits = [data_pc.lam[0:k] for data_pc, k in zip(data_pcs, dimensions)]
    U_all = torch.cat(U_splits, dim=1)
    M = U_all.T @ U_all
    values, vectors = torch.linalg.eigh(M)
    values = torch.flip(values, dims=[0])
    vectors = torch.flip(vectors, dims=[1])
    rho = (values[0:c_shared] - 1)/(n_sets - 1)
    V = np.sqrt(n_sets) * vectors[:, 0:c_shared]
    dimsum = np.concatenate([[0], np.cumsum(dimensions, 0)])
    V_splits = [V[i:j, :] for i, j in zip(dimsum[:-1], dimsum[1:])]

    # Dataset specific shared views.
    C_each = [U @ V_split for U, V_split in zip(U_splits, V_splits)]
    # Combined shared view.
    C_all = U_all @ (V * 1/(np.sqrt(1 + rho)))
    # Private views.
    _, P_perps = _make_P_mats(V_splits)
    weird_mats = [U @ P * l for U, P, l in zip(U_splits, P_perps, lam_splits)]
    weird_mat_svds = [torch.linalg.svd(wm) for wm in weird_mats]
    C_private = [wms.U[:, 0:c] for wms, c in zip(weird_mat_svds, c_private)]
    lam_private = [wms.S[0:c] for wms, c in zip(weird_mat_svds, c_private)]
    return MICCARes(C_all, C_each, rho, C_private, lam_private)


def micca(datasets: List[torch.Tensor],
          dimensions: List[int] = None,
          c_shared: int = None,
          c_private: List[int] = None) -> MICCARes:
    """Implementation of multiset informative CCA when PCs have been computed.

    Args:
        datasets: A list of two-D torch tensors with equal numbers of rows.
        dimensions: A list of integers. Dimension for each dataset to keep,
            full rank.
        c_shared: Integer, number of shared components to keep. None for
            sum of  dimensions in dimensions.
        c_private: List of integers, number of residual private components
            to keep for each dataset.
    """
    if dimensions is None:
        dimensions = [ds.size()[1] for ds in datasets]
    data_pcs = [pca(ds, k) for ds, k in zip(datasets, dimensions)]
    return micca_nopc(data_pcs, dimensions, c_shared, c_private)


# TODO(brielin): Should this bee List[torch.Tensor]?
#      Looks like no, would have to do lots of copies.
def EM_step(W: torch.Tensor,
            Phi: torch.Tensor,
            Sigma_tilde: torch.Tensor,
            p: List[int]) -> (torch.Tensor, torch.Tensor):
    """Computes one step of the Bach and Jordan EM update.

    Note that our Y has samples as rows so W is transpose of
    Bach and Jordan.

    Args:
        W: torch.Tensor dimensions d x p_t. p_t = sum(p).
        Phi: torch.Tensor dimensions p_t x p_t.
        Sigma_tilde: torch.Tensor dimensions p_t x p_t, empirical covariance
            matrix of data. Sigma_tilde = (Y.T @ Y) / (n - 1).
        p: Dimensionality of each dataset.
    """
    d, _ = W.size()
    Phi_inv = torch.inverse(Phi)
    M = torch.inverse(torch.eye(d) + W @ Phi_inv @ W.T)

    Q = torch.inverse(M + M @ W @ Phi_inv @ Sigma_tilde @ Phi_inv @ W.T @ M)
    W_next = Q @ M @ W @ Phi_inv @ Sigma_tilde
    Phi_next = Sigma_tilde - Sigma_tilde @ Phi_inv @ W.T @ M @ W
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    for i_l, i_r in zip(psum[:-1], psum[1:]):
        Phi_next[i_l:i_r, i_r:] = 0
        Phi_next[i_r:, i_l:i_r] = 0
    return W_next, Phi_next


#TODO(brielin): Flip these Ws to match math.
def EM_step_alt(W: torch.Tensor, Phi: torch.Tensor,
                Y: torch.Tensor, Sigma_tilde: torch.Tensor, p: List[int]):
    d = W.size()[0]
    n = Y.size()[0]
    S22_inv = torch.inverse(W.T @ W + Phi)
    E_z = Y @ S22_inv @ W.T  # nxd
    sum_E_zzT = n*torch.eye(d) - n*(W @ S22_inv @ W.T) + E_z.T @ E_z  # dxd
    W_next = torch.inverse(sum_E_zzT) @ (E_z.T @ Y) # d x p
    Phi_next = Sigma_tilde - W_next.T @ W_next
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    for i_l, i_r in zip(psum[:-1], psum[1:]):
        Phi_next[i_l:i_r, i_r:] = 0
        Phi_next[i_r:, i_l:i_r] = 0
    return W_next, Phi_next


def EM_step_stable(W: torch.Tensor, Phi: torch.Tensor,
                   Y: torch.Tensor, Sigma_tilde: torch.Tensor,
                   p: List[int], rcond: float = 1e-08):
    d = W.size()[0]
    n = Y.size()[0]
    Q_inv_WT = torch.linalg.lstsq(W.T @ W + Phi, W.T, rcond=rcond).solution
    E_z = Y @ Q_inv_WT
    sum_E_zzT = n*torch.eye(d) - n*(W @ Q_inv_WT) + E_z.T @ E_z
    W_next = torch.linalg.lstsq(sum_E_zzT, (E_z.T @ Y), rcond=rcond).solution
    Phi_next = Sigma_tilde - (Y.T @ E_z @ W_next)/n
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    for i_l, i_r in zip(psum[:-1], psum[1:]):
        Phi_next[i_l:i_r, i_r:] = 0
        Phi_next[i_r:, i_l:i_r] = 0
    return W_next, Phi_next


def fit_EM(datasets: List[torch.Tensor], d: int, niter: int = 100,
           W0: torch.Tensor = None, Phi0: torch.Tensor = None,
           method: str = "stable", print_iter = 100):
    p = [ds.shape[1] for ds in datasets]
    Y = torch.cat(datasets, 1).float()
    n = Y.shape[0]
    Sigma_tilde = (Y.T @ Y) / (n-1)

    if W0 or Phi0 is None:
        std_normal = torch.distributions.Normal(0, 1)
    if W0 is None:
        W0 = std_normal.sample([d, sum(p)])
    if Phi0 is None:
        Phi0 = torch.eye(sum(p))

    for i in range(niter):
        if method == "stable":
            W1, Phi1 = EM_step_stable(W0, Phi0, Y, Sigma_tilde, p)
        elif method == "BJ":
            W1, Phi1 = EM_step(W0, Phi0, Sigma_tilde, p)
        delta_W = torch.sum((W1 - W0)**2)
        delta_Phi = torch.sum((Phi1 - Phi0)**2)
        if i%print_iter == 0:
            print(delta_W, delta_Phi, loglik(W0.T @ W0 + Phi0, Sigma_tilde, n), loglik(W1.T @ W1 + Phi1, Sigma_tilde, n))
        W0 = W1
        Phi0 = Phi1
    return W0, Phi0


def posterior_z(Y: torch.Tensor, W: torch.Tensor, Phi: torch.Tensor,
                rcond: float = 1e-08):
    Q_inv_WT = torch.linalg.lstsq(W.T @ W + Phi, W.T, rcond=rcond).solution
    E_z = Y @ Q_inv_WT
    return E_z


def calc_feature_genvar(W: torch.Tensor, Phi: torch.Tensor):
    # Note: assumes feautures have been whitened first (does not whiten W).
    cov_mats = [(W[i:(i+1), :].T @ W[i:(i+1), :]).fill_diagonal_(1)
                for i in range(W.shape[0])]
    gen_vars = torch.tensor([torch.linalg.det(cov_mat) for cov_mat in cov_mats])
    total_gen_var = torch.linalg.det(W.T @ W + Phi)
    return gen_vars, total_gen_var


def loglik(Sigma: torch.Tensor, Sigma_hat: torch.Tensor, n: int) -> float:
    p = Sigma.shape[0]
    l = (n/2)*(p*np.log(2*np.pi) +
               torch.logdet(Sigma) +
               torch.trace(torch.linalg.lstsq(Sigma, Sigma_hat).solution))
    return l


def find_ML_params(
        Y: List[torch.Tensor], d = None) -> (torch.Tensor, torch.Tensor):
    """TODO(brielin): There are no ML params. Repurpose as initiailizer."""
    n_sets = len(Y)
    n = Y[0].size()[0]
    p = [Y_m.size()[1] for Y_m in Y]
    Sigma_tildes = [(Y_m.T @ Y_m) / (n - 1) for Y_m in Y]
    Y_svds = [torch.linalg.svd(Y_m, full_matrices=False) for Y_m in Y]
    Us, ls, Vs = map(list, zip(*Y_svds))

    U_all = torch.cat(Us, dim=1)
    M = U_all.T @ U_all
    values, vectors = torch.linalg.eigh(M)
    values = torch.flip(values, dims=[0])
    vectors = torch.flip(vectors, dims=[1])

    if d is None: d = min(p)
    rho = (values[0:d] - 1)/(n_sets - 1)
    vecs = np.sqrt(n_sets) * vectors[:, 0:d]
    p_sum = np.concatenate([[0], np.cumsum(p, 0)])
    vec_splits = [vecs[i:j, :] for i, j in zip(p_sum[:-1], p_sum[1:])]
    Fs = [((V.T * l) @ vec) for V, l, vec in zip(Vs, ls, vec_splits)]
    # Equal to diag(rho) @ F.T.
    W_hat = [(F * torch.sqrt(rho)).T for F in Fs]
    Phi_hat = [Sigma_tilde - W.T @ W
               for Sigma_tilde, W in zip(Sigma_tildes, W_hat)]
    W_hat = torch.cat(W_hat, dim = 1)
    Phi_hat = torch.block_diag(*Phi_hat)
    return W_hat, Phi_hat
