# pylint: disable=invalid-name
"""Multiset Informative Canonical Correlation Analysis model.

A basic implementation of multiset canonical correlation analysis as described
in Para bioRxiv 2018. Input datasets are optionally projected onto their top
principal components for "informative" analysis as described in Asendorf 2015.
"""

import numpy as np
import torch
from typing import List
from dataclasses import dataclass


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


@dataclass
class MICCARes:
    """Simple dataclass for storing MICCA results."""
    C_all: torch.tensor
    C_each: List[torch.tensor]
    rho: torch.Tensor
    C_private: List[torch.tensor]
    lam_private: torch.Tensor


def pca(X: torch.tensor, k: int = None) -> PCARes:
    """Basic PCA implementation.

    Args:
        X: Two-dimesional (n x d) torch.tensor.
        k: Integer. Number of pcs to keep. Default is min(n ,d).
    Returns:
        A PCARes instance.
    """
    n, d = X.size()
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

    pcs = vectors[:, 0:k] * values[0:k]
    var_exp = values[0:k] / torch.sum(values[1:k])
    return PCARes(pcs, var_exp, vectors, values, k, n, d)


def _make_pmat(X: torch.tensor, n: int) -> torch.tensor:
    nrow_x, _ = X.size()
    Z = torch.zeros([nrow_x, n - nrow_x])
    top = torch.cat([X @ X.T, Z ], dim = 1)
    bot = torch.cat([Z.transpose_(0, 1), torch.eye(n - nrow_x)], dim = 1)
    return torch.cat([top, bot], dim = 0)


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
            min of  dimensions in data_pcs.
        c_private: List of integers, number of residual private components
            to keep for each dataset.
    """
    n_sets = len(data_pcs)

    if dimensions is None:
        dimensions = [data_pc.k for data_pc in data_pcs]

    if c_shared is None:
        c_shared = min(dimensions)

    if c_private is None:
        c_private = dimensions - c_shared

    U_splits = [data_pc.U[:, 0:k] for data_pc, k in zip(data_pcs, dimensions)]
    U_all = torch.cat(U_splits, dim=1)
    M = U_all.T @ U_all
    values, vectors = torch.linalg.eigh(M)
    values = torch.flip(values, dims=[0])
    vectors = torch.flip(vectors, dims=[1])
    rho = values[0:c_shared] - 1
    V = np.sqrt(n_sets) * vectors[:, 0:c_shared]
    dimsum = np.concatenate([[0], np.cumsum(dimensions, 0)])
    V_splits = [V[i:j, :] for i, j in zip(dimsum[:-1], dimsum[1:])]

    # Dataset specific shared views.
    C_each = [U @ V for U, V in zip(U_splits, V_splits)]
    # Combined shared view.
    C_all = U_all @ (V * 1/(np.sqrt(1 + rho)))
    # Private views.
    # TODO(brielin): double check we want min(dimensions)?
    Vp = np.sqrt(n_sets) * vectors[:, c_shared:min(dimensions)]
    Vp_splits = [Vp[i:j, :] for i, j in zip(dimsum[:-1], dimsum[1:])]
    Vp_proj_mats = [_make_pmat(Vps, d) for Vps, d in zip(Vp_splits, dimensions)]
    C_p_svds = [torch.linalg.svd((dpc.U * dpc.lam) @ P, full_matrices = False)
           for dpc, P, cp in zip(data_pcs, Vp_proj_mats, c_private)]
    C_private = [C_p_svd.U[:, 1:cp] for C_p_svd, cp in zip(C_p_svds, c_private)]
    lam_private = [C_p_svd.S[1:cp] for C_p_svd, cp in zip(C_p_svds, c_private)]
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
            min of  dimensions in dimensions.
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


def find_ML_params(
        Y: List[torch.Tensor], d = None) -> (torch.Tensor, torch.Tensor):
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
    rho = values[0:d] - 1
    vecs = np.sqrt(n_sets) * vectors[:, 0:d]
    p_sum = np.concatenate([[0], np.cumsum(p, 0)])
    vec_splits = [vecs[i:j, :] for i, j in zip(p_sum[:-1], p_sum[1:])]
    Fs = [((V.T * l) @ vec)/(np.sqrt(n-1))
          for V, l, vec in zip(Vs, ls, vec_splits)]
    # Equal to diag(rho) @ F.T.
    W_hat = [(F * torch.sqrt(rho)).T for F in Fs]
    Phi_hat = [Sigma_tilde - W.T @ W
               for Sigma_tilde, W in zip(Sigma_tildes, W_hat)]
    W_hat = torch.cat(W_hat, dim = 1)
    Phi_hat = torch.block_diag(*Phi_hat)
    return W_hat, Phi_hat
