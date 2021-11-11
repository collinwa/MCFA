# pylint: disable=invalid-name
# pylint: disable-msg=not-callable

"""Multi-Set Canonical Correlation Analysis with Private Structure

An implementation of the graphical model using EM described in our paper
for isotropic / diagonal variance updates, with or without sparsity.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EStepRes:
    """Class for storing EStep results."""
    zzT: torch.tensor
    zxT: torch.tensor
    xxT: torch.tensor
    zmu: torch.tensor
    xmu: torch.tensor


@dataclass
class MStepRes:
    """Class for storing MStep results."""
    W: torch.tensor
    L: torch.tensor
    Phi: torch.tensor


def _E_step(W: torch.tensor,
            L: torch.tensor,
            Phi: torch.tensor,
            x_dims: torch.tensor,
            d: int,
            y_i: torch.tensor,
            device='cpu') -> EStepRes:
    """Implementation of E-Step for CCA with private-latent structure

    Args:
        W: A 2-dimensional torch.tensor (dim_total, d).
        L: A 2-dimensional torch.tensor (dim_total, k).
        Phi: A 2-dimensional torch.tensor (dim_total, dim_total).
        x_dims: A 1-dimensional torch.tensor (n_datasets, ).
        d: Integer. Shared space dimension.
        y_i: A 2-dimensional torch.tensor (dim_total x dim_total).
        device: String. Where to run torch model.

    Returns:
        An EStepRes instance.

    """

    # Consider random vector (z, x, y)
    # Sigma_22 = Var(y | W, L, Phi)
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + Phi).to(device)
    # print(sigma_22_inv)

    # sigma_12 = Cov((z, x), y)
    sigma_12 = torch.cat([W.T, L.T], axis=0).to(device)
    # sigma_11 = Var((z, x))
    sigma_11 = torch.eye(torch.sum(x_dims)+d).to(device)

    # E[z, x | y_i, params]
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ (y_i)
    # E[z | y_i, params]
    posterior_z_mean = posterior_z_x_mean[:d].to(device)
    # E[x | y_i, params]
    posterior_x_mean = posterior_z_x_mean[d:].to(device)

    # Var((z, x) | y_i, params)
    posterior_x1_cov = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
    # Cov(z, x | y_i, params)
    posterior_z_x_cov = posterior_x1_cov[:d, d:].to(device)
    # Var(z | y_i, params)
    posterior_z_z_cov = posterior_x1_cov[:d, :d].to(device)
    # Var(x | y_i, params)
    posterior_x_x_cov = posterior_x1_cov[d:, d:].to(device)

    # zmu_batched dimension: [n_samples, (d+k), 1]
    # [:, None] adds a batch dimension in the None index
    zmu_batched = posterior_z_mean.T[:, :, None].to(device)
    xmu_batched = posterior_x_mean.T[:, :, None].to(device)

    # Recall that <zx.T> w.r.t the posterior is not 0-mean w.r.t
    # the posterior distribution; it is more closely related
    # to the second moment. We therefore adjust the quantity
    # by the product of posterior means.
    # posterior <zx.T> = cov(z, x) + <z><x.T>

    # shape: (n_samples, z_dim, x_dim)
    posterior_zxT = (posterior_z_x_cov + \
                     zmu_batched @ xmu_batched.permute(0, 2, 1)).to(device)

    # shape: (n_samples, z_dim, z_dim)
    posterior_zzT = (posterior_z_z_cov + \
                     zmu_batched @ zmu_batched.permute(0, 2, 1)).to(device)

    # shape: (n_samples, x_dim, x_dim)
    posterior_xxT = (posterior_x_x_cov + \
                     xmu_batched @ xmu_batched.permute(0, 2, 1)).to(device)

    return EStepRes(posterior_zzT,
                    posterior_zxT,
                    posterior_xxT,
                    zmu_batched,
                    xmu_batched)


def _M_step(zxT: torch.tensor,
            zzT: torch.tensor,
            xxT: torch.tensor,
            zmu: torch.tensor,
            xmu: torch.tensor,
            y_i: torch.tensor,
            L_model: torch.tensor,
            W_model: torch.tensor,
            N: int,
            device: str = 'cpu') -> MStepRes:
    """Method for computing diagonal covariance M-Step in EM algorithm.
    Args:
        Rmk: All of the below quantities are inferred
             **per sample**. This means that every matmul
             is batched using the first dimension. <.> denotes
             the expectation w.r.t the posterior.

        zxT: torch.tensor. Inferred <zx.T> from E-Step.
        zzT: torch.tensor. Inferred <zz.T> from E-Step.

             Rmk: Note that this is not necessarily the
                  covariance of z since z is not 0-mean
                  w.r.t. the posterior (even though its
                  marginal distribution is 0-mean). The
                  same holds w.r.t. the other quantities
                  we infer that involve inner/outer
                  products of random vectors.

        xxT: torch.tensor. Inferred <xx.T> from E-Step.
        zmu: torch.tensor. Inferred <z> from E-Step.
        xmu: torch.tensor. Inferred <x> from E-Step.

        L_model: torch.tensor (p, k). Current L tensor.
        W_model: torch.tensor (p, d). Current W tensor.

    Returns:
        MStepRes instance.
    """

    # (n_samples, batch_dim, 1)
    y_i_batched = y_i[:, :, None].to(device)

    # compute the L update
    new_L = torch.sum(y_i_batched @ xmu.permute(0, 2, 1) - \
                      W_model @ zxT, axis=0) @ \
                      torch.inverse(torch.sum(xxT, axis=0)).to(device)

    # compute the W update
    new_W = torch.sum(y_i_batched @ zmu.permute(0, 2, 1) - \
                      L_model @ zxT.permute(0, 2, 1), axis=0) @ \
                      torch.inverse(torch.sum(zzT, axis=0)).to(device)

    # compute the new Phi update
    new_Phi = (1/N * torch.sum(y_i_batched @ y_i_batched.permute(0, 2, 1) + \
                                L_model @ xxT @ L_model.T + \
                                W_model @ zzT @ W_model.T + \
                                2 * L_model @ zxT.permute(0, 2, 1) @ \
                               W_model.T + \
                                -2 * y_i_batched @ zmu.permute(0, 2, 1) @ \
                               W_model.T + \
                                -2 * y_i_batched @ xmu.permute(0, 2, 1) @ \
                               L_model.T, axis=0)).to(device)

    # only update the diagonal components
    new_Phi = torch.diag(torch.diagonal(new_Phi)).to(device)

    return MStepRes(new_W, new_L, new_Phi)


def initialize_isotropic_model(y_dims: torch.tensor,
                               x_dims: torch.tensor,
                               datasets: List[torch.tensor],
                               d: int = 5,
                               std: float = 2.0,
                               mean: float = 0.0):
    """Initialize values for simulation of isotropic covariance MPCCA model.

    Args:
        y_dims: torch.tensor. List of dimensionality of datasets.
        x_dims: torch.tensor. List of private space dimensions.
        d: Integer. Dimension of shared space.
        datasets: List. List of torch.tensors (n x p_i) of each dataset.
        std: Float. Standard deviation of Gaussian from which to init.
        mean: Float. Mean of Gaussian from which to init.
    Returns:
        Stacked W and L matrix, and modeled isotropic variance.
    """

    Ws_to_stack = []
    Phis_to_stack = []
    Ls_to_stack = []

    for i, y_dim in enumerate(y_dims):
        # set W and L to standard normal initialization around 0
        cur_W = torch.nn.init.normal_(torch.zeros(y_dim, d),
                                      mean=mean,
                                      std=std)
        cur_L = torch.nn.init.normal_(torch.zeros(y_dim, x_dims[i]),
                                      mean=mean,
                                      std=std)

        # set Phi to the empirical covariance matrix
        cur_dataset = datasets[i] # (n_samples x dimension)
        ymu = torch.mean(cur_dataset, axis=0, keepdim=True)
        cur_dataset = cur_dataset - ymu

        # due to design matrix construction, covariance = 1/n(Y^T Y)
        cur_Phi = (1/cur_dataset.shape[0]) * cur_dataset.T @ cur_dataset

        Ws_to_stack.append(cur_W)
        Ls_to_stack.append(cur_L)
        Phis_to_stack.append(cur_Phi)

    # get the current phi matrix
    cur_Phi = torch.block_diag(*Phis_to_stack)

    # we're fixing an *isotropic* variance assumption here
    # so the model variance will be captured by a *scalar*
    variance = 1/cur_Phi.shape[0] * torch.einsum('ii->', cur_Phi)
    assert variance >= 0
    return torch.cat(Ws_to_stack, axis=0), \
        torch.block_diag(*Ls_to_stack), variance


def initialize_model(y_dims: torch.tensor,
                     x_dims: torch.tensor,
                     datasets: List[torch.tensor],
                     d: int = 5,
                     std: float = 2.0,
                     mean: float = 0.0) -> Tuple[torch.tensor,
                                                 torch.tensor,
                                                 torch.tensor]:
    """Method for initializing W, L, and Phi matrix randomly.

    Args:
        y_dims: torch.tensor (n_datasets, ). Dimension of each dataset.
        x_dims: torch.tensor (n_datasets, ). Dimension of each private space.
        d: Integer. Dimension of shared space
        std: Float. Stddev of Gaussian initializing distribution.
        mean: Float. mean of Gaussian initializing distribution.
    Returns:
        3-Tuple of initialized W, L, and Phi matrix.
    """
    Ws_to_stack = []
    Phis_to_stack = []
    Ls_to_stack = []

    for i, y_dim in enumerate(y_dims):
        # set W and L to standard normal initialization around 0
        cur_W = torch.nn.init.normal_(torch.zeros(y_dim, d),
                                      mean=mean,
                                      std=std)
        cur_L = torch.nn.init.normal_(torch.zeros(y_dim, x_dims[i]),
                                      mean=mean,
                                      std=std)

        # set Phi to the empirical covariance matrix
        cur_dataset = datasets[i] # (n_samples x dimension)
        ymu = torch.mean(cur_dataset, axis=0, keepdim=True)
        demean_dataset = cur_dataset - ymu
        # print(demean_dataset.shape)
        # due to design matrix construction, covariance = 1/n(Y^T Y)
        cur_Phi = torch.diag(torch.diagonal((1/demean_dataset.shape[0]) * \
                                            demean_dataset.T @ \
                                            demean_dataset))

        Ws_to_stack.append(cur_W)
        Ls_to_stack.append(cur_L)
        Phis_to_stack.append(cur_Phi)

        # construct joint matrices and return
        W_final = torch.cat(Ws_to_stack, axis=0).double()
        L_final = torch.block_diag(*Ls_to_stack).double()
        Phi_final = torch.block_diag(*Phis_to_stack).double()
    return W_final, L_final, Phi_final


def _isotropic_E_step(W: torch.tensor,
                      L: torch.tensor,
                      Phi: torch.tensor,
                      x_dims: torch.tensor,
                      d: int,
                      y_i: torch.tensor) -> EStepRes:
    """Method for computing the E-Step of the EM algorithm.
    Args:
        W: torch.tensor (p, d). Current W matrix.
        L: torch.tensor (p, k). Current L matrix.
        Phi: torch.tensor (p, p). Current Phi matrix.
        x_dims: torch.tensor (n, ). Dimension of
                private spaces.
        d: Integer. Dimension of shared space
        y_i: torch.tensor (p, n). Data Matrix of observed data
             with samples aligned as columns.

    Returns:
        EStepRes instance.
    """
    # Phi is a scalar now! (Phi = \sigma^2)
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + Phi*torch.eye(W.shape[0]))
    #     print(sigma_22_inv)

    # other necessary block matrices
    sigma_12 = torch.cat([W.T, L.T], axis=0)
    sigma_11 = torch.eye(torch.sum(x_dims)+d)

    # compute the posterior mean of z and x;
    # y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ (y_i)
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    # posterior covariance
    posterior_x1_cov = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_12.T
    posterior_z_x_cov = posterior_x1_cov[:d, d:]  # cross covariance
    posterior_z_z_cov = posterior_x1_cov[:d, :d]  # upper left block matrix
    posterior_x_x_cov = posterior_x1_cov[d:, d:]  # bottom right block matrix

    # need to batch zmu and xmu: [n_samples, <[z, x]>.shape, 1]
    zmu_batched = posterior_z_mean.T[:, :, None]
    xmu_batched = posterior_x_mean.T[:, :, None]

    # posterior <zx.T> = cov(z, x) + <z><x.T>
    # shape: (n_samples, z_dim, x_dim)
    posterior_zxT = posterior_z_x_cov + \
        zmu_batched @ \
        xmu_batched.permute(0, 2, 1)

    # shape: (n_samples, z_dim, z_dim)
    posterior_zzT = posterior_z_z_cov + \
        zmu_batched @ \
        zmu_batched.permute(0, 2, 1)

    # shape: (n_samples, x_dim, x_dim)
    posterior_xxT = posterior_x_x_cov + \
        xmu_batched @ \
        xmu_batched.permute(0, 2, 1)

    return EStepRes(posterior_zxT,
                    posterior_zzT,
                    posterior_xxT,
                    zmu_batched,
                    xmu_batched)


def _isotropic_M_step(zxT: torch.tensor,
                      zzT: torch.tensor,
                      xxT: torch.tensor,
                      zmu: torch.tensor,
                      xmu: torch.tensor,
                      y_i: torch.tensor,
                      L_model: torch.tensor,
                      W_model: torch.tensor,
                      N: int) -> MStepRes:
    """Method for computing isotropic M-Step in EM algorithm.
    Args:
        Rmk: All of the below quantities are inferred
             **per sample**. This means that every matmul
             is batched using the first dimension. <.> denotes
             the expectation w.r.t the posterior.

        zxT: torch.tensor. Inferred <zx.T> from E-Step.
        zzT: torch.tensor. Inferred <zz.T> from E-Step.

             Rmk: Note that this is not necessarily the
                  covariance of z since z is not 0-mean
                  w.r.t. the posterior (even though its
                  marginal distribution is 0-mean). The
                  same holds w.r.t. the other quantities
                  we infer that involve inner/outer
                  products of random vectors.

        xxT: torch.tensor. Inferred <xx.T> from E-Step.
        zmu: torch.tensor. Inferred <z> from E-Step.
        xmu: torch.tensor. Inferred <x> from E-Step.

        L_model: torch.tensor (p, k). Current L tensor.
        W_model: torch.tensor (p, d). Current W tensor.
        N: Integer. Total number of samples.
    Returns:
        MStepRes instance.
    """
    # (n_samples, batch_dim, 1)
    y_i_batched = y_i[:, :, None]
    new_L = torch.sum(y_i_batched @ \
                      xmu.permute(0, 2, 1) - \
                      W_model @ zxT, axis=0) @ \
                      torch.inverse(torch.sum(xxT, axis=0))

    new_W = torch.sum(y_i_batched @ \
                      zmu.permute(0, 2, 1) - \
                      L_model @ \
                      zxT.permute(0, 2, 1),
                      axis=0) @ \
                      torch.inverse(torch.sum(zzT, axis=0))

    # compute terms involving y_i_batched
    var_terms = []
    var_terms.append(1/2 * torch.sum(y_i_batched.permute(0, 2, 1) @ \
                                     y_i_batched))
    var_terms.append(-1 * torch.sum(y_i_batched.permute(0, 2, 1) @ \
                                    W_model @ zmu))
    var_terms.append(-1 * torch.sum(y_i_batched.permute(0, 2, 1) @ \
                                    L_model @ \
                                    xmu))

    # einsum does a batched trace
    var_terms.append(1/2 * torch.sum(torch.einsum('bii->b', W_model.T @ \
                                                  W_model @ \
                                                  zzT)))
    var_terms.append(1/2 * torch.sum(torch.einsum('bii->b', L_model.T @ \
                                                  L_model @ \
                                                  xxT)))
    var_terms.append(1/2 * torch.sum(torch.einsum('bii->b', L_model.T @ \
                                                  W_model @ \
                                                  zxT)))

    # Compute new Phi
    new_Phi = torch.sum(2 / (N * y_i.shape[1]) * torch.tensor(var_terms))
    return new_W, new_L, new_Phi


def fit_isotropic_model(y_dims: torch.tensor,
                        x_dims: torch.tensor,
                        datasets: List[torch.tensor],
                        d: int,
                        y_concat: torch.tensor,
                        N: int,
                        eps: float = 1e-6,
                        steps: int = 5000) -> Tuple[torch.tensor,
                                                    torch.tensor,
                                                    torch.tensor]:
    """Fit the EM model assuming isotropic noise variance.
    Args:
        y_dims: torch.tensor (n, ). Dimension of each dataset.
        x_dims: torch.tensor (n, ). Dimension of each private space.
        datasets: List[torch.tensor]. Each dataset with samples as rows.
        y_concat: torch.tensor (n, p). Concatenated observations
                  with samples as rows.
        N: Integer. Total number of samples.
        eps: Float. Terminate when ||A_t - A_{t-1}||_F <= eps.
        steps: Integer. Max number of EM iterations.

    Returns:
        Fitted W, L, and Phi matrix. For concatenated data
        in the order of y_concat.
    """

    torch.set_num_threads(20)

    y_concat_T = y_concat.T

    # initialize the model parameters
    W_model, L_model, Phi_model = initialize_isotropic_model(y_dims,
                                                             x_dims,
                                                             datasets,
                                                             d=d)

    #print(W_model.shape)
    #print(L_model.shape)
    #print(Phi_model)

    # store the update size
    W_diffs = []
    L_diffs = []
    Phi_diffs = []

    # iterate through E/M Steps
    for i in range(steps):
        # E-Step, then M-Step
        # infer missing data from the posterior
        estep = _isotropic_E_step(W_model,
                                  L_model,
                                  Phi_model,
                                  x_dims,
                                  d,
                                  y_concat_T)

        # update params by solving:
        # argmax_{W, L, Phi} E[L | y, W_t, Phi_t, L_t]
        mstep = _isotropic_M_step(estep.zxT,
                                  estep.zzT,
                                  estep.xxT,
                                  estep.zmu,
                                  estep.xmu,
                                  y_concat,
                                  L_model,
                                  W_model,
                                  N)

        # extract params
        W_tprime = mstep.W
        L_tprime = mstep.L
        Phi_tprime = mstep.Phi

        # compute updated L
        L_tupdate = torch.zeros_like(L_tprime)

        # careful when updating L_model
        # have to make sure to keep terms that allow
        # interaction of private structure across datasets fixed at 0
        #     L_model = L_tprime
        for j in range(len(y_dims)):
            # only update the specific L_i for dataset
            # zero-padded values ensure matmuls work as-is
            bot_y = 0 if j == 0 else torch.sum(y_dims[:j])
            bot_x = 0 if j == 0 else torch.sum(x_dims[:j])
            L_tupdate[bot_y:bot_y+y_dims[j],
                      bot_x:bot_x+x_dims[j]] = L_tprime[bot_y:bot_y+y_dims[j],
                                                        bot_x:bot_x+x_dims[j]]

        # F-norm between prev params and updated
        W_diff = torch.norm(W_tprime-W_model).item()
        L_diff = torch.norm(L_tupdate-L_model).item()
        Phi_diff = torch.norm(torch.tensor(Phi_tprime - Phi_model)).item()

        # store training stats
        W_diffs.append(W_diff)
        L_diffs.append(L_diff)
        Phi_diffs.append(Phi_diff)

        # update parameters
        W_model = W_tprime
        Phi_model = Phi_tprime
        L_model = L_tupdate

        # check for convergence
        if W_diff <= eps and L_diff <= eps and Phi_diff <= eps:
            print('Stopping... step between W_t, W_t+1 <= {}'.format(eps))
            break

        # debugging
        if i % 100 == 0:
            print("{}/{}: (Wtprime-Wt)_F: {}"
                  "(Ltprime-Lt)_F: {}"
                  "(Phi_tprime-Phi_t)_F: {}".format(i,
                                                    steps,
                                                    W_diff,
                                                    L_diff,
                                                    Phi_diff))

    # Save Training Stats
    n_steps = np.arange(len(W_diffs))
    plt.plot(n_steps, W_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(W_t - W_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./mesa_W_updates.png')
    plt.clf()

    n_steps = np.arange(len(L_diffs))
    plt.plot(n_steps, L_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(L_t - L_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./mesa_L_updates.png')
    plt.clf()

    n_steps = np.arange(len(Phi_diffs))
    plt.plot(n_steps, Phi_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(Phi_t - Phi_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./mesa_Phi_updates.png')
    plt.clf()

    return W_model, L_model, Phi_model


def project_latent(W: torch.tensor,
                   L: torch.tensor,
                   Phi: torch.tensor,
                   d: int,
                   y_concat: torch.tensor,
                   isotropic: bool = False) -> Tuple[torch.tensor,
                                                     torch.tensor]:
    """Method for projecting data into the latent space.

    Args:
        W: torch.tensor (p, d). Fitted W matrix.
        L: torch.tensor (p, k). Fitted L matrix.
        Phi: torch.tensor (p, p). Fitted Phi matrix.
        d: Integer. Dimension of shared space
        y_concat: torch.tensor (n, p). Design matrix with samples
                  aligned as rows.
        isotropic: Boolean. Whether we are using an isotropic noise
                   variance model.
    """
    y_concat_T = y_concat.T
    var_term = Phi*torch.eye(W.shape[0]) if isotropic else Phi
    sigma_22_inv = torch.inverse(W@W.T + L @ L.T + var_term)
    #     print(sigma_22_inv)

    # other necessary block matrices
    sigma_12 = torch.cat([W.T, L.T], axis=0)

    # compute the posterior mean of z and x;
    # y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ y_concat_T
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    return posterior_z_mean.T, posterior_x_mean.T


def project_latent_individual(W: torch.tensor,
                              L: torch.tensor,
                              Phi: torch.tensor,
                              d: int,
                              y_concat: torch.tensor,
                              x_dims: torch.tensor,
                              y_dims: torch.tensor,
                              nth_dataset: int,
                              isotropic: bool = False):

    """Method for projecting data into the latent space conditional
       only on the observed sample from nth_dataset.

    Args:
        W: torch.tensor (p, d). Fitted W matrix.
        L: torch.tensor (p, k). Fitted L matrix.
        Phi: torch.tensor (p, p). Fitted Phi matrix.
        d: Integer. Dimension of shared space
        y_concat: torch.tensor (n, p). Fully concatenated
                  design matrix with samples aligned as rows.
        x_dims: torch.tensor (n,). Dimension of each private space.
        y_dims: torch.tensor (n,). Dimension of each dataset.
        isotropic: Boolean. Whether we are using an isotropic noise
                   variance model.
    """
    # get dimension of dataset of interest
    prev_dims = torch.sum(y_dims[:nth_dataset]).item()
    cur_dim = y_dims[nth_dataset]
    cur_W = W[prev_dims:prev_dims+cur_dim, :]
    cur_L = L[prev_dims:prev_dims+cur_dim, :]
    cur_Phi = Phi

    # if isotropic cur_Phi is scalar
    # else cur_Phi is a block matrix
    if isotropic:
        var_term = cur_Phi * torch.eye(cur_W.shape[0])
    else:
        var_term = cur_Phi[prev_dims:prev_dims+cur_dim,
                           prev_dims:prev_dims+cur_dim]

    # transpose dataset for easier manipulation
    y_concat_T = y_concat[:, prev_dims:prev_dims+cur_dim].T
    sigma_22_inv = torch.inverse(cur_W@cur_W.T + cur_L @ cur_L.T + var_term)

    # other necessary block matrices
    sigma_12 = torch.cat([cur_W.T, cur_L.T], axis=0)

    # compute the posterior mean of z and x;
    # y should be a matrix with all samples aligned as columns
    posterior_z_x_mean = sigma_12 @ sigma_22_inv @ y_concat_T
    posterior_z_mean = posterior_z_x_mean[:d]
    posterior_x_mean = posterior_z_x_mean[d:]

    # slice x s.t. it contains the private space
    # of the desired dataset
    prev_x_dims = torch.sum(x_dims[:nth_dataset]).item()
    cur_x_dim = x_dims[nth_dataset]
    posterior_x_mean = posterior_x_mean[prev_x_dims:prev_x_dims+cur_x_dim, :]

    return posterior_z_mean.T, posterior_x_mean.T


def compute_ISC(datasets: List[torch.tensor]) -> Tuple[torch.tensor,
                                                       torch.tensor,
                                                       torch.tensor]:
    """Method for computing the quantity proportional to
       inter-set correlation after dataset-wise projection
       into the shared space.

    Args:
        datasets: List[torch.tensor]. List of inferred latent vectors
        for each dataset. That is, for dataset m, for each observed
        y_i we infer <z_i | y_i> (expected latent vector conditional
        only on y_i in dataset m).

        We do this for all samples in each dataset and concatentate them
        into a data matrix of shape (n, d), then put them into a list.

    Returns:
        rho: torch.tensor. Proportional to inter-set correlation.
        rb: torch.tensor. Numerator of inter-set correlation.
        rw: torch.tensor. Denominator of normalized variance.
    """
    # setup for processing and computation of ISC
    # datasets = [*all_ys]
    N_sets = len(datasets)

    # de-mean each individual CCA projection
    for i in range(N_sets):
        datasets[i] = datasets[i] - torch.mean(datasets[i],
                                               axis=1,
                                               keepdim=True)

    # ISC explained by each canonical component
    rb = torch.zeros(datasets[0].shape[1])
    rw = torch.zeros(datasets[0].shape[1])

    # compute the b/w set correlation
    for i in range(N_sets):
        # for j in range(i+1, N_sets):
        for j in range(N_sets):
            if i != j:
                rb += torch.sum(torch.mul(datasets[i], datasets[j]), axis=0)
        rw += torch.sum(torch.pow(datasets[i], 2), axis=0)
    rho = rb / (rw * (N_sets - 1))
    return rho, rb, rw


def fit_model(y_dims: torch.tensor,
              x_dims: torch.tensor,
              datasets: List[torch.tensor],
              d: int,
              y_concat: torch.tensor,
              N: int,
              eps: float = 1e-6,
              steps: int = 5000,
              device: str = 'cpu'):
    """Fit the EM model assuming diagonal noise variance.
    Args:
        y_dims: torch.tensor (n, ). Dimension of each dataset.
        x_dims: torch.tensor (n, ). Dimension of each private space.
        datasets: List[torch.tensor]. Each dataset with samples as rows.
        y_concat: torch.tensor (n, p). Concatenated observations
                  with samples as rows.
        N: Integer. Total number of samples.
        eps: Float. Terminate when ||A_t - A_{t-1}||_F <= eps.
        steps: Integer. Max number of EM iterations.

    Returns:
        Fitted W, L, and Phi matrix. For concatenated data
        in the order of y_concat.
    """

    torch.set_num_threads(20)
    device = torch.device(device)

    y_concat_T = y_concat.T.to(device)

    # initialize the model parameters
    W_model, L_model, Phi_model = initialize_model(y_dims,
                                                   x_dims,
                                                   datasets,
                                                   d=d)

    # move parameters to device
    W_model = W_model.to(device)
    L_model = L_model.to(device)
    Phi_model = Phi_model.to(device)
    y_concat = y_concat.to(device)
    y_concat_T = y_concat_T.to(device)

    #print(W_model.shape)
    #print(L_model.shape)
    #print(Phi_model)

    # store the update size
    W_diffs = []
    L_diffs = []
    Phi_diffs = []

    # iterate through E/M Steps
    for i in range(steps):
        # E-Step, then M-Step

        # infer missing data from posterior
        estep = _E_step(W_model,
                        L_model,
                        Phi_model,
                        x_dims,
                        d,
                        y_concat_T,
                        device=device)

        # maximize likelihood w.r.t. model params
        mstep = _M_step(estep.zxT,
                        estep.zzT,
                        estep.xxT,
                        estep.zmu,
                        estep.xmu,
                        y_concat,
                        L_model,
                        W_model,
                        N,
                        device=device)

        L_tprime = mstep.L
        W_tprime = mstep.W
        Phi_tprime = mstep.Phi

        # compute updated L
        L_tupdate = torch.zeros_like(L_tprime).to(device)

        # careful when updating L_model;
        # make sure that private structure doesn't interact
        # across datasets by fixing at 0
        #     L_model = L_tprime
        for j in range(len(y_dims)):
            # only update the specific L_i corresponding to each dataset; keep
            # zero-padded values that make the matrix multiplication work as 0
            bot_y = 0 if j == 0 else torch.sum(y_dims[:j])
            bot_x = 0 if j == 0 else torch.sum(x_dims[:j])
            L_tupdate[bot_y:bot_y+y_dims[j],
                      bot_x:bot_x+x_dims[j]] = L_tprime[bot_y:bot_y+y_dims[j],
                                                        bot_x:bot_x+x_dims[j]]

        # F-norm between prev params and updated
        W_diff = torch.norm(W_tprime-W_model).item()
        L_diff = torch.norm(L_tupdate-L_model).item()
        Phi_diff = torch.norm(Phi_tprime - Phi_model)

        # store training stats
        W_diffs.append(W_diff)
        L_diffs.append(L_diff)
        Phi_diffs.append(Phi_diff)

        # update parameters
        W_model = W_tprime.to(device)
        Phi_model = Phi_tprime.to(device)
        L_model = L_tupdate.to(device)

        # check for convergence
        if W_diff <= eps and L_diff <= eps and Phi_diff <= eps:
            print('Stopping... step between W_t, W_t+1 <= {}'.format(eps))
            break

        # debugging
        if i % 100 == 0:
            print("{}/{}: (Wtprime-Wt)_F: {}"
                  " (Ltprime-Lt)_F: {}"
                  " (Phi_tprime-Phi_t)_F: {}".format(i,
                                                     steps,
                                                     W_diff,
                                                     L_diff,
                                                     Phi_diff))

    # Save Training Stats
    n_steps = np.arange(len(W_diffs))
    plt.plot(n_steps, W_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(W_t - W_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./W_updates.png')
    plt.clf()

    n_steps = np.arange(len(L_diffs))
    plt.plot(n_steps, L_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(L_t - L_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./L_updates.png')
    plt.clf()

    n_steps = np.arange(len(Phi_diffs))
    plt.plot(n_steps, Phi_diffs)
    plt.grid()
    plt.xlabel('Training Step')
    plt.ylabel('(Phi_t - Phi_t-1)_F')
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    plt.savefig('./Phi_updates.png')
    plt.clf()

    return W_model, L_model, Phi_model


def compute_x_cov(posterior_x_mu: torch.tensor) -> Tuple[torch.tensor,
                                                         torch.tensor]:
    """Compute empirical covariance of predicted private space.
    Args:
        posterior_x_mu: torch.tensor (n_samples x private_dim).
    Returns:
        posterior_cov: posterior covariance matrix
        torch.diagonal(.): diagonal view of posterior cov
    """
    posterior_cov = posterior_x_mu.T @ posterior_x_mu
    return posterior_cov, torch.diagonal(posterior_cov)


def slice_private_projection(projection: torch.tensor,
                             x_dims: torch.tensor) -> List[torch.tensor]:
    """Slice the concatenated private space projections that
    are returned after conditioning on all data for a given sample

    Args:
        projection: torch.tensor of shape (n_samples x sum(x_dims))
        x_dims: torch tensor of size (n_datasets), each
                dataset private space dimension in order of concatenation
    returns:
        all_projections: list of tensors of length (n_datasets)
                         of private space projection for every dataset
    """

    all_projections = []
    for idx, dim in enumerate(x_dims):
        prev_dims = torch.sum(x_dims[:idx])
        all_projections.append(projection[:, prev_dims:prev_dims+dim])

    return all_projections

