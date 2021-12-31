# pylint: disable=invalid-name
# pylint: disable-msg=not-callable
# pylint: disable=no-member
# pylint: disable=too-many-arguments

"""Tests for Multi-Set Canonical Correlation Analysis with Private Structure.

Tests for recovery of rotationally invariant quantities under simulation
of the graphical model where we know the ground-truth values of W, L, and Phi.
"""
from typing import Tuple, List
import unittest
import torch
import torch.distributions as tdist
import torch.distributions.multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
import model

# set torch random seed
torch.manual_seed(1337)

# slightly weird simulation seed values
def generate_W_L_Phi_gaussian(y_dims: torch.tensor,
                              x_dims: torch.tensor,
                              d: int = 5,
                              W_stdev: float = 2.0,
                              L_stdev: float = 2.0,
                              max_noise: float = 5.0) -> Tuple[List[torch.tensor],
                                                               List[torch.tensor],
                                                               List[torch.tensor]]:
    """Generate ground truth W, L, and Phi for simulation tests.

    Args:
        y_dims: torch.tensor. Dimension of each dataset.
        x_dims: torch.tensor. Dimension of each private space.
        W_stdev: Float. Stddev of the Gaussian from which we sample
                 values of W.
        L_stdev: Float. Stddev of the Gaussian from which we sample L.
        max_noise: Float. Maximum value of Uniform from which we sample
                   noise variance.
    Returns:
        List of ground-truth W, L, and Phi matrices.
    """
    all_Ws = []
    all_Ls = []
    all_Phis = []

    for i, y_val in enumerate(y_dims):
        # declare new uniform distributions from which to
        # sample ground truth W, L, and Phi for each dataset
        W_uniform = tdist.normal.Normal(torch.zeros(y_val, d),
                                        torch.ones(y_val, d) * W_stdev)
        L_uniform = tdist.normal.Normal(torch.zeros(y_val, x_dims[i]),
                                        torch.ones(y_val, x_dims[i]) * L_stdev)
        Phi_uniform = tdist.uniform.Uniform(torch.zeros(y_val) + 1e-2,
                                            torch.ones(y_val) * max_noise)

        # store ground truth W, L, and Phi
        all_Ws.append(W_uniform.sample())
        all_Ls.append(L_uniform.sample())
        all_Phis.append(torch.diag(Phi_uniform.sample()))

    return all_Ws, all_Ls, all_Phis


def generate_W_L_Phi_identity(y_dims: torch.tensor,
                              x_dims: torch.tensor,
                              d: int = 5,
                              W_stdev: float = 2.0,
                              L_stdev: float = 2.0) -> Tuple[List[torch.tensor],
                                                             List[torch.tensor],
                                                             List[torch.tensor]]:

    """Generate ground truth W, L, and Phi for simulation tests.
       In this case, Phi is fixed to be the identity matrix.

    Args:
        y_dims: torch.tensor. Dimension of each dataset.
        x_dims: torch.tensor. Dimension of each private space.
        W_stdev: Float. Stddev of the Gaussian from which we sample
                 values of W.
        L_stdev: Float. Stddev of the Gaussian from which we sample L.
    Returns:
        List of ground-truth W, L, and Phi matrices.
    """
    all_Ws = []
    all_Ls = []
    all_Phis = []

    for i, y_val in enumerate(y_dims):
        # declare new uniform distributions from which to sample
        # ground truth W, L, and Phi for each dataset
        W_dist = tdist.normal.Normal(torch.zeros(y_val, d),
                                     torch.ones(y_val, d) * W_stdev)
        L_dist = tdist.normal.Normal(torch.zeros(y_val, x_dims[i]),
                                     torch.ones(y_val, x_dims[i]) * L_stdev)

        # store ground truth W, L, and Phi
        all_Ws.append(W_dist.sample())
        all_Ls.append(L_dist.sample())

        # append identity matrix instead of sampling uniform values
        all_Phis.append(torch.eye(y_val))

    return all_Ws, all_Ls, all_Phis


def generate_samples(all_Ws: List[torch.tensor],
                     all_Ls: List[torch.tensor],
                     all_Phis: List[torch.tensor],
                     y_dims: torch.tensor,
                     x_dims: torch.tensor,
                     N: int = 1000,
                     d: int = 5) -> List[torch.tensor]:
    """Generates datasets according to specified W, L, and Phi matrices.
    Args: TODO

    Returns:
        List of datasets generated according to MPCCA with private
        latent structure.
    """
    datasets = [torch.zeros(1, y_d) for y_d in y_dims]

    # z-distribution remains fixed
    z_distribution = mvn.MultivariateNormal(torch.zeros(d),
                                            torch.eye(d))

    # simulate the graphical model
    for sample in range(N):
        print('Generating sample: {}/{}'.format(sample, N), end='\r', flush=True)
        # for each sample, retrieve the latent z latent variable
        z = z_distribution.sample()
        # for each dataset, compute the dataset-specific mean
        # and variance, and obtain 1 sample
        for i, dim in enumerate(x_dims):
            x = mvn.MultivariateNormal(torch.zeros(dim),
                                       torch.eye(dim)).sample()
            y_i = mvn.MultivariateNormal(all_Ws[i] @ z + all_Ls[i] @ x,
                                         all_Phis[i]).sample()

            datasets[i] = torch.cat([datasets[i], y_i[None,:]])

    # slice out the dummy variable
    datasets = [dataset[1:] for dataset in datasets]
    return datasets


class TestMPCCAModel(unittest.TestCase):
    """Unit Test Class for checking EM convergence and
       recovery of rotationally invariant quantities.
    """

    def test_rotinv_recovery(self):
        """Test that the EM algorithm recovers the rotationally
        invariant quantities.
        """

        # set the parameters of the datasets
        y_dims = torch.tensor([40, 50, 80, 90])
        x_dims = torch.tensor([5, 5, 5, 5])
        d = 5
        N = 5000

        # generate ground truth params
        Ws, Ls, Phis = generate_W_L_Phi_gaussian(y_dims, x_dims, d)

        # generate datasets
        all_datasets = generate_samples(Ws,
                                        Ls,
                                        Phis,
                                        y_dims,
                                        x_dims,
                                        N=N,
                                        d=d)

        # make sure that the datasets have the right
        # dimensionality
        for i, dataset in enumerate(all_datasets):
            self.assertEqual(y_dims[i], dataset.shape[1])
            self.assertEqual(N, dataset.shape[0])

        # use gpu if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # create the y_concat tensor
        y_concat = torch.cat(all_datasets, axis=1).double()
        W_hat, L_hat, Phi_hat = model.fit_model(y_dims,
                                                x_dims,
                                                all_datasets,
                                                d,
                                                y_concat,
                                                N,
                                                eps=1e-6,
                                                steps=10000,
                                                device=device)


        # create block matrices that represent the full data
        W_GT = torch.cat(Ws, axis=0).double()
        L_GT = torch.block_diag(*Ls).double()
        Phi_GT = torch.block_diag(*Phis).double()

        # create rotationally invariant quantities
        W_rotinv = W_GT @ W_GT.T
        L_rotinv = L_GT @ L_GT.T
        W_rotinv_pred = W_hat @ W_hat.T
        L_rotinv_pred = L_hat @ L_hat.T

        # compute fnorms for debugging
        W_fnorm = torch.norm(W_rotinv - W_rotinv_pred.cpu())
        L_fnorm = torch.norm(L_rotinv - L_rotinv_pred.cpu())
        Phi_fnorm = torch.norm(Phi_GT - Phi_hat.cpu())
        print('-----------------------norms-----------------')
        print("(WW^T-hat(WW^T))_F: {} "
              "(LL^T-hat(LL^T))_F: {} "
              "(Phi-hat(Phi))_F: {}".format(W_fnorm, L_fnorm, Phi_fnorm))
        print( torch.norm(W_GT - W_hat.cpu()) )
        print( torch.norm(L_GT - L_hat.cpu()) )
        print('-----------------------norms-----------------')

        plt.scatter(W_rotinv.reshape(-1), W_rotinv_pred.reshape(-1).cpu(),
                    c='black')
        plt.grid()
        plt.savefig("./W_rotinv_debug.png")
        plt.clf()

        plt.scatter(L_rotinv.reshape(-1), L_rotinv_pred.reshape(-1).cpu(),
                    c='black')
        plt.grid()
        plt.savefig("./L_rotinv_debug.png")
        plt.clf()
    
        # test rotationally invariant quantities for closeness
        torch.testing.assert_close(
            W_rotinv,
            W_rotinv_pred.cpu(),
            check_stride=False)
        torch.testing.assert_close(
            L_rotinv,
            L_rotinv_pred.cpu(),
            check_stride=False)
        torch.testing.assert_close(
            Phi_GT,
            Phi_hat.cpu(),
            check_stride=False)


if __name__ == '__main__':
    unittest.main()
