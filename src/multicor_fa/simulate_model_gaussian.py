# pylint: disable=invalid-name
"""Functions to simulate from the mip-CCA model."""
import torch
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class MIPData:
    """Dataclass for storing simulated data."""
    Y: List[torch.tensor]
    X: List[torch.tensor]
    Z: torch.tensor


@dataclass
class MIPModel:
    """Dataclass for storing model parameters."""
    W: List[torch.tensor]
    L: List[torch.tensor]
    Phi: List[torch.tensor]
    k: List[int]
    p: List[int]
    d: int

    def simulate(self, n: int):
        """Simulates data from the given mip-CCA model parameters.

        Args:
            n: Integer. Number of samples to generate.
        """
        std_normal = torch.distributions.Normal(0, 1)
        Z = std_normal.sample([n, self.d])
        Y = []
        X = []
        # TODO(brielin): revisit this mess.
        for m, (W_m, Phi_m, p_m) in enumerate(zip(self.W, self.Phi, self.p)):
            if self.k is None:
                mvn = torch.distributions.MultivariateNormal(
                    torch.zeros(p_m), covariance_matrix = Phi_m)
                noise_m = mvn.sample([n])
                X = None
                Y_m = Z @ W_m + noise_m
            else:
                k_m = self.k[m]
                L_m = self.L[m]
                # TODO(brielin): you lost your phi!
                noise_m = std_normal.sample([n, p_m])
                X_m = std_normal.sample([n, k_m])
                X.append(X_m)
                Y_m = Z @ W_m + X_m @ L_m + noise_m
            Y.append(Y_m)
        return MIPData(Y, X, Z)


def generate_model(
        p: List[int], k: List[int] = None, d: int = None) -> MIPModel:
    """Generates mip-CCA model params W, L, Phi.

    Args:
        p: List of integers. Observed dimensionality of each dataset.
        k: List of integers. Private dimensionality of each dataset.
            None to have no private space.
        d: Integer. Share hidden space dimensionality, None to use min
             of p.
    """
    if d is None: d = min(p)
    std_normal = torch.distributions.Normal(0, 1)
    W = [std_normal.sample([d, p_m]) for p_m in p]
    Phi = [torch.sqrt((torch.ones([p_m]) - std_normal.sample([p_m]))**2)
           for p_m in p]

    if k is None:
        L = None
        l = int(np.floor(np.sqrt(min(p))))
        P = [std_normal.sample([l, p_m]) for p_m in p]
        Phi = [(P_m.T @ P_m)/l + torch.diag(Phi_m)
               for P_m, Phi_m in zip(P, Phi)]
    else:
        L = [std_normal.sample([k_m, p_m]) for p_m, k_m in zip(p, k)]
    return MIPModel(W, L, Phi, k, p, d)
