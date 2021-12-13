# pylint: disable=invalid-name,missing-class-docstring
"""Tests for micca_model.py"""
import micca_model
import numpy as np
import torch
import unittest


class TestMiccaModel(unittest.TestCase):

    def test_pca_methods_same(self):
        n = 3
        d = 5
        X = torch.Tensor(range(n*d)).reshape([n, d])
        pca_res = micca_model.pca(X)
        vectors, values, _ = torch.linalg.svd(X, full_matrices=False)
        values = values/np.sqrt(n)
        torch.testing.assert_close(
            torch.abs(pca_res.U), torch.abs(vectors), check_stride=False)


    def test_EM_step_runs(self):
        p = [3, 4, 5]
        p_t = sum(p)
        n = 10
        d = 2
        W = torch.ones((d, p_t))
        Phi = torch.eye(p_t)
        Y = torch.ones((n, p_t))
        Sigma_tilde = (Y.T @ Y) / (n - 1)
        W_next, Phi_next = micca_model.EM_step(W, Phi, Sigma_tilde, p)
        self.assertEqual(W_next.size(), torch.Size([d, p_t]))
        self.assertEqual(Phi_next.size(), torch.Size([p_t, p_t]))
        torch.testing.assert_close(
            Phi_next[0:p[0], p[0]:],
            torch.zeros((p[0], p[1]+p[2])),
            check_stride=False)
        torch.testing.assert_close(
            Phi_next[p[0]:, 0:p[0]],
            torch.zeros((p[1]+p[2], p[0])),
            check_stride=False)

    def test_EM_step_alt_runs(self):
        p = [3, 4, 5]
        p_t = sum(p)
        n = 10
        d = 2
        W = torch.ones((d, p_t))
        Phi = torch.eye(p_t)
        Y = torch.ones((n, p_t))
        Sigma_tilde = (Y.T @ Y) / (n - 1)
        W_next, Phi_next = micca_model.EM_step_alt(W, Phi, Y, Sigma_tilde, p)
        self.assertEqual(W_next.size(), torch.Size([d, p_t]))
        self.assertEqual(Phi_next.size(), torch.Size([p_t, p_t]))
        torch.testing.assert_close(
            Phi_next[0:p[0], p[0]:],
            torch.zeros((p[0], p[1]+p[2])),
            check_stride=False)
        torch.testing.assert_close(
            Phi_next[p[0]:, 0:p[0]],
            torch.zeros((p[1]+p[2], p[0])),
            check_stride=False)

    def test_find_ML_params_runs(self):
        n = 10
        p = 5
        d = 2
        Y1 = torch.ones((n, p))
        Y2 = torch.ones((n, p))
        W_hat, Phi_hat = micca_model.find_ML_params([Y1, Y2], d = d)
        self.assertEqual(W_hat.size(), torch.Size([d, 2*p]))
        self.assertEqual(Phi_hat.size(), torch.Size([2*p, 2*p]))



if __name__ == '__main__':
    unittest.main()
