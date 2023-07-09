# pylint: disable=invalid-name,missing-class-docstring
"""Tests for simulate_model_gaussian.py"""
import simulate_model_gaussian
import torch
import unittest


class TestSimulateModelGaussain(unittest.TestCase):

    def test_generate_model_runs(self):
        p = [3, 4, 5]
        k = [2, 2, 2]
        d = 2
        model = simulate_model_gaussian.generate_model(p, k, d)
        self.assertEqual(len(model.W), len(p))
        self.assertEqual(model.W[0].size(), torch.Size([d, p[0]]))
        self.assertEqual(model.L[0].size(), torch.Size([k[0], p[0]]))
        self.assertEqual(len(model.Phi[0].size()), 1)

    def test_generate_model_k_none(self):
        p = [3, 4, 5]
        k = None
        d = 2
        model = simulate_model_gaussian.generate_model(p, k, d)
        self.assertEqual(len(model.W), len(p))
        self.assertEqual(model.W[0].size(), torch.Size([d, p[0]]))
        self.assertIs(model.L, None)
        self.assertEqual(len(model.Phi[0].size()), 2)

    def test_MIPModel_simulate(self):
        p = [3, 4, 5]
        k = [2, 2, 2]
        d = 2
        n = 10
        model = simulate_model_gaussian.generate_model(p, k, d)
        data = model.simulate(n)
        self.assertEqual(len(data.Y), len(p))
        self.assertEqual(data.Z.size(), torch.Size([n, d]))
        self.assertEqual(data.Y[0].size(), torch.Size([n, p[0]]))
        self.assertEqual(data.X[0].size(), torch.Size([n, k[0]]))

    def test_MIPModel_simulate_k_none(self):
        p = [3, 4, 5]
        k = None
        d = 2
        n = 10
        model = simulate_model_gaussian.generate_model(p, k, d)
        data = model.simulate(n)
        self.assertEqual(len(data.Y), len(p))
        self.assertEqual(data.Z.size(), torch.Size([n, d]))
        self.assertEqual(data.Y[0].size(), torch.Size([n, p[0]]))
        self.assertIs(data.X, None)

