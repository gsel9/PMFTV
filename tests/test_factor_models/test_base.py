"""Test base functionality of factor models."""
import unittest

import numpy as np
import pytest
from hypothesis import given, strategies
from hypothesis.extra.numpy import array_shapes, arrays
from longimc import MatrixCompletionBase


class TestMatrixCompletionBase(unittest.TestCase):
    def test_init_matrices(self):
        "Ensure raises the correct error."

        with pytest.raises(NotImplementedError):
            model = MatrixCompletionBase(rank=None)
            model._init_matrices(None)

    def test_loss(self):
        "Ensure raises the correct error."

        with pytest.raises(NotImplementedError):
            model = MatrixCompletionBase(rank=None)
            model.loss()

    def test_run_step(self):
        "Ensure raises the correct error."

        with pytest.raises(NotImplementedError):
            model = MatrixCompletionBase(rank=None)
            model.run_step()

    @given(
        strategies.floats(min_value=0, max_value=1),
        strategies.floats(min_value=0, max_value=1),
        strategies.floats(min_value=0, max_value=1),
    )
    def test_set_params(self, lambda1, lambda2, lambda3):
        params = {"lambda1": lambda1, "lambda2": lambda2, "lambda3": lambda3}

        model = MatrixCompletionBase(rank=None)
        model.set_params(**params)

        for key, value in params.items():
            assert getattr(model, key) == value

    @given(
        strategies.integers(min_value=10, max_value=100),
        strategies.integers(min_value=10, max_value=100),
        strategies.integers(min_value=10, max_value=100),
    )
    def test_M(self, N, T, rank):
        U = np.random.random((N, rank))
        V = np.random.random((T, rank))
        M_gt = np.array(U @ V.T, dtype=np.float32)

        model = MatrixCompletionBase(rank=None)
        model.set_params(**{"U": U, "V": V})

        assert np.array_equal(model.M, M_gt)

    @given(strategies.data())
    def test_identity_weights(self, data):
        """Given an array with not missing, the weight matrix
        entries should all all equal one."""
        array = data.draw(
            arrays(
                float,
                array_shapes(min_dims=2, max_dims=2),
                elements=strategies.floats(1, 4),
            )
        )

        base = MatrixCompletionBase(rank=None)
        base.X = array

        W = base.identity_weights()
        assert np.all(W == 1)

    @given(
        strategies.integers(min_value=10, max_value=100),
        strategies.integers(min_value=10, max_value=100),
    )
    def test_init_basis(self, T, rank):
        model = MatrixCompletionBase(rank=rank)
        model.set_params(**{"T": T})

        assert np.shape(model.init_basis()) == (T, rank)

    @given(
        strategies.integers(min_value=10, max_value=100),
        strategies.integers(min_value=10, max_value=100),
    )
    def test_init_coefs(self, N, rank):
        model = MatrixCompletionBase(rank=rank)
        model.set_params(**{"N": N})

        assert np.shape(model.init_coefs()) == (N, rank)

    # TODO: need V, X. check that model.transform() output
    # has expected shape (N x T)
    def test_transform(self, V, X):
        "Test output from model.transform() has expected shape (N x T)."
        model = MatrixCompletionBase(rank=None)
        model.set_params(**{"V": V})

        assert np.shape(model.transform(X)) == np.shape(X)

    # TODO: mocking
    def test_fit(self):
        pass
