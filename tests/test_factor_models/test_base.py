"""Test base functionality of factor models."""
import unittest

import numpy as np

# import pytest
from hypothesis import given, strategies
from hypothesis.extra.numpy import array_shapes, arrays
from longimc import CMF


# TODO: find a way to parametrize test for multiple factor models
# eg: @pytest.mark.parametrize("factorizer", (CMF, WCMF), indirect=True)
class TestFactorModelsBase(unittest.TestCase):
    @given(
        strategies.floats(min_value=0, max_value=1),
        strategies.floats(min_value=0, max_value=1),
        strategies.floats(min_value=0, max_value=1),
    )
    def test_set_params(self, lambda1, lambda2, lambda3):
        params = {"lambda1": lambda1, "lambda2": lambda2, "lambda3": lambda3}

        model = CMF(rank=None)
        model.set_params(**params)

        for key, value in params.items():
            assert getattr(model, key) == value

    @given(strategies.data())
    def test_identity_weights(self, data):
        array = data.draw(
            arrays(
                float,
                array_shapes(min_dims=2, max_dims=2),
                elements=strategies.floats(1, 4),
            )
        )

        base = CMF(rank=None)
        base.X = array

        W = base.identity_weights()
        assert np.all(W == 1)
