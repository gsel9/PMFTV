import unittest

import numpy as np

# import pytest
from hypothesis import given, strategies
from hypothesis.extra.numpy import array_shapes, arrays
from longimc import CMF


# TODO: find a way to parametrize test for multiple factor models
# eg: @pytest.mark.parametrize("factorizer", (CMF, WCMF), indirect=True)
class TestFactorModels(unittest.TestCase):
    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_init_matrices(self, rank, data):
        array = data.draw(
            arrays(
                float,
                array_shapes(min_dims=2, max_dims=2),
                elements=strategies.floats(1, 4),
            )
        )

        model = CMF(rank=rank)
        model._init_matrices(array)

        for attr in ("X", "U", "V"):
            assert hasattr(model, attr)

    def test_init_basis(self):
        pass

    def test_init_coefs(self):
        pass

    def test_update_U(self):
        # exact_U = factorizer._update_U()
        # assert np.array_equal(exact_U, factorizer.U)
        pass

    def test_update_V(self):
        pass

    def test_update_S(self):
        pass

    def test_loss(self):
        pass

    # TODO: check step counter increased by one
    def test_run_step(self):
        pass

    @given(strategies.integers(min_value=1, max_value=10), strategies.data())
    def test_persistent_input(self, rank, data):
        """Test that models do not modify their input arguments.

        Models are expected to leave input variables like X, W, s_budged unmodified
        unless otherwise specified.
        >>> model = WCMF(X, V, W)
        >>> model.run_step()
        model.X should be the same as supplied during initialization.
        """

        array = data.draw(
            arrays(
                float,
                array_shapes(min_dims=2, max_dims=2),
                elements=strategies.floats(1, 4),
            )
        )

        array_initial = array.copy()

        model = CMF(rank=rank)
        model._init_matrices(array)

        assert np.array_equal(model.X, array_initial)
        model.run_step()
        assert np.array_equal(model.X, array_initial)
