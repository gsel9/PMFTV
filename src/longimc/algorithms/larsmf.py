# third party
import numpy as np
from sklearn.linear_model import LassoLars

# local
from .cmf import CMF


class LarsMF(CMF):
    r"""Computes Lasso path using LARS algorithm for sparsity in female-
    specific coefficients.

    .. math::
       \min F(\mathbf{U}, \mathbf{V}) + R(\mathbf{U}, \mathbf{V})

    Computes the Lasso path using the LARS algorithm.

    """

    def __init__(
        self,
        rank,
        W=None,
        n_iter=100,
        n_iter_U=100,
        gamma=1.0,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        random_state=42,
        missing_value=0,
    ):
        super().__init__(
            rank=rank,
            W=W,
            n_iter=n_iter,
            gamma=gamma,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            random_state=random_state,
            missing_value=missing_value,
        )

        self.n_iter_U = n_iter_U

        self.alpha = 1e-20
        self.alphas = None

    def _update_U(self):
        # NOTE:
        # * Transposing S \approx UV^\top yields a LS problem
        #   s^\top \approx Vu^\top on the required scikit-learn form
        #   y = Xw, giving a transposed solution to our problem.
        # * Algorithm iterates until all variables in acive set or
        #   n_iter > max_iter. Thus, max_iter < rank impose sparsity.
        # * LassoLars is subclass of Lars which has argument `n_nonzero_coefs`.
        #   In LassoLars, `n_nonzero_coefs=max_iter`.
        # * Weights might blow-up if fitting intercept.

        reg = LassoLars(
            alpha=self.alpha,
            fit_path=False,
            normalize=True,
            fit_intercept=False,
            max_iter=self.n_iter_U,
        ).fit(self.V, self.S.T)

        self.U = reg.coef_
        self.alphas = np.squeeze(reg.alphas_)

    def loss(self):
        "Evaluate the optimization objective"

        loss = np.square(np.linalg.norm(self.mask * (self.X - self.U @ self.V.T)))
        loss += sum(self.alphas * np.linalg.norm(self.U, ord=1, axis=1))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss
