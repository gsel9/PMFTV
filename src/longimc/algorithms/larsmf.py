# third party
import numpy as np
from sklearn.linear_model import LassoLars

# local
from .cmf import CMF


class LarsMF(CMF):
    r"""Computes Lasso path using LARS algorithm for sparsity in female-
    specific coefficients.

    .. math::
       \min F(\mathbf{U}, \mathbf{V}}) + \mathcal{R}(\mathbf{U}, \mathbf{V}})

    """

    def __init__(
        self,
        X_train,
        V_init,
        R=None,
        J=None,
        K=None,
        rank=None,
        max_iter=None,
        lambda0=1.0,
        lambda1=None,
        lambda2=1.0,
        lambda3=0.0,
        name="MFLars",
        verbose=0,
    ):
        CMF.__init__(
            self,
            name=name,
            X_train=X_train,
            V_init=V_init,
            R=R,
            J=J,
            K=K,
            rank=rank,
            lambda0=lambda0,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
        )

        self.max_iter = max_iter
        self.verbose = verbose

        self.alpha = 1e-20
        self.alphas = None

    def loss(self):
        # Updates to S occurs only at validation scores so must compare against U, V.
        frob_tensor = self.O_train * (self.X_train - self.U @ self.V.T)
        loss_frob = np.square(np.linalg.norm(frob_tensor)) / np.sum(self.O_train)

        loss_reg1 = sum(self.alphas * np.linalg.norm(self.U, ord=1, axis=1))
        loss_reg2 = self.lambda2 * np.square(np.linalg.norm(self.V))
        loss_reg3 = self.lambda3 * np.square(np.linalg.norm(self.R @ self.V))

        return loss_frob + loss_reg1 + loss_reg2 + loss_reg3

    def _update_U(self):
        # NOTE:
        # * Transposing S \approx UV^\top yields a LS problem
        #   s^\top \approx Vu^\top on the required scikit-learn form
        #   y = Xw, giving a transposed solution to our problem.
        # * Algorithm iterates until all variables in acive set or
        #   n_iter > max_iter. Thus, max_iter < rank impose sparsity.
        # * LassoLars is subclass of Lars which has argument `n_nonzero_coefs`.
        #   In LassoLars, `n_nonzero_coefs=max_iter`.
        # * Weights tend to blow-up if fitting intercept.

        # NOTE: Computes Lasso path using LARS.
        reg = LassoLars(
            alpha=self.alpha,
            fit_path=False,
            normalize=True,
            fit_intercept=False,
            max_iter=self.max_iter,
        ).fit(self.V, self.S.T)

        self.U = reg.coef_
        self.alphas = np.squeeze(reg.alphas_)
