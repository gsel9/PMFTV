# third party
import numpy as np
import tensorflow as tf
from lmc.algorithms import utils

# local
from ._base import MatrixCompletionBase


class WCMC(MatrixCompletionBase):
    """Matrix factorization with L2 and convolutional regularization.
    Factor updates are based on gradient descent approximations, permitting
    an arbitrary weight matrix in the discrepancy term.

    Args:
        X: Sparse data matrix used to estimate factor matrices
        V: Initial estimate for basic vectors
        config: Configuration model.
          shift is ignored in the WCMF factorizer.
    """

    def __init__(
        self,
        rank,
        W=None,
        n_iter=100,
        gamma=1.0,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        random_state=42,
        missing_value=None,
    ):
        super().__init__(
            rank=rank,
            W=W,
            n_iter=n_iter,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            random_state=random_state,
            missing_value=missing_value,
        )
        self.gamma = gamma

    def _init_matrices(self, X):
        self.X = X
        self.S = self.X.copy()
        self.mask = (self.X != 0).astype(np.float32)

        self.N, self.T = np.shape(X)

        self.V = self.init_basis()
        self.U = self.init_coefs()
        # run one exact update to improve U initalization
        self._solve_U_exact()

        K = utils.finite_difference_matrix(self.T)
        D = utils.laplacian_kernel_matrix(self.T, self.gamma)
        self.KD = K @ D

        # the minimum value for the basic profiles
        min_value = np.min(self.X[self.X != self.missing_value])
        self.J = utils.basis_baseline_value(self.V.shape, min_value)

        if self.W is None:
            self.W = self.identity_weights()

    def _update_V(self):
        def _loss_V():
            frob_tensor = tf.multiply(W, X - (U @ tf.transpose(V)))
            frob_loss = tf.square(tf.norm(frob_tensor))

            l2_loss = self.config.lambda2 * tf.square(tf.norm(V - J))

            conv_loss = self.config.lambda3 * tf.square(tf.norm(tf.matmul(self.KD, V)))

            return frob_loss + l2_loss + conv_loss

        V = tf.Variable(self.V, dtype=tf.float32)

        J = tf.cast(self.J, dtype=tf.float32)
        W = tf.cast(self.W, dtype=tf.float32)
        X = tf.cast(self.X, dtype=tf.float32)
        U = tf.cast(self.U, dtype=tf.float32)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        for _ in tf.range(self.config.iter_V):
            optimiser.minimize(_loss_V, [V])

        self.V = V.numpy()

    def _solve_U_exact(self):
        """Solve for U at a fixed V.

        V is assumed to be initialized."""
        U = np.empty((self.N, self.config.rank))

        for n in range(self.N):
            U[n] = (
                self.V.T
                @ (self.W[n] * self.X[n])
                @ np.linalg.inv(self.V.T @ (self.W[n][:, None] * self.V) + self.I_l1)
            )
        self.U = U

    def _update_U(self):
        def _loss_U():
            frob_tensor = tf.multiply(W, X - tf.matmul(U, V, transpose_b=True))
            frob_loss = tf.square(tf.norm(frob_tensor))

            return frob_loss + self.config.lambda1 * tf.square(tf.norm(U))

        U = tf.Variable(self.U, dtype=tf.float32)

        W = tf.cast(self.W, dtype=tf.float32)
        X = tf.cast(self.X, dtype=tf.float32)
        V = tf.cast(self.V, dtype=tf.float32)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        for _ in tf.range(self.config.iter_U):
            optimiser.minimize(_loss_U, [U])

        self.U = U.numpy()

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.square(np.linalg.norm(self.W * (self.X - self.U @ self.V.T)))
        loss += self.config.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.config.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.config.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss

    def run_step(self):
        "Run one step of alternating minimization"

        self._update_U()
        self._update_V()


class WCMFADMM(MatrixCompletionBase):
    """Matrix factorization with L2 and convolutional regularization.
    Factor updates are based on gradient descent approximations, permitting
    an arbitrary weight matrix in the discrepancy term.

    Args:
        X: Sparse data matrix used to estimate factor matrices
        V: Initial estimate for basic vectors
        config: Configuration model.
          shift is ignored in the WCMF factorizer.
    """

    def __init__(
        self,
        rank,
        beta=1.0,
        W=None,
        n_iter=100,
        gamma=1.0,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        random_state=42,
        missing_value=None,
    ):
        super().__init__(
            rank=rank,
            W=W,
            n_iter=n_iter,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            random_state=random_state,
            missing_value=missing_value,
        )
        self.beta = beta
        self.gamma = gamma

    def _init_matrices(self, X):
        self.X = X
        self.S = self.X.copy()
        self.Z = self.X.copy()
        self.mask = (self.X != 0).astype(np.float32)

        self.N, self.T = np.shape(X)
        self.nz_rows, self.nz_cols = np.nonzero(X)

        self.P = np.zeros(X.shape)

        self.V = self.init_basis()
        self.U = self.init_coefs()

        K = utils.finite_difference_matrix(self.T)
        D = utils.laplacian_kernel_matrix(self.T, self.gamma)

        # pre-compute static variables
        self.KD = K @ D
        self.DTKTKD = self.KD.T @ self.KD

        self.ZbPT = np.transpose(self.beta * self.Z - self.P)
        self.L2, self.Q2 = np.linalg.eigh(self.lambda3 * self.DTKTKD)

        # the minimum value for the basic profiles
        min_value = np.min(self.X[self.X != self.missing_value])
        self.J = utils.basis_baseline_value(self.V.shape, min_value)

        if self.W is None:
            self.W = self.identity_weights()

        self.W2 = np.multiply(self.W, self.W)
        self.WY = np.multiply(self.W2, self.X)
        self.denom = self.beta * np.ones((self.N, self.T)) + self.W2

        self.I_l1 = self.lambda1 * np.identity(self.r)
        self.I_l2 = self.lambda2 * np.identity(self.r)

    def _update_V(self):
        self.ZbPT = (self.beta * self.Z - self.P).T
        L1, Q1 = np.linalg.eigh(self.beta * (self.U.T @ self.U) + self.I_l2)

        V_hat = (
            (self.Q2.T @ (self.ZbPT @ self.U + self.lambda2 * self.J))
            @ Q1
            / np.add.outer(self.L2, L1)
        )

        self.V = self.Q2 @ (V_hat @ Q1.T)

    def _update_U(self):
        UT = np.linalg.solve(
            self.beta * (self.V.T @ self.V) + self.I_l1, self.V.T @ self.ZbPT
        )
        self.U = np.transpose(UT)

    def _update_Z(self):
        self.UV = self.U @ self.V.T
        self.Z = (self.beta * self.UV + self.P + self.WY) / self.denom

    def _update_P(self):
        self.P = self.P - self.beta * (self.Z - self.UV)

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.square(np.linalg.norm(self.W * (self.X - self.U @ self.V.T)))
        loss += self.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss

    def run_step(self):
        "Perform one step of alternating minimization"

        self._update_U()
        self._update_V()
        self._update_Z()
        self._update_P()

        self.n_iter_ += 1
