import numpy as np

from .base import MFBase 


class MFTV(MFBase):

    def __init__(self, X_train, V_init, R=None, J=None, rank=5, num_iter=100, name="MFTV",
                 lambda0=1.0, lambda1=1.0, lambda2=1.0, lambda3=1.0, gamma=0.5):

        MFBase.__init__(self)

        self.X_train = X_train
        self.V = V_init
        self.rank = rank
        self.num_iter = num_iter
        self.name = name

        # Regularization parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.sigma = gamma
        self.tau = gamma

        self.init_matrices(J, R)

        self.n_iter_ = 0

    def init_matrices(self, J, R):

        self.S = self.X_train.copy()
        self.Ir = np.eye(self.rank) * (1 + self.lambda2 * self.tau)

        self.O_train = np.zeros_like(self.X_train)
        self.O_train[self.X_train.nonzero()] = 1

        self.T = int(self.X_train.shape[1])

        self.J = np.zeros((self.T, self.rank)) if J is None else J
        self.R = np.eye(self.T) if R is None else R

        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X_train)

    def _update_V(self):
        # NOTE: If self.n_iter_ > 0: uses solutions from previous run in initialisation.
        # Re-initialising dual variable with zeros gives best performance. 

        A = np.linalg.inv(self.tau * self.U.T @ self.U + self.Ir)

        # Dual and auxillary variables.
        self.Y = np.zeros_like(self.V)
        V_bar = self.V

        for i in range(self.num_iter):

            # Solve for dual variable.
            self.Y = self.proj_inf_ball(self.Y + self.sigma * self.R @ V_bar)

            # Solve for primal variable.
            V_next = A @ (self.tau * self.U.T @ self.S + self.V.T - self.tau * self.Y.T @ self.R)
            V_next = np.transpose(V_next)

            # NOTE: Using theta = 1.
            V_bar = 2 * V_next - self.V

            self.V = V_next

    def proj_inf_ball(self, X):
        """Projecting X onto the infininty ball of radius lambda2 
        amounts to element-wise clipping at +/- radius as given by 

            np.minimum(np.abs(X), radius) * np.sign(X)
        
        """
        return np.clip(X, a_min=-1.0 * self.lambda3, a_max=self.lambda3)
        
    def _update_U(self):

        U = np.linalg.solve(
            self.V.T @ self.V + (self.lambda1 / self.lambda0) * np.identity(self.rank),
            self.V.T @ self.S.T,
        )
        self.U = np.transpose(U)

    def _update_S(self):

        self.S = self.U @ self.V.T
        self.S[self.nonzero_rows, self.nonzero_cols] = self.X_train[self.nonzero_rows, self.nonzero_cols]

    def loss(self):

        # Updates to S occurs only at validation scores so must compare against U, V.
        frob_tensor = self.O_train * (self.X_train - self.U @ self.V.T)
        loss_frob = np.square(np.linalg.norm(frob_tensor)) / np.sum(self.O_train)

        loss_reg1 = self.lambda1 * np.square(np.linalg.norm(self.U))
        loss_reg2 = self.lambda2 *  np.square(np.linalg.norm(self.V))
        loss_reg3 = self.lambda3 *  np.linalg.norm(self.R @ self.V, ord=1)

        return loss_frob + loss_reg1 + loss_reg2 + loss_reg3

    def train(self):
        """Reconstruct risk profiles."""
       
        self._update_U()
        self._update_V()
        self._update_S()

        self.n_iter_ += 1
