import numpy as np

from scipy.linalg import solve_sylvester

from .base import MFBase


class MFConv(MFBase):
    """
    Temporal Matrix Completion (MC) Classifier.

    DGDClassifier fits a matrix decomposition model M = U V^T, 
    as the hidden matrix from which X_train is assumed to be entrywise
    sampled through the discrete gaussian distribution. Targets are
    predicted by computing their similarity (likelihood) to fitted profiles
    (u1 V^T, ... un V^T) in the training set and computing the probability
    that u1 V^T can result in state z at time t. 

    Parameters
    ----------
    lambda0 : float, default=1.0
        Regularization parameter
    
    lambda1 : float, default=0.0
        Regularization parameter
    
    lambda2 : float, default=0.0
        Regularization parameter
    
    lambda3 : float, default=0.0
        Regularization parameter

    K : int, default=5
        Rank estimate of decomposition

    theta : float, default=2.5
        Parameter for prediction using the dicrete gaussian distribution

    domain_z : array of shape=(n_classes_z), default=np.arange(1, 10),
        Allowed integer classes.

    z_to_event_mapping : map, default=None
        Mapping from allowed integer classes to allowed event classes.

    domain_event : array of shape(n_classes_e), default=None
        Allowed event classes.
        
    z_to_binary_mapping : map, default=None
        Mapping from allowed integer classes to True/False.
    
    T : float, default=100
        Time granularity.

    R : array of shape (T, T), default=None
        Linear mapping used in regularization term.

    J : array of shape (T, K), default=None
        Offset matrix used in regularization term.

    C : array of shape (T, T), default=None
        Linear mapping used in regularization term.

    max_iter : int, default=100-
        Maximum number of n_iter_s for the solver.

    tol : float, default=1e-4
        Stopping critertion. 


    Attributes
    ----------
    U : array of shape (n_samples, K)
        Estimated profile weights for the matrix decomposition problem.
        
    V : array of shape (T, K)
        Estimated time profiles for the matrix decomposition problem.

    n_iter_ : int
        Actual number of iterations .
        
    """

    def __init__(self, X_train, V_init, R=None, J=None, K=None, rank=5, name="MFConv",
                 lambda0=1.0, lambda1=1.0, lambda2=1.0, lambda3=0.0, init_matrices=True):

        self.X_train = X_train

        self.r = rank
        self.V = V_init
        self.name = name

        # Regularization parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        if init_matrices:
            self._init_matrices(K, J, R)

        self.n_iter_ = 0

    def _init_matrices(self, K, J, R):

        self.S = self.X_train.copy()

        self.O_train = np.zeros_like(self.X_train)
        self.O_train[self.X_train.nonzero()] = 1

        self.T = int(self.X_train.shape[1])

        self.J = np.zeros((self.T, self.r)) if J is None else J
        self.K = np.eye(self.T) if K is None else K
        self.R = np.eye(self.T) if R is None else R

        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X_train)

        # Code optimization: static variables are computed and stored
        self.RTCTCR = (self.K @ self.R).T @ (self.K @ self.R)
        self.L2, self.Q2 = np.linalg.eigh((self.lambda3 / self.lambda0) * self.RTCTCR)

    def _update_V(self):

        L1, Q1 = np.linalg.eigh(self.U.T @ self.U + (self.lambda2 / self.lambda0) * np.identity(self.r))
        V_hat = (self.Q2.T @ (self.S.T @ self.U + (self.lambda2 / self.lambda0) * self.J)) @ Q1 / np.add.outer(self.L2, L1)
        V = self.Q2 @ (V_hat @ Q1.T)
        self.V = V

    def _update_U(self):
        
        U = np.linalg.solve(
            self.V.T @ self.V + (self.lambda1 / self.lambda0) * np.identity(self.r),
            self.V.T @ self.S.T,
        )
        self.U = np.transpose(U)

    def _update_S(self):

        S = self.U @ self.V.T
        S[self.nonzero_rows, self.nonzero_cols] = self.X_train[self.nonzero_rows, self.nonzero_cols]

        self.S = S

    def loss(self):

        # Updates to S occurs only at validation scores so must compare against U, V.
        frob_tensor = self.O_train * (self.X_train - self.U @ self.V.T)
        loss_frob = np.square(np.linalg.norm(frob_tensor)) / np.sum(self.O_train)

        loss_reg1 = self.lambda1 * np.square(np.linalg.norm(self.U))
        loss_reg2 = self.lambda2 *  np.square(np.linalg.norm(self.V))
        loss_reg3 = self.lambda3 *  np.square(np.linalg.norm(self.R @ self.V))

        return loss_frob + loss_reg1 + loss_reg2 + loss_reg3

    def train(self):
        """Fit model.

        Fits model using X_train as input

        Parameters
        ----------
        X_train : array_like, shape (n_samples_train, time_granularity)
            The training set.

        Returns
        -------
        self
            Fitted estimator
        """
       
        self._update_U()
        self._update_V()
        self._update_S()

        self.n_iter_ += 1


class WeightedMFConv(MFConv):
    """Weight the discrepancy S - UV^T to focus more attention on reconstructing 
    specific samples.

    NOTE:
        * Implementation performs inversion fo weights.
    """

    def __init__(self, X_train, V_init, R=None, W=None, K=None, rank=None,
                 lambda0=1.0, lambda1=None, lambda2=1.0, lambda3=0.0, name="WMFConv", verbose=0):

        MFConv.__init__(self, name=name, X_train=X_train, V_init=V_init, rank=rank, 
                        lambda0=lambda0, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                        init_matrices=False)

        self.W = W
        self.w = np.diag(self.W)

        self._init_matrices(R, K)

    def _init_matrices(self, R, K):

        self.S = self.X_train.copy()

        self.N, self.T = np.shape(self.X_train)
        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X_train)

        self.O_train = np.zeros_like(self.X_train)
        self.O_train[self.X_train.nonzero()] = 1

        self.R = np.eye(self.T) if R is None else R
        self.K = np.eye(self.T) if K is None else K

        self.RTCTCR = (self.K @ self.R).T @ (self.K @ self.R)

        self.L2_U, self.Q2_U = np.linalg.eigh(self.lambda1 * np.linalg.inv(self.W))
        self.L2_V, self.Q2_V = np.linalg.eigh((self.lambda3 / self.lambda0) * self.RTCTCR)
        
        # HACK: skip inversion and assumes weight matrix represents inversion of squared weights. 
        # self.L2_U, self.Q2_U = np.linalg.eigh(self.lambda1 * self.W)

        # TEMP:
        self.J = np.zeros((self.T, self.r))
        self.J2 = (self.lambda2 / self.lambda0) * self.J

    def _update_V(self):

        # NOTE: Replace A @ W @ B with A @ (self.w2[:, None] * B) for speed-up.
        H = self.w[:, None] * self.U
        
        L1_V, Q1_V = np.linalg.eigh(self.U.T @ H + self.lambda2 * np.eye(self.r))
        V_hat = (self.Q2_V.T @ self.S.T @ H) @ Q1_V / np.add.outer(self.L2_V, L1_V)
        self.V = self.Q2_V @ (V_hat @ Q1_V.T)

    def _update_U(self):

        L1_U, Q1_U = np.linalg.eigh(self.V.T @ self.V)
        U_hat = (self.Q2_U.T @ self.S @ self.V @ Q1_U) / np.add.outer(self.L2_U, L1_U)
        self.U = self.Q2_U @ (U_hat @ Q1_U.T)

    def loss(self):

        frob_tensor = self.W @ (self.O_train * (self.X_train - self.U @ self.V.T))
        loss_frob = np.square(np.linalg.norm(frob_tensor)) / np.sum(self.O_train)

        # if self.n_iter_ % 20 == 0:
        #     print("-" * 20)
        #     print("loss_frob", loss_frob)
        #     print("norm U", np.square(np.linalg.norm(self.U)))
        #     print("norm V", np.square(np.linalg.norm(self.V)))
        #     print("conv", np.square(np.linalg.norm(self.R @ self.V)))
        #     print("-" * 20)

        loss_reg1 = self.lambda1 * np.square(np.linalg.norm(self.U))
        loss_reg2 = self.lambda2 *  np.square(np.linalg.norm(self.V))
        loss_reg3 = self.lambda3 *  np.square(np.linalg.norm(self.R @ self.V))

        return loss_frob + loss_reg1 + loss_reg2 + loss_reg3


# ERROR: Kronecker product matrices wont fit into computer RAM.
# class WeightedMFConv(MFConv):

#     def __init__(self, X_train, V_init, R=None, W=None, rank=None,
#                  lambda0=1.0, lambda1=None, lambda2=1.0, lambda3=0.0, name="WMFConv", verbose=0):

#         MFConv.__init__(self, name=name, X_train=X_train, V_init=V_init, rank=rank, 
#                         lambda0=lambda0, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
#                         init_matrices=False)

#         # TEMP:
#         self.W = X_train #W

#         self._init_matrices(R)

#     def _init_matrices(self, R):

#         self.N, self.T = np.shape(self.X_train)

#         self.S = self.X_train.copy()
#         self.U = np.zeros((self.N, self.r))

#         self.O_train = np.zeros_like(self.X_train)
#         self.O_train[self.X_train.nonzero()] = 1

#         self.R = np.eye(self.T) if R is None else R

#         self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X_train)

#         # Static variables.
#         self.Ip = np.eye(self.T)
#         self.Ir = np.eye(self.r)

#         self.Irp = np.kron(self.Ir, self.Ip) 
#         self.IrRTR = np.kron(self.Ir, self.R.T @ self.R)

#         self.Dw = np.diag((self.W).flatten(order="F"))

#     def _update_V(self):
#         # see https://math.stackexchange.com/questions/3548885/sylvesters-equation-with-hadamard-product

#         H = np.kron(self.U.T, self.Ip) @ self.Dw @ np.kron(self.U, self.Ip) + self.Irp + self.IrRTR
#         C = (W.T ** 2 * self.S.T) @ self.U

#         self.V = np.linalg.solve(a=H, b=C.flatten(order='F')).reshape((self.T, self.r), order='F')

#     def _update_U(self):
#         # NOTE: Optimality criterion is decoupled over the rows of U such that the solution can 
#         # be computed row-wise.
        
#         for i in range(self.N):

#             A = np.linalg.inv(self.V.T @ np.diag(self.W[i] ** 2) @ self.V + self.lambda1 * self.Ir)
#             self.U[i] = self.V.T @ np.diag(self.W[i] ** 2) @ self.S[i] @ A

#     def loss(self):

#         # Updates to S occurs only at validation scores so must compare against U, V.
#         frob_tensor = self.W * self.O_train * (self.X_train - self.U @ self.V.T)
#         loss_frob = np.square(np.linalg.norm(frob_tensor)) / np.sum(self.O_train)

#         loss_reg1 = self.lambda1 * np.square(np.linalg.norm(self.U))
#         loss_reg2 = self.lambda2 *  np.square(np.linalg.norm(self.V))
#         loss_reg3 = self.lambda3 *  np.square(np.linalg.norm(self.R @ self.V))

#         return loss_frob + loss_reg1 + loss_reg2 + loss_reg3
