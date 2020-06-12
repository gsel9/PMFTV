import numpy as np

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
                 lambda0=1.0, lambda1=1.0, lambda2=1.0, lambda3=0.0):

        self.X_train = X_train

        self.r = rank
        self.V = V_init
        self.name = name

        # Regularization parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.init_matrices(K, J, R)

        self.n_iter_ = 0

    def init_matrices(self, K, J, R):

        self.S = self.X_train.copy()

        self.O_train = np.zeros_like(self.X_train)
        self.O_train[self.X_train.nonzero()] = 1

        self.T = int(self.X_train.shape[1])

        self.K = np.identity(self.T) if K is None else K
        self.J = np.zeros((self.T, self.r)) if J is None else J
        self.R = np.eye(self.T) if R is None else R

        self.nonzero_rows, self.nonzero_cols = np.nonzero(self.X_train)

        # Code optimization: static variables are computed and stored
        self.RTCTCR = (self.K @ self.R).T @ (self.K @ self.R)
        self.L2, self.Q2 = np.linalg.eigh((self.lambda3 / self.lambda0) * self.RTCTCR)

    def _update_V(self):

        L1, Q1 = np.linalg.eigh(self.U.T @ self.U + (self.lambda2 / self.lambda0) * np.identity(self.r))

        # For efficiency purposes, these need to be evaluated in order
        hatV = ((self.Q2.T @ (self.S.T @ self.U + (self.lambda2 / self.lambda0) * self.J)) @ Q1 / np.add.outer(self.L2, L1))
        V = self.Q2 @ (hatV @ Q1.T)

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
