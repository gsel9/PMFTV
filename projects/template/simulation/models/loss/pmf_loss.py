import numpy as np

from .base import BaseLoss
from ..special_matrices import finite_difference_matrix, laplacian_kernel_matrix


def set_loss(model_config, X_true, O):

    loss_fn = PMFLoss

    print(f"Fetching loss: {loss_fn.__name__}")

    loss = loss_fn(
        X_true=X_true, O=O,
        lambda1=model_config.lambda1,
        lambda2=model_config.lambda2,
        lambda3=model_config.lambda3,
    )
    return loss


class PMFLoss(BaseLoss):

    def __init__(self, O, X_true, lambda1, lambda2, lambda3, name="PMFLoss", **kwargs):

        super().__init__(name=name)

        self.X_true = X_true
        self.O = O

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        K = laplacian_kernel_matrix(self.X_true.shape[1])
        R = finite_difference_matrix(self.X_true.shape[1])
        self.KR = K @ R

    def _build_loss(self, M_pred, U, V, *args):
        
        # Updates to S occurs only at validation scores so must compare against U, V.
        frob_tensor = self.O * (self.X_true - M_pred)
        loss_frob = np.square(np.linalg.norm(frob_tensor, 'fro')) / np.sum(self.O)

        loss_reg1 = self.lambda1 * np.square(np.linalg.norm(U, 'fro'))
        loss_reg2 = self.lambda2 *  np.square(np.linalg.norm(V, 'fro'))
        loss_reg3 = self.lambda3 *  np.square(np.linalg.norm(self.KR @ V, 'fro'))

        return loss_frob + loss_reg1 + loss_reg2 + loss_reg3
    