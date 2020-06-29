import numpy as np
import tensorflow as tf

from .base import BaseLoss
from ..special_matrices import finite_difference_matrix, laplacian_kernel_matrix


DTYPE = tf.float32


# NB: For optimal predictive ability the loss should probably be modified to be a function 
# of predicted diagnoses rather than reconstructed samples, at least if theta is to be updated, 
# the prediction method should probably be called in reconstruction method.
def get_gdl_loss_fn(model_config, X_true, O, Lr, Lc):

    loss_fn = OriginalLoss

    if model_config.tv_gamma is not None:

        if model_config.conv_gamma is not None:
            loss_fn = TVConvolutionLoss

        else:
            loss_fn = TVLoss

    if model_config.conv_gamma is not None:
        loss_fn =  ConvolutionLoss

    print(f"Fetching loss: {loss_fn.__name__}")

    loss = loss_fn(
        X_true=X_true, O=O, Lr=Lr, Lc=Lc,
        tv_gamma=model_config.tv_gamma,
        row_gamma=model_config.row_gamma,
        col_gamma=model_config.col_gamma,
        conv_gamma=model_config.conv_gamma,
        epsilon=1e-9
    )
    return loss


class OriginalLoss(BaseLoss):

    def __init__(self, X_true, O, Lr, Lc, col_gamma, row_gamma, name="OriginalLoss", **kwargs):

        super().__init__(name=name)

        self.X_true = tf.constant(X_true, dtype=DTYPE)
        self.O = tf.constant(O, dtype=DTYPE)
        self.Lr = tf.constant(Lr, dtype=DTYPE)
        self.Lc = tf.constant(Lc, dtype=DTYPE)
        self.col_gamma = col_gamma
        self.row_gamma = row_gamma

    def _build_loss(self, M_pred, M_pred_adj, *args, **kwargs):

        frob_tensor = tf.multiply(self.O, self.X_true - M_pred_adj)
        loss_frob = tf.square(tf.norm(frob_tensor)) / tf.reduce_sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(self.X_true, self.Lc), M_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(M_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(M_pred, self.Lr, transpose_a=True), M_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(M_pred, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col


class TVLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, col_gamma, row_gamma, epsilon=1e-9, name="TVLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma = col_gamma
        self.row_gamma = row_gamma
        self.tv_gamma = tv_gamma
        self.epsilon = epsilon

    def _build_loss(self, M_pred, M_pred_adj, *args, **kwargs):

        frob_tensor = tf.multiply(self.O, self.X_true - M_pred_adj)
        loss_frob = tf.square(tf.norm(frob_tensor)) / tf.reduce_sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(self.X_true, self.Lc), M_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(M_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(M_pred, self.Lr, transpose_a=True), M_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(M_pred, out_type=DTYPE)
        
        col_diff_tensor = tf.sqrt(np.diff(V, axis=1) ** 2 + self.epsilon ** 2)
        loss_tv = self.tv_gamma * tf.reduce_sum(col_diff_tensor) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_tv 


class ConvolutionLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, col_gamma, row_gamma, name="ConvolutionLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma = col_gamma
        self.row_gamma = row_gamma
        self.conv_gamma = conv_gamma
 
        C = laplacian_kernel_matrix(self.O.shape[1])
        R = finite_difference_matrix(self.O.shape[1])
        self.CR = C @ R

    def _build_loss(self, M_pred, M_pred_adj, V, *args):
        
        frob_tensor = tf.multiply(self.O, self.X_true - M_pred_adj)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(self.X_true, self.Lc), M_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(M_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(M_pred, self.Lr, transpose_a=True), M_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(M_pred, out_type=DTYPE)

        loss_conv = self.conv_gamma * tf.square(frobenius_norm(self.CR @ V)) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_conv 


class TVConvolutionLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, tv_gamma, col_gamma, row_gamma, epsilon=1e-9, name="TVConvolutionLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma = col_gamma
        self.row_gamma = row_gamma
        self.tv_gamma = tv_gamma
        self.conv_gamma = conv_gamma
        self.epsilon = epsilon

        C = laplacian_kernel_matrix(self.O.shape[1])
        R = finite_difference_matrix(self.O.shape[1])
        self.CR = C @ R

    def _build_loss(self, M_pred, M_pred_adj, V, *args):
        
        frob_tensor = tf.multiply(self.O, self.X_true - M_pred_adj)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(self.X_true, self.Lc), M_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(M_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(M_pred, self.Lr, transpose_a=True), M_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(M_pred, out_type=DTYPE)
        
        loss_conv = self.conv_gamma * tf.square(frobenius_norm(self.CR @ V)) / tf.size(V, out_type=DTYPE)

        col_diff_tensor = tf.sqrt(np.diff(V, axis=1) ** 2 + self.epsilon ** 2)
        loss_tv = self.tv_gamma * tf.reduce_sum(col_diff_tensor) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_conv + loss_tv
