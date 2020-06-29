from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


SEED = 0
tf.random.set_seed(SEED)

DTYPE = tf.float32


def get_gdl_loss_fn(loss, Lr, Lc, col_gamma, row_gamma):

    if loss == "original":
        return OriginalLoss(Lr, Lc, col_gamma, row_gamma)

    raise ValueError(f"Unknown loss: {loss}")


class BaseLoss(ABC):

    def __init__(self, name="loss_function"):

        self.name = name

    def __call__(self, Y_true, M_pred, M_pred_adj, **kwargs):
        return self._build_loss(Y_true, M_pred, M_pred_adj, **kwargs)

    @abstractmethod
    def _build_loss(self, Y_true, M_pred, M_pred_adj):
        pass


class OriginalLoss(BaseLoss):

    def __init__(self, Lr, Lc, col_gamma, row_gamma, name="OriginalLoss", **kwargs):

        super().__init__(name=name)

        self.Lr = tf.constant(Lr, dtype=DTYPE) 
        self.Lc = tf.constant(Lc, dtype=DTYPE)
        self.col_gamma = tf.constant(col_gamma, dtype=DTYPE)
        self.row_gamma = tf.constant(row_gamma, dtype=DTYPE)

    def _build_loss(self, Y_true, M_pred, M_pred_adj, *args):
        
        O = tf.ones_like(Y_true) * tf.cast(tf.not_equal(Y_true, 1), DTYPE)
        
        frob_tensor = tf.multiply(O, Y_true - M_pred_adj)
        loss_frob = tf.square(tf.norm(frob_tensor)) / tf.reduce_sum(O)
        
        trace_col_tensor = tf.matmul(tf.matmul(M_pred, self.Lc), M_pred, transpose_b=True)
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
        self.col_gamma
        self.row_gamma
        self.tv_gamma = tv_gamma
        self.epsilon = epsilon

    def _build_loss(self, Y_true, Y_pred, V, *args):
        
        frob_tensor = tf.multiply(self.O, Y_true - self.Y_pred)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(Y_pred, self.Lc), Y_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(Y_pred, self.Lr, transpose_a=True), Y_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        col_diff_tensor = tf.sqrt(np.diff(V, axis=1) ** 2 + self.epsilon ** 2)
        loss_tv = self.tv_gamma * tf.reduce_sum(col_diff_tensor) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_tv 


class ConvolutionLoss(BaseLoss):

    def __init__(self, O, Lr, Lc, col_gamma, row_gamma, name="ConvolutionLoss", **kwargs):

        super().__init__(name=name)

        self.O = O
        self.Lr = Lr 
        self.Lc = Lc
        self.col_gamma
        self.row_gamma
        self.conv_gamma = conv_gamma
 
        C = laplacian_kernel_matrix(self.Y_true.shape[1])
        R = finite_difference_matrix(self.Y_true.shape[1])
        self.CR = C @ R

    def _build_loss(self, Y_true, Y_pred, V, *args):
        
        frob_tensor = tf.multiply(self.O, Y_true - self.Y_pred)
        loss_frob = tf.square(frobenius_norm(frob_tensor)) / np.sum(self.O)
        
        trace_col_tensor = tf.matmul(tf.matmul(Y_pred, self.Lc), Y_pred, transpose_b=True)
        loss_trace_col = self.col_gamma * tf.linalg.trace(trace_col_tensor) / tf.size(Y_pred, out_type=DTYPE)
        
        trace_row_tensor = tf.matmul(tf.matmul(Y_pred, self.Lr, transpose_a=True), Y_pred)
        loss_trace_row = self.row_gamma * tf.linalg.trace(trace_row_tensor) / tf.size(Y_pred, out_type=DTYPE)

        loss_conv = self.conv_gamma * tf.square(frobenius_norm(self.CR @ V)) / tf.size(V, out_type=DTYPE)

        return loss_frob + loss_trace_row + loss_trace_col + loss_conv 
